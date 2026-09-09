"""MicStream's worker, driven directly — no PortAudio, no device, no state machine.

Moved out of tests/test_session.py with ai/mic_stream.py. _process_block is the seam these use:
it is the whole fan-out (gate, resample, high-pass, wake, VAD, capture) with the audio handed in
rather than arriving from a callback.
"""

import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ai.audio import FrameAssembler
from ai.mic_device import MicChoice
from ai.mic_stream import MicStream
from config.wake import CAPTURE_QUEUE_BLOCKS

T0 = 1000.0


class TestMicStreamFanOut(unittest.TestCase):
    """MicStream's worker, driven directly — no PortAudio, no device."""

    def _stream(self, muted=False, wake_hit=False):
        events = {"wakes": [], "audio": []}
        mic = MicStream(
            on_wake=lambda now: events["wakes"].append(now),
            on_audio=lambda pcm, now: events["audio"].append(pcm),
            muted=lambda now: muted,
        )
        mic._stream = object()          # pretend open
        mic._decim = None               # 16 kHz already
        mic.wake = MagicMock()
        mic.wake.ready = True
        mic.wake.frame_length = 512
        mic.wake.process.return_value = wake_hit
        mic._wake_frames = FrameAssembler(512)
        return mic, events

    def _pump(self, mic, blocks, now=T0):
        """Push blocks through the callback, then run the fan-out on each — same path the worker
        thread takes, without needing a thread."""
        for b in blocks:
            mic._on_block(b.reshape(-1, 1), len(b), None, None)
        while not mic._blocks.empty():
            mic._process_block(mic._blocks.get_nowait(), now)

    def test_gated_blocks_are_counted_and_skip_all_dsp(self):
        mic, events = self._stream(muted=True)
        self._pump(mic, [np.ones(1536, dtype="int16")])
        self.assertEqual(mic.muted_blocks, 1)
        self.assertEqual(events["audio"], [])
        mic.wake.process.assert_not_called()

    def test_ungated_blocks_reach_both_consumers(self):
        mic, events = self._stream(muted=False)
        self._pump(mic, [np.ones(1536, dtype="int16")])
        self.assertEqual(len(events["audio"]), 1)
        mic.wake.process.assert_called()

    def test_wake_detection_is_reported(self):
        mic, events = self._stream(muted=False, wake_hit=True)
        self._pump(mic, [np.ones(1536, dtype="int16")])
        self.assertEqual(len(events["wakes"]), 3, "1536 samples = 3 wake frames")

    def test_the_asr_branch_reaches_the_capture_but_not_the_vad_or_the_wake_word(self):
        """The fan-out split, asserted rather than assumed.

        _asr_signal is the identity today, so nothing observable depends on it yet — which is
        exactly why it needs a test. The moment a denoiser goes in there, this is what catches it
        leaking onto the wake path (models trained on unprocessed audio) or the VAD path (whose
        floors are absolute int16 levels tuned against the un-enhanced signal)."""
        mic, events = self._stream(muted=False)
        mic._armed = True
        # The high-pass would take a constant block to zero, which says nothing about routing.
        # Off here so each consumer's input is literally the block that was pushed in.
        mic._hpf = None
        mic._asr_signal = lambda pcm: np.full_like(pcm, 7)

        self._pump(mic, [np.ones(1536, dtype="int16")])

        captured, _rate = mic.harvest_utterance()
        self.assertTrue(np.all(captured == 7), "the capture buffer must get the ASR branch")
        self.assertTrue(np.all(events["audio"][0] == 1), "the VAD must get the raw signal")
        frame = mic.wake.process.call_args[0][0]
        self.assertTrue(np.all(frame == 1), "the wake engine must get the raw signal")

    def test_asr_signal_is_the_identity_today(self):
        mic, _ = self._stream()
        block = np.arange(64, dtype="int16")
        np.testing.assert_array_equal(mic._asr_signal(block), block)

    def test_callback_drops_oldest_when_the_queue_is_full(self):
        # Blocking in the PortAudio callback would xrun the device; stale audio is worth less than
        # fresh audio for wake spotting.
        mic, _ = self._stream()
        for _ in range(CAPTURE_QUEUE_BLOCKS + 5):
            mic._on_block(np.ones((1536, 1), dtype="int16"), 1536, None, None)
        self.assertGreater(mic.dropped_blocks, 0)
        self.assertLessEqual(mic._blocks.qsize(), CAPTURE_QUEUE_BLOCKS)

    def test_callback_takes_only_the_live_channel_and_copies(self):
        mic, _ = self._stream()
        mic._take_channel = 0
        indata = np.zeros((10, 2), dtype="int16")
        indata[:, 0] = np.arange(10)
        indata[:, 1] = 999
        mic._on_block(indata, 10, None, None)
        got = mic._blocks.get_nowait()
        np.testing.assert_array_equal(got, np.arange(10))
        indata[:] = 0
        np.testing.assert_array_equal(got, np.arange(10), "PortAudio reuses its buffer")

    def test_overflow_is_counted(self):
        mic, _ = self._stream()
        status = MagicMock()
        status.input_overflow = True
        # PortAudio's callback signature is (indata, frames, time, status) — status is 4th.
        mic._on_block(np.ones((1536, 1), dtype="int16"), 1536, None, status)
        self.assertEqual(mic.overflows, 1)

    def test_arm_and_harvest_round_trip(self):
        mic, _ = self._stream(muted=False)
        mic.arm_utterance(preroll=False)
        self._pump(mic, [np.ones(1536, dtype="int16")])
        audio, rate = mic.harvest_utterance()
        self.assertEqual(rate, 16000)
        self.assertEqual(audio.size, 1536, "this fake stream is already 16 kHz — no decimation")
        self.assertFalse(mic.armed)

    def test_arm_fails_when_the_stream_is_closed(self):
        mic, _ = self._stream()
        mic._stream = None
        self.assertFalse(mic.arm_utterance())

    def test_nothing_is_buffered_until_armed(self):
        # With the mic always open, an unconditional append would leak within minutes.
        mic, _ = self._stream(muted=False)
        self._pump(mic, [np.ones(1536, dtype="int16")])
        audio, _ = mic.harvest_utterance()
        self.assertEqual(audio.size, 0)

    def test_frame_size_follows_the_winning_engine(self):
        """Regression: MicStream.__init__ sizes the assembler before wake.open() knows which tier
        won. Porcupine is 512 both times so it worked by luck; openWakeWord wants 1280 and used to
        be fed 512-sample frames forever — scores pinned near zero, no wake ever, sess_wake_ok True.
        """
        mic, _ = self._stream(muted=False)
        self.assertEqual(mic._wake_frames.size, 512, "constructed before open(), so the default")

        seen = []
        mic.wake = MagicMock()
        mic.wake.ready = True
        mic.wake.frame_ready = True
        mic.wake.frame_length = 1280            # as if openWakeWord won
        mic.wake.process.side_effect = lambda f: seen.append(len(f)) or False

        # No explicit sync call: the hot path must self-heal.
        self._pump(mic, [np.ones(1536, dtype="int16"), np.ones(1536, dtype="int16")])
        self.assertEqual(mic._wake_frames.size, 1280)
        self.assertTrue(seen, "the engine must actually be fed")
        self.assertTrue(all(n == 1280 for n in seen), f"wrong frame sizes: {seen}")

    def test_sync_wake_geometry_is_explicit_too(self):
        mic, _ = self._stream()
        mic.wake = MagicMock()
        mic.wake.frame_length = 1280
        mic.sync_wake_geometry()
        self.assertEqual(mic._wake_frames.size, 1280)

    def test_sync_wake_geometry_never_sizes_to_zero(self):
        # FrameAssembler(0) divides by zero on every push.
        mic, _ = self._stream()
        mic.wake = MagicMock()
        mic.wake.frame_length = 0
        mic.sync_wake_geometry()
        self.assertGreaterEqual(mic._wake_frames.size, 1)

    def test_utterance_tier_is_never_fed_frames(self):
        mic, _ = self._stream(muted=False)
        mic.wake = MagicMock()
        mic.wake.ready = True
        mic.wake.frame_ready = False           # utterance tier
        mic.wake.frame_length = 512
        self._pump(mic, [np.ones(1536, dtype="int16")])
        mic.wake.process.assert_not_called()

    def test_reset_dsp_also_resets_the_engine(self):
        mic, _ = self._stream()
        mic.wake = MagicMock()
        mic.wake.frame_length = 512
        mic.reset_dsp()
        mic.wake.reset.assert_called_once()

    def test_preroll_seeds_the_utterance(self):
        mic, _ = self._stream(muted=False)
        self._pump(mic, [np.full(1536, 7, dtype="int16")])
        mic.arm_utterance(preroll=True)
        audio, _ = mic.harvest_utterance()
        self.assertGreater(audio.size, 0, "speech before the VAD confirmed onset must survive")


class TestReopenRescansDevices(unittest.TestCase):
    """reopen() is the one place in the process where PortAudio's device list can be refreshed.

    PortAudio snapshots that list at Pa_Initialize and never updates it, so a card plugged in after
    startup is invisible for the life of the process — which is why the dashboard's "find the
    microphone again" button could re-run the entire discovery path and still not find new hardware.
    Refreshing means tearing the host API down and building it again, which invalidates open streams,
    so the only safe window is between closing the old one and opening the new one.
    """

    def _mic(self):
        mic = MicStream()
        mic._stream = object()
        return mic

    def test_the_device_list_is_refreshed_before_the_new_stream_is_opened(self):
        calls = []
        with patch("ai.mic_stream.refresh_devices", side_effect=lambda: calls.append("refresh")), \
             patch.object(MicStream, "_close_stream",
                          side_effect=lambda: calls.append("close")) as _c, \
             patch.object(MicStream, "open", side_effect=lambda: calls.append("open") or True):
            self._mic().reopen()
        self.assertEqual(calls, ["close", "refresh", "open"])

    def test_no_stream_is_open_while_the_rescan_runs(self):
        # The invariant that makes the rescan safe. Asserted on the object rather than on call
        # order, because "close was called first" and "nothing is open" are not the same claim.
        seen = {}
        mic = self._mic()
        with patch("ai.mic_stream.refresh_devices",
                   side_effect=lambda: seen.update(stream=mic._stream)), \
             patch.object(MicStream, "open", return_value=True):
            mic.reopen()
        self.assertIsNone(seen["stream"])

    def test_a_failed_rescan_does_not_stop_the_reopen(self):
        # A stale device list means the newly plugged mic may still be invisible, not that recovery
        # should be abandoned — whatever the old list still offers is better than no stream.
        with patch("ai.mic_stream.refresh_devices", return_value=False), \
             patch.object(MicStream, "open", return_value=True) as opened:
            self.assertTrue(self._mic().reopen())
        opened.assert_called_once()

class TestPulseIsKeptOffTheCardBeingOpened(unittest.TestCase):
    """open() must not hand the chosen mic's card back to pulse on its way to opening it.

    pulse re-grabs a resumed card immediately on this build (no module-suspend-on-idle), so the
    resume that used to run here for every non-I2S device took the USB mic away between the probe
    that found it live and the open that needed it: "Device unavailable" [-9985], every attempt,
    forever, on 2026-08-27. `is_i2s` was the wrong test - hw:3,0 is raw and exclusive either way.

    open() now delegates the whole resolve — route, suspend, probe, and the keep_card-aware resume
    — to resolve_capture_device() (ai/mic_device.py, S16), so these assert on the MicChoice it
    hands back rather than on the individual steps it used to inline. The keep_card behavior
    itself is covered directly in tests/test_mic_device.py::TestTwoPhaseResolve.
    """

    def _open(self, choice):
        sd = MagicMock()
        with (
            patch("ai.mic_stream.resolve_capture_device", return_value=choice),
            patch.dict("sys.modules", {"sounddevice": sd}),
        ):
            ok = MicStream().open()
        return ok, sd

    def test_a_raw_usb_mic_keeps_its_own_card_suspended(self):
        ok, sd = self._open(MicChoice(25, 48000, 1, 0, "int16", False, "usb", "3"))
        self.assertTrue(ok)
        self.assertEqual(sd.InputStream.call_args.kwargs["device"], 25)

    def test_a_raw_i2s_mic_keeps_its_own_card_suspended(self):
        ok, sd = self._open(MicChoice(6, 48000, 2, 0, "int16", True, "i2s", "APE"))
        self.assertTrue(ok)
        self.assertEqual(sd.InputStream.call_args.kwargs["device"], 6)

    def test_a_pulse_mediated_device_hands_every_card_back(self):
        ok, sd = self._open(MicChoice(None, 16000, 1, 0, "int16", False))
        self.assertTrue(ok)
        self.assertIsNone(sd.InputStream.call_args.kwargs["device"])


class TestPortAudioReinitCannotLandInsideAnOpen(unittest.TestCase):
    """refresh_devices() and open() are on different threads and must never overlap.

    The stall watchdog reopens from the audio worker while the dashboard's re-resolve button can be
    opening from the HTTP thread. Terminating the PortAudio host API mid-Pa_OpenStream does not fail
    cleanly: it surfaces as "Illegal combination of I/O devices" [-9993] - an error about duplex
    device pairs, raised on a stream that has no output side - which is what sent the 2026-08-27
    investigation looking for a hardware fault that was not there.
    """

    def test_another_thread_cannot_reinit_portaudio_while_open_runs(self):
        import ai.mic_device as mic_device
        blocked = {}

        def resolve():
            # What refresh_devices() would be doing, from where it would be doing it: another
            # thread. Re-entrancy makes the owning thread the wrong place to ask.
            def probe():
                got = mic_device.pa_lock.acquire(blocking=False)
                blocked["locked_out"] = not got
                if got:
                    mic_device.pa_lock.release()
            t = threading.Thread(target=probe)
            t.start()
            t.join(5)
            return MicChoice(None, 16000, 1, 0, "int16", False)

        with (
            patch("ai.mic_stream.resolve_capture_device", side_effect=resolve),
            patch.dict("sys.modules", {"sounddevice": MagicMock()}),
        ):
            MicStream().open()
        self.assertTrue(blocked["locked_out"],
                        "open() must hold pa_lock across resolve and Pa_OpenStream")

if __name__ == "__main__":
    unittest.main()
