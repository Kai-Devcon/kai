"""The process's single always-open capture stream, fanned out to several consumers.

Capture is the raw ALSA hw:APE device with PulseAudio suspended on it (config/voice.py), and a raw
hw device admits exactly ONE opener — so a self-contained wake module that opened its own 16 kHz
mic would fail intermittently and re-trigger the pulse-grabs-the-card-at-44.1k garble. Everything
that wants microphone audio therefore comes through here.

The shape that makes it safe: the PortAudio callback stays dumb (slice, copy, enqueue) and one
worker thread does all the work — gate, resample, high-pass, then fan out to the wake engine, the
VAD and the utterance buffer. Nothing on the callback can block, and nothing here knows about
conversation states; it only knows whether capture is armed and whether the mic is gated shut.
Consumers are pulled, not pushed: the session hands in callbacks and MicStream calls them.

Split out of ai/session.py, which owns the state machine that drives it. The seam is the three
callbacks in __init__ — on_wake, on_audio, muted — so the FSM can be tested against a fake mic and
this can be driven block-by-block with no PortAudio at all (see _process_block).

The DSP primitives themselves live in ai/audio.py and the device discovery in ai/mic_device.py;
this module is the wiring between them and the one place a block of audio is allowed to be dropped.
"""

from __future__ import annotations

import os
import queue
import threading
import time

import numpy as np

from ai.audio import (
    CaptureBuffer, Decimator, FrameAssembler, HighPass, RingPreroll, WakeDetector, rms,
)
from ai.mic_device import pa_lock, refresh_devices, resolve_capture_device
from config.voice import SAMPLE_RATE
from config.wake import (
    CAPTURE_BLOCKSIZE, CAPTURE_HARD_CAP_S, CAPTURE_QUEUE_BLOCKS, MIC_HIGHPASS_HZ, MIC_INPUT_GAIN,
    MIC_REOPEN_BACKOFF_S, MIC_STALL_S, PREROLL_S,
)


class MicStream:
    """The process's single always-open capture stream, fanned out to several consumers.

    Consumers are pulled, not pushed: the session hands in callbacks and MicStream calls them from
    its worker thread. Nothing here knows about conversation states — it only knows whether capture
    is armed and whether the mic is currently gated shut.
    """

    def __init__(self, on_wake=None, on_audio=None, muted=None) -> None:
        self._on_wake = on_wake or (lambda now: None)
        self._on_audio = on_audio or (lambda pcm, now: None)
        self._muted = muted or (lambda now: False)

        self._lock = threading.Lock()
        self._stream = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._blocks: queue.Queue = queue.Queue(maxsize=CAPTURE_QUEUE_BLOCKS)

        self.rate = SAMPLE_RATE          # the rate we EMIT, always 16 kHz
        self._capture_rate = SAMPLE_RATE
        self._take_channel = 0
        self._channels = 1
        self._dtype = "int16"
        self._device = None
        self._is_i2s = False
        self._kind = "other"

        self._decim: Decimator | None = None
        # Built once, not per open: it runs at the EMIT rate (always 16 kHz), so unlike the decimator
        # it does not depend on which device won. None when MIC_HIGHPASS_HZ is 0.
        self._hpf = HighPass() if MIC_HIGHPASS_HZ > 0 else None
        self.wake = WakeDetector()
        # Guards the wake engine against being torn down mid-process(). Porcupine's handle owns native
        # memory, so freeing it while the audio worker is inside process() is a use-after-free, not just
        # a race. Held only around the framing/process block and around reload_wake() — deliberately
        # NOT the same lock as self._lock, and never taken on the PortAudio callback (_on_block only
        # enqueues), so it cannot xrun the device.
        self._wake_lock = threading.Lock()
        self._wake_frames = FrameAssembler(self.wake.frame_length)
        self._preroll = RingPreroll(PREROLL_S, SAMPLE_RATE)
        self._capture = CaptureBuffer(SAMPLE_RATE, CAPTURE_HARD_CAP_S)
        self._armed = False
        self._wake_enabled = True
        # True across the close/open window of a watchdog reopen. Without it the session sees a
        # momentarily-dead stream and declares mic_lost, ending a conversation that was fine.
        self.reopening = False

        # Counters, all published on /params — a misfire has to be diagnosable over ssh with no
        # debugger, which means every drop and every suppression is counted rather than ignored.
        self.blocks = 0
        self.dropped_blocks = 0
        self.overflows = 0
        self.muted_blocks = 0
        self.reopens = 0
        self.last_block_t = 0.0
        self.last_rms = 0.0
        self.error: str | None = None

    # ── lifecycle ───────────────────────────────────────────────────────────

    def open(self) -> bool:
        """Resolve a mic and open the stream. False (with .error set) if it can't be opened."""
        # Held across resolve AND open, not just the open: refresh_devices() re-initialises the
        # PortAudio host API, and it runs on the OTHER thread — the dashboard's re-resolve button
        # while the audio worker's stall watchdog is here, or the reverse. A re-init landing inside
        # Pa_OpenStream surfaces as "Illegal combination of I/O devices" [-9993] on an input-only
        # stream, which reads as a device fault and is not one. See pa_lock in ai/mic_device.py.
        with pa_lock:
            return self._open_locked()

    def _open_locked(self) -> bool:
        import sounddevice as sd

        # Same sequence as VoiceAssistant.ensure_input_resolved() (both call ai/mic_device):
        # route, take the card off pulse so
        # the raw hw probe can open at 48 kHz, hand pulse back if we land elsewhere.
        # Each step is logged because every one of them can block on external state (amixer, pulse,
        # ALSA device contention), and without this a hang here is indistinguishable from a hang
        # anywhere else in startup.
        # resolve_capture_device() owns the whole sequence now — the I2S route, and BOTH pulse
        # states. It has to: the pulse-mediated route needs sources UP and every raw route needs
        # them suspended, so "suspend, then resolve, then resume" could only ever serve one of them
        # and the code here used to hard-code the raw one. See S16 / resolve_capture_device.
        print("[mic] resolving input device…", flush=True)
        mic = resolve_capture_device()
        print(f"[mic] resolved device={mic.device} rate={mic.rate} ch={mic.channels} "
              f"kind={mic.kind} i2s={mic.is_i2s} — opening stream…", flush=True)

        with self._lock:
            self._device, self._capture_rate = mic.device, mic.rate
            self._channels, self._take_channel = mic.channels, mic.take_channel
            self._dtype, self._is_i2s = mic.dtype, mic.is_i2s
            self._kind = mic.kind
            # Only the raw I2S device is rate-locked to 48 kHz. USB/default go through pulse's
            # plughw, which resamples for us — so ask for 16 kHz directly and skip the decimator,
            # which deletes the 44.1 kHz non-integer-ratio problem instead of solving it.
            if mic.rate == SAMPLE_RATE:
                self._decim = None
            else:
                try:
                    self._decim = Decimator(mic.rate, SAMPLE_RATE, gain=MIC_INPUT_GAIN)
                except ValueError as exc:
                    self.error = f"cannot resample {mic.rate} Hz: {exc}"
                    print(f"[mic] ERROR: {self.error}", flush=True)
                    return False

        # Any environment the choice carries has to be in place for the OPEN, not only the probe:
        # the pulse route names its source with PULSE_SOURCE, and a stream created without it records
        # from pulse's default instead — the same card by luck, or a different one entirely. Set for
        # the life of the process on purpose, because the watchdog reopens this stream without going
        # back through resolve.
        if mic.env:
            os.environ.update(mic.env)
            print(f"[mic] capture environment: "
                  f"{', '.join(f'{k}={v}' for k, v in mic.env.items())}", flush=True)

        try:
            stream = sd.InputStream(
                samplerate=self._capture_rate, channels=self._channels, dtype=self._dtype,
                device=self._device, blocksize=CAPTURE_BLOCKSIZE, callback=self._on_block,
            )
            stream.start()
        except Exception as exc:
            self.error = f"could not open microphone: {exc}"
            print(f"[mic] ERROR: {self.error}", flush=True)
            return False

        with self._lock:
            self._stream = stream
            self.error = None
        self.last_block_t = time.monotonic()
        print(f"[mic] open: device={self._device} ({self._kind}) {self._capture_rate} Hz "
              f"x{self._channels} -> {SAMPLE_RATE} Hz"
              f"{'' if self._decim else ' (no resample)'}", flush=True)
        return True

    def start(self) -> bool:
        """Open the stream and start the fan-out worker."""
        if not self.open():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="kai-audio")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        self._close_stream()
        self.wake.close()   # Porcupine holds native memory; leaking it across restarts is a leak

    def _close_stream(self) -> None:
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def reopen(self) -> bool:
        """Tear the stream down and open it again, after a stall. Keeps the worker running."""
        with pa_lock:            # close -> refresh -> open is one atomic step; see open()
            return self._reopen_locked()

    def _reopen_locked(self) -> bool:
        self.reopens += 1
        print(f"[mic] reopening (attempt {self.reopens})", flush=True)
        self.reopening = True
        try:
            self._close_stream()
            while not self._blocks.empty():     # stale audio must not survive the gap
                try:
                    self._blocks.get_nowait()
                except queue.Empty:
                    break
            # The only safe window for this in the whole process: PortAudio's device list is a
            # snapshot taken at Pa_Initialize, and refreshing it means tearing the host API down and
            # building it again — which invalidates open streams. Ours is closed, right here, and
            # nowhere else. Without it a re-resolve re-runs the entire discovery path against a list
            # that predates the mic being plugged in, which is why the dashboard button could not
            # find new hardware and why hot-plug would not have worked either.
            refresh_devices()
            ok = self.open()
            if ok:
                self.reset_dsp()
            return ok
        finally:
            self.reopening = False

    # ── the PortAudio callback: stays dumb, never blocks ────────────────────

    def _on_block(self, indata, frames, time_info, status) -> None:
        if status is not None and getattr(status, "input_overflow", False):
            self.overflows += 1
        col = self._take_channel
        # .copy() is required — PortAudio reuses indata's buffer once we return.
        chunk = indata[:, col:col + 1].copy().ravel()
        try:
            self._blocks.put_nowait(chunk)
        except queue.Full:
            # Drop the oldest and keep the newest: for wake-word spotting, fresh audio is worth more
            # than complete audio, and blocking here would xrun the device.
            self.dropped_blocks += 1
            try:
                self._blocks.get_nowait()
                self._blocks.put_nowait(chunk)
            except (queue.Empty, queue.Full):
                pass

    # ── the fan-out worker ──────────────────────────────────────────────────

    def _worker(self) -> None:
        backoff_i = 0
        while not self._stop.is_set():
            try:
                chunk = self._blocks.get(timeout=0.2)
            except queue.Empty:
                # Nothing arriving. If pulse re-grabbed the suspended APE card (a settings panel, a
                # stray pactl, a USB re-enumeration) the stream goes silent with NO exception, so
                # without this Kai would go permanently deaf with no signal at all.
                now = time.monotonic()
                if now - self.last_block_t > MIC_STALL_S and not self._stop.is_set():
                    delay = MIC_REOPEN_BACKOFF_S[min(backoff_i, len(MIC_REOPEN_BACKOFF_S) - 1)]
                    print(f"[mic] WARNING: no audio for {now - self.last_block_t:.1f}s — "
                          f"reopening in {delay:.0f}s", flush=True)
                    if self._stop.wait(delay):
                        return
                    if self.reopen():
                        backoff_i = 0
                        self.last_block_t = time.monotonic()
                    else:
                        backoff_i += 1
                continue

            backoff_i = 0
            self._process_block(chunk, time.monotonic())

    def _process_block(self, chunk: np.ndarray, now: float) -> None:
        """Gate, resample, and fan one captured block out to the wake word, the VAD and the buffer.

        Split out of _worker so the fan-out can be driven directly in tests with no device, no
        PortAudio and no thread."""
        self.last_block_t = now
        self.blocks += 1

        # The self-hearing gate, enforced in exactly one place and BEFORE any DSP: no resample, no
        # Porcupine, no VAD, no append. Skipping the decimation saves the CPU too.
        if self._muted(now):
            self.muted_blocks += 1
            return

        pcm = self._decim.feed(chunk) if self._decim else chunk.astype(np.int16)
        if pcm.size == 0:
            return
        # After decimation, before anything measures or stores it — so the wake engine, the VAD and
        # the audio Whisper transcribes all see the same de-rumbled signal. See MIC_HIGHPASS_HZ.
        if self._hpf is not None:
            pcm = self._hpf.feed(pcm)
        # DC-blocked, deliberately: this number is published as sess_rms and is what VAD_RMS_FLOOR
        # gets tuned against, so it MUST be the same quantity SpeechGate compares to that floor.
        # Measured raw it reads roughly 2x higher on this mic (the INMP441 carries a large standing
        # offset), which made the floor look comfortably clear while not a single frame passed it.
        self.last_rms = rms(pcm.astype(np.float64) - float(np.mean(pcm)))

        # frame_ready, not ready: the utterance tier is "ready" but has no frames to push, and
        # framing blocks to feed a no-op would be pure waste.
        if self._wake_enabled and self.wake.frame_ready:
            with self._wake_lock:
                # Re-check under the lock: a reload may have closed the engine while we waited.
                if self.wake.frame_ready:
                    if self._wake_frames.size != self.wake.frame_length:
                        # Self-heal. The assembler is sized in __init__, before open() knows which tier
                        # won and therefore what frame size it wants (512 for Porcupine, 1280 for
                        # openWakeWord). One integer compare per block makes that ordering bug
                        # structurally impossible rather than something a future refactor must remember.
                        self.sync_wake_geometry()
                    # Count the hits rather than stopping at the first: openWakeWord keeps a rolling
                    # feature history, so every frame must be fed, and the caller's contract is one
                    # _on_wake per firing frame (the session's WAKE_REFRACTORY_S is what collapses
                    # them into a single wake).
                    fired = sum(1 for frame in self._wake_frames.push(pcm)
                                if self.wake.process(frame))
                else:
                    fired = 0
            # Outside the lock: _on_wake runs the session's own machinery (and takes its lock), so
            # holding the wake lock across it would invert the lock order against reload_wake().
            for _ in range(fired):
                self._on_wake(now)

        # ── the fan-out splits here ──────────────────────────────────────────
        # Everything above this line is shared. Below it there are two consumers with genuinely
        # different needs, and _asr_signal is the seam between them. See its docstring.
        asr = self._asr_signal(pcm)
        self._preroll.push(asr)
        with self._lock:
            armed = self._armed
        if armed:
            self._capture.push(asr)
        # `pcm`, not `asr`: the VAD keeps the un-enhanced signal (see _asr_signal).
        self._on_audio(pcm, now)

    def _asr_signal(self, pcm: np.ndarray) -> np.ndarray:
        """The branch of the fan-out that ends up in Whisper. Identity today.

        The seam is here rather than added later because the split is structural, not speculative.
        Noise suppression — spectral subtraction, RNNoise, speex, anything of that family — reliably
        helps ASR on a noisy room and reliably HURTS acoustic wake-word models, which were trained
        on unprocessed audio and see a denoiser's artefacts as a different signal. One buffer cannot
        serve both, so the two paths have to diverge before any of it can be tried.

        The VAD stays on the un-enhanced side too, for a separate reason: VAD_RMS_FLOOR and
        VAD_RMS_FLOOR_HOLD are absolute int16 levels, measured on this exact chain (config/wake.py
        documents the room and the date), and sess_rms is published as the number they are tuned
        against. Putting a gain-changing stage in front of the gate would invalidate both at once
        and the symptom would be a floor that no longer means anything, not an error.

        Rules for anything added here:
          * It must be block-continuous — carried state, overlap-save — exactly like Decimator and
            HighPass. A stateless per-block filter injects a discontinuity every 32 ms; see the
            Decimator docstring for why that is worse than it sounds.
          * It must be reset in reset_dsp(), or residue from before a mute rings into the first
            frames back.
          * It must not change length, or the pre-roll arithmetic stops lining up with the capture.
        """
        return pcm

    # ── consumer controls (called from the session thread) ──────────────────

    def sync_wake_geometry(self) -> None:
        """Re-size the wake frame assembler to whatever the tier that actually WON reports.

        Must happen after wake.open(): the frame length isn't known until then, and a stale size
        means the engine is fed wrong-shaped frames and silently never fires — with sess_wake_ok
        still reading True, which is the nastiest way for this to fail."""
        with self._lock:
            want = max(1, self.wake.frame_length)
            if self._wake_frames.size != want:
                self._wake_frames = FrameAssembler(want)

    def reset_dsp(self) -> None:
        """Drop every scrap of buffered/filtered audio. Called when the mic un-mutes: pre-mute
        residue and half-frames would otherwise trip a speech onset on the first frame back."""
        if self._decim is not None:
            self._decim.reset()
        if self._hpf is not None:
            self._hpf.reset()          # 255 taps of ring-out would otherwise trip the first onset
        self._wake_frames.reset()
        self._preroll.reset()
        # openWakeWord keeps its own audio/feature history, which must go for the same reason.
        self.wake.reset()

    def set_wake_enabled(self, enabled: bool) -> None:
        self._wake_enabled = enabled

    def set_wake_sensitivity(self, value: float) -> bool:
        """Retune the wake word. Returns True if the engine was reloaded to apply it.

        Called from a Flask request thread, which is why the reload is guarded: see reload_wake.
        """
        if not self.wake.set_sensitivity(value):
            return False        # the live tier compares per frame — nothing to reopen
        self.reload_wake()
        return True

    def reload_wake(self) -> None:
        """Close and reopen the wake engine, safely, while audio keeps flowing.

        The audio worker is inside wake.process() at 30+ blocks a second, and close() frees Porcupine's
        native handle — so this MUST hold _wake_lock. create() costs ~10-50 ms, which delays at most a
        block or two; those are already counted as sess_blocks_dropped / sess_audio_overflows. Wake
        detection is disabled across the window so no frame reaches a half-built engine.
        """
        was_enabled = self._wake_enabled
        self._wake_enabled = False
        try:
            with self._wake_lock:
                self.wake.close()
                if not self.wake.open():
                    print(f"[mic] WARNING: wake engine did not come back after a reload "
                          f"({self.wake.unavailable}) — push-to-talk is unaffected", flush=True)
                    return
            # After open(), never before: the winning tier decides the frame size.
            self.sync_wake_geometry()
        finally:
            self._wake_enabled = was_enabled and self.wake.ready

    def arm_utterance(self, preroll: bool = False) -> bool:
        """Begin buffering audio into the utterance buffer. `preroll` seeds it with the ring, so
        speech that arrived before the VAD confirmed onset isn't clipped off the front."""
        with self._lock:
            if self._stream is None:
                return False
            self._capture.reset()
            self._armed = True
        if preroll:
            self._capture.prepend(self._preroll.take())
        return True

    def harvest_utterance(self) -> tuple[np.ndarray, int]:
        """Stop buffering and return (audio, rate). Always 16 kHz mono int16."""
        with self._lock:
            self._armed = False
        return self._capture.take(), SAMPLE_RATE

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    @property
    def capture_seconds(self) -> float:
        return self._capture.seconds

    @property
    def capture_truncated(self) -> bool:
        return self._capture.truncated

    @property
    def live(self) -> bool:
        with self._lock:
            return self._stream is not None

    # Which mic actually won, published so /audio/reresolve can report what it landed on and the
    # dashboard can say "I2S" or "fallback" rather than just "ok". Read under the lock because
    # open() writes them from the session-start thread while a request thread may be reading.
    #
    # Named `capture_rate`, NOT `rate`: `self.rate` is already taken and means the opposite end of
    # the pipeline — the rate this class EMITS, always 16 kHz. Shadowing it with the rate the
    # hardware is opened at would be a genuinely confusing bug to chase.
    @property
    def device(self) -> int | None:
        with self._lock:
            return self._device

    @property
    def capture_rate(self) -> int:
        with self._lock:
            return self._capture_rate

    @property
    def is_i2s(self) -> bool:
        with self._lock:
            return self._is_i2s

    # Which mic this is — "i2s", "usb" or "other". Separate from is_i2s, which answers the narrower
    # operational question "does this device need pulse suspended". The dashboard needs to name the
    # mic now that USB is a choice rather than a fallback, and "not I2S" is not a name.
    @property
    def kind(self) -> str:
        with self._lock:
            return self._kind
