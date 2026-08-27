import json
import os
import random
import shutil
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from ai import filler
from ai import session as sess_mod
from ai.session import (
    STATE_ACK, STATE_BUSY, STATE_COOLDOWN, STATE_DISABLED, STATE_IDLE,
    STATE_LISTEN_SPEECH, STATE_LISTEN_WAIT, STATE_SCAN_CHECK, STATE_SCAN_SPEECH, STATE_SPEAKING,
    ConversationSession,
)
from ai.voice_assistant import STATUS_RECORDING
import settings
from config.filler import (
    BANK_LINE_RETRIES, BANK_PASSES, FILLER_DELAY_JITTER_S, FILLER_MAX_LINE_S, FILLER_MAX_SILENCE_S,
    FILLER_MAX_STALL_S, FILLER_MIN_GAP_S, FILLER_PLAYBACK_START_BUDGET_S,
)
from config.thinking import THINKING_SOUND_DELAY_S, THINKING_SOUND_TEXT
from config.voice import MIC_HOTPLUG_COOLDOWN_S
from config.wake import (
    GREETING_TEXT,
    MAX_UTTERANCE_S, MIN_UTTERANCE_S, SESSION_BUSY_MAX_S, SESSION_MAX_ERROR_STREAK,
    SESSION_MAX_NO_SPEECH_STREAK, SESSION_NO_FACE_S, SESSION_NO_SPEECH_S,
    SESSION_SPEAK_GRACE_S, SESSION_SPEAK_MAX_UNKNOWN_S,
    SESSION_START_ATTEMPTS, SESSION_START_BACKOFF_S, WAKE_ACK_MAX_S,
    VAD_HANGOVER_S, VAD_RMS_FLOOR, VAD_RMS_FLOOR_HOLD, WAKE_REFRACTORY_S, WAKE_SCAN_HANGOVER_S,
    WAKE_WHISPER_CHECK_MAX_S, WAKE_WHISPER_COOLDOWN_S,
    WAKE_WHISPER_MAX_UTTERANCE_S, WAKE_WHISPER_MIN_UTTERANCE_S,
)

T0 = 1000.0


class FakeMic:
    """Stands in for MicStream. Records arm/harvest so capture lifecycle is observable."""

    def __init__(self, wake_ready=True, wake_kind="frame", wake_frame=512,
                 wake_engine="porcupine"):
        self.wake = MagicMock()
        self.wake.ready = wake_ready
        self.wake.frame_ready = wake_ready and wake_kind == "frame"
        self.wake.unavailable = None
        # Concrete, not MagicMocks: these land on /params, which is a JSON stream.
        self.wake.kind = wake_kind
        self.wake.engine = wake_engine if wake_ready else ""
        self.wake.frame_length = wake_frame
        self.wake.last_score = 0.0
        self.wake.sensitivity = 0.5
        self.wake.tiers = {wake_engine: "ok"} if wake_ready else {}
        self.live = True
        self.reopening = False
        self.kind = "i2s"          # concrete for the same reason as wake.kind: it lands on /params
        self.last_rms = 0.0
        self.muted_blocks = 0
        self.reopens = 0
        self.overflows = 0
        self.dropped_blocks = 0
        self.capture_seconds = 0.0
        self.capture_truncated = False
        self.armed = False
        self.arms = 0
        self.harvests = 0
        self.dsp_resets = 0
        self.wake_enabled = None
        self.wake_sensitivities: list = []
        self.wake_reload_needed = False
        self.audio = np.ones(16000, dtype="int16")
        self.started = True
        self.geometry_syncs = 0
        # What resolution landed on, reported by /audio/reresolve so the operator can see WHICH mic
        # came back rather than just that something did.
        self.device = 5
        self.capture_rate = 48000
        self.is_i2s = True
        self.reopens_requested = 0
        self.reopen_ok = True

    def start(self):
        return self.started

    def reopen(self):
        self.reopens_requested += 1
        return self.reopen_ok

    def stop(self):
        pass

    def set_wake_enabled(self, enabled):
        self.wake_enabled = enabled

    def set_wake_sensitivity(self, value):
        """Records the request and reports whether an engine reload was needed, like the real one."""
        self.wake_sensitivities.append(value)
        return self.wake_reload_needed

    def sync_wake_geometry(self):
        self.geometry_syncs += 1

    def arm_utterance(self, preroll=False):
        self.armed = True
        self.arms += 1
        return True

    def harvest_utterance(self):
        self.armed = False
        self.harvests += 1
        return self.audio, 16000

    def reset_dsp(self):
        self.dsp_resets += 1


class FakeVoice:
    """Stands in for VoiceAssistant: epoch bookkeeping plus records of what was spoken."""

    def __init__(self):
        self._epoch = 0
        self.stt_ready = True               # the turn model is loaded
        self.scan_ready = True              # the tiny wake-scan model is pre-warmed
        self.speaking = False
        self.audio_end = None               # monotonic end of the audio playing, None = unknown
        self.speech_cuts = 0                # times stop_speech() cut a line short
        self.gate_open = True               # mic_muted() returns `not gate_open`
        self.turns = []
        self.spoken = []
        self.spoken_wavs = []
        self.history_resets = 0
        self.turn_status_clears = 0
        # Who the assistant believes it is talking to (ai/identity.py). Published as sess_person and
        # forgotten by reset_history() with everything else the next person must not inherit.
        self.person_name = None
        self.status = "idle"        # the assistant's own per-turn status
        self.attached = None
        self.turn_result = {"status": "ok"}
        self.last_on_done = None
        # Whisper wake tier seams.
        self.transcribes = []
        self.said = []
        self.say_result = {"status": "ok"}
        self.last_scan_on_done = None

    # epoch
    @property
    def epoch(self):
        return self._epoch

    def bump_epoch(self):
        self._epoch += 1
        return self._epoch

    def attach_mic(self, mic):
        self.attached = mic

    def reset_history(self):
        self.history_resets += 1
        self._epoch += 1
        self.person_name = None

    def clear_turn_status(self):
        self.turn_status_clears += 1
        if self.status in ("done", "error"):
            self.status = "idle"

    def get_status(self):
        return {"voice_status": self.status}

    # speech
    def speech_in_flight(self):
        return self.speaking

    def audio_ends_at(self):
        # None = "length not known", which is the real assistant's answer for the whole synthesis
        # window and for a silent pantomime. Tests that exercise the duration-sized speak deadline
        # set this to an absolute time on the fake clock.
        return getattr(self, "audio_end", None)

    def stop_speech(self):
        # Deliberately does NOT clear `speaking`: the real assistant releases _tts_active from the
        # cut worker's finally, under a generation check, so that a cut cannot report silence on
        # behalf of the line that replaced it.
        self.speech_cuts += 1
        self.audio_end = None

    def last_language(self):
        # Whisper's label for the previous utterance, which the filler bank reads to pick a
        # language. Tests that care set self.language; the default matches a fresh boot.
        return getattr(self, "language", "")

    def mic_muted(self, now=None):
        return not self.gate_open

    def speak_text(self, text, epoch=None):
        self.spoken.append(text)
        self.speaking = True

    def speak_wav(self, wav, jaw_text, epoch=None):
        self.spoken_wavs.append((wav, jaw_text))
        self.speaking = True

    # turns
    def process_utterance(self, audio, rate, epoch=None, on_done=None):
        self.turns.append({"samples": int(audio.size), "rate": rate, "epoch": epoch})
        self.last_on_done = on_done
        return dict(self.turn_result)

    # whisper wake tier
    def transcribe_async(self, audio, rate, on_done, token=None, log_language=True):
        self.transcribes.append({"samples": int(audio.size), "rate": rate, "token": token,
                                 "log_language": log_language})
        self.last_scan_on_done = on_done   # the test invokes it, so the fake clock stays in charge

    def say(self, text, use_llm=True, epoch=None, on_done=None):
        self.said.append({"text": text, "use_llm": use_llm, "epoch": epoch})
        self.last_on_done = on_done
        return dict(self.say_result)


class FakePresence:
    """(visible, since, is_fresh) with all three independently settable."""

    def __init__(self, visible=False, is_fresh=True):
        self.visible = visible
        self.is_fresh = is_fresh
        self.since = 0.0
        self.raise_on_call = False

    def __call__(self, now):
        if self.raise_on_call:
            raise RuntimeError("camera thread died")
        return self.visible, self.since, self.is_fresh


def _lock_probing_recorder(session, held):
    """A recorder double whose record() reports whether the session lock was held when it ran.

    The probe runs on ANOTHER thread on purpose: session._lock is an RLock, so acquiring it from
    the calling thread would succeed whether or not the bug is present and the test would pass
    vacuously. Acquire and release both happen on the probe thread — RLock ownership is per-thread,
    so releasing from anywhere else raises.
    """
    def _record(*_args, **_kwargs):
        got = []

        def _probe():
            acquired = session._lock.acquire(blocking=False)
            got.append(acquired)
            if acquired:
                session._lock.release()

        probe = threading.Thread(target=_probe)
        probe.start()
        probe.join(timeout=2.0)
        held.append(not (got and got[0]))
        return "clip"

    return SimpleNamespace(record=_record, annotate=lambda *a, **k: None,
                           status=lambda: {"enabled": False, "written": 0, "skipped": 0,
                                           "mb": 0.0, "error": ""})


class FakeWatcher:
    """Stands in for CardWatcher. `fire` is a one-shot: set it, and the next poll reports a change.

    Every session test gets one of these, not a real watcher — partly so the fake clock stays in
    charge, and partly because the real one reads /proc/asound/cards, which does not exist on a dev
    box and would have every test in this module log itself disabled."""

    def __init__(self, active=True):
        self.active = active
        self.fire = False
        self.changes = 0
        self.polls = []

    def poll(self, now):
        self.polls.append(now)
        if not (self.active and self.fire):
            return False
        self.fire = False
        self.changes += 1
        return True


class SessionCase(unittest.TestCase):
    """Every test drives session.tick(now) with a fake clock — no threads, no sleeping."""

    def setUp(self):
        # tts is a module-level singleton reached directly by the session for stop()/is_playing().
        for name, kw in (("stop", {}), ("is_playing", {"return_value": False}),
                         ("quiet_since", {"return_value": float("inf")}),
                         ("prewarm_canned", {"return_value": {}})):
            p = patch(f"ai.session.tts.{name}", **kw)
            setattr(self, f"mock_{name}", p.start())
            self.addCleanup(p.stop)
        # _prewarm_bank paces itself with REAL sleeps: 0.3 s between lines, and up to 40 x 0.25 s
        # waiting for quiet, across 52 lines x 3 passes. That pacing exists to keep Piper off the
        # CPU the vision loop and Ollama share; against a mocked synth it buys nothing and costs
        # ~36 s per test that reaches a re-warm, which is most of what made this module take
        # twenty minutes. Patched here rather than per-test so no future test pays it either.
        for name, value in (("BANK_SYNTH_GAP_S", 0.0), ("BANK_QUIET_POLL_S", 0.0),
                            ("BANK_QUIET_WAIT_TRIES", 1), ("GREETING_POLL_S", 0.0)):
            p = patch(f"ai.session.{name}", value)
            p.start()
            self.addCleanup(p.stop)
        # The startup greeting is OFF for every test that doesn't ask for it. _speak_greeting waits
        # for the audio to finish, and FakeVoice.speak_text latches `speaking` True with nothing to
        # clear it — so leaving it on would park _warm_all for GREETING_QUIET_WAIT_S and then hand
        # _prewarm_bank a session that never looks quiet. TestGreeting turns it on and drives the
        # flag itself.
        p = patch("ai.session.GREETING_ENABLED", False)
        p.start()
        self.addCleanup(p.stop)
        # The cross-process greeting latch is a FILE (see GREETING_REPEAT_SUPPRESS_S). Pointed at a
        # fresh temp path for every test, because the real one lives in /tmp and is shared with the
        # robot: a test that wrote it would suppress the greeting on the next boot, and a test that
        # read it would pass or fail depending on how recently Kai had been restarted.
        self._greet_stamp_dir = tempfile.mkdtemp(prefix="kai-greet-stamp-")
        self.greet_stamp = os.path.join(self._greet_stamp_dir, "kai_greeted.stamp")
        self.addCleanup(shutil.rmtree, self._greet_stamp_dir, True)
        p = patch("ai.session.GREETING_STAMP_PATH", self.greet_stamp)
        p.start()
        self.addCleanup(p.stop)

    def make(self, wake_ready=True, visible=False, is_fresh=True, enabled=True, canned=True,
             wake_kind="frame"):
        self.mic = FakeMic(wake_ready=wake_ready, wake_kind=wake_kind,
                           wake_engine="whisper" if wake_kind == "utterance" else "porcupine")
        self.voice = FakeVoice()
        self.presence = FakePresence(visible=visible, is_fresh=is_fresh)
        self.hotplug = FakeWatcher()
        s = ConversationSession(self.voice, presence=self.presence, mic=self.mic, enabled=enabled,
                                hotplug=self.hotplug)
        s._set_state(STATE_IDLE if (enabled and wake_ready) else STATE_DISABLED, T0)
        if canned:
            # The four core lines, warm. NOT the filler bank: _canned_lines() is core-only since the
            # 44-line burst was split out into _prewarm_bank, and the bank is a separate, slower,
            # best-effort warm that a session is expected to run without. TestFiller adds it.
            # Built FROM _canned_lines() rather than listed, so a line added to it cannot leave the
            # fixture half-warm and quietly change what the re-warm retry tests assert.
            s._canned = {k: f"/tmp/{k}.wav" for k in s._canned_lines()}
            s._canned.update({"ack": "/tmp/ack.wav", "no_speech": "/tmp/ns.wav",
                              "error": "/tmp/err.wav", "thinking": "/tmp/hmm.wav"})
        return s


    # helpers -----------------------------------------------------------------

    def wake(self, s, at=T0):
        """Wake, then let the ack finish and the tail clear, landing in LISTEN_WAIT."""
        s._last_wake_t = at - WAKE_REFRACTORY_S - 1
        self.assertTrue(s.on_wake(at))
        return self.finish_speaking(s, at)

    def finish_speaking(self, s, at):
        """Advance through ACK/SPEAKING -> COOLDOWN -> LISTEN_WAIT."""
        self.voice.speaking = False
        s.tick(at)                      # ACK/SPEAKING -> COOLDOWN
        self.voice.gate_open = True
        s.tick(at)                      # COOLDOWN -> LISTEN_WAIT
        return at

    def fake_gate(self, **attrs):
        """A MagicMock gate whose numeric attributes are CONCRETE.

        Several of them are read straight onto /params, which is a JSON stream — a bare MagicMock
        serializes as a TypeError and takes the status endpoint down rather than failing visibly
        here. Every gate attribute get_status() touches must be set in this one place; adding a
        field to get_status() without adding it here is the bug this helper exists to prevent.
        """
        gate = MagicMock()
        gate.last_rms = 5000.0
        gate.onsets = 1
        gate.vad_available = True
        gate.rms_floor = VAD_RMS_FLOOR
        gate.ambient = 0.0
        gate.open_floor = VAD_RMS_FLOOR
        gate.hold_floor = VAD_RMS_FLOOR_HOLD
        for name, value in attrs.items():
            setattr(gate, name, value)
        return gate

    def speak_into(self, s, at, loud_s=1.0):
        """Simulate a real utterance: VAD onset, then a hangover after `loud_s` of speech."""
        s._gate = self.fake_gate()
        s._gate.speech_duration.return_value = loud_s
        s._gate.update.return_value = "onset"
        s._on_audio(np.ones(320, dtype="int16"), at)
        s._gate.update.return_value = "hangover"
        s._on_audio(np.ones(320, dtype="int16"), at + loud_s)
        return at + loud_s


class TestReresolveMic(SessionCase):
    """reresolve_mic() — recover a microphone without restarting the process.

    The case that matters is the one the watchdog cannot reach. Its `mic_lost` check only fires for
    a stream that DIED and is explicitly gated on `state != STATE_DISABLED`, but a mic that never
    came up at all leaves the session in exactly STATE_DISABLED — so the situation most in need of
    a second look was the one nothing ever looked at again.
    """

    def _running(self, s):
        """Pretend the tick thread is alive, which is how reresolve_mic tells the two cases apart."""
        s._thread = MagicMock()
        s._thread.is_alive.return_value = True
        return s

    def test_a_live_session_reopens_the_stream_rather_than_starting_a_second_one(self):
        s = self._running(self.make())
        out = s.reresolve_mic()
        self.assertTrue(out["ok"])
        self.assertEqual(self.mic.reopens_requested, 1)
        self.assertFalse(out["restarted_session"])

    def test_reports_which_mic_it_landed_on(self):
        # "It worked" is not enough: Kai being back on the fallback dongle when it should be on the
        # I2S mic is a different situation with a different next step.
        s = self._running(self.make())
        self.mic.device, self.mic.capture_rate, self.mic.is_i2s = 0, 48000, False
        out = s.reresolve_mic()
        self.assertEqual((out["device"], out["rate"], out["is_i2s"]), (0, 48000, False))

    def test_a_session_that_never_started_is_started_not_reopened(self):
        # No tick thread: start() is the correct call, and it is safe here precisely because a
        # failed start() returns before creating any thread.
        s = self.make(enabled=True)
        s._thread = None
        with patch.object(s, "start", return_value=True) as start:
            out = s.reresolve_mic()
        start.assert_called_once_with()
        self.assertEqual(self.mic.reopens_requested, 0)
        self.assertTrue(out["restarted_session"])

    def test_a_recovered_mic_leaves_disabled_behind_so_the_robot_reports_that_it_can_hear(self):
        # Without this the stream is open, the wake engine is ready, and sess_state still says
        # "disabled" — the dashboard would show a deaf robot that is actually listening.
        s = self._running(self.make())
        s._set_state(STATE_DISABLED, T0)
        s.reresolve_mic()
        self.assertEqual(s._state, STATE_IDLE)

    def test_a_failed_reopen_reports_the_reason_and_does_not_claim_to_be_idle(self):
        s = self._running(self.make())
        s._set_state(STATE_DISABLED, T0)
        self.mic.reopen_ok = False
        self.mic.error = "could not open microphone: Device unavailable"
        out = s.reresolve_mic()
        self.assertFalse(out["ok"])
        self.assertIn("Device unavailable", out["error"])
        self.assertEqual(s._state, STATE_DISABLED)

    def test_it_never_raises_because_a_recovery_control_that_500s_is_one_more_thing_to_fix(self):
        s = self._running(self.make())
        with patch.object(self.mic, "reopen", side_effect=RuntimeError("portaudio exploded")):
            out = s.reresolve_mic()
        self.assertFalse(out["ok"])
        self.assertIn("portaudio exploded", out["error"])

    def test_the_result_is_json_safe_because_it_goes_straight_out_of_a_flask_route(self):
        s = self._running(self.make())
        json.dumps(s.reresolve_mic())


class TestWakeAcceptance(SessionCase):
    def test_wake_from_idle_enters_ack_and_speaks_the_cached_line(self):
        s = self.make()
        self.assertTrue(s.on_wake(T0))
        self.assertEqual(s.state, STATE_ACK)
        self.assertEqual(self.voice.spoken_wavs, [("/tmp/ack.wav", "Yes?")])
        self.assertEqual(self.voice.spoken, [], "the ack must not be synthesized live")

    def test_wake_falls_back_to_live_synth_when_the_cache_missed(self):
        s = self.make(canned=False)
        s.on_wake(T0)
        self.assertEqual(self.voice.spoken, ["Yes?"])

    def test_wake_bumps_the_epoch(self):
        s = self.make()
        before = self.voice.epoch
        s.on_wake(T0)
        self.assertGreater(self.voice.epoch, before)

    def test_refractory_suppresses_a_double_fire(self):
        # A drawn-out "Heeey Kaaai" trips Porcupine twice.
        s = self.make()
        self.assertTrue(s.on_wake(T0))
        self.assertFalse(s.on_wake(T0 + WAKE_REFRACTORY_S / 2))
        self.assertEqual(s.get_status()["sess_wake_rejected_refractory"], 1)

    def test_wake_accepted_again_after_the_refractory_window(self):
        s = self.make()
        self.wake(s, T0)
        self.assertTrue(s.on_wake(T0 + WAKE_REFRACTORY_S + 0.1))

    def test_wake_while_busy_is_ignored(self):
        # The Ollama call is a blocking non-streaming POST — not cancellable, so there is nothing
        # useful to do with a wake here.
        s = self.make()
        s._set_state(STATE_BUSY, T0)
        self.assertFalse(s.on_wake(T0 + 5))
        self.assertEqual(s.state, STATE_BUSY)
        self.assertEqual(s.get_status()["sess_wake_rejected_busy"], 1)

    def test_wake_while_speaking_is_ignored_without_barge_in(self):
        s = self.make()
        s._set_state(STATE_SPEAKING, T0)
        self.assertFalse(s.on_wake(T0 + 1))
        self.assertEqual(s.state, STATE_SPEAKING)
        self.assertEqual(s.get_status()["sess_wake_rejected_speaking"], 1)

    def test_wake_during_cooldown_is_ignored(self):
        s = self.make()
        s._set_state(STATE_COOLDOWN, T0)
        self.assertFalse(s.on_wake(T0 + 0.1))

    def test_wake_while_listening_resets_the_clock_without_replaying_the_ack(self):
        s = self.make()
        at = self.wake(s, T0)
        s.tick(at + 5)
        self.voice.spoken_wavs.clear()
        self.assertTrue(s.on_wake(at + 6))
        self.assertEqual(s.state, STATE_LISTEN_WAIT)
        self.assertEqual(self.voice.spoken_wavs, [])
        # the silence timer restarted, so it must not fire at what would have been 15 s
        s.tick(at + 6 + SESSION_NO_SPEECH_S - 1)
        self.assertEqual(s.state, STATE_LISTEN_WAIT)

    def test_wake_before_whisper_is_warm_is_ignored(self):
        # Otherwise the first wake looks like a hang and blows every timer.
        s = self.make()
        self.voice.stt_ready = False
        self.assertFalse(s.on_wake(T0))
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(s.get_status()["sess_wake_rejected_not_ready"], 1)

    def test_wake_ignored_when_disabled(self):
        s = self.make(wake_ready=False)
        self.assertEqual(s.state, STATE_DISABLED)
        self.assertFalse(s.on_wake(T0))

    def test_wake_mid_utterance_starts_over_cleanly(self):
        # Saying "hey Kai" in the middle of a sentence means start again, so the half-captured
        # utterance must be dropped rather than carried into the new session.
        s = self.make()
        at = self.wake(s, T0)
        s._gate = self.fake_gate()
        s._gate.update.return_value = "onset"
        s._on_audio(np.ones(320, dtype="int16"), at + 1)
        self.assertEqual(s.state, STATE_LISTEN_SPEECH)
        before = self.mic.harvests
        self.assertTrue(s.on_wake(at + 2 + WAKE_REFRACTORY_S))
        self.assertEqual(s.state, STATE_ACK)
        self.assertGreater(self.mic.harvests, before, "the partial utterance must be discarded")
        self.assertFalse(self.mic.armed)
        self.assertEqual(self.voice.turns, [], "and never sent to Whisper")


class TestAckAndCooldown(SessionCase):
    def test_ack_holds_while_speech_is_in_flight(self):
        s = self.make()
        s.on_wake(T0)
        self.voice.speaking = True
        s.tick(T0 + 0.1)
        self.assertEqual(s.state, STATE_ACK, "must not leave ACK during the Piper run")

    def test_ack_exits_when_speech_finishes(self):
        s = self.make()
        s.on_wake(T0)
        self.voice.speaking = False
        s.tick(T0 + 0.5)
        self.assertEqual(s.state, STATE_COOLDOWN)

    def test_ack_times_out_if_playback_never_reports_done(self):
        s = self.make()
        s.on_wake(T0)
        self.voice.speaking = True
        s.tick(T0 + WAKE_ACK_MAX_S + 0.1)
        self.assertEqual(s.state, STATE_COOLDOWN)
        # The jaw has to be cut with the audio, so stop_speech() rather than a bare tts.stop().
        self.assertEqual(self.voice.speech_cuts, 1)

    def test_a_long_reply_is_not_cut_off_by_the_unknown_duration_backstop(self):
        # The regression this suite existed without: SESSION_SPEAK_MAX_UNKNOWN_S (20 s) is armed
        # from on_done, BEFORE Piper starts, and was the only deadline a healthy reply ever got —
        # while TTS_MAX_SPOKEN_CHARS allows ~31 s of speech. Every long answer was guillotined
        # mid-sentence by the guard against a WEDGED paplay.
        s = self.make()
        s._set_state(STATE_SPEAKING, T0)
        s._speak_deadline = T0 + SESSION_SPEAK_MAX_UNKNOWN_S
        self.voice.speaking = True
        self.voice.audio_end = T0 + 31.0        # a full-length reply, as the assistant measured it
        for i in range(1, 31):
            s.tick(T0 + i)
            self.assertEqual(s.state, STATE_SPEAKING, f"cut off at {i}s into a 31s reply")
        self.assertEqual(self.voice.speech_cuts, 0)

    def test_a_reply_past_its_own_measured_end_is_still_cut(self):
        # The backstop has to survive: a paplay that wedges overruns the end time it published.
        s = self.make()
        s._set_state(STATE_SPEAKING, T0)
        s._speak_deadline = T0 + SESSION_SPEAK_MAX_UNKNOWN_S
        self.voice.speaking = True
        self.voice.audio_end = T0 + 4.0
        s.tick(T0 + 4.0 + SESSION_SPEAK_GRACE_S - 0.1)
        self.assertEqual(s.state, STATE_SPEAKING)   # still inside the allowed overrun
        s.tick(T0 + 4.0 + SESSION_SPEAK_GRACE_S + 0.1)
        self.assertEqual(s.state, STATE_COOLDOWN)
        self.assertEqual(self.voice.speech_cuts, 1)

    def test_an_unmeasurable_reply_falls_back_to_the_armed_cap(self):
        # A silent pantomime, or a WAV whose header wouldn't read, publishes no end time.
        s = self.make()
        s._set_state(STATE_SPEAKING, T0)
        s._speak_deadline = T0 + SESSION_SPEAK_MAX_UNKNOWN_S
        self.voice.speaking = True
        self.voice.audio_end = None
        s.tick(T0 + SESSION_SPEAK_MAX_UNKNOWN_S - 0.1)
        self.assertEqual(s.state, STATE_SPEAKING)
        s.tick(T0 + SESSION_SPEAK_MAX_UNKNOWN_S + 0.1)
        self.assertEqual(s.state, STATE_COOLDOWN)

    def test_cooldown_holds_while_the_amp_settles(self):
        # paplay exits once the WAV is in the sink buffer, before the amp is actually quiet.
        s = self.make()
        s.on_wake(T0)
        self.voice.speaking = False
        s.tick(T0 + 0.5)
        self.voice.gate_open = False          # still inside TTS_TAIL_MUTE_S
        s.tick(T0 + 0.6)
        self.assertEqual(s.state, STATE_COOLDOWN)

    def test_cooldown_exits_and_flushes_the_dsp(self):
        s = self.make()
        s.on_wake(T0)
        self.voice.speaking = False
        s.tick(T0 + 0.5)
        before = self.mic.dsp_resets
        self.voice.gate_open = True
        s.tick(T0 + 1.0)
        self.assertEqual(s.state, STATE_LISTEN_WAIT)
        self.assertGreater(self.mic.dsp_resets, before,
                           "pre-mute residue must not trip an onset on the first frame back")

    def test_listen_wait_arms_capture(self):
        s = self.make()
        self.wake(s, T0)
        self.assertTrue(self.mic.armed)


class TestSessionEndNoSpeech(SessionCase):
    def test_ends_after_the_silence_window(self):
        s = self.make()
        at = self.wake(s, T0)
        s.tick(at + SESSION_NO_SPEECH_S - 0.1)
        self.assertEqual(s.state, STATE_LISTEN_WAIT, "must not fire early")
        s.tick(at + SESSION_NO_SPEECH_S)
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(s.get_status()["sess_end_reason"], "no_speech")

    def test_forgets_the_conversation_exactly_once(self):
        s = self.make()
        at = self.wake(s, T0)
        s.tick(at + SESSION_NO_SPEECH_S)
        self.assertEqual(self.voice.history_resets, 1)
        s.tick(at + SESSION_NO_SPEECH_S + 5)
        self.assertEqual(self.voice.history_resets, 1, "an idle session must not keep resetting")

    def test_disarms_capture_on_end(self):
        s = self.make()
        at = self.wake(s, T0)
        s.tick(at + SESSION_NO_SPEECH_S)
        self.assertFalse(self.mic.armed)


class TestSessionEndNoFace(SessionCase):
    def test_ends_after_a_seen_face_leaves(self):
        s = self.make(visible=True)
        at = self.wake(s, T0)
        s.tick(at)                              # sees the face -> face_ever_seen
        self.presence.visible = False
        s.tick(at + 0.1)                        # absence clock starts
        s.tick(at + 0.1 + SESSION_NO_FACE_S)
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(s.get_status()["sess_end_reason"], "no_face")

    def test_wake_with_no_face_ever_seen_is_governed_only_by_silence(self):
        # The dark-room / next-room case: the camera must not hang up on someone it never saw.
        s = self.make(visible=False)
        at = self.wake(s, T0)
        s.tick(at + SESSION_NO_FACE_S + 1)
        self.assertEqual(s.state, STATE_LISTEN_WAIT)
        self.assertFalse(s.get_status()["sess_face_ever_seen"])
        s.tick(at + SESSION_NO_SPEECH_S)
        self.assertEqual(s.get_status()["sess_end_reason"], "no_speech")

    def test_stale_feed_is_unknown_not_absent(self):
        # face_track stops calling mark() entirely on a camera stall; a dead camera must never end a
        # session.
        s = self.make(visible=True)
        at = self.wake(s, T0)
        s.tick(at)
        self.presence.visible = False
        self.presence.is_fresh = False
        for i in range(int(SESSION_NO_FACE_S) + 3):
            s.tick(at + 0.1 + i)
        self.assertEqual(s.state, STATE_LISTEN_WAIT)
        self.assertEqual(s.get_status()["sess_face_present"], "unknown")

    def test_absence_clock_resets_when_the_face_returns(self):
        s = self.make(visible=True)
        at = self.wake(s, T0)
        s.tick(at)
        self.presence.visible = False
        s.tick(at + 1)
        self.presence.visible = True
        s.tick(at + SESSION_NO_FACE_S - 1)      # glanced back
        self.presence.visible = False
        s.tick(at + SESSION_NO_FACE_S)
        self.assertEqual(s.state, STATE_LISTEN_WAIT, "a head turn must not end a conversation")

    def test_broken_presence_feed_fails_open(self):
        s = self.make(visible=True)
        at = self.wake(s, T0)
        s.tick(at)
        self.presence.raise_on_call = True
        for i in range(int(SESSION_NO_FACE_S) + 3):
            s.tick(at + 1 + i)
        self.assertEqual(s.state, STATE_LISTEN_WAIT)

    def test_note_face_also_feeds_presence(self):
        s = self.make(visible=False)
        s.note_face(True, T0)
        self.assertTrue(s.get_status()["sess_face_ever_seen"])


class TestTimersAreScopedToStates(SessionCase):
    """The central rigor claim: a timer armed in LISTEN_WAIT cannot fire anywhere else. Deadlines are
    measured from state entry, so there is no code path that could leak one into another state."""

    def test_silence_timer_does_not_fire_while_thinking(self):
        s = self.make()
        at = self.wake(s, T0)
        s._set_state(STATE_BUSY, at)
        for i in range(int(SESSION_NO_SPEECH_S) + 5):
            s.tick(at + i)
        self.assertEqual(s.state, STATE_BUSY)

    def test_face_timer_does_not_fire_while_thinking(self):
        # The 50 s cold-model-load case: someone asks a question and steps out of frame to wait.
        s = self.make(visible=True)
        at = self.wake(s, T0)
        s.tick(at)
        s._set_state(STATE_BUSY, at)
        self.presence.visible = False
        for i in range(int(SESSION_NO_FACE_S) + 5):
            s.tick(at + i)
        self.assertEqual(s.state, STATE_BUSY)
        self.assertEqual(s.get_status()["sess_no_face_left_s"], 0.0)

    def test_neither_timer_fires_while_speaking(self):
        s = self.make(visible=True)
        at = self.wake(s, T0)
        s.tick(at)
        self.presence.visible = False
        s._set_state(STATE_SPEAKING, at)
        s._speak_deadline = at + 1000
        self.voice.speaking = True
        for i in range(int(SESSION_NO_SPEECH_S) + 5):
            s.tick(at + i)
        self.assertEqual(s.state, STATE_SPEAKING)

    def test_absence_clock_is_cleared_on_re_entering_listen_wait(self):
        # Someone who stepped away during a long think gets a fresh window, not an instant hang-up.
        s = self.make(visible=True)
        at = self.wake(s, T0)
        s.tick(at)
        self.presence.visible = False
        s.tick(at + 1)
        s._set_state(STATE_BUSY, at + 2)
        for i in range(int(SESSION_NO_FACE_S) + 3):
            s.tick(at + 2 + i)
        s._enter_listen_wait(at + 30)
        s.tick(at + 30)
        self.assertEqual(s.state, STATE_LISTEN_WAIT)
        s.tick(at + 30 + SESSION_NO_FACE_S - 0.5)
        self.assertEqual(s.state, STATE_LISTEN_WAIT)

    def test_busy_times_out_only_past_the_ollama_ceiling(self):
        s = self.make()
        at = self.wake(s, T0)
        s._set_state(STATE_BUSY, at)
        s.tick(at + SESSION_BUSY_MAX_S - 1)
        self.assertEqual(s.state, STATE_BUSY)
        s.tick(at + SESSION_BUSY_MAX_S)
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(s.get_status()["sess_end_reason"], "busy_timeout")


class TestUtterance(SessionCase):
    def test_onset_arms_capture_with_preroll_and_enters_listen_speech(self):
        s = self.make()
        at = self.wake(s, T0)
        s._gate = self.fake_gate(last_rms=4000.0)
        s._gate.update.return_value = "onset"
        before = self.mic.arms
        s._on_audio(np.ones(320, dtype="int16"), at + 1)
        self.assertEqual(s.state, STATE_LISTEN_SPEECH)
        self.assertGreater(self.mic.arms, before)

    def test_hangover_runs_the_turn(self):
        s = self.make()
        at = self.wake(s, T0)
        self.speak_into(s, at + 1, loud_s=1.5)
        self.assertEqual(s.state, STATE_BUSY)
        self.assertEqual(len(self.voice.turns), 1)
        self.assertEqual(self.voice.turns[0]["rate"], 16000)

    def test_turn_carries_the_current_epoch(self):
        s = self.make()
        at = self.wake(s, T0)
        self.speak_into(s, at + 1)
        self.assertEqual(self.voice.turns[0]["epoch"], s.get_status()["sess_epoch"])

    def test_short_blip_is_discarded_without_running_whisper(self):
        # Breaks the loop: hiss -> empty transcript -> "didn't catch that" -> amp hiss -> forever.
        s = self.make()
        at = self.wake(s, T0)
        self.speak_into(s, at + 1, loud_s=MIN_UTTERANCE_S / 2)
        self.assertEqual(self.voice.turns, [], "a blip must not cost a Whisper run")
        self.assertEqual(s.state, STATE_LISTEN_WAIT)
        self.assertEqual(s.get_status()["sess_discarded_short"], 1)

    def test_discarded_blip_restarts_the_silence_window(self):
        s = self.make()
        at = self.wake(s, T0)
        blip_end = self.speak_into(s, at + 1, loud_s=MIN_UTTERANCE_S / 2)
        s.tick(blip_end + SESSION_NO_SPEECH_S - 0.5)
        self.assertEqual(s.state, STATE_LISTEN_WAIT)

    def test_max_utterance_forces_the_turn(self):
        s = self.make()
        at = self.wake(s, T0)
        s._gate = self.fake_gate()
        s._gate.update.return_value = "onset"
        s._gate.speech_duration.return_value = MAX_UTTERANCE_S
        s._on_audio(np.ones(320, dtype="int16"), at + 1)
        s.tick(at + 1 + MAX_UTTERANCE_S)
        self.assertEqual(s.state, STATE_BUSY)
        self.assertEqual(len(self.voice.turns), 1)

    def test_max_utterance_dispatches_without_holding_the_session_lock(self):
        """S1. The lock is an RLock, so a nested `with` inside a method called from tick() is
        re-entrant — which meant the code _finish_utterance places *after* its own block, on the
        stated grounds that a WAV write must not land on the tick thread's critical section, was
        still inside tick()'s. UtteranceRecorder.record() writes up to CAPTURE_HARD_CAP_S of audio,
        and the audio worker needs this same lock ~30 times a second for the VAD.

        Checked from ANOTHER thread on purpose: an RLock is re-entrant, so acquiring it from this
        one would succeed whether or not the bug is present, and the test would pass vacuously."""
        s = self.make()
        at = self.wake(s, T0)
        s._gate = self.fake_gate()
        s._gate.update.return_value = "onset"
        s._gate.speech_duration.return_value = MAX_UTTERANCE_S
        s._on_audio(np.ones(320, dtype="int16"), at + 1)

        held = []
        s._recorder = _lock_probing_recorder(s, held)
        s.tick(at + 1 + MAX_UTTERANCE_S)

        self.assertEqual(s.state, STATE_BUSY, "the turn must still be forced")
        self.assertEqual(held, [False], "the WAV write ran while tick() held the session lock")

    def test_scan_too_long_dispatches_without_holding_the_session_lock(self):
        """The same seam on the wake-scan path.

        Weaker than the turn case above, and deliberately asserted differently. _finish_scan's
        `too_long` branch discards the audio and returns BEFORE it records anything or dispatches
        Whisper, so this path never actually reached disk I/O — the ticket overstated it. What is
        asserted here is the structural property: tick() hands the dispatch out with the lock
        released, so the branch stays safe if it ever grows work the way the turn path has."""
        s = self.make(wake_kind="utterance")
        s._gate = self.fake_gate()
        s._gate.update.return_value = "onset"
        s._voice.scan_ready = True
        s._on_audio(np.ones(320, dtype="int16"), T0 + 1)
        self.assertEqual(s.state, STATE_SCAN_SPEECH)

        held = []

        def _probe_finish(now, reason):
            got = []

            def _probe():
                acquired = s._lock.acquire(blocking=False)
                got.append(acquired)
                if acquired:
                    s._lock.release()

            probe = threading.Thread(target=_probe)
            probe.start()
            probe.join(timeout=2.0)
            held.append(not (got and got[0]))

        s._finish_scan = _probe_finish
        s.tick(T0 + 1 + WAKE_WHISPER_MAX_UTTERANCE_S + 1)

        self.assertEqual(held, [False], "_finish_scan ran while tick() held the session lock")

    def test_rejected_turn_returns_to_listening(self):
        s = self.make()
        at = self.wake(s, T0)
        self.voice.turn_result = {"error": "busy: thinking"}
        self.speak_into(s, at + 1)
        self.assertEqual(s.state, STATE_LISTEN_WAIT)


class TestTurnOutcomes(SessionCase):
    def _to_busy(self, s):
        at = self.wake(s, T0)
        return self.speak_into(s, at + 1)

    def test_done_speaks_the_reply(self):
        s = self.make()
        self._to_busy(s)
        epoch = self.voice.turns[0]["epoch"]
        s._on_turn_done(epoch, "done")
        self.assertEqual(s.state, STATE_SPEAKING)
        self.assertEqual(s.get_status()["sess_turns"], 1)

    def test_second_turn_needs_no_wake_word(self):
        s = self.make()
        self._to_busy(s)
        s._on_turn_done(self.voice.turns[0]["epoch"], "done")
        at = self.finish_speaking(s, T0 + 10)
        self.assertEqual(s.state, STATE_LISTEN_WAIT)
        self.speak_into(s, at + 1)
        self.assertEqual(len(self.voice.turns), 2)
        self.assertEqual(s.get_status()["sess_wake_count"], 1)

    def test_empty_speaks_the_cached_line_not_the_ui_text(self):
        # NO_SPEECH_RESPONSE is "(didn't catch that — try again)" — UI text, with parentheses and an
        # em dash that espeak-ng would voice.
        s = self.make()
        self._to_busy(s)
        s._on_turn_done(self.voice.turns[0]["epoch"], "empty")
        self.assertEqual(self.voice.spoken_wavs[-1][0], "/tmp/ns.wav")
        self.assertEqual(s.state, STATE_SPEAKING)

    def test_repeated_empties_end_the_session(self):
        s = self.make()
        for _ in range(SESSION_MAX_NO_SPEECH_STREAK):
            self.voice.turns.clear()
            at = self.finish_speaking(s, T0) if s.state != STATE_IDLE else self.wake(s, T0)
            if s.state == STATE_IDLE:
                at = self.wake(s, T0)
            self.speak_into(s, at + 1)
            s._on_turn_done(self.voice.turns[0]["epoch"], "empty")
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(s.get_status()["sess_end_reason"], "no_speech_streak")

    def test_a_good_turn_clears_the_empty_streak(self):
        s = self.make()
        at = self.wake(s, T0)
        self.speak_into(s, at + 1)
        s._on_turn_done(self.voice.turns[0]["epoch"], "empty")
        at = self.finish_speaking(s, T0 + 5)
        self.voice.turns.clear()
        self.speak_into(s, at + 1)
        s._on_turn_done(self.voice.turns[0]["epoch"], "done")
        self.assertEqual(s._no_speech_streak, 0)

    def test_repeated_errors_end_the_session(self):
        s = self.make()
        for _ in range(SESSION_MAX_ERROR_STREAK):
            if s.state == STATE_IDLE:
                at = self.wake(s, T0)
            else:
                at = self.finish_speaking(s, T0)
            self.voice.turns.clear()
            self.speak_into(s, at + 1)
            s._on_turn_done(self.voice.turns[0]["epoch"], "error")
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(s.get_status()["sess_end_reason"], "error_streak")

    def test_stale_result_is_dropped_silently(self):
        s = self.make()
        self._to_busy(s)
        s._on_turn_done(self.voice.turns[0]["epoch"] - 1, "done")
        self.assertEqual(s.state, STATE_BUSY, "a superseded turn must not drive the FSM")
        self.assertGreaterEqual(s.get_status()["sess_stale_results"], 1)

    def test_result_arriving_after_the_session_ended_is_dropped(self):
        s = self.make()
        self._to_busy(s)
        epoch = self.voice.turns[0]["epoch"]
        s.end_session("manual")
        s._on_turn_done(epoch, "done")
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(s.get_status()["sess_turns"], 0)


class TestPushToTalk(SessionCase):
    def test_start_from_idle_records(self):
        s = self.make()
        self.assertEqual(s.request_ptt_start(), {"status": "ok"})
        self.assertEqual(s.state, STATE_LISTEN_SPEECH)
        self.assertTrue(self.mic.armed)

    def test_start_rejected_while_busy(self):
        s = self.make()
        s._set_state(STATE_BUSY, T0)
        self.assertIn("error", s.request_ptt_start())
        self.assertEqual(s.state, STATE_BUSY)

    def test_start_rejected_during_the_ack(self):
        s = self.make()
        s.on_wake(T0)
        self.assertIn("error", s.request_ptt_start())

    def test_start_interrupts_a_reply(self):
        # A button press is unambiguous intent, unlike an acoustic wake word during playback.
        s = self.make()
        s._set_state(STATE_SPEAKING, T0)
        self.assertEqual(s.request_ptt_start(), {"status": "ok"})
        # Audio AND jaw — a reply interrupted by the button must not go on mouthing itself.
        self.assertEqual(self.voice.speech_cuts, 1)
        self.assertEqual(s.state, STATE_LISTEN_SPEECH)

    def test_start_takes_over_a_hands_free_session(self):
        s = self.make()
        self.wake(s, T0)
        self.assertEqual(s.request_ptt_start(), {"status": "ok"})
        self.assertEqual(s.state, STATE_LISTEN_SPEECH)

    def test_start_bumps_the_epoch(self):
        s = self.make()
        before = self.voice.epoch
        s.request_ptt_start()
        self.assertGreater(self.voice.epoch, before)

    def test_stop_runs_the_turn(self):
        s = self.make()
        s.request_ptt_start()
        self.assertEqual(s.request_ptt_stop(), {"status": "ok"})
        self.assertEqual(s.state, STATE_BUSY)
        self.assertEqual(len(self.voice.turns), 1)

    def test_stop_rejected_when_not_recording(self):
        s = self.make()
        self.assertIn("error", s.request_ptt_stop())

    def test_stop_while_hands_free_listening_ends_the_session(self):
        # LISTEN_WAIT projects onto voice_status="recording", so the dashboard shows a live mic
        # button. Tapping it must do something sensible, not 400.
        s = self.make()
        self.wake(s, T0)
        self.assertEqual(s.state, STATE_LISTEN_WAIT)
        self.assertEqual(s.request_ptt_stop(), {"status": "ok"})
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(s.get_status()["sess_end_reason"], "ptt_stop")
        self.assertEqual(self.voice.history_resets, 1)
        self.assertEqual(self.voice.turns, [], "nothing was said, so nothing to transcribe")

    def test_short_manual_utterance_is_kept(self):
        # The human said "stop", so honour it — the blip guard is only for VAD-opened turns.
        s = self.make()
        s.request_ptt_start()
        s._gate = self.fake_gate()
        s._gate.speech_duration.return_value = 0.0
        s.request_ptt_stop()
        self.assertEqual(len(self.voice.turns), 1)

    def test_manual_turn_ignores_the_max_utterance_cap(self):
        s = self.make()
        s.request_ptt_start()
        s.tick(T0 + MAX_UTTERANCE_S + 5)
        self.assertEqual(s.state, STATE_LISTEN_SPEECH)
        self.assertEqual(self.voice.turns, [])


class TestMicMuted(SessionCase):
    def test_muted_in_every_speech_state(self):
        s = self.make()
        for state in (STATE_ACK, STATE_BUSY, STATE_SPEAKING, STATE_COOLDOWN):
            s._set_state(state, T0)
            self.assertTrue(s.mic_muted(T0), f"{state} must gate the mic")

    def test_open_while_idle_and_listening(self):
        s = self.make()
        for state in (STATE_IDLE, STATE_LISTEN_WAIT, STATE_LISTEN_SPEECH):
            s._set_state(state, T0)
            self.assertFalse(s.mic_muted(T0), f"{state} must not gate the mic")

    def test_defers_to_the_assistant_gate_when_listening(self):
        s = self.make()
        s._set_state(STATE_LISTEN_WAIT, T0)
        self.voice.gate_open = False
        self.assertTrue(s.mic_muted(T0))

    def test_busy_is_muted_even_though_no_audio_plays_yet(self):
        # BUSY covers the STT+LLM window. Leaving the mic open there would let Kai's own hum, or the
        # user's follow-up, open a turn he cannot service.
        s = self.make()
        s._set_state(STATE_BUSY, T0)
        self.voice.gate_open = True
        self.assertTrue(s.mic_muted(T0))


class TestMicLost(SessionCase):
    def test_session_ends_when_the_stream_dies(self):
        s = self.make()
        at = self.wake(s, T0)
        self.mic.live = False
        s.tick(at + 1)
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(s.get_status()["sess_end_reason"], "mic_lost")

    def test_a_watchdog_reopen_does_not_end_the_session(self):
        # The stream is briefly gone while MicStream re-suspends pulse and reopens. That's recovery,
        # not a lost mic — treating it as one would drop a conversation that was about to be fine.
        s = self.make()
        at = self.wake(s, T0)
        self.mic.live = False
        self.mic.reopening = True
        s.tick(at + 1)
        self.assertEqual(s.state, STATE_LISTEN_WAIT)
        self.mic.live = True
        self.mic.reopening = False
        s.tick(at + 2)
        self.assertEqual(s.state, STATE_LISTEN_WAIT)


class TestWakeUnavailableFallsBackCleanly(SessionCase):
    def test_start_turns_hands_free_off_rather_than_leaving_it_never_ready(self):
        # "enabled but never ready" reads as broken. Off + push-to-talk reads as a configuration.
        s = self.make(wake_ready=False)
        self.mic.wake.open.return_value = False
        with patch("ai.session.threading.Thread"):
            self.assertTrue(s.start())
        self.assertFalse(s.enabled)
        self.assertEqual(s.state, STATE_DISABLED)
        self.assertTrue(s.ready, "push-to-talk is a working configuration, so report ready")

    def test_push_to_talk_still_works_with_no_wake_word(self):
        s = self.make(wake_ready=False)
        self.mic.wake.open.return_value = False
        with patch("ai.session.threading.Thread"):
            s.start()
        self.assertEqual(s.request_ptt_start(), {"status": "ok"})
        self.assertEqual(s.request_ptt_stop(), {"status": "ok"})
        self.assertEqual(len(self.voice.turns), 1)


class TestStatusProjection(SessionCase):
    """The dashboard reads voice_status by exact string match, so the session projects onto the six
    values it already knows rather than introducing new ones."""

    def test_listening_maps_to_recording(self):
        s = self.make()
        self.wake(s, T0)
        status = s.get_status()
        self.assertEqual(status["voice_status"], STATUS_RECORDING)
        self.assertFalse(status["voice_speaking"])

    def test_speech_states_report_speaking(self):
        s = self.make()
        for state in (STATE_ACK, STATE_SPEAKING, STATE_COOLDOWN):
            s._set_state(state, T0)
            self.assertTrue(s.get_status()["voice_speaking"], state)

    def test_idle_leaves_the_assistants_own_values_alone(self):
        s = self.make()
        status = s.get_status()
        self.assertNotIn("voice_status", status)
        self.assertNotIn("voice_speaking", status)

    def test_busy_defers_to_a_live_assistant_status(self):
        # transcribing -> thinking is the assistant's own progression; overriding it would lose the
        # distinction the dashboard shows.
        s = self.make()
        s._set_state(STATE_BUSY, T0)
        for live in ("transcribing", "thinking"):
            self.voice.status = live
            status = s.get_status(T0)
            self.assertNotIn("voice_status", status, live)
            self.assertFalse(status["voice_speaking"])

    def test_busy_masks_a_previous_turns_terminal_status(self):
        """The duplicate-bubble bug: _finish_utterance enters BUSY inside the lock and only then calls
        into the assistant. In that gap voice_status fell back to the LAST turn's "done", and coming
        from LISTEN_* ("recording") the dashboard read that as a new completed turn and re-posted the
        previous question and answer verbatim."""
        s = self.make()
        s._set_state(STATE_BUSY, T0)
        for stale in ("done", "error", "idle"):
            self.voice.status = stale
            self.assertEqual(s.get_status(T0)["voice_status"], "transcribing", stale)

    def test_a_new_session_does_not_inherit_the_last_turns_status(self):
        # ACK/SPEAKING/COOLDOWN don't override voice_status, so a stale "done" carried into a new
        # session would surface as a fresh transition into "done".
        s = self.make()
        self.voice.status = "done"
        s._last_wake_t = T0 - 100
        self.assertTrue(s.on_wake(T0))
        self.assertEqual(s.state, STATE_ACK)
        self.assertEqual(self.voice.status, "idle")

    def test_session_end_does_not_replay_the_last_turn_as_a_new_one(self):
        """The bug this pins: LISTEN_WAIT projects "recording", so when the session ended the
        projection stopped overriding and un-masked the assistant's stale "done". The dashboard
        appends a chat bubble on the transition INTO "done", so it posted the same question and answer
        a second time, ~15 s after the reply, on every hands-free turn."""
        s = self.make()
        at = self.wake(s, T0)
        # Finish a turn so the assistant is left sitting on DONE, as it would be in real life.
        self.speak_into(s, at + 1)
        s._on_turn_done(self.voice.turns[0]["epoch"], "done")
        at = self.finish_speaking(s, at + 5)
        self.assertEqual(s.get_status(at)["voice_status"], STATUS_RECORDING)

        # Now let it time out.
        before = self.voice.turn_status_clears
        s.tick(at + SESSION_NO_SPEECH_S)
        self.assertEqual(s.state, STATE_IDLE)
        self.assertGreater(self.voice.turn_status_clears, before,
                           "ending the session must retire the finished turn's status")
        # With the status retired there is no override and no stale "done" underneath it.
        self.assertNotIn("voice_status", s.get_status(at + SESSION_NO_SPEECH_S))

    def test_every_key_is_json_safe(self):
        import json

        s = self.make()
        self.wake(s, T0)
        json.dumps(s.get_status())   # /params is an SSE JSON stream; must not raise

    def test_countdowns_are_exposed_while_waiting(self):
        # A misfire has to be visible BEFORE it happens, which is what the countdowns are for.
        s = self.make()
        at = self.wake(s, T0)
        s.tick(at + 5)
        status = s.get_status(now=at + 5)
        self.assertAlmostEqual(status["sess_no_speech_left_s"], SESSION_NO_SPEECH_S - 5, places=1)

    def test_face_countdown_is_exposed_once_absence_starts(self):
        s = self.make(visible=True)
        at = self.wake(s, T0)
        s.tick(at)
        self.presence.visible = False
        s.tick(at + 1)
        status = s.get_status(now=at + 3)
        self.assertAlmostEqual(status["sess_no_face_left_s"], SESSION_NO_FACE_S - 2, places=1)

    def test_wake_error_is_surfaced(self):
        s = self.make(wake_ready=False)
        self.mic.wake.unavailable = "no access key"
        self.assertEqual(s.get_status()["sess_wake_error"], "no access key")


class TestReady(SessionCase):
    def test_not_ready_without_a_live_stream(self):
        s = self.make()
        self.mic.live = False
        self.assertFalse(s.ready)

    def test_not_ready_before_whisper_loads(self):
        s = self.make()
        self.voice.stt_ready = False
        self.assertFalse(s.ready)

    def test_not_ready_when_the_wake_word_failed_but_hands_free_is_on(self):
        s = self.make(wake_ready=False)
        self.assertFalse(s.ready)

    def test_ready_without_a_wake_word_when_hands_free_is_off(self):
        s = self.make(wake_ready=False, enabled=False)
        self.assertTrue(s.ready, "push-to-talk only is a working configuration")

    def test_ready_when_everything_is_up(self):
        self.assertTrue(self.make().ready)


class TestStart(SessionCase):
    def test_attaches_the_mic_to_the_assistant(self):
        s = self.make()
        with patch("ai.session.threading.Thread"):
            self.assertTrue(s.start())
        self.assertIs(self.voice.attached, self.mic)

    def test_failure_to_open_the_stream_is_reported(self):
        s = self.make()
        self.mic.started = False
        with patch("ai.session.threading.Thread"):
            self.assertFalse(s.start())

    def test_wake_failure_still_brings_the_stream_up_for_push_to_talk(self):
        s = self.make(wake_ready=False)
        with patch("ai.session.threading.Thread"):
            self.assertTrue(s.start())
        self.assertIs(self.voice.attached, self.mic)
        self.assertEqual(s.state, STATE_DISABLED)
        self.assertFalse(self.mic.wake_enabled)

    def test_syncs_wake_geometry_after_opening_the_engine(self):
        # Order matters: before open() the winning tier — and so the frame size — isn't known yet.
        order = []
        s = self.make()
        self.mic.wake.open.side_effect = lambda: order.append("open") or True
        self.mic.sync_wake_geometry = lambda: order.append("sync")
        with patch("ai.session.threading.Thread"):
            s.start()
        self.assertEqual(order, ["open", "sync"])

    def test_legacy_capture_disables_the_shared_stream(self):
        s = self.make()
        with patch("ai.session.MIC_LEGACY_CAPTURE", True), patch("ai.session.threading.Thread"):
            self.assertFalse(s.start())
        self.assertIsNone(self.voice.attached)


class TestStop(SessionCase):
    """R7. Piper and paplay are child PROCESSES, not threads — the interpreter exiting does not take
    them with it. A teardown that leaves them running means the replacement process (autostart.sh
    relaunches within seconds) starts talking over audio the previous one left in the air, with its
    own self-hearing gate wide open because it knows nothing about that sound."""

    def test_stop_cancels_speech_in_flight(self):
        s = self.make()
        self.mock_stop.reset_mock()
        s.stop()
        self.mock_stop.assert_called()

    def test_stop_cancels_speech_before_releasing_the_mic(self):
        # Ordering is the point: the mic close ends this process's claim on the audio devices, and a
        # synth still running past it is exactly the orphan this closes.
        s = self.make()
        order = []
        self.mock_stop.side_effect = lambda: order.append("tts.stop")
        self.mic.stop = lambda: order.append("mic.stop")
        s.stop()
        self.assertEqual(order, ["tts.stop", "mic.stop"])

    def test_stop_is_idempotent_and_safe_on_a_session_that_never_started(self):
        s = self.make()
        s.stop()
        s.stop()


class TestGreeting(SessionCase):
    """"Hi, I'm Kai" on boot: once per process, on the warm thread, and out of the bank's way."""

    def setUp(self):
        super().setUp()
        # On, and with the post-speech wait collapsed — the wait is exercised on its own below.
        for name, value in (("GREETING_ENABLED", True), ("GREETING_QUIET_WAIT_S", 0.0)):
            p = patch(f"ai.session.{name}", value)
            p.start()
            self.addCleanup(p.stop)

    def test_the_greeting_is_spoken(self):
        s = self.make()
        s._speak_greeting()
        self.assertEqual(self.voice.spoken, [GREETING_TEXT])

    def test_it_is_spoken_once_per_process(self):
        # start() is re-entered by reresolve_mic() every time an operator retries a mic that failed
        # to come up at boot, which on this robot is a normal thing to do more than once. Greeting
        # the room again on each attempt would turn a recovery button into a talking one.
        s = self.make()
        s._speak_greeting()
        s._speak_greeting()
        self.assertEqual(self.voice.spoken, [GREETING_TEXT])

    def test_disabling_it_boots_silently(self):
        s = self.make()
        with patch("ai.session.GREETING_ENABLED", False):
            s._speak_greeting()
        self.assertEqual(self.voice.spoken, [])

    def test_blank_text_is_treated_as_off(self):
        s = self.make()
        with patch("ai.session.GREETING_TEXT", "   "):
            s._speak_greeting()
        self.assertEqual(self.voice.spoken, [])

    def test_it_never_posts_a_chat_bubble(self):
        # speak_text, not say(): nobody took this turn, so it must not appear in the dashboard
        # transcript — the same contract the ack and the filler lines hold to.
        s = self.make()
        s._speak_greeting()
        self.assertEqual(self.voice.said, [])

    def test_it_lands_between_the_core_lines_and_the_bank(self):
        # Not before the core lines (a wake during the greeting still needs "Yes?" on disk), and not
        # after the bank, which runs for minutes — a hello that late is not a greeting.
        order = []
        s = self.make()
        s._canned = {}
        self.mock_prewarm_canned.side_effect = (
            lambda lines, *a, **kw: order.append("warm:" + ",".join(lines)) or {})
        self.voice.speak_text = lambda text, epoch=None: order.append("greet")
        s._warm_all()
        self.assertEqual(order[:2], ["warm:ack,no_speech,error,thinking", "greet"])
        self.assertTrue(all(o.startswith("warm:filler_") for o in order[2:]))

    def test_it_waits_for_the_audio_before_the_bank_synthesises(self):
        # tts has ONE synth slot and _begin_speech cuts whatever is playing, so returning while the
        # greeting is still going would have the bank's first line kill it mid-sentence.
        polls = []

        def _still_playing():
            polls.append(1)
            return len(polls) < 3

        s = self.make()
        self.voice.speech_in_flight = _still_playing
        with patch("ai.session.GREETING_QUIET_WAIT_S", 5.0):
            s._speak_greeting()
        self.assertEqual(len(polls), 3)

    def test_the_wait_is_bounded(self):
        # A playback that never reports done costs the bank a delay, not the whole warm.
        s = self.make()
        self.voice.speech_in_flight = lambda: True
        s._speak_greeting()          # GREETING_QUIET_WAIT_S is 0.0 here — must return, not hang
        self.assertEqual(self.voice.spoken, [GREETING_TEXT])

    def test_the_warm_thread_runs_even_with_presynth_off(self):
        # ACK_PRESYNTH is a debugging switch for the CANNED lines; turning it off must not also
        # silence the greeting, which is not cached at all.
        s = self.make()
        with patch("ai.session.ACK_PRESYNTH", False), \
             patch("ai.session.threading.Thread") as thread:
            s.start()
        self.assertIn("kai-ack-warm", [c.kwargs.get("name") for c in thread.call_args_list])

    def test_presynth_off_greets_without_warming_anything(self):
        s = self.make()
        with patch("ai.session.ACK_PRESYNTH", False):
            s._warm_all()
        self.assertEqual(self.voice.spoken, [GREETING_TEXT])
        self.assertEqual(self.mock_prewarm_canned.call_args_list, [])


class TestGreetingAcrossProcesses(SessionCase):
    """The second way to greet twice: a DIFFERENT process starting right after this one.

    The in-memory latch cannot see that. Measured on the robot 2026-08-11 — a segfault at the
    greeting, an automatic relaunch, and the whole line spoken again 57 s later. See
    GREETING_REPEAT_SUPPRESS_S in config/wake.py.
    """

    def setUp(self):
        super().setUp()
        for name, value in (("GREETING_ENABLED", True), ("GREETING_QUIET_WAIT_S", 0.0)):
            p = patch(f"ai.session.{name}", value)
            p.start()
            self.addCleanup(p.stop)

    def stamp(self, age_s):
        """Write the stamp as though a previous process greeted `age_s` seconds ago."""
        import time as _time
        with open(self.greet_stamp, "w") as fh:
            fh.write("previous process\n")
        when = _time.time() - age_s
        os.utime(self.greet_stamp, (when, when))

    def test_a_relaunch_inside_the_window_stays_quiet(self):
        self.stamp(10.0)
        self.make()._speak_greeting()
        self.assertEqual(self.voice.spoken, [])

    def test_a_relaunch_after_the_window_greets_normally(self):
        # Restarting to hear a change is normal, and must not be answered with silence.
        self.stamp(sess_mod.GREETING_REPEAT_SUPPRESS_S + 5.0)
        self.make()._speak_greeting()
        self.assertEqual(self.voice.spoken, [GREETING_TEXT])

    def test_a_first_ever_boot_greets(self):
        # No stamp at all: /tmp cleared by a reboot, or a robot that has never greeted.
        self.assertFalse(os.path.exists(self.greet_stamp))
        self.make()._speak_greeting()
        self.assertEqual(self.voice.spoken, [GREETING_TEXT])

    def test_the_stamp_is_written_before_the_audio_starts(self):
        # THE case this exists for: the crashing run died partway through its own greeting. A stamp
        # written only after the audio finished would never have been written by that run at all,
        # and the relaunch would have greeted anyway.
        seen = []
        s = self.make()
        self.voice.speak_text = lambda text, epoch=None: seen.append(
            os.path.exists(self.greet_stamp))
        s._speak_greeting()
        self.assertEqual(seen, [True])

    def test_a_stamp_dated_in_the_future_still_greets(self):
        # This board has no RTC battery, so the clock steps when NTP lands. An unreadable age must
        # err toward greeting: a missing greeting is the failure this feature must not cause.
        self.stamp(-3600.0)
        self.make()._speak_greeting()
        self.assertEqual(self.voice.spoken, [GREETING_TEXT])

    def test_zero_suppression_restores_greeting_on_every_start(self):
        self.stamp(1.0)
        with patch("ai.session.GREETING_REPEAT_SUPPRESS_S", 0.0):
            self.make()._speak_greeting()
        self.assertEqual(self.voice.spoken, [GREETING_TEXT])

    def test_an_unwritable_stamp_never_costs_the_greeting(self):
        # Best-effort in the one direction that matters: no stamp means a possible duplicate, and a
        # duplicate greeting is far cheaper than a silent boot.
        with patch("ai.session.GREETING_STAMP_PATH", os.path.join(self.greet_stamp, "nope", "x")):
            self.make()._speak_greeting()
        self.assertEqual(self.voice.spoken, [GREETING_TEXT])

    def test_a_suppressed_greeting_still_leaves_a_working_session(self):
        # The stamp gates the greeting and nothing else: the warm thread goes on to the bank, and a
        # wake arriving straight after still gets its cached "Yes?".
        self.stamp(1.0)
        s = self.make()
        s._warm_all()
        self.assertEqual(self.voice.spoken, [])
        self.wake(s, T0)
        self.assertEqual([wav for wav, _ in self.voice.spoken_wavs], ["/tmp/ack.wav"])


class TestWhisperTierScan(SessionCase):
    """The utterance tier: capture in idle, transcribe, match, and only then wake. Every guard here
    exists to stop a room full of conversation from costing a Whisper run per sentence."""

    def make_scan(self, **kw):
        s = self.make(wake_kind="utterance", **kw)
        s._set_state(STATE_IDLE, T0)
        return s

    def onset(self, s, at):
        """Drive a VAD onset into the session."""
        s._gate = self.fake_gate()
        s._gate.update.return_value = "onset"
        s._on_audio(np.ones(320, dtype="int16"), at)

    def hangover(self, s, at, spoken_s=1.0):
        s._gate.speech_duration.return_value = spoken_s
        s._gate.update.return_value = "hangover"
        s._on_audio(np.ones(320, dtype="int16"), at)

    def scan(self, s, at, spoken_s=1.0):
        self.onset(s, at)
        self.hangover(s, at + spoken_s, spoken_s=spoken_s)

    # ── entry ───────────────────────────────────────────────────────────────

    def test_frame_tier_never_scans_in_idle(self):
        s = self.make()                      # kind == "frame"
        s._set_state(STATE_IDLE, T0)
        self.onset(s, T0 + 1)
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(self.voice.transcribes, [])

    def test_onset_in_idle_captures_with_preroll(self):
        s = self.make_scan()
        before = self.mic.arms
        self.onset(s, T0 + 1)
        self.assertEqual(s.state, STATE_SCAN_SPEECH)
        self.assertGreater(self.mic.arms, before)

    def test_a_scan_uses_the_short_hangover_not_the_turn_one(self):
        """The turn's 1.5 s exists so a speaker pausing mid-sentence isn't cut off. Nobody pauses
        mid-thought while saying "hey", and the clock is charged to EVERY wake — 1.5 s of hangover
        plus the transcribe plus the cooldown was ~3.2 s of deafness after you finished speaking, so
        a natural immediate retry landed inside the dead window."""
        s = self.make_scan()
        self.onset(s, T0 + 1)
        s._gate.set_hangover.assert_called_with(WAKE_SCAN_HANGOVER_S)

    def test_a_turn_gets_the_long_hangover_back(self):
        # One gate serves both jobs, so the scan path above must not leave the short clock set on a
        # real conversation — that would cut people off mid-sentence.
        s = self.make()
        at = self.wake(s, T0)
        s._gate = self.fake_gate()
        s._gate.update.return_value = "onset"
        s._on_audio(np.ones(320, dtype="int16"), at + 1)
        self.assertEqual(s.state, STATE_LISTEN_SPEECH)
        s._gate.set_hangover.assert_called_with(VAD_HANGOVER_S)

    def test_hangover_transcribes(self):
        s = self.make_scan()
        self.scan(s, T0 + 1, spoken_s=1.2)
        self.assertEqual(s.state, STATE_SCAN_CHECK)
        self.assertEqual(len(self.voice.transcribes), 1)
        self.assertEqual(self.voice.transcribes[0]["rate"], 16000)

    def test_disabled_hands_free_never_scans(self):
        s = self.make_scan(enabled=False)
        self.onset(s, T0 + 1)
        self.assertEqual(self.voice.transcribes, [])

    # ── cost guards ─────────────────────────────────────────────────────────

    def test_blip_is_discarded_without_running_whisper(self):
        # WAKE_WHISPER_MIN_UTTERANCE_S, not MIN_UTTERANCE_S: the scan path has its own, much lower
        # floor (0.15 vs 0.35) because a bare "hey" is only ~0.3 s of audio. This test used the turn
        # constant and silently stopped testing a blip the moment they diverged.
        s = self.make_scan()
        self.scan(s, T0 + 1, spoken_s=WAKE_WHISPER_MIN_UTTERANCE_S / 2)
        self.assertEqual(self.voice.transcribes, [], "a blip must not cost an STT run")
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(s.get_status(T0 + 5)["sess_scan_skip_short"], 1)

    def test_long_utterance_is_discarded_without_running_whisper(self):
        # Somebody mid-conversation is not saying a two-word phrase.
        s = self.make_scan()
        self.scan(s, T0 + 1, spoken_s=WAKE_WHISPER_MAX_UTTERANCE_S + 1)
        self.assertEqual(self.voice.transcribes, [])
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(s.get_status(T0 + 5)["sess_scan_skip_long"], 1)

    def test_long_utterance_backs_off_further(self):
        s = self.make_scan()
        self.scan(s, T0 + 1, spoken_s=WAKE_WHISPER_MAX_UTTERANCE_S + 1)
        ready_in = s.get_status(T0 + 2)["sess_scan_ready_in_s"]
        self.assertGreater(ready_in, WAKE_WHISPER_COOLDOWN_S)

    def test_max_utterance_tick_forces_the_discard(self):
        s = self.make_scan()
        self.onset(s, T0 + 1)
        s._gate.speech_duration.return_value = 2.0
        s.tick(T0 + 1 + WAKE_WHISPER_MAX_UTTERANCE_S)
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(self.voice.transcribes, [])

    def test_cooldown_blocks_a_second_scan(self):
        s = self.make_scan()
        self.scan(s, T0 + 1, spoken_s=1.0)
        s._on_scan_done(self.voice.transcribes[0]["token"], "nothing relevant", "")
        self.assertEqual(s.state, STATE_IDLE)
        self.onset(s, T0 + 2.2)              # inside WAKE_WHISPER_COOLDOWN_S
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(len(self.voice.transcribes), 1)
        self.assertGreaterEqual(s.get_status(T0 + 2.2)["sess_scan_skip_cooldown"], 1)

    def test_no_scan_before_whisper_is_warm(self):
        # Otherwise the first check pays the model load inside its own timeout and always fails.
        s = self.make_scan()
        self.voice.scan_ready = False
        self.onset(s, T0 + 1)
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(s.get_status(T0 + 1)["sess_scan_skip_no_model"], 1)

    # ── outcomes ────────────────────────────────────────────────────────────

    def test_no_match_is_completely_silent(self):
        s = self.make_scan()
        self.scan(s, T0 + 1)
        token = self.voice.transcribes[0]["token"]
        s._on_scan_done(token, "so anyway the weather is nice", "")
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(self.voice.spoken, [], "nothing may be spoken")
        self.assertEqual(self.voice.spoken_wavs, [], "not even the ack")
        self.assertEqual(self.voice.turns, [])
        self.assertEqual(s.get_status(T0 + 5)["sess_scan_matches"], 0)

    def test_phrase_alone_acks_and_listens(self):
        s = self.make_scan()
        self.scan(s, T0 + 1)
        s._on_scan_done(self.voice.transcribes[0]["token"], "Hey Kai", "")
        self.assertEqual(s.state, STATE_ACK)
        self.assertEqual(self.voice.spoken_wavs, [("/tmp/ack.wav", "Yes?")])
        self.assertEqual(s.get_status(T0 + 5)["sess_scan_matches"], 1)

    def test_phrase_plus_command_answers_in_one_breath(self):
        s = self.make_scan()
        self.scan(s, T0 + 1)
        s._on_scan_done(self.voice.transcribes[0]["token"], "Hey Kai, what time is it?", "")
        self.assertEqual(s.state, STATE_BUSY, "no ack — straight to the turn")
        self.assertEqual(self.voice.spoken_wavs, [])
        self.assertEqual(len(self.voice.said), 1)
        self.assertEqual(self.voice.said[0]["text"], "what time is it?",
                         "the wake phrase must be stripped before the LLM sees it")

    def test_one_breath_turn_carries_the_new_epoch(self):
        s = self.make_scan()
        self.scan(s, T0 + 1)
        s._on_scan_done(self.voice.transcribes[0]["token"], "hey kai tell me a joke", "")
        self.assertEqual(self.voice.said[0]["epoch"], s.get_status(T0 + 2)["sess_epoch"])

    def test_one_breath_turn_completes_through_on_turn_done(self):
        s = self.make_scan()
        self.scan(s, T0 + 1)
        s._on_scan_done(self.voice.transcribes[0]["token"], "hey kai tell me a joke", "")
        s._on_turn_done(self.voice.said[0]["epoch"], "done")
        self.assertEqual(s.state, STATE_SPEAKING)
        self.assertEqual(s.get_status(T0 + 3)["sess_turns"], 1)

    def test_say_refusing_falls_back_to_the_ack(self):
        # Ignoring say()'s error would leave the session in BUSY for SESSION_BUSY_MAX_S — two
        # minutes of a robot that looks hung.
        s = self.make_scan()
        self.voice.say_result = {"error": "busy: thinking"}
        self.scan(s, T0 + 1)
        s._on_scan_done(self.voice.transcribes[0]["token"], "hey kai what time is it", "")
        self.assertEqual(s.state, STATE_ACK)
        self.assertEqual(self.voice.spoken_wavs, [("/tmp/ack.wav", "Yes?")])

    def test_transcription_error_returns_to_idle(self):
        s = self.make_scan()
        self.scan(s, T0 + 1)
        s._on_scan_done(self.voice.transcribes[0]["token"], "", "ctranslate2 exploded")
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(self.voice.spoken, [])

    # ── invalidation ────────────────────────────────────────────────────────

    def test_stale_token_is_dropped(self):
        s = self.make_scan()
        self.scan(s, T0 + 1)
        s._on_scan_done(self.voice.transcribes[0]["token"] - 1, "hey kai", "")
        self.assertEqual(s.state, STATE_SCAN_CHECK, "a superseded check must not drive the FSM")
        self.assertEqual(self.voice.spoken_wavs, [])

    def test_check_timeout_orphans_the_worker(self):
        s = self.make_scan()
        self.scan(s, T0 + 1)
        token = self.voice.transcribes[0]["token"]
        s.tick(T0 + 2 + WAKE_WHISPER_CHECK_MAX_S)
        self.assertEqual(s.state, STATE_IDLE)
        s._on_scan_done(token, "hey kai", "")           # late result
        self.assertEqual(s.state, STATE_IDLE, "the late result must be ignored")
        self.assertEqual(self.voice.spoken_wavs, [])

    def test_ptt_during_a_check_drops_the_result(self):
        s = self.make_scan()
        self.scan(s, T0 + 1)
        token = self.voice.transcribes[0]["token"]
        s.request_ptt_start()
        self.assertEqual(s.state, STATE_LISTEN_SPEECH)
        s._on_scan_done(token, "hey kai what time is it", "")
        self.assertEqual(s.state, STATE_LISTEN_SPEECH, "PTT must win")
        self.assertEqual(self.voice.said, [])

    def test_session_end_during_a_check_drops_the_result(self):
        s = self.make_scan()
        self.scan(s, T0 + 1)
        token = self.voice.transcribes[0]["token"]
        s.end_session("manual")
        s._on_scan_done(token, "hey kai", "")
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(self.voice.spoken_wavs, [])

    def test_manual_wake_during_a_check_drops_the_result(self):
        s = self.make_scan()
        self.scan(s, T0 + 1)
        token = self.voice.transcribes[0]["token"]
        s._set_state(STATE_IDLE, T0 + 3)                # as _on_scan_done would
        s._last_wake_t = T0 - 100
        self.assertTrue(s.on_wake(T0 + 3))
        s._on_scan_done(token, "hey kai what time is it", "")
        self.assertEqual(self.voice.said, [])

    # ── the scan states must not disturb anything else ──────────────────────

    def test_scan_states_do_not_mute_the_mic(self):
        # Muting during a scan would make the tier unable to hear the utterance it is capturing.
        s = self.make_scan()
        for state in (STATE_SCAN_SPEECH, STATE_SCAN_CHECK):
            s._set_state(state, T0)
            self.assertFalse(s.mic_muted(T0), f"{state} must not gate the mic")

    def test_scan_states_project_exactly_as_idle(self):
        s = self.make_scan()
        for state in (STATE_SCAN_SPEECH, STATE_SCAN_CHECK):
            s._set_state(state, T0)
            status = s.get_status(T0)
            self.assertNotIn("voice_status", status, state)
            self.assertNotIn("voice_speaking", status, state)

    def test_scan_states_are_not_speech_states(self):
        for state in (STATE_SCAN_SPEECH, STATE_SCAN_CHECK):
            self.assertNotIn(state, sess_mod._SPEECH_STATES)

    def test_no_session_timers_fire_during_a_scan(self):
        s = self.make_scan()
        self.onset(s, T0 + 1)
        s._gate.speech_duration.return_value = 0.5
        for i in range(int(SESSION_NO_SPEECH_S) + 5):
            s.tick(T0 + 1 + i * 0.1)                     # stay under the max-utterance cap
        self.assertEqual(self.voice.history_resets, 0, "a scan is not a session")

    def test_transcript_is_only_logged_when_explicitly_enabled(self):
        # It is speech nobody addressed to Kai, so logging it is opt-in. (It is currently ON in
        # config while the phrase aliases are being tuned — this pins the behaviour of the flag
        # itself rather than whatever it happens to be set to.)
        s = self.make_scan()
        self.scan(s, T0 + 1)
        with patch("ai.session.WAKE_WHISPER_LOG_TEXT", False), patch("builtins.print") as mock_print:
            s._on_scan_done(self.voice.transcribes[0]["token"], "my bank password is hunter2", "")
        logged = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertNotIn("hunter2", logged)

    def test_transcript_is_logged_when_enabled(self):
        s = self.make_scan()
        self.scan(s, T0 + 1)
        with patch("ai.session.WAKE_WHISPER_LOG_TEXT", True), patch("builtins.print") as mock_print:
            s._on_scan_done(self.voice.transcribes[0]["token"], "some overheard sentence", "")
        logged = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("overheard", logged)

    def test_matched_phrase_is_reported_on_params(self):
        s = self.make_scan()
        self.scan(s, T0 + 1)
        s._on_scan_done(self.voice.transcribes[0]["token"], "Hey Kai", "")
        status = s.get_status(T0 + 5)
        self.assertEqual(status["sess_scan_checks"], 1)
        self.assertEqual(status["sess_scan_matches"], 1)
        self.assertIn("Kai", status["sess_scan_last_text"])
        self.assertGreaterEqual(status["sess_scan_last_ms"], 0)


class TestWakeEngineReporting(SessionCase):
    def test_engine_fields_are_exposed(self):
        s = self.make()
        status = s.get_status(T0)
        self.assertEqual(status["sess_wake_engine"], "porcupine")
        self.assertEqual(status["sess_wake_kind"], "frame")
        self.assertEqual(status["sess_wake_frame"], 512)
        self.assertEqual(status["sess_wake_tried"], "porcupine=ok")

    def test_utterance_tier_reports_its_kind(self):
        s = self.make(wake_kind="utterance")
        status = s.get_status(T0)
        self.assertEqual(status["sess_wake_engine"], "whisper")
        self.assertEqual(status["sess_wake_kind"], "utterance")


class TestThinkingSound(SessionCase):
    """The "Hmm..." played during the STT + LLM pause. Decoration on top of BUSY: it must not become a
    state, must not touch the turn status, and must not fire on a reply that came back quickly.

    Since the filler bank landed this is the FALLBACK path — it runs when the bank is off or has
    nothing for the turn's language — so these run with FILLER_ENABLED patched off. That is the
    condition under which the behaviour below is still the contract; TestFiller covers the other."""

    def setUp(self):
        super().setUp()
        # settings holds module-level state shared across the whole test process; persist=False keeps
        # every write out of the operator's real ~/.config/kai/settings.json.
        settings._reset_for_tests()
        self.addCleanup(settings._reset_for_tests)
        patcher = patch("ai.session.FILLER_ENABLED", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _busy(self, s):
        at = self.wake(s, T0)
        return self.speak_into(s, at + 1)

    def test_plays_the_cached_line_after_the_delay(self):
        s = self.make()
        at = self._busy(s)
        s.tick(at + THINKING_SOUND_DELAY_S)
        self.assertEqual(self.voice.spoken_wavs[-1], ("/tmp/hmm.wav", THINKING_SOUND_TEXT))
        self.assertEqual(s.state, STATE_BUSY, "the sound is not a state change")

    def test_silent_before_the_delay(self):
        # The whole point of the delay: a fast reply must produce no sound at all.
        s = self.make()
        at = self._busy(s)
        before = list(self.voice.spoken_wavs)
        s.tick(at + THINKING_SOUND_DELAY_S - 0.01)
        self.assertEqual(self.voice.spoken_wavs, before)

    def test_plays_at_most_once_per_turn(self):
        s = self.make()
        at = self._busy(s)
        for i in range(20):
            s.tick(at + THINKING_SOUND_DELAY_S + i * 0.05)
        played = [w for w in self.voice.spoken_wavs if w[0] == "/tmp/hmm.wav"]
        self.assertEqual(len(played), 1)

    def test_rearms_on_the_next_turn(self):
        s = self.make()
        at = self._busy(s)
        s.tick(at + THINKING_SOUND_DELAY_S)
        s._on_turn_done(self.voice.turns[0]["epoch"], "done")
        at = self.finish_speaking(s, at + 10)
        at = self.speak_into(s, at + 1)
        s.tick(at + THINKING_SOUND_DELAY_S)
        played = [w for w in self.voice.spoken_wavs if w[0] == "/tmp/hmm.wav"]
        self.assertEqual(len(played), 2)

    def test_off_is_silent(self):
        s = self.make()
        settings.set_many({"thinking_sounds": False}, persist=False)
        at = self._busy(s)
        for i in range(20):
            s.tick(at + THINKING_SOUND_DELAY_S + i * 0.1)
        self.assertEqual([w for w in self.voice.spoken_wavs if w[0] == "/tmp/hmm.wav"], [])

    def test_leaves_the_turn_status_alone_so_no_chat_bubble_appears(self):
        # It goes through _speak_canned, never say(): a voice_response here would make the dashboard
        # post "Kai: Hmm..." as a real reply, and voice_speaking=True would flip the mic button.
        s = self.make()
        at = self._busy(s)
        s.tick(at + THINKING_SOUND_DELAY_S)
        self.assertEqual(self.voice.said, [], "the hmm must never be routed through say()")
        status = s.get_status(at + THINKING_SOUND_DELAY_S)
        self.assertFalse(status["voice_speaking"], "voice_speaking would flip the mic button")
        self.assertNotEqual(status.get("voice_status"), "done",
                            "the dashboard posts a chat bubble on the transition into done")

    def test_status_stays_json_safe(self):
        s = self.make()
        at = self._busy(s)
        s.tick(at + THINKING_SOUND_DELAY_S)
        json.dumps(s.get_status(at + THINKING_SOUND_DELAY_S))

    def test_busy_timeout_still_fires(self):
        # The sound must not disturb the deadline that state shares.
        s = self.make()
        at = self._busy(s)
        s.tick(at + THINKING_SOUND_DELAY_S)
        s.tick(at + SESSION_BUSY_MAX_S)
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(s.get_status()["sess_end_reason"], "busy_timeout")

    def test_is_prewarmed_with_the_other_canned_lines(self):
        s = self.make()
        s._prewarm_canned()
        lines = self.mock_prewarm_canned.call_args[0][0]
        self.assertEqual(lines["thinking"], THINKING_SOUND_TEXT)


class TestFiller(SessionCase):
    """The filler bank driving a BUSY turn: one opener, then stalls on a loop until the reply lands.

    Like the "Hmm" it supersedes, this is decoration on top of BUSY — not a state, not a status
    change, and silent on a reply that came back fast. What is new and worth pinning is the LOOP:
    it must keep talking for an arbitrarily long wait, it must never start a line over one already
    playing, and no gap it leaves may exceed the ceiling the whole feature exists for.

    The session's Random is seeded per test so a draw cannot make a run flaky.
    """

    def setUp(self):
        super().setUp()
        settings._reset_for_tests()
        self.addCleanup(settings._reset_for_tests)
        # ONE flag for "audio is in flight", driving both seams the loop reads. False by default,
        # which means playback ends the instant it starts -- the worst case for the gap contract,
        # since every silence is then a full drawn one with no audio covering any of it.
        self.playing = False
        patcher = patch("ai.session.tts.is_playing", side_effect=lambda: self.playing)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make(self, *a, **kw):
        s = super().make(*a, **kw)
        s._filler_rng = random.Random(4242)
        # The bank, warm. SessionCase seeds only the four core lines, and the loop selects from
        # CACHED keys alone (filler is never synthesised live), so a cold bank here would make
        # every test below silently exercise the "Hmm" fallback while appearing to test the bank.
        s._canned.update({k: f"/tmp/{k}.wav" for k in filler.canned_lines()})
        # FakeVoice latches speaking=True in speak_wav and nothing in a tick loop ever clears it,
        # so a fake left to itself reports the opener still playing forever and no stall could
        # follow it. Driving speech_in_flight from the same flag as is_playing keeps the two
        # consistent; the one test that needs them to DISAGREE says so explicitly.
        self.voice.speech_in_flight = lambda: self.playing
        return s

    def _busy(self, s):
        at = self.wake(s, T0)
        return self.speak_into(s, at + 1)

    def _filler_wavs(self):
        return [w for w in self.voice.spoken_wavs if str(w[0]).startswith("/tmp/filler_")]

    # the opener ---------------------------------------------------------------

    def test_silent_before_the_drawn_delay(self):
        # The head-start is the whole reason a fast reply never hears filler.
        s = self.make()
        at = self._busy(s)
        s.tick(at + FILLER_DELAY_JITTER_S[0] - 0.01)
        self.assertEqual(self._filler_wavs(), [])

    def test_opener_plays_once_the_delay_passes(self):
        s = self.make()
        at = self._busy(s)
        s.tick(at + FILLER_DELAY_JITTER_S[1])
        self.assertEqual(len(self._filler_wavs()), 1)
        self.assertTrue(self._filler_wavs()[0][0].startswith("/tmp/filler_op_"))
        self.assertEqual(s.state, STATE_BUSY, "the filler is not a state change")

    def test_exactly_one_opener_per_turn(self):
        s = self.make()
        at = self._busy(s)
        self.playing = True                      # nothing else can start while it plays
        for i in range(40):
            s.tick(at + FILLER_DELAY_JITTER_S[1] + i * 0.05)
        openers = [w for w in self._filler_wavs() if "filler_op_" in w[0]]
        self.assertEqual(len(openers), 1)

    def test_the_hmm_does_not_also_play(self):
        # Both firing would talk over each other; the filler supersedes it rather than joining it.
        s = self.make()
        at = self._busy(s)
        for i in range(40):
            s.tick(at + THINKING_SOUND_DELAY_S + i * 0.05)
        self.assertEqual([w for w in self.voice.spoken_wavs if w[0] == "/tmp/hmm.wav"], [])

    def test_off_falls_back_to_the_hmm(self):
        s = self.make()
        with patch("ai.session.FILLER_ENABLED", False):
            at = self._busy(s)
            s.tick(at + THINKING_SOUND_DELAY_S)
        self.assertEqual(self._filler_wavs(), [])
        self.assertIn("/tmp/hmm.wav", [w[0] for w in self.voice.spoken_wavs])

    def test_think_out_loud_off_is_completely_silent(self):
        s = self.make()
        settings.set_many({"thinking_sounds": False}, persist=False)
        at = self._busy(s)
        for i in range(40):
            s.tick(at + FILLER_DELAY_JITTER_S[1] + i * 0.1)
        self.assertEqual(self._filler_wavs(), [])
        self.assertEqual([w for w in self.voice.spoken_wavs if w[0] == "/tmp/hmm.wav"], [])

    # the stall loop -----------------------------------------------------------

    def test_stalls_keep_coming_for_a_long_wait(self):
        # The failure this exists for: an opener that covers 6 s and then 20 s of dead air.
        s = self.make()
        at = self._busy(s)
        for i in range(300):
            s.tick(at + FILLER_DELAY_JITTER_S[1] + i * 0.1)
        stalls = [w for w in self._filler_wavs() if "filler_st_" in w[0]]
        self.assertGreater(len(stalls), 5)

    def test_never_starts_a_line_while_one_is_playing(self):
        s = self.make()
        at = self._busy(s)
        s.tick(at + FILLER_DELAY_JITTER_S[1])            # opener
        before = len(self._filler_wavs())
        self.playing = True
        for i in range(60):
            s.tick(at + FILLER_DELAY_JITTER_S[1] + 1 + i * 0.5)
        self.assertEqual(len(self._filler_wavs()), before, "overlapping lines garble each other")

    def test_the_overlap_guard_covers_the_gap_before_playback_starts(self):
        # Bug 3, heard on the robot as fillers talking over each other. The loop used to guard on
        # tts.is_playing(), which only goes true once a playback PROCESS exists -- and speak_wav
        # hands off to a worker thread first. At 20 Hz that left several ticks where a line had been
        # started but was not yet "playing", and every one of them started another. So this is the
        # one case where the two seams must DISAGREE: nothing is playing, speech is in flight, and
        # nothing new may start.
        s = self.make()
        at = self._busy(s)
        s.tick(at + FILLER_DELAY_JITTER_S[1])            # opener
        before = len(self._filler_wavs())
        self.voice.speech_in_flight = lambda: True       # started; no playback process yet
        self.playing = False                             # is_playing() alone would say "go"
        for i in range(60):
            s.tick(at + FILLER_DELAY_JITTER_S[1] + 1 + i * 0.5)
        self.assertEqual(len(self._filler_wavs()), before,
                         "a line was started while another was still being handed to playback")

    def test_no_gap_exceeds_the_ceiling(self):
        # THE contract. Walked at 20 Hz (SESSION_TICK_HZ) with playback ending immediately, which
        # is the worst case: every gap is a full drawn silence with no audio covering it.
        s = self.make()
        at = self._busy(s)
        spoken, step = [], 1.0 / sess_mod.SESSION_TICK_HZ
        for i in range(400):
            now = at + i * step
            n = len(self._filler_wavs())
            s.tick(now)
            if len(self._filler_wavs()) > n:
                spoken.append(now)
        self.assertGreater(len(spoken), 5, "nothing was said, so nothing was measured")
        gaps = [b - a for a, b in zip(spoken, spoken[1:])]
        self.assertLess(max(gaps + [spoken[0] - at]), FILLER_MAX_SILENCE_S,
                        "a stretch of dead air exceeded the ceiling the bank exists to hold")

    def test_stalls_do_not_repeat_before_the_bank_is_exhausted(self):
        s = self.make()
        at = self._busy(s)
        for i in range(400):
            s.tick(at + i * 0.05)
        stalls = [w[0] for w in self._filler_wavs() if "filler_st_" in w[0]]
        first_pass = stalls[:min(len(stalls), 4)]
        self.assertEqual(len(set(first_pass)), len(first_pass))

    def test_a_lap_never_reopens_with_the_line_that_just_played(self):
        # The robot bug of 2026-08-09, walked end to end: the same short filler twice in one
        # exchange. A pool small enough to lap inside one wait is the trigger -- ceb and en each
        # held four lines against a wait that spends three or four -- and at the boundary the
        # rebuilt queue could put the line still ringing in the room next out. Three warm stalls
        # here so the boundary is crossed many times in one turn rather than at most once.
        s = self.make()
        keep = {"filler_op_tl_0", "filler_st_tl_0", "filler_st_tl_1", "filler_st_tl_2"}
        s._canned = {k: v for k, v in s._canned.items()
                     if not k.startswith("filler_") or k in keep}
        at = self._busy(s)
        # Bisaya has nothing warm here, and a turn that latched onto it would go silent and test
        # nothing. The share is a product choice, not part of this contract.
        with patch("ai.filler.FILLER_CEB_SHARE", 0.0):
            for i in range(800):
                s.tick(at + i * 0.05)
        stalls = [w[0] for w in self._filler_wavs() if "filler_st_" in w[0]]
        self.assertGreater(len(stalls), 6, "the pool never lapped, so nothing was tested")
        self.assertTrue(all(a != b for a, b in zip(stalls, stalls[1:])),
                        f"a stall repeated back to back: {stalls}")

    # the gap floor ------------------------------------------------------------

    def test_no_two_lines_land_closer_than_the_floor(self):
        # The other half of the contract from test_no_gap_exceeds_the_ceiling, and walked the same
        # way: 20 Hz, playback ending immediately, so every gap is a full drawn silence. Without a
        # floor the ceiling alone would push these toward zero and the bank would read as a queue
        # draining rather than as Kai thinking.
        s = self.make()
        at = self._busy(s)
        spoken, step = [], 1.0 / sess_mod.SESSION_TICK_HZ
        for i in range(400):
            now = at + i * step
            n = len(self._filler_wavs())
            s.tick(now)
            if len(self._filler_wavs()) > n:
                spoken.append(now)
        self.assertGreater(len(spoken), 5, "nothing was said, so nothing was measured")
        gaps = [b - a for a, b in zip(spoken, spoken[1:])] + [spoken[0] - at]
        # One tick of slack: the loop can only act on a tick boundary, so a gap drawn at exactly
        # the floor is observed up to one step late, never early.
        self.assertGreaterEqual(min(gaps), FILLER_MIN_GAP_S - step,
                                "a line came back before the listener got a beat of quiet")

    def test_the_ceiling_still_wins_if_the_floor_is_raised_past_it(self):
        # Dead air is the failure this module was built for. A floor raised past the ceiling must
        # not be able to break that promise -- it may only pin the gap, never extend it.
        s = self.make()
        ceiling = FILLER_MAX_SILENCE_S - FILLER_PLAYBACK_START_BUDGET_S
        with patch("ai.session.FILLER_MIN_GAP_S", ceiling + 5.0):
            drawn = [s._filler_gap(FILLER_DELAY_JITTER_S) for _ in range(50)]
        self.assertTrue(all(g == ceiling for g in drawn), f"the ceiling leaked: {set(drawn)}")

    # no repeats within one conversation ---------------------------------------

    def _next_turn(self, s, at):
        """Finish the turn in flight and start another one in the SAME conversation."""
        s._on_turn_done(self.voice.turns[-1]["epoch"], "done")
        at = self.finish_speaking(s, at + 10)
        return self.speak_into(s, at + 1)

    def _talk(self, s, at, turns=5, ticks=120):
        """Drive `turns` consecutive turns, returning every filler WAV played across all of them."""
        for _ in range(turns):
            for i in range(ticks):
                s.tick(at + FILLER_DELAY_JITTER_S[1] + i * 0.05)
            at = self._next_turn(s, at + FILLER_DELAY_JITTER_S[1] + ticks * 0.05)
        return [w[0] for w in self._filler_wavs()]

    def test_an_opener_is_never_reused_within_one_conversation(self):
        # avoid= only ever excluded the line before, so turn 3 could repeat turn 1 at 1-in-12.
        s = self.make()
        self.voice.language = "en"          # 4 openers, so a repeat is 1-in-4 if unguarded
        at = self._busy(s)
        played = [w for w in self._talk(s, at, turns=4) if "filler_op_" in w]
        self.assertEqual(len(played), 4, "every turn must still get an opener")
        self.assertEqual(len(set(played)), len(played), f"an opener came back: {played}")

    def test_a_stall_is_never_reused_within_one_conversation(self):
        # The queue is rebuilt per turn, so a fresh shuffle knows nothing about the last one's.
        # "Sandali ha" opening the stalls of three turns running is exactly what this prevents.
        # Tagalog with the Bisaya route off, so all 12 stalls come from one bank and three turns
        # cannot lap it -- an exhausted bank is ALLOWED to repeat, and would hide the regression.
        s = self.make()
        with patch("ai.filler.FILLER_CEB_SHARE", 0.0):
            at = self._busy(s)
            played = [w for w in self._talk(s, at, turns=3, ticks=60) if "filler_st_" in w]
        self.assertGreater(len(played), 3, "nothing was said across three turns, so nothing held")
        self.assertLess(len(played), 12, "the bank lapped, so a repeat here would be legitimate")
        self.assertEqual(len(set(played)), len(played), f"a stall came back: {played}")

    def test_a_new_conversation_gets_the_whole_bank_back(self):
        # The scope is the conversation, not the process: a demo would otherwise exhaust 12 Tagalog
        # stalls in one afternoon and spend the rest of it on the fallback path.
        s = self.make()
        self.voice.language = "en"
        at = self._busy(s)
        self._talk(s, at, turns=4)
        self.assertTrue(s._filler_used_openers and s._filler_used_stalls)
        s._end_session(at + 500, "no_speech")
        s._last_wake_t = at + 500 - WAKE_REFRACTORY_S - 1
        s.on_wake(at + 501)
        self.assertEqual(s._filler_used_openers, set(), "a new conversation starts fresh")
        self.assertEqual(s._filler_used_stalls, set())

    def test_waking_again_mid_conversation_keeps_the_history(self):
        # on_wake from LISTEN_WAIT is "give me my time back", not a new conversation -- it does not
        # reach _begin_session, so the bank must not quietly reset under a repeated "hey Kai".
        s = self.make()
        at = self._busy(s)
        self._talk(s, at, turns=2)
        used = set(s._filler_used_openers)
        self.assertTrue(used)
        s._enter_listen_wait(at + 200)      # _talk leaves a turn in flight; park it where a
                                            # repeated "hey Kai" is accepted rather than rejected
        s._last_wake_t = at + 200 - WAKE_REFRACTORY_S - 1
        s.on_wake(at + 201)
        self.assertEqual(s._filler_used_openers, used)

    def test_a_conversation_longer_than_the_bank_keeps_talking(self):
        # The no-repeat rule is a preference. Going silent once everything has been heard would
        # trade a small tell for the dead air the whole module exists to prevent.
        s = self.make()
        self.voice.language = "en"          # only 4 of each, so 6 turns must exhaust both
        at = self._busy(s)
        s._filler_used_openers.update(k for k in filler.canned_lines() if "_op_" in k)
        s._filler_used_stalls.update(k for k in filler.canned_lines() if "_st_" in k)
        for i in range(120):
            s.tick(at + FILLER_DELAY_JITTER_S[1] + i * 0.05)
        self.assertTrue(self._filler_wavs(), "an exhausted bank went silent instead of repeating")

    # per-turn state -----------------------------------------------------------

    def test_the_next_turn_starts_over(self):
        s = self.make()
        at = self._busy(s)
        s.tick(at + FILLER_DELAY_JITTER_S[1])
        s._on_turn_done(self.voice.turns[0]["epoch"], "done")
        at = self.finish_speaking(s, at + 10)
        at = self.speak_into(s, at + 1)
        before = len(self._filler_wavs())
        s.tick(at + FILLER_DELAY_JITTER_S[1])
        after = [w for w in self._filler_wavs()[before:] if "filler_op_" in w[0]]
        self.assertEqual(len(after), 1, "a second turn must get its own opener")

    def test_language_is_latched_for_the_whole_turn(self):
        # Reading it fresh per line would let STT landing mid-turn switch languages between the
        # opener and the stalls, which is worse than being consistently wrong.
        s = self.make()
        self.voice.language = "en"
        at = self._busy(s)
        s.tick(at + FILLER_DELAY_JITTER_S[1])
        self.voice.language = "tl"               # STT lands, reporting something else
        for i in range(200):
            s.tick(at + FILLER_DELAY_JITTER_S[1] + i * 0.1)
        langs = {w[0].rsplit("_", 2)[-2] for w in self._filler_wavs()}
        self.assertEqual(len(langs), 1, f"a turn switched language mid-way: {langs}")

    def test_leaves_the_turn_status_alone_so_no_chat_bubble_appears(self):
        s = self.make()
        at = self._busy(s)
        s.tick(at + FILLER_DELAY_JITTER_S[1])
        self.assertEqual(self.voice.said, [], "filler must never be routed through say()")
        status = s.get_status(at + FILLER_DELAY_JITTER_S[1])
        self.assertFalse(status["voice_speaking"], "voice_speaking would flip the mic button")
        self.assertNotEqual(status.get("voice_status"), "done")

    def test_busy_timeout_still_fires(self):
        s = self.make()
        at = self._busy(s)
        s.tick(at + FILLER_DELAY_JITTER_S[1])
        s.tick(at + SESSION_BUSY_MAX_S)
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(s.get_status()["sess_end_reason"], "busy_timeout")

    # prewarm ------------------------------------------------------------------

    def test_the_core_prewarm_does_not_carry_the_bank(self):
        # The 44-line burst is the bug this split replaces: it put a second Piper on the CPU beside
        # a live reply's, made tts.stop() kill the wrong process, and cost 24.6 s to first audio.
        # The four core lines go out on their own, and "Yes?" is never queued behind 40 fillers.
        s = self.make()
        s._prewarm_canned()
        keys = list(self.mock_prewarm_canned.call_args[0][0])
        self.assertEqual(keys, ["ack", "no_speech", "error", "thinking"])

    def test_the_bank_is_warmed_one_line_at_a_time(self):
        # Not "the bank gets warmed" but "never in a burst" -- one synth call per line IS the fix.
        s = self.make()
        s._canned = {}
        s._prewarm_bank()
        sizes = {len(c[0][0]) for c in self.mock_prewarm_canned.call_args_list}
        self.assertEqual(sizes, {1}, "a multi-line call is the burst that corrupted a live reply")
        asked = {k for c in self.mock_prewarm_canned.call_args_list for k in c[0][0]}
        self.assertEqual(asked, set(filler.canned_lines()))

    def test_the_core_lines_are_warmed_before_the_bank(self):
        # Strict ordering on one thread: the bank must never be synthesising while "Yes?" is still
        # not on disk. A missing filler line costs variety; a missing ack costs hands-free itself.
        s = self.make()
        s._canned = {}
        s._warm_all()
        keys = [k for c in self.mock_prewarm_canned.call_args_list for k in c[0][0]]
        self.assertEqual(keys[:4], ["ack", "no_speech", "error", "thinking"])
        self.assertTrue(all(k.startswith("filler_") for k in keys[4:]))

    def test_the_bank_does_not_synthesise_while_a_turn_is_live(self):
        # The quiet gate. BUSY means a reply is being synthesised through tts's single _synth_proc
        # slot, and starting a bank line there is precisely what made stop() kill the wrong process.
        s = self.make()
        s._canned = {}
        s._set_state(STATE_BUSY, T0)
        s._prewarm_bank()
        self.assertEqual(self.mock_prewarm_canned.call_args_list, [])

    def test_prewarm_off_leaves_the_bank_cold(self):
        # Off means SILENT, not slow: an unwarmed line is skipped, never synthesised on demand.
        s = self.make()
        s._canned = {}
        with patch("ai.session.FILLER_PREWARM", False):
            s._prewarm_bank()
        self.assertEqual(self.mock_prewarm_canned.call_args_list, [])

    def _synthesises_everything(self):
        """Make the mocked prewarm behave like a Piper that succeeds on every line."""
        self.mock_prewarm_canned.side_effect = (
            lambda lines, *a, **kw: {k: f"/tmp/{k}.wav" for k in lines})

    def _fails_then_succeeds(self, key: str, failures: int):
        """Make the mocked prewarm kill `key` its first `failures` times, as tts.stop() does when a
        turn starts, and succeed on everything else."""
        seen = {"n": 0}

        def synth(lines, *a, **kw):
            asked = list(lines)
            if asked == [key]:
                seen["n"] += 1
                if seen["n"] <= failures:
                    return {}
            return {k: f"/tmp/{k}.wav" for k in lines}

        self.mock_prewarm_canned.side_effect = synth
        return seen

    def test_a_killed_synthesis_is_retried_on_the_spot(self):
        # The bias measured on the robot 2026-08-09: openers failed on 55% of lines against the
        # stalls' 16%, because they are the longest and tts.stop() kills the one _synth_proc when a
        # turn starts. Deferring to the next pass is minutes away, so the opener tier sat empty
        # while the stalls filled -- and the turn opened with a stall instead of the long line.
        # Pinned to ONE pass, because the later passes are exactly what used to paper over this:
        # the line does come back eventually, minutes later, which is no use to the turn happening
        # now. What is being asserted is that it comes back inside the pass that lost it.
        s = self.make()
        s._canned = {}
        self._fails_then_succeeds("filler_op_tl_0", failures=1)
        with patch("ai.session.BANK_PASSES", 1):
            s._prewarm_bank()
        self.assertIn("filler_op_tl_0", s._canned, "a killed line waited for the next pass")

    def test_the_retry_happens_before_the_next_line_is_touched(self):
        # "Retried" is not enough: retried IMMEDIATELY is the fix. A retry that queues behind the
        # rest of the bank is the deferral this replaces, just with a shorter name.
        s = self.make()
        s._canned = {}
        self._fails_then_succeeds("filler_op_tl_0", failures=1)
        s._prewarm_bank()
        asked = [k for c in self.mock_prewarm_canned.call_args_list for k in c[0][0]]
        self.assertEqual(asked[:2], ["filler_op_tl_0", "filler_op_tl_0"])

    def test_retries_are_bounded(self):
        # A line that never comes back must not hold the bank hostage: everything behind it is
        # still unwarmed, and dead air is the failure the whole module exists to prevent.
        s = self.make()
        s._canned = {}
        self._fails_then_succeeds("filler_op_tl_0", failures=99)
        s._prewarm_bank()
        attempts = [k for c in self.mock_prewarm_canned.call_args_list
                    for k in c[0][0] if k == "filler_op_tl_0"]
        self.assertEqual(len(attempts), (BANK_LINE_RETRIES + 1) * BANK_PASSES)
        self.assertIn("filler_op_tl_1", s._canned, "the rest of the bank warmed anyway")

    def test_a_line_over_the_cap_is_not_retried(self):
        # Deterministic: it will measure the same length every time. Spending the retry budget on
        # it takes the window away from the lines a retry can actually save.
        s = self.make()
        s._canned = {}
        self._synthesises_everything()
        with patch("ai.session.tts.wav_duration", return_value=FILLER_MAX_LINE_S + 1.0):
            s._prewarm_bank()
        attempts = [k for c in self.mock_prewarm_canned.call_args_list
                    for k in c[0][0] if k == "filler_op_tl_0"]
        self.assertEqual(len(attempts), BANK_PASSES, "a rejected line was retried within the pass")

    def test_a_line_that_never_finds_quiet_is_not_retried(self):
        # The robot is busy RIGHT NOW, so an immediate retry is just as futile as the wait that
        # just timed out -- and every retry is another BANK_QUIET_WAIT_TRIES of standing still.
        s = self.make()
        s._canned = {}
        self._synthesises_everything()
        s._set_state(STATE_BUSY, T0)
        s._prewarm_bank()
        self.assertEqual(self.mock_prewarm_canned.call_args_list, [])

    def test_a_line_over_the_cap_is_never_cached(self):
        # The cap is enforced HERE, at prewarm, rather than warned about: the loop only ever
        # selects from cached keys, so a rejected line is unreachable rather than merely flagged.
        s = self.make()
        s._canned = {}
        self._synthesises_everything()
        with patch("ai.session.tts.wav_duration", return_value=FILLER_MAX_LINE_S + 1.0):
            s._prewarm_bank()
        self.assertEqual([k for k in s._canned if k.startswith("filler_")], [])

    def test_stalls_are_held_to_the_tighter_cap(self):
        # A stall must stay interruptible, so it gets FILLER_MAX_STALL_S and not the opener's
        # ceiling. At a length between the two, the openers survive and the stalls do not.
        s = self.make()
        s._canned = {}
        self._synthesises_everything()
        between = (FILLER_MAX_STALL_S + FILLER_MAX_LINE_S) / 2
        with patch("ai.session.tts.wav_duration", return_value=between):
            s._prewarm_bank()
        warm = [k for k in s._canned if k.startswith("filler_")]
        self.assertTrue(warm, "the openers are under the line cap and must still be cached")
        self.assertTrue(all(k.startswith(filler.OPENER_PREFIX) for k in warm))

    def test_a_measurement_failure_keeps_the_line(self):
        # wav_duration returns 0.0 on a header it cannot read. Dropping the whole bank because the
        # measurement broke would be a far worse failure than one long line.
        s = self.make()
        s._canned = {}
        self._synthesises_everything()
        with patch("ai.session.tts.wav_duration", return_value=0.0):
            s._prewarm_bank()
        self.assertEqual(len([k for k in s._canned if k.startswith("filler_")]),
                         len(filler.canned_lines()))

    # degrading when the bank is cold ------------------------------------------

    def test_a_cache_miss_is_silent_and_never_synthesises(self):
        # The bug this replaces: a miss fell through to live synthesis, which writes tts's FIXED
        # shared _RAW_WAV / _OUTPUT_WAV -- the same two files the reply was about to write. On the
        # robot the reply thread died with EOFError inside wave.open and sox reported "RIFF header
        # not found". Even without the collision, a live Piper run costs the reply 0.5-1.5 s of the
        # CPU it is competing for, so filler that synthesises itself lengthens the wait it covers.
        s = self.make()
        s._canned = {k: v for k, v in s._canned.items() if not k.startswith("filler_")}
        at = self._busy(s)
        for i in range(40):
            s.tick(at + FILLER_DELAY_JITTER_S[1] + i * 0.1)
        self.assertEqual(self._filler_wavs(), [])
        self.assertEqual(self.voice.spoken, [], "filler must never reach live synthesis")

    def test_a_cold_bank_hands_the_turn_back_to_the_hmm(self):
        # Silence is right for ONE missing line; silence for the whole turn is the dead air the
        # bank exists to remove. This is the regression guard for latching _filler_opened on the
        # empty draw: that consumed the turn's only False tick, and on any turn whose delay drew
        # under THINKING_SOUND_DELAY_S the fallback never got a tick to fire on.
        for draw in (FILLER_DELAY_JITTER_S[0], THINKING_SOUND_DELAY_S - 0.01,
                     FILLER_DELAY_JITTER_S[1]):
            with self.subTest(draw=draw):
                s = self.make()
                s._canned = {k: v for k, v in s._canned.items() if not k.startswith("filler_")}
                at = self._busy(s)
                s._filler_delay = draw          # after _arm_filler, which BUSY entry just ran
                for i in range(40):
                    s.tick(at + draw + i * 0.05)
                self.assertIn("/tmp/hmm.wav", [w[0] for w in self.voice.spoken_wavs],
                              "a cold bank left the turn completely silent")

    def test_warm_stalls_still_run_when_every_opener_is_missing(self):
        # Observed on the robot 2026-08-09: the length cap rejected all 20 openers, so the opener
        # branch returned False on every tick and the stall loop below it was unreachable. The turn
        # got one "Hmm" and then nothing, with 10 perfectly good stalls sitting warm on disk.
        #
        # This is the exact shape a partly warm boot produces too, since openers are the long lines
        # and therefore the ones most likely to be rejected or still cold.
        s = self.make()
        s._canned = {k: v for k, v in s._canned.items()
                     if not k.startswith(filler.OPENER_PREFIX)}
        at = self._busy(s)
        for i in range(200):
            s.tick(at + FILLER_DELAY_JITTER_S[1] + i * 0.05)
        self.assertIn("/tmp/hmm.wav", [w[0] for w in self.voice.spoken_wavs],
                      "the Hmm must still cover the opening")
        stalls = [w for w in self._filler_wavs() if "filler_st_" in w[0]]
        self.assertGreater(len(stalls), 2, "warm stalls were stranded behind a cold opener tier")

    def test_the_hmm_only_hands_off_once_it_has_actually_played(self):
        # The other half of the same branch. If the latch did not wait for the Hmm, a turn whose
        # delay drew under THINKING_SOUND_DELAY_S would spend its one False tick too early and go
        # silent end to end -- no opener, no Hmm, no stalls.
        s = self.make()
        s._canned = {k: v for k, v in s._canned.items()
                     if not k.startswith(filler.OPENER_PREFIX)}
        at = self._busy(s)
        s._filler_delay = THINKING_SOUND_DELAY_S - 0.2      # draws below the fallback's own delay
        s.tick(at + s._filler_delay)
        self.assertFalse(s._filler_opened, "latched before the Hmm had any chance to play")
        for i in range(200):
            s.tick(at + THINKING_SOUND_DELAY_S + i * 0.05)
        self.assertIn("/tmp/hmm.wav", [w[0] for w in self.voice.spoken_wavs])
        self.assertTrue([w for w in self._filler_wavs() if "filler_st_" in w[0]])

    def test_a_half_warm_bank_uses_only_the_lines_it_has(self):
        # The normal state for the first minutes after boot, since the bank warms one line at a
        # time between turns. Choosing an uncached key would spend a slot and play nothing --
        # a gap exactly where the ceiling says there must not be one.
        s = self.make()
        keep = {f"{filler.OPENER_PREFIX}_tl_0", f"{filler.STALL_PREFIX}_tl_0",
                f"{filler.STALL_PREFIX}_tl_1"}
        s._canned = {k: v for k, v in s._canned.items()
                     if not k.startswith("filler_") or k in keep}
        at = self._busy(s)
        for i in range(200):
            s.tick(at + FILLER_DELAY_JITTER_S[1] + i * 0.05)
        played = {w[0] for w in self._filler_wavs()}
        self.assertTrue(played, "a partly warm bank must still talk")
        self.assertTrue(played <= {f"/tmp/{k}.wav" for k in keep},
                        f"an uncached line was selected: {played}")


class TestLiveSettings(SessionCase):
    """The dashboard-settable knobs that reach the session. Called from a Flask request thread."""

    def test_hands_free_off_lands_in_disabled(self):
        s = self.make()
        out = s.set_hands_free(False)
        self.assertFalse(out["hands_free"])
        self.assertEqual(s.state, STATE_DISABLED)
        self.assertFalse(s.enabled)
        self.assertFalse(self.mic.wake_enabled,
                         "the audio worker must stop feeding the wake engine")

    def test_hands_free_off_mid_conversation_ends_it(self):
        # Through the ordinary teardown, so playback stops, the buffer is dropped and the epoch bumps
        # — not a second, parallel shutdown path.
        s = self.make()
        self.wake(s)
        self.assertEqual(s.state, STATE_LISTEN_WAIT)
        s.set_hands_free(False)
        self.assertEqual(s.state, STATE_DISABLED)
        self.assertEqual(s.get_status()["sess_end_reason"], "hands_free_off")

    def test_hands_free_back_on_returns_to_idle(self):
        s = self.make(enabled=False)
        self.assertEqual(s.state, STATE_DISABLED)
        out = s.set_hands_free(True)
        self.assertTrue(out["hands_free"])
        self.assertTrue(out["wake_live"])
        self.assertEqual(s.state, STATE_IDLE)
        self.assertTrue(self.mic.wake_enabled)

    def test_hands_free_on_opens_the_engine_if_it_never_was(self):
        # Hands-free may have been off at startup, in which case wake.open() has not run. Geometry must
        # sync AFTER open(), since the winning tier decides the frame size.
        s = self.make(enabled=False, wake_ready=False)
        self.mic.wake.ready = False
        s.set_hands_free(True)
        self.mic.wake.open.assert_called_once()
        self.assertTrue(self.mic.geometry_syncs >= 1)

    def test_hands_free_on_with_no_usable_engine_stays_disabled(self):
        # "Off" is honest; "on but never ready" reads as broken.
        s = self.make(enabled=False, wake_ready=False)
        self.mic.wake.ready = False
        self.mic.wake.open.return_value = False
        out = s.set_hands_free(True)
        self.assertFalse(out["wake_live"])
        self.assertEqual(s.state, STATE_DISABLED)

    def test_rms_floor_reaches_the_gate_and_is_reported(self):
        # sess_rms + sess_rms_floor is the mic-tuning pair; reporting the config constant while the
        # gate used a different value would make the one number an operator trusts a lie.
        s = self.make()
        s.set_rms_floor(1234.0)
        self.assertEqual(s._gate.rms_floor, 1234.0)
        self.assertEqual(s.get_status()["sess_rms_floor"], 1234.0)

    def test_wake_sensitivity_is_delegated_to_the_mic(self):
        s = self.make()
        s.set_wake_sensitivity(0.8)
        self.assertEqual(self.mic.wake_sensitivities, [0.8])

    def test_reprewarm_clears_the_cached_lines(self):
        # Otherwise "Yes?" keeps the old voice forever after a volume or rate change.
        s = self.make()
        self.assertTrue(s._canned)
        s.reprewarm_canned()
        self.assertFalse(s._canned, "the stale WAVs must go immediately, before re-synthesis")

    def test_rewarm_waits_for_a_reply_to_finish(self):
        # tts has one synth slot and stop() cancels it, so re-synthesising during a reply gets killed
        # and the ack silently goes missing — which puts the dead air back before "Yes?".
        s = self.make()
        self.voice.speaking = True
        with patch("ai.session.tts.is_playing", return_value=False), \
             patch("ai.session.REWARM_QUIET_WAIT_S", 0.4), \
             patch("ai.session.REWARM_RETRY_S", 0.01), \
             patch.object(s, "_prewarm_canned") as warm:
            s._rewarm_when_quiet()
        self.assertTrue(warm.called, "it must still run once the wait budget expires")

    def test_rewarm_retries_when_the_ack_did_not_land(self):
        s = self.make(canned=False)
        with patch("ai.session.tts.is_playing", return_value=False), \
             patch("ai.session.REWARM_QUIET_WAIT_S", 0.0), \
             patch("ai.session.REWARM_RETRY_S", 0.01), \
             patch.object(s, "_prewarm_canned") as warm:
            s._rewarm_when_quiet()
        self.assertEqual(warm.call_count, 2, "a cancelled ack synthesis gets one more chance")

    def test_rewarm_does_not_retry_when_the_ack_landed(self):
        s = self.make()          # canned already populated, including "ack"
        with patch("ai.session.tts.is_playing", return_value=False), \
             patch("ai.session.REWARM_QUIET_WAIT_S", 0.0), \
             patch("ai.session.REWARM_RETRY_S", 0.01), \
             patch.object(s, "_prewarm_canned") as warm:
            s._rewarm_when_quiet()
        self.assertEqual(warm.call_count, 1)

    def test_rewarm_retries_when_any_line_is_missing_not_just_the_ack(self):
        # The check used to be `"ack" not in self._canned`, so a re-warm that lost ANY other line
        # never retried — observed as "cached spoken lines: ack, error, no_speech" with `thinking`
        # silently gone. `thinking` is the likeliest casualty: it is the length-fitted line, so it
        # costs several Piper passes and has the widest window in which stop() can cancel it.
        s = self.make()
        s._canned = {k: "/tmp/x.wav" for k in s._canned_lines() if k != "thinking"}
        with patch("ai.session.tts.is_playing", return_value=False), \
             patch("ai.session.REWARM_QUIET_WAIT_S", 0.0), \
             patch("ai.session.REWARM_RETRY_S", 0.01), \
             patch.object(s, "_prewarm_canned") as warm:
            s._rewarm_when_quiet()
        self.assertEqual(warm.call_count, 2, "a missing 'thinking' must get the same retry as ack")

    def test_status_stays_json_safe_with_the_new_keys(self):
        import json

        s = self.make()
        s.set_rms_floor(900.0)
        json.dumps(s.get_status())

    def test_person_name_is_published(self):
        """sess_person is read from the assistant rather than mirrored on the session, so the
        dashboard cannot show a name the prompt no longer carries. See docs/tickets/S12."""
        s = self.make()
        self.assertEqual(s.get_status()["sess_person"], "")
        self.voice.person_name = "Jhondel"
        self.assertEqual(s.get_status()["sess_person"], "Jhondel")

    def test_person_name_survives_a_wake_mid_conversation(self):
        """Saying "hey Kai" again while it is already listening continues the conversation and keeps
        its history (see _begin_session's note). The name has to follow the same rule, or repeating
        the wake word mid-chat quietly turns you back into a stranger."""
        s = self.make()
        self.wake(s, T0)
        self.voice.person_name = "Jhondel"
        self.wake(s, T0 + 2)
        self.assertEqual(s.get_status(T0 + 3)["sess_person"], "Jhondel")

    def test_person_name_is_gone_after_the_session_ends(self):
        s = self.make()
        self.wake(s, T0)
        self.voice.person_name = "Jhondel"
        s.end_session("manual")
        self.assertEqual(s.get_status()["sess_person"], "",
                         "the next person must not inherit the last one's name")


class TestSessionStartRetrySchedule(unittest.TestCase):
    """The start-retry window must outlast startup contention, which is the only thing it exists for.

    2026-08-07: it was 5 attempts x 5 s = 25 s, and that boot's log had `[llm] MODEL RELOADED:
    26215ms` inside exactly that window, plus MediaPipe init and a HuggingFace fetch. Every attempt
    lost the I2S probe, Kai fell back to the 44.1 kHz USB dongle (undecimatable to 16 kHz, so NO mic
    opened), and then gave up permanently — hands-free dead for the whole run, with no error.

    A retry schedule shorter than the storm it is racing cannot win, so the span is the assertion.
    """

    def _span(self):
        sched = SESSION_START_BACKOFF_S
        return sum(sched[min(i, len(sched) - 1)] for i in range(SESSION_START_ATTEMPTS - 1))

    def test_the_window_outlasts_a_slow_model_load(self):
        # The measured storm was ~26 s for the LLM alone, with other init overlapping it. 25 s was
        # not enough; anything under ~2 minutes is betting against the machine.
        self.assertGreater(self._span(), 120.0,
                           "retries that all land inside the boot storm cannot succeed")

    def test_the_first_retry_is_still_prompt(self):
        # The common case is a transient loss of the single-opener device (a previous face_track
        # still exiting). That must recover in seconds, not after a long backoff.
        self.assertLessEqual(SESSION_START_BACKOFF_S[0], 5.0)

    def test_the_schedule_never_decreases(self):
        # A backoff that dips would spend its late, valuable attempts at early, useless intervals.
        for a, b in zip(SESSION_START_BACKOFF_S, SESSION_START_BACKOFF_S[1:]):
            self.assertLessEqual(a, b)

    def test_there_are_more_attempts_than_schedule_entries(self):
        # The last value is meant to repeat (same idiom as MIC_REOPEN_BACKOFF_S). If the schedule
        # were longer than the attempt count, its tail would be dead config.
        self.assertGreater(SESSION_START_ATTEMPTS, len(SESSION_START_BACKOFF_S))


class TestMicError(SessionCase):
    def test_reports_the_underlying_mic_error(self):
        # Surfaced so the retry loop can log the reason on the SAME line as the retry. The reason
        # and the retry landing in separate lines is what made this failure read as unexplained.
        s = self.make()
        s._mic.error = "cannot resample 44100 Hz: decimation needs an integer ratio"
        self.assertIn("44100", s.mic_error())

    def test_is_empty_when_there_is_no_error(self):
        s = self.make()
        s._mic.error = None
        self.assertEqual(s.mic_error(), "")


class TestMicHotPlug(SessionCase):
    """Plugging a mic into a running robot, and when it is safe to act on that.

    The watcher only reports that the set of sound cards changed; every decision below is the
    session's, because it is the only thing that knows whether Kai is mid-conversation.
    """

    def test_a_settled_change_re_resolves_the_mic(self):
        s = self.make()
        with patch.object(s, "reresolve_mic", return_value={"ok": True, "kind": "usb",
                                                            "device": 1}) as re:
            self.hotplug.fire = True
            s.tick(T0 + 1)
        re.assert_called_once()
        self.assertEqual(s.get_status()["sess_mic_hotswaps"], 1)

    def test_nothing_happens_without_a_change(self):
        s = self.make()
        with patch.object(s, "reresolve_mic") as re:
            for i in range(10):
                s.tick(T0 + i)
        re.assert_not_called()

    def test_a_change_mid_turn_is_deferred_not_dropped(self):
        # Re-resolving takes the mic away for seconds. Doing that during an utterance would drop the
        # turn on the floor and the person talking would have no idea why — but discarding the
        # change would mean the mic they just plugged in never gets picked up, because the watcher
        # reports each change exactly once.
        s = self.make()
        at = self.wake(s, T0)
        self.assertEqual(s.state, STATE_LISTEN_WAIT)
        with patch.object(s, "reresolve_mic", return_value={"ok": True}) as re:
            self.hotplug.fire = True
            s.tick(at + 1)
            re.assert_not_called()
            s._end_session(at + 2, "test")              # back to idle
            s.tick(at + 3)
            re.assert_called_once()

    def test_a_deaf_robot_is_the_best_moment_to_act(self):
        # STATE_DISABLED is where a robot with no working mic sits, so it must count as quiet.
        s = self.make(wake_ready=False)
        self.assertEqual(s.state, STATE_DISABLED)
        with patch.object(s, "reresolve_mic", return_value={"ok": True}) as re:
            self.hotplug.fire = True
            s.tick(T0 + 1)
        re.assert_called_once()

    def test_the_cooldown_suppresses_a_second_swap(self):
        # Anti-flap: a cable with a bad contact must not be able to spend the session tearing the
        # capture stream down.
        s = self.make()
        with patch.object(s, "reresolve_mic", return_value={"ok": True}) as re:
            self.hotplug.fire = True
            s.tick(T0 + 1)
            self.hotplug.fire = True
            s.tick(T0 + 2)
            self.assertEqual(re.call_count, 1)
            self.hotplug.fire = True
            s.tick(T0 + 1 + MIC_HOTPLUG_COOLDOWN_S + 1)
            self.assertEqual(re.call_count, 2)

    def test_a_change_during_the_cooldown_is_honoured_when_it_expires(self):
        # The pending flag survives the cooldown, so a real change inside one is delayed, not lost.
        s = self.make()
        with patch.object(s, "reresolve_mic", return_value={"ok": True}) as re:
            self.hotplug.fire = True
            s.tick(T0 + 1)
            self.hotplug.fire = True                    # arrives mid-cooldown
            s.tick(T0 + 2)
            self.assertEqual(re.call_count, 1)
            s.tick(T0 + 1 + MIC_HOTPLUG_COOLDOWN_S + 1)  # no new change, but one is still pending
            self.assertEqual(re.call_count, 2)

    def test_the_re_resolve_runs_outside_the_session_lock(self):
        # It tears the stream down, re-scans PortAudio and runs liveness probes — seconds of
        # blocking work, several hundred tick periods. The lock is an RLock, so doing it inside the
        # tick's locked section would have been legal and would have starved the audio worker,
        # which needs that same lock ~30 times a second for the VAD.
        s = self.make()
        held = {"free": False}

        def check():
            # The tick thread owns the RLock re-entrantly if it never released it, so asking for it
            # again from HERE would succeed no matter what. Another thread failing to take it is the
            # only honest test. Acquire and release both happen on that thread — an RLock can only
            # be released by its owner.
            import threading as _t

            def grab():
                if s._lock.acquire(timeout=0.5):
                    held["free"] = True
                    s._lock.release()

            t = _t.Thread(target=grab)
            t.start()
            t.join()
            return {"ok": True}

        with patch.object(s, "reresolve_mic", side_effect=check):
            self.hotplug.fire = True
            s.tick(T0 + 1)
        self.assertTrue(held["free"], "the session lock was still held during the re-resolve")

    def test_a_failed_swap_is_reported_and_changes_no_state(self):
        # The cards changed, we looked, nothing usable came back. Not a reason to end anything.
        s = self.make()
        with patch.object(s, "reresolve_mic",
                          return_value={"ok": False, "error": "every device read as silent"}):
            self.hotplug.fire = True
            s.tick(T0 + 1)
        self.assertEqual(s.state, STATE_IDLE)
        self.assertEqual(s.get_status()["sess_mic_hotswaps"], 1)

    def test_the_watcher_is_polled_with_the_ticks_clock(self):
        # No time.monotonic() of its own — that is what lets these tests step it without sleeping.
        s = self.make()
        s.tick(T0 + 7)
        self.assertEqual(self.hotplug.polls[-1], T0 + 7)

    def test_status_reports_whether_hot_plug_is_being_watched(self):
        # It disables itself where /proc/asound/cards does not exist, and an operator staring at a
        # robot that will not notice a new mic needs to be able to see that.
        s = self.make()
        status = s.get_status()
        self.assertIn("sess_mic_hotplug_watching", status)
        self.assertIsInstance(status["sess_mic_hotplug_watching"], bool)
        self.assertEqual(status["sess_mic_kind"], "i2s")

class TestWatchingForAMicWhenStartupGaveUp(SessionCase):
    """The case the tick loop cannot cover: start() failed, so there is no tick loop.

    That is exactly the state a robot boots into when its mic is missing, which makes it the most
    likely moment for someone to plug one in — so "hot-plug works everywhere except when Kai is
    already deaf" would have been the wrong shape of feature.
    """

    def test_it_starts_the_session_when_a_mic_appears(self):
        s = self.make()
        stop = threading.Event()
        with patch.object(s, "reresolve_mic", return_value={"ok": True}) as re:
            self.hotplug.fire = True
            self.assertTrue(s.watch_for_a_mic(stop))
        re.assert_called_once()

    def test_it_keeps_waiting_while_nothing_changes(self):
        s = self.make()
        stop = threading.Event()
        with patch.object(s, "reresolve_mic") as re, \
             patch.object(stop, "wait", side_effect=lambda t: stop.set()):
            self.assertFalse(s.watch_for_a_mic(stop))
        re.assert_not_called()

    def test_it_keeps_waiting_when_the_mic_that_appeared_is_unusable(self):
        # A card changed but nothing usable came back. Not a reason to stop watching — the next
        # change might be the real mic.
        s = self.make()
        stop = threading.Event()
        with patch.object(s, "reresolve_mic", return_value={"ok": False}) as re, \
             patch.object(stop, "wait", side_effect=lambda t: stop.set()):
            self.hotplug.fire = True
            self.assertFalse(s.watch_for_a_mic(stop))
        re.assert_called_once()

    def test_it_returns_immediately_when_hot_plug_is_unavailable(self):
        # No /proc/asound/cards — nothing to watch, so blocking a thread on it forever would be a
        # thread that can never do anything.
        s = self.make()
        self.hotplug.active = False
        with patch.object(s, "reresolve_mic") as re:
            self.assertFalse(s.watch_for_a_mic(threading.Event()))
        re.assert_not_called()

    def test_stopping_ends_the_watch(self):
        s = self.make()
        stop = threading.Event()
        stop.set()
        self.assertFalse(s.watch_for_a_mic(stop))


class TestReresolveRescansDevices(SessionCase):
    def test_the_never_started_path_re_scans_portaudio_first(self):
        # MicStream.reopen() does this for the running case. Nothing did it for this one, so the
        # button most needed when Kai has no mic would have searched a device list snapshotted
        # before the mic was plugged in.
        s = self.make()
        calls = []
        with patch("ai.session.refresh_devices", side_effect=lambda: calls.append("refresh")), \
             patch.object(s, "start", side_effect=lambda: calls.append("start") or True):
            s.reresolve_mic()
        self.assertEqual(calls, ["refresh", "start"])

    def test_the_running_path_leaves_the_rescan_to_the_stream(self):
        # reopen() owns the only window where it is safe — after the stream is closed. Doing it here
        # too would be a second rescan against a stream that is still open.
        s = self.make()
        s._thread = MagicMock()
        s._thread.is_alive.return_value = True
        with patch("ai.session.refresh_devices") as refresh:
            s.reresolve_mic()
        refresh.assert_not_called()
        self.assertEqual(self.mic.reopens_requested, 1)

if __name__ == "__main__":
    unittest.main()
