"""Hands-free conversation: a wake word, turn-taking, and a session that ends itself.

ConversationSession owns the state machine, every timer, and all the counters. It does not import
Flask, cv2, mediapipe or face_track; presence arrives as an injected callable and the microphone as
an injected object, so the session-end rules are testable with a fake clock and a fake camera.

The mic it drives is ai/mic_stream.MicStream — the process's single always-open capture stream.
The seam between them is three callbacks (on_wake, on_audio, muted): the stream knows nothing about
conversation states, and this file knows nothing about PortAudio.

Two design choices carry most of the weight:

* Timers are deadlines scoped to a single state, never accumulators that pause and resume. Entering a
  state arms its timers; leaving destroys them. This makes "the 15 s silence timer must not fire while
  Kai is thinking" true by construction rather than by remembering to pause it on ~20 transition
  edges — one missed resume is a session that never ends, one missed pause is a session killed in the
  middle of a 50 s cold model load. A single 20 Hz tick thread owns the clock; there is no
  threading.Timer anywhere, because a cancelled-but-already-fired Timer is a double transition.

* Presence is three-valued. "No face" and "no idea" are different: the face feed stops entirely on a
  camera stall or with --no-camera, and reading that silence as absence would end every conversation
  whenever the camera hiccuped.
"""

from __future__ import annotations

import os
import random
import threading
import time

import numpy as np

from ai import filler, tts
from ai.audio import SpeechGate
from ai.audio_debug import UtteranceRecorder
from ai.mic_device import refresh_devices
from ai.mic_hotplug import CardWatcher
from ai.mic_stream import MicStream
from ai.voice_assistant import (
    STATUS_DONE, STATUS_ERROR, STATUS_IDLE, STATUS_RECORDING, STATUS_TRANSCRIBING,
)
from ai.wake_phrase import match_wake_phrase
from config.filler import (
    BANK_LINE_RETRIES, BANK_PASSES, BANK_QUIET_POLL_S, BANK_QUIET_WAIT_TRIES, BANK_SYNTH_GAP_S,
    FILLER_DEFAULT_LANG, FILLER_DELAY_JITTER_S, FILLER_ENABLED, FILLER_MAX_LINE_S,
    FILLER_MAX_SILENCE_S, FILLER_MAX_STALL_S, FILLER_MIN_GAP_S, FILLER_OPENER_COOLDOWN_TURNS,
    FILLER_PLAYBACK_START_BUDGET_S, FILLER_PREWARM, FILLER_STALL_GAP_JITTER_S,
)
from config.thinking import THINKING_SOUND_DELAY_S, THINKING_SOUND_TARGET_S, THINKING_SOUND_TEXT
from config.voice import MIC_HOTPLUG_COOLDOWN_S, MIC_HOTPLUG_POLL_S, SAMPLE_RATE
from config.wake import (
    ACK_PRESYNTH, ACK_WAV_DIR, CANNED_ERROR, CANNED_NO_SPEECH,
    DEBUG_CAPTURE_DIR, DEBUG_CAPTURE_ENABLED,
    DEBUG_CAPTURE_KINDS, DEBUG_CAPTURE_MAX_FILES, DEBUG_CAPTURE_MAX_MB, GREETING_ENABLED,
    GREETING_REPEAT_SUPPRESS_S, GREETING_STAMP_PATH, GREETING_TEXT, HANDS_FREE_ENABLED,
    MAX_UTTERANCE_S, MIC_LEGACY_CAPTURE,
    MIN_UTTERANCE_S, SESSION_BUSY_MAX_S, SESSION_MAX_ERROR_STREAK,
    SESSION_MAX_NO_SPEECH_STREAK, SESSION_NO_FACE_S, SESSION_NO_SPEECH_S,
    SESSION_SPEAK_GRACE_S, SESSION_SPEAK_MAX_UNKNOWN_S, SESSION_TICK_HZ,
    VAD_HANGOVER_S, WAKE_ACK_MAX_S, WAKE_ACK_TEXT, WAKE_ALLOW_BARGE_IN,
    WAKE_REFRACTORY_S, WAKE_SCAN_HANGOVER_S,
    WAKE_WHISPER_CHECK_MAX_S, WAKE_WHISPER_COOLDOWN_S, WAKE_WHISPER_LOG_CHARS,
    WAKE_WHISPER_LOG_TEXT, WAKE_WHISPER_LONG_COOLDOWN_S, WAKE_WHISPER_MAX_UTTERANCE_S,
    WAKE_WHISPER_MIN_UTTERANCE_S,
)
# PULL, not a set_* method: thinking_sounds' only effect is being read, once per turn, so there is
# nothing for a callback to push. settings.py imports nothing from ai/, so this is not a cycle.
import settings

# ── Session states ────────────────────────────────────────────────────────────────────────────
# Plain strings, matching voice_assistant.py's STATUS_* convention, so they serialize onto /params
# for free. These are a SEPARATE axis from voice_status: that field is a wire contract the dashboard
# reads by exact string match (a chat bubble is appended on the transition into "done", the mic
# button is gated on "recording"/"thinking"), so new values there would silently break the UI.
# _project_status() below maps session state onto the six values the dashboard already knows.
STATE_DISABLED = "disabled"          # no key / no .ppn / hands-free off — push-to-talk only
STATE_IDLE = "idle"                  # listening for "Hey Kai" and nothing else
STATE_ACK = "ack"                    # playing the cached "Yes?"
STATE_COOLDOWN = "cooldown"          # Kai's audio has stopped but the amp hasn't settled
STATE_LISTEN_WAIT = "listen_wait"    # mic open, waiting for the user to start talking
STATE_LISTEN_SPEECH = "listen_speech"  # capturing an utterance
STATE_BUSY = "busy"                  # STT + LLM running
STATE_SPEAKING = "speaking"          # playing a reply
# The whisper wake tier only. It cannot decide from a frame, so it captures a candidate utterance
# while idle and transcribes it to see whether the wake phrase was in there.
STATE_SCAN_SPEECH = "scan_speech"    # capturing a candidate utterance while idle
STATE_SCAN_CHECK = "scan_check"      # transcribing it to check for the phrase

# States in which Kai's own audio could reach the mic, so the gate must hold it shut.
# The SCAN states are deliberately NOT here: Kai isn't speaking, and muting during a scan would make
# the tier unable to hear the very utterance it is capturing. (Adding new states to this tuple is the
# natural instinct — don't.)
_SPEECH_STATES = (STATE_ACK, STATE_BUSY, STATE_SPEAKING, STATE_COOLDOWN)
_SCAN_STATES = (STATE_SCAN_SPEECH, STATE_SCAN_CHECK)

# Re-synthesising the canned lines after a voice change shares tts's single synth slot, so it waits
# for any reply in flight rather than being cancelled by one. Implementation cadence, not a knob —
# same reasoning as RECONNECT_INTERVAL in servo/servo.py.
REWARM_QUIET_WAIT_S = 20.0   # give a long reply time to finish before re-synthesising
REWARM_RETRY_S = 1.5         # one retry if the ack still got cancelled

# The greeting is spoken on the same warm thread that synthesises the bank, so the thread waits for
# it to finish playing before starting the next Piper run — otherwise the bank's first line lands on
# tts's single synth slot mid-greeting and _begin_speech's stop() kills one of the two. Bounded, for
# the same reason REWARM_QUIET_WAIT_S is: a playback that never reports done must not park the warm
# thread forever, and the bank is best-effort anyway.
GREETING_QUIET_WAIT_S = 20.0
GREETING_POLL_S = 0.25

# How often a persistently broken presence feed may log. Reached from the 20 Hz tick and from every
# /params snapshot, so an unthrottled line here would run at hundreds per minute — the same failure
# NO_FACE_LOG_INTERVAL_S exists to prevent (see config/tracking.py: NO FACE was 58% of a 1.5-hour
# log). Implementation cadence, not a knob.
PRESENCE_ERROR_LOG_S = 60.0


def _greeting_age(now: float | None = None) -> float | None:
    """Seconds since ANY Kai process last spoke the greeting, or None if that isn't known.

    Wall clock (the stamp's mtime), deliberately not monotonic: the whole point is that a different
    process wrote it, and monotonic clocks are not comparable across processes.

    A stamp dated in the FUTURE reads as unknown rather than as recent. That happens for real on this
    board — no RTC battery, so the clock steps when NTP lands — and unknown is the safe direction:
    the failure this feature must never cause is a robot that stops greeting."""
    try:
        mtime = os.stat(GREETING_STAMP_PATH).st_mtime
    except OSError:
        return None            # never greeted, /tmp cleared by a reboot, or the path is unreadable
    age = (time.time() if now is None else now) - mtime
    return age if age >= 0.0 else None


def _mark_greeting_spoken() -> None:
    """Record that the greeting is being spoken NOW, for the next process to read.

    Written BEFORE the audio starts, not after: the case this exists for is a process that dies
    partway through the greeting, and one that stamped only on completion would never stamp at all —
    which is precisely the run whose relaunch must stay quiet.

    Best-effort. A stamp that cannot be written costs a duplicated greeting, so it must never cost
    the greeting itself."""
    try:
        os.makedirs(os.path.dirname(GREETING_STAMP_PATH) or ".", exist_ok=True)
        with open(GREETING_STAMP_PATH, "w") as fh:
            fh.write(f"{time.time():.0f}\n")   # mtime is what's read; the text is for a human
    except OSError as exc:
        print(f"[session] WARNING: could not write the greeting stamp "
              f"({GREETING_STAMP_PATH}: {exc}) — a crash-relaunch may greet twice", flush=True)


class ConversationSession:
    """The hands-free state machine: wake word in, spoken reply out, and a session that ends itself.

    `presence` is a callable returning (visible, seconds_since_seen, is_fresh) — vision/presence.py's
    snapshot, injected so this class never imports the tracking stack and can be tested with a fake.
    `hotplug` is a CardWatcher, injected for the same reason: the fake-clock tests step it through an
    enumeration without touching /proc or sleeping.
    """

    def __init__(self, assistant, presence=None, mic=None, enabled: bool | None = None,
                 hotplug=None) -> None:
        self._voice = assistant
        self._presence = presence
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.enabled = HANDS_FREE_ENABLED if enabled is None else enabled
        self._mic = mic if mic is not None else MicStream(
            on_wake=self.on_wake, on_audio=self._on_audio, muted=self.mic_muted,
        )
        self._gate = SpeechGate(rate=SAMPLE_RATE)
        # Mic hot-plug. The watcher only reports that the card set changed; every decision about
        # whether it is a good moment to act on that lives in _tick, because this class is the only
        # thing that knows whether Kai is mid-conversation.
        self._hotplug = hotplug if hotplug is not None else CardWatcher()
        self._hotplug_pending = False   # a change seen mid-turn, waiting for an idle tick
        self._hotplug_last = 0.0        # when the last automatic re-resolve ran (cooldown)
        self._hotswaps = 0
        self._canned: dict[str, object] = {}
        # The startup greeting is once per PROCESS, not once per start(). start() is re-entered by
        # reresolve_mic() whenever the operator retries a mic that failed to come up at boot — which
        # on this robot is a normal thing to do more than once (see reresolve_mic) — and greeting the
        # room again on each attempt would turn a recovery button into a talking one.
        self._greeted = False

        self._state = STATE_DISABLED
        self._state_since = 0.0
        self._epoch = 0
        self._manual_end = False       # push-to-talk turn: no max-utterance cap, human ends it
        self._played_thinking = False   # the "hmm" is once per turn; _set_state re-arms it on BUSY
        self._speak_deadline = 0.0

        # Filler bank state. All of it is re-armed by _arm_filler on every entry to BUSY, so a turn
        # can never inherit the previous turn's queue or its already-spoken flag. _filler_rng is a
        # private Random rather than the module-level one: the sweep in app/control_loop.py draws
        # from its own too, and sharing global state across two independent randomised features
        # makes each one's behaviour depend on how often the other ran.
        self._filler_rng = random.Random()
        self._filler_lang = FILLER_DEFAULT_LANG   # latched at the opener, so a turn never switches
        self._filler_delay = 0.0                  # drawn per turn, clamped against the ceiling
        self._filler_opened = False
        self._filler_queue: list[str] = []
        self._filler_next_at: float | None = None  # None = waiting for playback to end
        # The openers of the last FILLER_OPENER_COOLDOWN_TURNS exchanges, oldest first, so a long
        # line sits out a couple of turns before it can come back. One entry (the old
        # _filler_last_opener) only ruled out back-to-back, which on the four-line ceb/en pools left
        # A B A B available for the rest of a conversation once the used-set had emptied.
        self._filler_recent_openers: list[str] = []
        self._filler_last_stall = ""               # the same, for stalls; see the note below
        # Every filler key already spent in THIS conversation, cleared by _begin_session. Scoped to
        # the conversation rather than the turn because the stall queue is rebuilt per turn and a
        # fresh shuffle knows nothing about the last one's — without these, "Sandali ha" could
        # comfortably open the stalls of three turns running and the bank would sound half its size.
        self._filler_used_openers: set[str] = set()
        self._filler_used_stalls: set[str] = set()
        # Stalls spent in THIS turn, cleared by _arm_filler. The conversation set above is the
        # stronger preference but it empties permanently once a conversation has been through the
        # bank, and a single set going empty used to mean the full bank came straight back — the
        # line that just played included. This is the weaker promise that survives that: nothing
        # twice inside one wait, which is where the ear actually notices. Heard on the robot
        # 2026-08-09 as the same stall twice in one exchange, on the small ceb/en banks that lap
        # inside a single wait. _filler_last_stall guards the seam neither set can: the very first
        # pop after a rebuild. It deliberately survives _arm_filler, like _filler_recent_openers.
        self._filler_turn_stalls: set[str] = set()

        # Per-session facts, reset on every accepted wake.
        self._face_ever_seen = False
        self._face_absent_since: float | None = None
        self._no_speech_streak = 0
        self._error_streak = 0
        self._turns = 0

        self._last_wake_t = -WAKE_REFRACTORY_S
        self._wakes = 0
        self._rejected = {"busy": 0, "speaking": 0, "refractory": 0, "not_ready": 0}
        self._stale_results = 0
        self._discarded_short = 0
        self._end_reason = ""
        self._last_heartbeat = 0.0
        # Last presence-feed failure and when it was logged — see _note_presence_error.
        self._presence_error = ""
        self._presence_error_t = -PRESENCE_ERROR_LOG_S
        # Wall time of the whole turn (utterance handed over -> worker reported back). The PER-STAGE
        # breakdown lives on the VoiceAssistant, which is the only thing that can see the stages
        # apart — see VoiceAssistant.stage_timings(). Kept as "llm_ms" here because the dashboard
        # field is named that, but it has always been STT+RAG+LLM; sess_last_llm_ms below now
        # reports the real LLM number and this one is published as sess_last_turn_ms.
        self._timings = {"stt_ms": 0, "llm_ms": 0}
        self._stage_ms: dict = {}
        self._turn_started = 0.0

        # Whisper wake tier. _scan_token invalidates an in-flight check the way _epoch invalidates a
        # turn; _scan_ready_at is the cooldown floor that bounds how often Whisper can run.
        self._scan_token = 0
        self._scan_ready_at = 0.0
        self._scans = 0
        self._scan_matches = 0
        self._scan_skipped = {"cooldown": 0, "short": 0, "long": 0, "no_model": 0}
        self._scan_last_ms = 0
        self._scan_last_text = ""

        # Off unless DEBUG_CAPTURE_ENABLED. Constructed either way so every call site is
        # unconditional — a recorder that has to be null-checked at four sites is a recorder that
        # will one day be forgotten at the fifth.
        self._recorder = UtteranceRecorder(
            DEBUG_CAPTURE_DIR, enabled=DEBUG_CAPTURE_ENABLED, max_files=DEBUG_CAPTURE_MAX_FILES,
            max_mb=DEBUG_CAPTURE_MAX_MB, kinds=DEBUG_CAPTURE_KINDS)
        # clip id of the turn / scan currently in flight, so the transcript can be attached to the
        # audio it came from when it arrives on a worker thread.
        self._turn_clip = ""
        self._scan_clip = ""

    def _clip_context(self) -> dict:
        """The live tuning numbers, snapshotted alongside a recorded clip.

        This is the half that makes the corpus worth having. A WAV on its own tells you what the
        room sounded like; these tell you what Kai believed about it at that instant — which floor
        was actually in force after ambient adaptation, how far the room had drifted from the
        configured one, which wake tier was running. Reproducing a failure offline needs both."""
        return {
            "rms": round(self._mic.last_rms, 1),
            "ambient": round(self._gate.ambient, 1),
            "open_floor": round(self._gate.open_floor, 1),
            "hold_floor": round(self._gate.hold_floor, 1),
            "wake_engine": self._mic.wake.engine,
            "state": self._state,
        }

    # ── lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Open the mic, arm the wake word, and start the tick thread.

        Returns False if the audio device could not be opened at all. A missing wake word is NOT a
        failure: the stream still comes up and push-to-talk keeps working over it."""
        if MIC_LEGACY_CAPTURE:
            print("[session] MIC_LEGACY_CAPTURE is set — per-turn capture, hands-free off", flush=True)
            return False
        if not self._mic.start():
            return False
        self._voice.attach_mic(self._mic)

        if self.enabled and not self._mic.wake.open():
            # Every tier in the chain failed — no key, no .ppn, no trained model, wrong platform,
            # packages missing. Turn hands-free off outright rather than leaving it "on but never
            # ready", so sess_state, sess_ready and the log all agree on "push-to-talk only" instead
            # of reading as broken. Each tier's own reason is in wake.tiers / sess_wake_tried.
            self.enabled = False
        # After open(), never before: the winning tier decides the frame size.
        self._mic.sync_wake_geometry()
        self._mic.set_wake_enabled(self.enabled and self._mic.wake.ready)
        with self._lock:
            self._set_state(STATE_IDLE if self._wake_live() else STATE_DISABLED, time.monotonic())

        if ACK_PRESYNTH or GREETING_ENABLED:
            threading.Thread(target=self._warm_all, daemon=True, name="kai-ack-warm").start()

        self._stop.clear()
        self._thread = threading.Thread(target=self._tick_loop, daemon=True, name="kai-session")
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        # After the tick thread is down (so nothing can start a new line behind us) and before the
        # mic goes: cancel any synth or playback still in flight. Every other teardown path already
        # does this — _end_session, and face_track.run()'s finally — but stop() is reachable on its
        # own, and a speech worker outliving the session it belongs to is how audio from one run
        # ends up playing into the next. tts.stop() is a no-op when nothing is running.
        tts.stop()
        self._mic.stop()

    def reresolve_mic(self) -> dict:
        """Re-run mic discovery and bring the capture stream back, without restarting the process.

        This exists because the two mics fail in ways that a *later* look would fix, and nothing
        short of a restart used to take that later look. Measured across three boots on 2026-08-09:
        the INMP441 read as exact digital silence on one, timed out on the liveness probe on the
        next, and worked immediately on a third — while `arecord` on the raw device found real audio
        every single time. Whatever the boot race is, the device is usually fine a minute later.

        The existing watchdog cannot cover this. It reopens a stream that DIED (session.py's
        `mic_lost` check), and it is explicitly gated on `state != STATE_DISABLED` — but a mic that
        never came up at all leaves the session in exactly STATE_DISABLED, so the one case that
        needs a second look is the one case the watchdog skips. Hence an operator-driven retry.

        Two situations, told apart by whether the tick thread is running:
          - session live  -> reopen the stream (MicStream.reopen, the same path the watchdog uses)
          - session never started -> start() it, which is safe precisely because a failed start()
            returns before any thread is created (see start(): `if not self._mic.start(): return`)

        Returns a dict for the route to serialize. Never raises: this is a recovery button, and a
        recovery button that can 500 is one more thing to recover from.
        """
        running = self._thread is not None and self._thread.is_alive()
        try:
            if running:
                print("[mic] re-resolve requested from the dashboard (session is live)", flush=True)
                ok = self._mic.reopen()
            else:
                print("[mic] re-resolve requested from the dashboard (session was not running)",
                      flush=True)
                # MicStream.reopen() re-scans PortAudio for us on the branch above; nothing does on
                # this one, because start() opens a stream rather than reopening one. Without this
                # the never-started case — the case this button most exists for — would re-run the
                # whole discovery path against the device list PortAudio snapshotted at startup, and
                # a mic plugged in since then would still be invisible. Safe here by definition:
                # the session is not running, so there is no stream to invalidate.
                refresh_devices()
                ok = self.start()
        except Exception as exc:                      # pragma: no cover - defensive
            return {"status": "error", "ok": False, "restarted_session": False,
                    "error": f"{type(exc).__name__}: {exc}"}

        if ok and running:
            # The stream is back but the state machine was parked. Re-derive it the same way start()
            # does, so sess_state stops saying "disabled" on a robot that can now hear.
            with self._lock:
                if self._state == STATE_DISABLED and self._wake_live():
                    self._set_state(STATE_IDLE, time.monotonic())

        # `kind` names the mic; `is_i2s` is kept for the callers that only ever asked "raw device or
        # not". The dashboard needs the name now that a USB mic is something an operator chooses
        # rather than what Kai settles for — "not the I2S mic" was an adequate label only while the
        # alternative had no name worth saying.
        return {"status": "ok" if ok else "error", "ok": bool(ok),
                "restarted_session": not running and bool(ok),
                "device": self._mic.device, "rate": self._mic.capture_rate,
                "kind": self._mic.kind, "is_i2s": self._mic.is_i2s, "live": self._mic.live,
                "error": "" if ok else (self.mic_error() or "no reason reported")}

    def watch_for_a_mic(self, stop) -> bool:
        """Blocking. Keep looking for a microphone until one comes up or `stop` is set.

        For the one case the tick loop cannot cover: start() failed every attempt, so there is no
        tick thread, so nothing is polling the hot-plug watcher — and that is precisely the state a
        robot boots into when its mic is missing, which makes it the single most likely moment for
        someone to plug one in. Without this, "plug a mic into a running Kai" works in every
        situation except the one where the robot is already deaf.

        Driven by face_track's session-start thread once its retries are exhausted, so it costs a
        thread that had finished its work rather than a new one. Returns True if a mic came up.
        Never raises: reresolve_mic() cannot, and a watcher that cannot read its file simply
        reports that hot-plug is unavailable and this returns.
        """
        if not self._hotplug.active:
            print("[mic] no microphone, and hot-plug watching is unavailable — the dashboard's "
                  "'find the microphone again' button is the way back", flush=True)
            return False
        print("[mic] no microphone; watching for one to be plugged in", flush=True)
        while not stop.is_set():
            if self._hotplug.poll(time.monotonic()) and self.reresolve_mic().get("ok"):
                print(f"[mic] a microphone appeared — now on the {self._mic.kind} mic", flush=True)
                return True
            # Waits on the stop event rather than sleeping, so shutdown is immediate rather than up
            # to a poll interval late.
            stop.wait(MIC_HOTPLUG_POLL_S)
        return False

    def mic_error(self) -> str:
        """Why the last mic open failed, or "" if it didn't. Public so the start-retry loop can put
        the reason on the same line as the retry — the reason and the retry landing in separate log
        lines is what made a boot-time capture failure read as unexplained."""
        return self._mic.error or ""

    def _wake_live(self) -> bool:
        return bool(self.enabled and self._mic.wake.ready)

    def _canned_lines(self) -> dict[str, str]:
        """The four lines that MUST be on disk for the session to behave. Single source of truth so
        _rewarm_when_quiet can check what it actually got against what it asked for.

        Deliberately NOT the filler bank. The bank is 52 more lines and warms separately, one line
        at a time between turns (_prewarm_bank), because synthesising it in one burst put a second
        Piper on the CPU next to a live reply — measured on the robot 2026-08-07 as a 12.3 s synth
        on a turn that should take ~1 s, plus a corrupted output WAV. A missing filler line costs
        variety; a missing "Yes?" costs the whole hands-free illusion, so they do not share a
        retry policy and they must not share a burst."""
        return {"ack": WAKE_ACK_TEXT, "no_speech": CANNED_NO_SPEECH, "error": CANNED_ERROR,
                "thinking": THINKING_SOUND_TEXT}

    def _prewarm_canned(self) -> None:
        """Synthesize the ack and the failure lines once, at startup. Live synthesis would put
        0.5-1.5 s of dead air between "Hey Kai" and "Yes?", which is most of what makes hands-free
        feel broken.

        Merges rather than replaces, so a bank line already warmed is not thrown away by a re-warm
        of the four core lines."""
        lines = self._canned_lines()
        # Only the filler is length-pinned: it exists to cover a pause, so it is sized to one.
        got = tts.prewarm_canned(lines, ACK_WAV_DIR,
                                 targets={"thinking": THINKING_SOUND_TARGET_S})
        with self._lock:
            self._canned.update(got)
        if got:
            print(f"[session] cached spoken lines: {', '.join(sorted(got))}", flush=True)

    def _warm_all(self) -> None:
        """The four core lines first and fast, then the greeting, then the bank slowly. Strict
        ordering, one thread: the bank must never be synthesising while "Yes?" still is not on disk.

        The greeting sits between the two halves rather than at either end, and both sides of that
        are deliberate. Not before the core lines: a wake arriving during the greeting must still
        find "Yes?" on disk. Not after the bank: the bank runs for MINUTES, and a hello that arrives
        five minutes after the robot booted is not a greeting."""
        if ACK_PRESYNTH:
            self._prewarm_canned()
        self._speak_greeting()
        if ACK_PRESYNTH:
            self._prewarm_bank()

    def _speak_greeting(self) -> None:
        """Say hello to the room, once per process, then wait for the audio to finish.

        The wait is what makes this safe to call on the warm thread: tts has ONE synth slot and
        _begin_speech cuts whatever is playing, so returning while the greeting is still going would
        have _prewarm_bank's first line kill it mid-sentence. Bounded by GREETING_QUIET_WAIT_S so a
        playback that never reports done costs the bank a delay, not the whole warm.

        Best-effort in every direction: a failed synth just means no greeting (VoiceAssistant._speak
        falls back to the silent jaw pantomime), and nothing downstream depends on it having played.
        """
        if not (GREETING_ENABLED and GREETING_TEXT.strip()):
            return
        with self._lock:
            if self._greeted:
                return
            self._greeted = True
        # Two latches, because there are two ways to greet twice. The one above is this process
        # re-entering start() (reresolve_mic). This one is a DIFFERENT process starting shortly after
        # ours — a crash-relaunch, or an operator double-tapping /restart — which the in-memory latch
        # cannot see. See GREETING_REPEAT_SUPPRESS_S in config/wake.py for the measurement.
        age = _greeting_age()
        if GREETING_REPEAT_SUPPRESS_S > 0.0 and age is not None and age < GREETING_REPEAT_SUPPRESS_S:
            print(f"[session] greeting suppressed — a Kai process greeted {age:.0f}s ago, inside the "
                  f"{GREETING_REPEAT_SUPPRESS_S:.0f}s window (GREETING_REPEAT_SUPPRESS_S). If this "
                  f"was not a restart you asked for, look above for a non-zero exit: this is what a "
                  f"crash-and-relaunch looks like from here.", flush=True)
            return
        _mark_greeting_spoken()
        print(f"[session] greeting: {GREETING_TEXT}", flush=True)
        self._voice.speak_text(GREETING_TEXT)
        deadline = time.monotonic() + GREETING_QUIET_WAIT_S
        while time.monotonic() < deadline and (tts.is_playing()
                                               or self._voice.speech_in_flight()):
            time.sleep(GREETING_POLL_S)

    def _within_length_cap(self, key: str, wav) -> bool:
        """True if the synthesised line is short enough to cache. Rejected lines are never cached,
        and the bank only ever selects from cached keys, so this is the enforcement point for the
        length cap rather than a warning about it.

        Measured from the WAV rather than counted from the text, because speaking rate is a live
        dashboard setting: the same sentence crosses the cap at one rate and not another, and a
        word-count rule would be right at 1.0 and wrong everywhere else. A line that cannot be
        measured (0.0 from a bad header) is kept — silently dropping the whole bank because
        wav_duration failed would be a far worse failure than a long line."""
        cap = FILLER_MAX_STALL_S if key.startswith(filler.STALL_PREFIX) else FILLER_MAX_LINE_S
        secs = tts.wav_duration(wav)
        if secs > cap:
            print(f"[session] filler line {key} is {secs:.1f}s (cap {cap:.1f}s) — not cached; "
                  f"shorten it in config/filler.py", flush=True)
            return False
        return True

    def _quiet_for_synth(self) -> bool:
        """True when starting a Piper run would not land next to one already going.

        The whole reason _prewarm_bank exists. tts publishes ONE _synth_proc handle and stop() kills
        whatever is in it, so a background synth started during a turn both competes for CPU with
        the reply's synth and makes stop() kill the wrong process."""
        if tts.is_playing() or self._voice.speech_in_flight():
            return False
        with self._lock:
            return self._state not in (STATE_BUSY, STATE_SPEAKING, STATE_ACK)

    def _warm_one(self, key: str, text: str) -> str:
        """Caller does NOT hold the lock. One attempt at warming one bank line. Returns what the
        caller should do about it, which is the whole reason this is separate from _prewarm_bank:

          "cached" — on disk, measured, selectable.
          "retry"  — synthesis was reached and did not produce a file. Worth trying again NOW: we
                     had a quiet window a moment ago, and the overwhelmingly common cause is a
                     turn starting and tts.stop() killing the one _synth_proc. That is transient
                     and it is not evenly distributed — the longer the line, the wider the target.
          "skip"   — nothing to be gained by trying again in this pass. Two very different cases
                     that share an answer: no quiet window ever came (the robot is busy right now,
                     so an immediate retry is just as futile), or the line synthesised fine and
                     the length cap rejected it (deterministic — it will measure the same forever).
        """
        for _ in range(BANK_QUIET_WAIT_TRIES):
            if self._quiet_for_synth():
                break
            time.sleep(BANK_QUIET_POLL_S)
        else:
            return "skip"
        got = tts.prewarm_canned({key: text}, ACK_WAV_DIR)
        if not got:
            return "retry"
        if not self._within_length_cap(key, got[key]):
            return "skip"
        with self._lock:
            self._canned.update(got)
        return "cached"

    def _prewarm_bank(self) -> None:
        """Warm the filler bank in the background, ONE line at a time, only while nothing is
        speaking. Runs for minutes rather than seconds, and that is the point.

        Started after the core lines so it can never delay "Yes?". Every line is synthesised on its
        own, with a quiet check in front of it, because the failure this replaces was a single
        44-line burst: it ran straight through a live turn, 17 lines failed in a row, the reply's
        own synthesis was corrupted mid-write, and the turn took 24 s to first audio.

        Best-effort throughout. A line that fails is simply absent, and _speak_filler stays silent
        for it rather than synthesising it live — filler must never cost the reply anything."""
        if not (FILLER_ENABLED and FILLER_PREWARM):
            return
        lines = filler.canned_lines()
        for attempt in range(BANK_PASSES):
            done = 0
            for key, text in lines.items():
                with self._lock:
                    if key in self._canned:
                        continue
                # Retried HERE rather than by the next pass, because a pass is minutes long and
                # the wait is not spent evenly: the longest lines are the likeliest to be killed
                # mid-synthesis, so deferring them systematically empties the opener tier while
                # the stalls fill up. See BANK_LINE_RETRIES for the measurement.
                for _ in range(BANK_LINE_RETRIES + 1):
                    outcome = self._warm_one(key, text)
                    # A breath between attempts, so a long warm never monopolises the CPU the
                    # vision loop and Ollama are also on.
                    time.sleep(BANK_SYNTH_GAP_S)
                    if outcome == "cached":
                        done += 1
                    if outcome != "retry":
                        break
            with self._lock:
                cached = {k for k in self._canned if k.startswith("filler_")}
            # Per language, not just the total: a turn draws stalls from ONE language, so a healthy
            # total can still hide a two-line pool that repeats inside a single wait — which is
            # exactly what was heard on the robot before the queue grew its back-to-back guard.
            # The length cap drops lines silently, so this is the only place the real pool shows up.
            pools = ", ".join(f"{lang} {ops}op/{sts}st"
                              for lang, (ops, sts) in filler.warm_counts(cached).items())
            print(f"[session] filler bank: {len(cached)}/{len(lines)} lines cached "
                  f"(pass {attempt + 1}, +{done}) [{pools}]", flush=True)
            if len(cached) >= len(lines):
                return

    def reprewarm_canned(self) -> None:
        """Re-synthesize the cached lines because the voice changed. Returns immediately.

        Without this, changing TTS volume or speaking rate from the dashboard would leave "Yes?" and
        the failure lines in the OLD voice forever — every reply retuned except the one you hear most.
        Piper costs ~1s per line, hence the thread; the settings callback that calls this is debounced
        so a dragged slider does not queue a dozen of them.
        """
        with self._lock:
            self._canned = {}
        threading.Thread(target=self._rewarm_when_quiet, daemon=True, name="kai-ack-rewarm").start()

    def _rewarm_when_quiet(self) -> None:
        """Wait for any reply in flight, then re-synthesize — and check it actually landed.

        tts has a SINGLE synth slot and stop() cancels it, so a turn starting mid-prewarm kills the
        canned synthesis and the line silently goes missing (observed: "cached spoken lines: error,
        no_speech" with no ack). _speak_canned then falls back to live synthesis, which puts the
        0.5-1.5s of dead air between "Hey Kai" and "Yes?" back — the exact thing prewarming exists to
        remove. So: wait for quiet, then verify and retry once.

        Verifies EVERY key that was requested, not just ack. `thinking` is in fact the likeliest
        casualty — it is the length-fitted line, so it costs several Piper passes — and an ack-only
        check let it go missing silently ("cached spoken lines: ack, error, no_speech") without ever
        firing the retry.
        """
        deadline = time.monotonic() + REWARM_QUIET_WAIT_S
        while time.monotonic() < deadline and (tts.is_playing() or self._voice.speech_in_flight()):
            time.sleep(0.25)
        self._prewarm_canned()
        with self._lock:
            missing = sorted(set(self._canned_lines()) - set(self._canned))
        if missing:
            print(f"[session] re-warm incomplete, retrying: missing {', '.join(missing)}", flush=True)
            time.sleep(REWARM_RETRY_S)
            self._prewarm_canned()
        # The bank rides the voice change too, otherwise every filler line stays in the OLD voice
        # while the four core lines retune — the same inconsistency reprewarm exists to prevent.
        # After the core retry, never before it, and paced by _prewarm_bank's own quiet checks.
        self._prewarm_bank()

    # ── inputs ──────────────────────────────────────────────────────────────

    def note_face(self, visible: bool, now: float | None = None) -> None:
        """Optional push-style presence input, for callers not using the injected snapshot."""
        now = time.monotonic() if now is None else now
        with self._lock:
            self._apply_presence(visible, True, now)

    def _apply_presence(self, visible: bool, is_fresh: bool, now: float) -> None:
        """Fold one presence reading into the session. Caller holds the lock."""
        if not is_fresh:
            # UNKNOWN, not absent. Fail open: clear the absence clock so a camera stall can never
            # expire a session, and leave face_ever_seen as it was.
            self._face_absent_since = None
            return
        if visible:
            self._face_ever_seen = True
            self._face_absent_since = None
        elif self._face_ever_seen and self._face_absent_since is None:
            self._face_absent_since = now

    def on_wake(self, now: float | None = None, command: str = "") -> bool:
        """The wake word fired. True if it was accepted and a session began.

        `command` is the whisper tier's one-breath payload: whatever the speaker said after the wake
        phrase, already transcribed. When present the ack is skipped and the turn runs straight from
        that text — the frame tiers always pass "" and behave exactly as before."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if now - self._last_wake_t < WAKE_REFRACTORY_S:
                self._rejected["refractory"] += 1   # a drawn-out "Heeey Kaaai" fires twice
                return False
            if self._state in (STATE_BUSY,):
                # The Ollama call is a blocking non-streaming POST and is not cancellable, so
                # accepting would mean either queueing or abandoning a reply that arrives anyway.
                self._rejected["busy"] += 1
                self._log(f"wake ignored while {self._state}")
                return False
            if self._state in (STATE_ACK, STATE_SPEAKING, STATE_COOLDOWN):
                if not WAKE_ALLOW_BARGE_IN:
                    self._rejected["speaking"] += 1
                    return False
            if self._state == STATE_DISABLED or not self.ready:
                self._rejected["not_ready"] += 1
                self._log("wake ignored — still warming up")
                return False

            self._last_wake_t = now
            self._wakes += 1
            if self._state == STATE_LISTEN_WAIT:
                # Already listening: just give the user their time back, don't replay the ack.
                self._set_state(STATE_LISTEN_WAIT, now)
                return True

            if command and self._state == STATE_IDLE:
                # One breath: "Hey Kai, what time is it?" — the transcript already exists, so going
                # back through process_utterance would transcribe the same audio twice and add
                # 0.5-1.5 s to the interaction that is meant to feel fastest.
                self._begin_session(now, ack=False)
                self._start_text_turn(command, now)
                return True

            self._begin_session(now)
            return True

    def _start_text_turn(self, text: str, now: float) -> None:
        """Run a turn from already-transcribed text. Caller holds the lock."""
        self._set_state(STATE_BUSY, now)
        self._turn_started = now
        self._log(f"one-breath turn: {text[:60]!r}")
        result = self._voice.say(text, use_llm=True, epoch=self._epoch,
                                 on_done=self._on_turn_done)
        if "error" in result:
            # say() refuses while the assistant is mid-status. Falling through would leave us in BUSY
            # for SESSION_BUSY_MAX_S — two minutes of a robot that looks hung. Degrade to the
            # ordinary two-step wake instead.
            self._log(f"one-breath turn rejected: {result['error']} — acking instead")
            self._set_state(STATE_ACK, now)
            self._speak_canned("ack", WAKE_ACK_TEXT)

    def _begin_session(self, now: float, ack: bool = True) -> None:
        """Start a fresh session at the ack. Caller holds the lock.

        Reachable from LISTEN_SPEECH too — someone can say "hey Kai" in the middle of a sentence to
        start over — so it disarms and drops whatever was being captured rather than carrying a
        half-utterance into the new session."""
        self._epoch = self._voice.bump_epoch()
        self._mic.harvest_utterance()   # disarm + discard
        self._mic.reset_dsp()
        self._manual_end = False
        self._face_ever_seen = False
        self._face_absent_since = None
        self._no_speech_streak = 0
        self._error_streak = 0
        self._turns = 0
        self._end_reason = ""
        # A new conversation gets the whole bank back. Deliberately HERE and not in _arm_filler:
        # per-turn is the scope these already had implicitly, and it is what let a line repeat every
        # turn. Deliberately not never-reset either — a demo would exhaust 12 Tagalog stalls in one
        # long afternoon and spend the rest of it on the fallback path. A conversation is the span
        # over which one listener would actually notice the repeat.
        #
        # Note this is NOT reached by a wake that lands in LISTEN_WAIT (see on_wake): saying "hey
        # Kai" again while it is already listening continues the conversation, so it keeps its
        # history. _filler_recent_openers deliberately survives, since it guards the one case a
        # cleared set cannot — the same opener landing either side of the seam.
        self._filler_used_openers.clear()
        self._filler_used_stalls.clear()
        self._gate.reset()
        self._scan_token += 1   # any check still in flight belongs to the old session
        # A new session must not inherit the previous turn's completed status. ACK/SPEAKING/COOLDOWN
        # don't override voice_status, so a stale "done" carried in here would surface as a fresh
        # transition into "done" and re-post the last exchange.
        self._voice.clear_turn_status()
        if ack:
            self._set_state(STATE_ACK, now)
            self._speak_canned("ack", WAKE_ACK_TEXT)
        # With ack=False the caller sets the next state (see _start_text_turn).

    def _on_audio(self, pcm: np.ndarray, now: float) -> None:
        """Called from the audio worker for every ungated block. Runs the VAD when we're listening.

        The whole body is under the lock: SpeechGate is not thread-safe, and the tick thread calls
        reset() on it at every state change. Contention is negligible — this is ~30 calls/second
        doing a few hundred microseconds of work on 320-sample frames."""
        pending = None
        with self._lock:
            state = self._state

            if state == STATE_IDLE:
                # Only the utterance tier listens while idle; the frame tiers spot the phrase
                # acoustically and never need the VAD here.
                if not self._scan_tier():
                    return
                if self._gate.update(pcm, now) != "onset":
                    return
                if now < self._scan_ready_at:
                    self._scan_skipped["cooldown"] += 1     # bounds the STT duty cycle
                    self._gate.reset()
                    return
                if not self._voice.scan_ready:
                    self._scan_skipped["no_model"] += 1      # still pre-warming
                    self._gate.reset()
                    return
                # A wake scan is one short phrase, not a sentence someone might pause inside — so it
                # runs on its own, much shorter trailing-silence clock. See WAKE_SCAN_HANGOVER_S.
                self._gate.set_hangover(WAKE_SCAN_HANGOVER_S)
                self._mic.arm_utterance(preroll=True)
                self._set_state(STATE_SCAN_SPEECH, now)
                return

            if state == STATE_SCAN_SPEECH:
                if self._gate.update(pcm, now) != "hangover":
                    return
                pending = "scan"

            elif state in (STATE_LISTEN_WAIT, STATE_LISTEN_SPEECH):
                event = self._gate.update(pcm, now)
                if event == "onset" and state == STATE_LISTEN_WAIT:
                    # Back to the turn clock — the scan path above may have left the short one set.
                    self._gate.set_hangover(VAD_HANGOVER_S)
                    self._mic.arm_utterance(preroll=True)
                    self._set_state(STATE_LISTEN_SPEECH, now)
                    self._log(f"speech onset (rms={self._gate.last_rms:.0f})")
                    return
                if event != "hangover" or state != STATE_LISTEN_SPEECH:
                    return
                pending = "turn"
            else:
                return

        # Dispatched outside the lock: both of these re-enter it and call into the assistant.
        if pending == "scan":
            self._finish_scan(now, reason="hangover")
        else:
            self._finish_utterance(now, reason="hangover")

    def _scan_tier(self) -> bool:
        """True when the live wake engine is the utterance-kind one, so idle audio must be scanned."""
        return bool(self.enabled and self._mic.wake.ready and self._mic.wake.kind == "utterance")

    def _finish_scan(self, now: float, reason: str) -> None:
        """Close a candidate utterance and either transcribe it or throw it away.

        The two discard paths are what keep this tier affordable: a blip and a long stretch of
        conversation both cost nothing, because neither reaches Whisper at all."""
        with self._lock:
            if self._state != STATE_SCAN_SPEECH:
                return                              # superseded by PTT / manual wake / session end
            audio, rate = self._mic.harvest_utterance()
            spoken_s = self._gate.speech_duration(now)
            self._gate.reset()

            if spoken_s < WAKE_WHISPER_MIN_UTTERANCE_S:
                self._scan_skipped["short"] += 1
                self._scan_ready_at = now + WAKE_WHISPER_COOLDOWN_S
                self._set_state(STATE_IDLE, now)
                return
            if spoken_s > WAKE_WHISPER_MAX_UTTERANCE_S or reason == "too_long":
                # Somebody talking continuously for >6 s is not saying a two-word wake phrase. Back
                # off rather than re-arming into the middle of their next sentence.
                self._scan_skipped["long"] += 1
                self._scan_ready_at = now + WAKE_WHISPER_LONG_COOLDOWN_S
                self._set_state(STATE_IDLE, now)
                return

            # Before the transition: the context describes the audio that was just captured, and
            # after _set_state it would record the state we are moving TO instead.
            context = self._clip_context()
            self._scan_token += 1
            token = self._scan_token
            self._scans += 1
            self._set_state(STATE_SCAN_CHECK, now)

        self._voice.transcribe_async(audio, rate, on_done=self._on_scan_done,
                                     token=token, log_language=False)
        # After the dispatch, never before: transcribe_async only spawns a thread, so recording here
        # costs the check nothing, while recording inside the lock above would put a WAV write on the
        # tick thread in the middle of a state transition.
        self._scan_clip = self._recorder.record(audio, rate, "scan", reason=reason,
                                                spoken_s=round(spoken_s, 3), **context)

    def _on_scan_done(self, token, text: str, error: str) -> None:
        """The candidate's transcript came back. Runs on the STT worker thread."""
        now = time.monotonic()
        command = ""
        # Attached to the recorded clip in the finally below. Set before each return rather than at
        # one exit because the outcomes ARE the labels — "Whisper heard this and it did not match"
        # is the single most useful thing the corpus can say about a missed wake.
        note: dict | None = None
        try:
            with self._lock:
                if token != self._scan_token or self._state != STATE_SCAN_CHECK:
                    self._stale_results += 1
                    return                              # superseded; touch nothing
                self._scan_last_ms = int((now - self._state_since) * 1000)
                self._scan_ready_at = now + WAKE_WHISPER_COOLDOWN_S
                if error:
                    self._log(f"wake check failed: {error}")
                    note = {"outcome": "error", "error": error}
                    self._set_state(STATE_IDLE, now)
                    return

                match = match_wake_phrase(text)
                if match is None:
                    # Silent discard. A sentence that wasn't addressed to Kai must be
                    # indistinguishable from silence — no ack, nothing spoken, and no log line
                    # unless asked for.
                    if WAKE_WHISPER_LOG_TEXT:
                        self._scan_last_text = text[:WAKE_WHISPER_LOG_CHARS]
                        self._log(f"no wake phrase in {self._scan_last_text!r} "
                                  f"({self._scan_last_ms}ms)")
                    # The transcript IS written to the clip index here even though
                    # WAKE_WHISPER_LOG_TEXT may be off. That switch keeps overheard speech out of
                    # the shared log; debug capture is an explicit, off-by-default opt-in that is
                    # already storing the audio itself, so withholding the text would only make the
                    # recording harder to use without making it any more private.
                    note = {"outcome": "no_match", "text": text, "ms": self._scan_last_ms}
                    self._set_state(STATE_IDLE, now)
                    return

                self._scan_matches += 1
                self._scan_last_text = match.phrase[:WAKE_WHISPER_LOG_CHARS]
                self._log(f"wake phrase matched ({match.score:.2f}) in {self._scan_last_ms}ms"
                          + (" + command" if match.command else ""))
                command = match.command
                note = {"outcome": "match", "text": text, "score": round(match.score, 3),
                        "command": match.command, "ms": self._scan_last_ms}
                # Back to idle first: on_wake() guards against several states and special-cases
                # others; letting it see scan_check would mean a new arm in a function that already
                # has five.
                self._set_state(STATE_IDLE, now)
        finally:
            # Outside the lock by construction — the annotate appends to a file, and the session
            # lock is held by the 20 Hz tick thread.
            if note is not None:
                self._recorder.annotate(self._scan_clip, **note)

        self.on_wake(now, command=command)

    def request_ptt_start(self) -> dict:
        """Dashboard mic button / spacebar. Takes precedence over hands-free — a button press is
        unambiguous intent, so it even interrupts a reply, which the wake word deliberately cannot."""
        now = time.monotonic()
        with self._lock:
            if self._state in (STATE_ACK, STATE_BUSY):
                return {"error": f"busy: {self._state}"}
            if self._state == STATE_SPEAKING:
                self._cut_speech()
            self._epoch = self._voice.bump_epoch()
            self._scan_token += 1   # a scan result landing mid-PTT must be dropped, not acted on
            self._manual_end = True
            self._gate.reset()
            self._mic.reset_dsp()
            if not self._mic.arm_utterance(preroll=True):
                return {"error": "Could not arm the microphone"}
            self._set_state(STATE_LISTEN_SPEECH, now)
            self._log("push-to-talk start")
        return {"status": "ok"}

    def request_ptt_stop(self) -> dict:
        now = time.monotonic()
        with self._lock:
            if self._state == STATE_LISTEN_WAIT:
                # Hands-free listening projects onto voice_status="recording", so the dashboard mic
                # button reads "listening" here even though no utterance has started. Tapping it
                # means "stop listening" — end the session rather than returning an error for a
                # button the UI is actively inviting the user to press.
                self._end_session(now, "ptt_stop")
                return {"status": "ok"}
            if self._state != STATE_LISTEN_SPEECH:
                return {"error": f"cannot stop recording while {self._state}"}
        self._finish_utterance(now, reason="ptt_stop")
        return {"status": "ok"}

    # ── turns ───────────────────────────────────────────────────────────────

    def _finish_utterance(self, now: float, reason: str) -> None:
        """Close the current utterance and either run it or throw it away."""
        with self._lock:
            if self._state != STATE_LISTEN_SPEECH:
                return
            manual = self._manual_end
            self._manual_end = False
            audio, rate = self._mic.harvest_utterance()
            spoken_s = self._gate.speech_duration(now)
            truncated = self._mic.capture_truncated
            # Too short to be speech: discard WITHOUT running Whisper or the LLM. This is what breaks
            # the self-sustaining loop — hiss opens a turn, the transcript comes back empty, Kai says
            # "didn't catch that", the amp re-primes the hiss, forever — and each lap would otherwise
            # cost a full Whisper run.
            if not manual and spoken_s < MIN_UTTERANCE_S:
                self._discarded_short += 1
                self._log(f"discarded {spoken_s * 1000:.0f}ms blip (< {MIN_UTTERANCE_S * 1000:.0f}ms)")
                self._gate.reset()
                self._enter_listen_wait(now)
                return
            # Before the transition — see the same call in _finish_scan.
            context = self._clip_context()
            self._set_state(STATE_BUSY, now)
            self._turn_started = now
            epoch = self._epoch
            self._log(f"turn: {audio.size / max(1, rate):.1f}s audio ({reason}"
                      f"{', TRUNCATED' if truncated else ''})")

        result = self._voice.process_utterance(audio, rate, epoch=epoch, on_done=self._on_turn_done)
        # After the dispatch: process_utterance only spawns the worker, so this adds nothing to the
        # latency a person feels, and it stays off the tick thread's critical section.
        self._turn_clip = self._recorder.record(audio, rate, "turn", reason=reason,
                                                spoken_s=round(spoken_s, 3), manual=manual,
                                                truncated=truncated, **context)
        if "error" in result:
            with self._lock:
                self._log(f"turn rejected: {result['error']}")
                self._enter_listen_wait(time.monotonic())

    def _on_turn_done(self, epoch: int | None, outcome: str) -> None:
        """Called from the turn worker when STT+LLM finish. Runs on that thread, not the tick."""
        now = time.monotonic()
        note: dict | None = None      # see _on_scan_done; annotated in the finally, off the lock
        try:
            with self._lock:
                if epoch is not None and epoch != self._epoch:
                    self._stale_results += 1
                    return
                if self._state != STATE_BUSY:
                    return
                self._timings["llm_ms"] = int((now - self._turn_started) * 1000)
                # Pull the per-stage split the worker just measured. Best-effort: a faked/stubbed
                # assistant in the tests need not provide it, and a missing breakdown must not be
                # able to fail a turn that otherwise succeeded.
                try:
                    self._stage_ms = self._voice.stage_timings()
                    self._timings["stt_ms"] = int(self._stage_ms.get("stt_ms", 0))
                except (AttributeError, TypeError, ValueError):
                    pass
                # Set once here, before the outcome branches: `outcome` is already known and every
                # branch below returns, so this is the only point all of them pass through.
                note = {"outcome": outcome, "stt_ms": self._timings["stt_ms"],
                        "turn_ms": self._timings["llm_ms"]}

                if outcome == "done":
                    self._no_speech_streak = 0
                    self._error_streak = 0
                    self._turns += 1
                    self._enter_speaking(now)
                    return
                if outcome == "stale":
                    self._stale_results += 1
                    return
                if outcome == "empty":
                    self._no_speech_streak += 1
                    if self._no_speech_streak >= SESSION_MAX_NO_SPEECH_STREAK:
                        self._end_session(now, "no_speech_streak")
                        return
                    self._speak_canned("no_speech", CANNED_NO_SPEECH)
                    self._enter_speaking(now, canned=True)
                    return
                # error
                self._error_streak += 1
                if self._error_streak >= SESSION_MAX_ERROR_STREAK:
                    self._end_session(now, "error_streak")
                    return
                self._speak_canned("error", CANNED_ERROR)
                self._enter_speaking(now, canned=True)
        finally:
            if note is not None:
                # Merged, not two ** expansions: a key present in both would raise a TypeError from
                # inside a finally, which would replace the turn's real control flow.
                self._recorder.annotate(self._turn_clip, **{**note, **self._transcript_note()})

    def _transcript_note(self) -> dict:
        """What Whisper made of the last turn, plus how loud it was. For the clip index only.

        Read off the assistant rather than passed in: _on_turn_done is handed an outcome and
        nothing else, and widening that callback's signature to carry debug data into a feature
        that is off by default is the wrong trade. Fully guarded — a stubbed assistant in the tests
        has neither method, and a missing debug annotation must never fail a turn that worked."""
        note: dict = {}
        try:
            note["text"] = self._voice.get_status().get("voice_transcript", "")
        except (AttributeError, TypeError):
            pass
        try:
            levels = self._voice.input_levels()
            note["asr_rms"] = round(levels["rms"]["turn"], 5)
            note["asr_gain"] = round(levels["gain"]["turn"], 2)
        except (AttributeError, TypeError, KeyError):
            pass
        return note

    def _enter_speaking(self, now: float, canned: bool = False) -> None:
        """Caller holds the lock. Sets a hard deadline so a wedged paplay cannot deafen Kai for good
        — a failure that would otherwise be invisible, precisely because the mute gate is keyed on
        playback.

        This is the FALLBACK deadline only. It has to be, because nothing here can know how long the
        reply takes to say: on_done fires from the turn worker the moment the reply text exists, and
        _speak() hands synthesis to a thread, so at this point Piper has not started. Once the WAV
        exists the assistant publishes its real end time and _speaking_deadline() prefers that."""
        self._set_state(STATE_SPEAKING, now)
        self._speak_deadline = now + SESSION_SPEAK_MAX_UNKNOWN_S
        if canned:
            self._speak_deadline = now + WAKE_ACK_MAX_S + SESSION_SPEAK_GRACE_S

    def _speaking_deadline(self) -> float:
        """Caller holds the lock. When STATE_SPEAKING gives up and cuts the audio.

        The WAV's own end plus SESSION_SPEAK_GRACE_S once the assistant knows it, else the fallback
        _enter_speaking armed. That is what SESSION_SPEAK_GRACE_S ("allowed overrun past the WAV's
        own duration") was always written for, and until now it reached only the canned branch.

        Why this matters rather than being tidiness: SESSION_SPEAK_MAX_UNKNOWN_S is 20 s and was the
        ONLY deadline a healthy reply ever got, while TTS_MAX_SPOKEN_CHARS lets a reply run to 500
        characters — around 90 words, or ~31 s at SPEAK_SEC_PER_WORD. The clock also starts before
        synthesis, so Piper's run came out of the same 20 s. Every long reply was therefore cut off
        mid-sentence by the guard against a WEDGED paplay, with the jaw (sized to the real audio)
        carrying on over the silence. The backstop still works: a paplay that wedges publishes no
        end time, or overruns the one it published, and both land back on a deadline.

        Guarded because a stubbed assistant in the tests need not publish an end time at all, and a
        missing one must only ever mean "fall back to the armed cap"."""
        try:
            ends_at = self._voice.audio_ends_at()
        except (AttributeError, TypeError):
            ends_at = None
        if ends_at is None:
            return self._speak_deadline
        return ends_at + SESSION_SPEAK_GRACE_S

    def _cut_speech(self) -> None:
        """Stop Kai talking, jaw included. See VoiceAssistant.stop_speech — a bare tts.stop() leaves
        the mouth miming audio that has already been killed. Guarded so a stub without the method
        still cuts the audio."""
        try:
            self._voice.stop_speech()
        except (AttributeError, TypeError):
            tts.stop()

    # ── filler ──────────────────────────────────────────────────────────────

    def _arm_filler(self) -> None:
        """Caller holds the lock. Draw this turn's filler timing and clear last turn's state.

        Called from _set_state on every entry to BUSY, alongside _played_thinking, so a new turn
        cannot inherit a half-consumed stall queue.

        The delay is CLAMPED rather than trusted to the jitter range: the range is a tuning knob
        someone will widen one day, and both bounds it lives between are promises to the listener.
        See _filler_gap."""
        self._filler_delay = self._filler_gap(FILLER_DELAY_JITTER_S)
        self._filler_opened = False
        self._filler_queue = []
        self._filler_next_at = None
        # NOT _filler_last_stall, which has to survive the turn boundary to guard the seam.
        self._filler_turn_stalls.clear()

    def _filler_gap(self, jitter: tuple[float, float]) -> float:
        """One drawn silence, clamped into [FILLER_MIN_GAP_S, ceiling]. The single place both
        contracts are applied, so the opener delay and the stall gap cannot drift apart.

        The ceiling (FILLER_MAX_SILENCE_S minus the playback-start reservation) is the point of the
        whole module: no edit anywhere may put more than that much dead air in front of a listener,
        and budgeting the playback start is what makes that a bound on AUDIBLE silence rather than
        on the timer alone.

        The floor is the counterweight. Without it the ceiling pushes every gap toward zero — the
        safest way to never be quiet too long is to never be quiet — and filler arriving the instant
        the previous line stops reads as a queue draining rather than as thinking, with no room left
        for the real answer to land in a gap instead of on top of a line.

        The ceiling wins if they ever conflict. A floor raised past it would be one config edit
        silently breaking the promise the ceiling exists to keep, and dead air is the failure this
        module was built for; a gap fractionally shorter than someone wanted is not."""
        lo, hi = jitter
        ceiling = FILLER_MAX_SILENCE_S - FILLER_PLAYBACK_START_BUDGET_S
        return min(max(self._filler_rng.uniform(lo, hi), FILLER_MIN_GAP_S), ceiling)

    def _filler_enabled(self) -> bool:
        """Rides the existing "Think out loud" dashboard toggle rather than adding a twelfth live
        knob: to a listener this IS the thinking sound, just a talking one, and someone who turned
        that off wants Kai quiet while it thinks — not quiet in one way and chatty in another."""
        return FILLER_ENABLED and bool(settings.get("thinking_sounds"))

    def _speak_filler(self, key: str) -> None:
        """Caller holds the lock. Play one filler line from the cache, or stay SILENT.

        Deliberately not _speak_canned: a cache miss here must never fall through to live synthesis.
        The reply's own synthesis goes through tts.synthesize(), which writes fixed shared paths
        (_RAW_WAV / _OUTPUT_WAV) — so a filler synthesising live during BUSY writes the same two
        files the reply is about to, and both come out corrupt. Observed on the robot 2026-08-07:
        the reply thread died with EOFError inside wave.open, and sox reported "RIFF header not
        found", on a turn that took 24.6 s to first audio.

        Even without the collision it would be wrong. Filler exists to mask latency; a live Piper
        run costs 0.5-1.5 s of CPU that the reply is competing for, so a filler that synthesises
        itself makes the wait it is covering longer. Silence is the correct degradation."""
        wav = self._canned.get(key)
        if wav is not None:
            self._voice.speak_wav(wav, filler.text_for(key), epoch=self._epoch)

    def _tick_filler(self, now: float, elapsed: float) -> bool:
        """Caller holds the lock. Drive the filler for one tick of a BUSY turn. True if the filler
        is handling this turn, so the caller leaves the old "Hmm" alone.

        The shape, and why it is written against tts.is_playing() rather than a schedule: the whole
        contract is about SILENCE, and silence is the gap between one line ending and the next
        starting. A precomputed schedule would drift the moment a line synthesised long, or came
        back from the cache at a different length in a different voice. Reading playback state
        instead means the gap is measured from where it actually is.

            elapsed >= delay          -> opener, once
            playback ends             -> draw a gap, arm _filler_next_at
            now >= _filler_next_at    -> one stall, disarm, wait for playback to end again

        Nothing here ends the turn or touches state: the real reply arriving calls tts.stop() and
        leaves BUSY, which strands whatever is playing mid-word. That is intended — the stalls are
        short precisely so being cut off reads as natural rather than as a glitch."""
        if not self._filler_enabled():
            return False

        if not self._filler_opened:
            if elapsed < self._filler_delay:
                return True                      # still inside the deliberate head-start
            # Language is latched HERE, at the one moment the turn starts talking, and reused for
            # every stall after it. Reading it fresh per line would let a turn open in Tagalog and
            # continue in English the instant STT lands, which is worse than being wrong twice.
            warm = {k for k in self._canned if k.startswith("filler_")}
            self._filler_lang = filler.pick_lang(self._voice.last_language(), self._filler_rng)
            key = filler.pick_opener(self._filler_lang, self._filler_rng,
                                     avoid=self._filler_recent_openers, have=warm,
                                     used=self._filler_used_openers)
            if not key:
                # Nothing warm for this language — early in a boot, or every opener was rejected by
                # the length cap. Hand the OPENING of the turn back to the "Hmm", which is always
                # cached, rather than saying nothing.
                #
                # Latching is gated on the "Hmm" having actually played, and both halves of that
                # matter because each was a real bug:
                #
                #   Latching unconditionally: the fallback fires only on a tick where this returns
                #   False AND elapsed >= THINKING_SOUND_DELAY_S. Latching here spends the turn's one
                #   False tick, so any turn whose filler delay drew under 0.6 went silent end to end
                #   — the exact dead air this module exists to remove.
                #
                #   Never latching: _filler_opened stays False forever, so this branch returns False
                #   on every tick and the stall loop below is unreachable. Observed on the robot
                #   2026-08-09, where the length cap rejected all 20 openers: the turn got one "Hmm"
                #   and then nothing, with 10 perfectly good stalls sitting warm on disk.
                #
                # Gating on _played_thinking gets both: the "Hmm" covers the opening, then the
                # stalls carry the rest of the wait exactly as they would after a real opener. Until
                # it has played, staying unopened also lets a bank that finishes warming mid-turn
                # still open properly.
                if self._played_thinking:
                    self._filler_opened = True
                    return True
                return False
            self._filler_opened = True
            self._filler_recent_openers.append(key)
            # Keep the last COOLDOWN entries; a cooldown of 0 empties the list, which switches the
            # window off entirely rather than pinning it at back-to-back.
            del self._filler_recent_openers[:-FILLER_OPENER_COOLDOWN_TURNS or None]
            self._filler_used_openers.add(key)
            self._speak_filler(key)
            return True

        # Past the opener: keep stalls coming until the reply lands. Never while something is still
        # speaking — that is what would turn two lines into one garbled overlap.
        #
        # speech_in_flight(), NOT tts.is_playing(). is_playing() only goes true once a playback
        # PROCESS exists, and speak_wav hands off to a worker thread first; at 20 Hz that leaves
        # several ticks where a line has been started but is not yet "playing", and each of those
        # ticks happily started another one. That is the overlap heard on the robot — three stalls
        # talking over each other. speech_in_flight covers the whole span from before synthesis to
        # the end of playback, which is exactly the window in which nothing else may start.
        if self._voice.speech_in_flight() or tts.is_playing():
            self._filler_next_at = None
            return True
        if self._filler_next_at is None:
            self._filler_next_at = now + self._filler_gap(FILLER_STALL_GAP_JITTER_S)
            return True
        if now >= self._filler_next_at:
            if not self._filler_queue:
                warm = {k for k in self._canned if k.startswith("filler_")}
                self._filler_queue = filler.stall_queue(self._filler_lang, self._filler_rng,
                                                        have=warm, used=self._filler_used_stalls,
                                                        turn_used=self._filler_turn_stalls,
                                                        avoid=self._filler_last_stall)
            if self._filler_queue:
                key = self._filler_queue.pop()
                self._filler_used_stalls.add(key)
                self._filler_turn_stalls.add(key)
                self._filler_last_stall = key
                self._speak_filler(key)
            self._filler_next_at = None
        return True

    def _speak_canned(self, key: str, fallback: str) -> None:
        """Play a pre-synthesized line, or synthesize it live if the cache missed. Caller holds the
        lock. Never routed through say(): that sets _status/_response, which would make the dashboard
        post a spurious "Kai: Yes?" chat bubble on every single wake."""
        wav = self._canned.get(key)
        if wav is not None:
            self._voice.speak_wav(wav, fallback, epoch=self._epoch)
        else:
            self._voice.speak_text(fallback, epoch=self._epoch)

    # ── state + timers ──────────────────────────────────────────────────────

    def _set_state(self, state: str, now: float) -> None:
        """Caller holds the lock. Entering a state arms its deadlines; there is nothing to cancel,
        because every deadline is measured from _state_since."""
        if state != self._state:
            self._log(f"{self._state} -> {state}")
        if state == STATE_BUSY:
            # Re-armed here rather than at each of the two BUSY entry points (_start_text_turn and
            # _finish_utterance), so a third one can never forget to.
            self._played_thinking = False
            self._arm_filler()
        self._state = state
        self._state_since = now

    def _enter_listen_wait(self, now: float) -> None:
        """Caller holds the lock. Re-entering resets the silence clock AND the absence clock, so
        someone who stepped out of frame during a 50 s cold load gets a fresh 8 s rather than being
        hung up on the instant Kai finishes."""
        self._face_absent_since = None
        self._gate.reset()
        self._mic.arm_utterance(preroll=False)
        self._set_state(STATE_LISTEN_WAIT, now)

    def _tick_loop(self) -> None:
        interval = 1.0 / max(1, SESSION_TICK_HZ)
        next_tick = time.monotonic()
        while not self._stop.is_set():
            next_tick += interval
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                if self._stop.wait(sleep):
                    return
            else:
                next_tick = time.monotonic()   # fell behind; don't spin trying to catch up
            try:
                self.tick(time.monotonic())
            except Exception as exc:
                print(f"[session] ERROR in tick: {exc}", flush=True)

    def tick(self, now: float) -> None:
        """Advance every timer. `now` is passed in so tests drive a fake clock with no sleeping.

        The two capture timeouts below do NOT finish their utterance inline. `self._lock` is an
        RLock, so a nested `with` inside a method called from here is re-entrant — which means the
        code those methods carefully place *after* their own `with` block, on the stated grounds
        that "recording inside the lock would put a WAV write on the tick thread", was still inside
        THIS one. UtteranceRecorder.record() writes up to CAPTURE_HARD_CAP_S of audio, and the audio
        worker needs this same lock ~30 times a second for the VAD. So the timeout paths hand back a
        pending dispatch and it runs once the lock is released, which is exactly what _on_audio
        already does with its own `pending`.
        """
        pending: tuple[str, str] | None = None
        hotswap = False
        with self._lock:
            if self._presence is not None:
                try:
                    visible, _since, is_fresh = self._presence(now)
                    self._apply_presence(visible, is_fresh, now)
                except Exception as exc:
                    self._face_absent_since = None   # a broken feed must fail open
                    self._note_presence_error(exc, now)

            state = self._state
            elapsed = now - self._state_since

            if state == STATE_ACK:
                # Exit on speech_in_flight, NOT on tts.is_playing(): playback hasn't started yet
                # while Piper runs, and quiet_since() is already finite from the PREVIOUS reply —
                # either check alone would leave ACK the instant it was entered.
                if not self._voice.speech_in_flight():
                    self._set_state(STATE_COOLDOWN, now)
                elif elapsed >= WAKE_ACK_MAX_S:
                    self._cut_speech()
                    self._log("ack timed out")
                    self._set_state(STATE_COOLDOWN, now)

            elif state == STATE_SPEAKING:
                if not self._voice.speech_in_flight():
                    self._set_state(STATE_COOLDOWN, now)
                elif now >= self._speaking_deadline():
                    # Backstop only — the normal exit is above. A wedged paplay would otherwise keep
                    # the mic muted forever, i.e. deafen Kai for good, and invisibly, because the
                    # mute gate is keyed on playback.
                    self._cut_speech()
                    self._log("speak timed out")
                    self._set_state(STATE_COOLDOWN, now)

            elif state == STATE_COOLDOWN:
                # The assistant-level gate, deliberately not self.mic_muted(): that one reports True
                # for every state in _SPEECH_STATES, COOLDOWN included, so it could never clear here.
                if not self._voice.mic_muted(now):
                    self._mic.reset_dsp()   # residue from the mute window must not trip an onset
                    self._enter_listen_wait(now)

            elif state == STATE_LISTEN_WAIT:
                if elapsed >= SESSION_NO_SPEECH_S:
                    self._end_session(now, "no_speech")
                elif (self._face_ever_seen and self._face_absent_since is not None
                        and now - self._face_absent_since >= SESSION_NO_FACE_S):
                    # Only armed here, and only after a first sighting — so waking from the next room
                    # or in the dark gets the full no-speech window instead of dying in 8 s.
                    self._end_session(now, "no_face")

            elif state == STATE_LISTEN_SPEECH:
                if not self._manual_end and elapsed >= MAX_UTTERANCE_S:
                    self._log("max utterance reached")
                    pending = ("turn", "max_utterance")   # dispatched below, off the lock

            elif state == STATE_SCAN_SPEECH:
                if elapsed >= WAKE_WHISPER_MAX_UTTERANCE_S:
                    pending = ("scan", "too_long")        # dispatched below, off the lock

            elif state == STATE_SCAN_CHECK:
                if elapsed >= WAKE_WHISPER_CHECK_MAX_S:
                    self._scan_token += 1        # orphan the worker's result
                    self._log("wake check timed out")
                    self._scan_ready_at = now + WAKE_WHISPER_LONG_COOLDOWN_S
                    self._set_state(STATE_IDLE, now)

            elif state == STATE_BUSY:
                # "Hmm..." once per turn, after a delay, so quick replies stay silent. Deliberately
                # NOT a state change: this is decoration on top of BUSY, so every BUSY timer and the
                # mic mute (BUSY is already in _SPEECH_STATES) keep working untouched, and
                # _project_status keeps reporting voice_speaking=False so no chat bubble appears.
                # The filler bank supersedes the "Hmm" when it is on: an opener then stalls on a
                # loop, which covers a wait of any length instead of only the first 1.5 s. It
                # returns False when it has nothing to say (bank off, or empty for this language),
                # and the old single sound runs exactly as it did before.
                if not self._tick_filler(now, elapsed):
                    if (not self._played_thinking and elapsed >= THINKING_SOUND_DELAY_S
                            and settings.get("thinking_sounds")):
                        self._played_thinking = True
                        self._speak_canned("thinking", THINKING_SOUND_TEXT)
                if elapsed >= SESSION_BUSY_MAX_S:
                    # Above OLLAMA_TIMEOUT_S + STT, so this only fires when a worker is genuinely
                    # wedged in native code, where nothing raises.
                    self._end_session(now, "busy_timeout")

            # A watchdog reopen briefly has no stream; that is recovery in progress, not a lost mic,
            # and ending the session there would drop a conversation that was about to be fine.
            if not self._mic.live and not self._mic.reopening and state != STATE_DISABLED:
                self._end_session(now, "mic_lost")

            hotswap = self._check_hotplug(now)
            self._heartbeat(now)

        # Outside the lock, by construction rather than by convention: both of these harvest audio,
        # spawn a worker and write a WAV, and the audio worker is waiting on this lock at ~30 Hz.
        if pending is not None:
            kind, reason = pending
            if kind == "turn":
                self._finish_utterance(now, reason=reason)
            else:
                self._finish_scan(now, reason=reason)

        # Out here for the same reason and a stronger one: a re-resolve tears the stream down,
        # re-scans PortAudio and runs liveness probes, which is seconds of blocking work — several
        # hundred tick periods — and the lock is an RLock, so doing it above would have been legal
        # and would have starved the audio worker the whole time.
        if hotswap:
            self._run_hotswap(now)

    def _check_hotplug(self, now: float) -> bool:
        """Caller holds the lock. True if the tick should re-resolve the mic once it releases it.

        The watcher reports a settled card change; this decides whether now is the moment. Two things
        make it not the moment, and they are different:

          * Kai is mid-turn. Re-resolving takes the microphone away for seconds, so doing it during a
            wake check, an utterance or a reply would drop that turn on the floor — and the person
            talking would have no idea why. The change is LATCHED rather than dropped: whoever
            plugged the mic in still gets it, at the next quiet moment, which is at most one turn
            away. (Latched, not re-polled: the watcher reports each change exactly once, so a
            dropped report would never come back.)
          * The cooldown has not elapsed. Anti-flap — see MIC_HOTPLUG_COOLDOWN_S. A cable with a bad
            contact, or a hub re-enumerating under load, must not be able to spend the session
            tearing the capture stream down. The pending flag survives the cooldown, so a real change
            during one is honoured when it expires rather than being lost.

        STATE_DISABLED counts as quiet on purpose: it is the state a robot with no working mic sits
        in, so it is the single most valuable moment to act on someone plugging one in.
        """
        if self._hotplug.poll(now):
            self._hotplug_pending = True
        if not self._hotplug_pending:
            return False
        if self._state not in (STATE_IDLE, STATE_DISABLED):
            return False
        if self._hotplug_last and now - self._hotplug_last < MIC_HOTPLUG_COOLDOWN_S:
            return False
        self._hotplug_pending = False
        self._hotplug_last = now
        return True

    def _run_hotswap(self, now: float) -> None:
        """Caller does NOT hold the lock. Re-resolve after a card change, and say what happened.

        Deliberately the same path as the dashboard button — reresolve_mic() — rather than a second
        way to reopen the mic. It already handles both "the session is live" and "the session never
        started", already re-derives sess_state out of STATE_DISABLED, and already cannot raise. The
        only thing hot-plug adds is noticing; nothing about the recovery itself is new.
        """
        self._hotswaps += 1
        result = self.reresolve_mic()
        if result.get("ok"):
            self._log(f"hot-plug: now on the {result.get('kind', '?')} mic "
                      f"(device {result.get('device')})")
        else:
            # Not an error worth ending anything over: the cards changed, we looked, nothing usable
            # came back. Says so once, and the next change (or the dashboard button) tries again.
            self._log(f"hot-plug: no usable microphone after the card change "
                      f"({result.get('error') or 'no reason reported'})")

    def _end_session(self, now: float, reason: str) -> None:
        """Caller holds the lock. Back to idle, with the conversation forgotten."""
        self._end_reason = reason
        self._log(f"session end: {reason} after {self._turns} turn(s)")
        # Orphan any wake check still in flight, so a late match can't start a session on top of one
        # that just ended.
        self._scan_token += 1
        self._scan_ready_at = now + WAKE_WHISPER_COOLDOWN_S
        self._cut_speech()
        self._mic.harvest_utterance()   # disarm and drop whatever was buffered
        self._gate.reset()
        self._mic.reset_dsp()
        self._manual_end = False
        # Bumps the epoch too, so a reply still in flight can't append itself to the next person's
        # conversation.
        self._voice.reset_history()
        # Retire the finished turn's status. Without this, leaving LISTEN_WAIT (projected as
        # "recording") back to IDLE un-masks a stale "done" and the dashboard posts the last
        # question and answer a second time.
        self._voice.clear_turn_status()
        self._epoch = self._voice.epoch
        self._set_state(STATE_IDLE if self._wake_live() else STATE_DISABLED, now)

    def end_session(self, reason: str = "manual") -> dict:
        with self._lock:
            self._end_session(time.monotonic(), reason)
        return {"status": "ok"}

    # ── live settings (called from a Flask request thread) ───────────────────

    def set_hands_free(self, enabled: bool) -> dict:
        """Turn the wake word on or off while running.

        Off ends any conversation in progress through the ordinary _end_session teardown (which stops
        playback, drops the buffer and resets the gate) and lands in STATE_DISABLED — the state that
        already exists for "push-to-talk only", so sess_state, sess_ready and the dashboard all agree
        it is off rather than broken.

        On has to cope with never having opened the engine: hands-free may have been off at startup, in
        which case wake.open() has not run. If no tier can start, we stay disabled and sess_wake_error
        explains why, rather than claiming to be listening.
        """
        enabled = bool(enabled)
        with self._lock:
            was = self.enabled
            self.enabled = enabled

        if enabled and not self._mic.wake.ready:
            # Order matters: the winning tier decides the frame size, so geometry syncs AFTER open().
            self._mic.wake.open()
            self._mic.sync_wake_geometry()

        live = self._wake_live()
        self._mic.set_wake_enabled(live)

        now = time.monotonic()
        with self._lock:
            if not enabled and self._state not in (STATE_IDLE, STATE_DISABLED):
                self._end_session(now, "hands_free_off")
            if not enabled:
                self._set_state(STATE_DISABLED, now)
            elif live:
                self._set_state(STATE_IDLE, now)

        if was != enabled:
            print(f"[session] hands-free {'on' if enabled else 'off'}"
                  + ("" if live or not enabled else f" (but no wake engine: {self._mic.wake.unavailable})"),
                  flush=True)
        return {"status": "ok", "hands_free": enabled, "wake_live": live}

    def set_rms_floor(self, floor: float) -> None:
        """Retune the VAD speech-onset floor. Under the lock, because SpeechGate is not thread-safe and
        the audio worker touches it on every block."""
        with self._lock:
            self._gate.set_rms_floor(float(floor))
        print(f"[session] mic noise floor -> {float(floor):g}", flush=True)

    def set_wake_sensitivity(self, value: float) -> None:
        """Retune wake-word sensitivity. Some tiers compare per frame (free); Porcupine bakes it in at
        create() time and needs a guarded engine reload — see MicStream.reload_wake."""
        if self._mic.set_wake_sensitivity(float(value)):
            print(f"[session] wake sensitivity -> {float(value):g} (engine reloaded)", flush=True)
        else:
            print(f"[session] wake sensitivity -> {float(value):g}", flush=True)

    # ── gate ────────────────────────────────────────────────────────────────

    def mic_muted(self, now: float | None = None) -> bool:
        """True while Kai's own audio could be reaching the mic. The audio worker drops blocks on
        this, before any DSP.

        FSM state is primary and tts.is_playing() secondary, not the other way round: ACK and
        SPEAKING are entered BEFORE synthesis begins, and that pre-playback window is exactly where
        `voice_speaking` and `is_playing()` both still read false."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._state in _SPEECH_STATES:
                return True
        return self._voice.mic_muted(now)

    # ── status + logging ────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        """Everything a wake needs to be answerable rather than looking like a hang: the stream is
        live, the model is loaded, and (if hands-free is on) Porcupine came up."""
        if not self._mic.live:
            return False
        if self.enabled and not self._mic.wake.ready:
            return False
        return self._voice.stt_ready

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def owns_capture(self) -> bool:
        """True once the shared stream is up, meaning push-to-talk must be routed through here
        rather than letting the assistant open a second stream on the same device."""
        return self._mic.live

    def _project_status(self) -> tuple[str | None, bool | None]:
        """Map session state onto the dashboard's voice_status/voice_speaking contract.

        Returns (status_override, speaking_override); None means "leave the assistant's own value".
        This is what lets the existing dashboard show hands-free state with zero JS changes."""
        state = self._state
        if state in (STATE_DISABLED, STATE_IDLE):
            return None, None
        if state in _SCAN_STATES:
            # A scan is speculative — nobody may have addressed Kai at all. Project it exactly as
            # idle so the dashboard's WakeChip falls through to its idle entry and the mic button
            # doesn't flicker on every nearby sentence.
            return None, None
        if state in (STATE_LISTEN_WAIT, STATE_LISTEN_SPEECH):
            return STATUS_RECORDING, False
        if state == STATE_BUSY:
            # Normally defer to the assistant's own transcribing -> thinking progression. But never
            # let a PREVIOUS turn's terminal status show through: _finish_utterance enters BUSY inside
            # the lock and only then calls into the assistant, and in that gap voice_status would fall
            # back to the last turn's "done". Coming from LISTEN_* ("recording") that reads as a fresh
            # transition INTO "done", and the dashboard re-posts the previous question and answer
            # verbatim — which is exactly the duplicate-bubble bug.
            live = self._voice.get_status().get("voice_status")
            if live in (STATUS_DONE, STATUS_ERROR, STATUS_IDLE):
                return STATUS_TRANSCRIBING, False
            return None, False
        if state in (STATE_ACK, STATE_SPEAKING, STATE_COOLDOWN):
            return None, True
        return None, None

    def get_status(self, now: float | None = None) -> dict:
        """Additive sess_* keys for /params, plus the projected voice_status/voice_speaking.

        Every counter here exists because a misfire has to be diagnosable over ssh. The pair that
        actually gets used most is sess_rms + sess_rms_floor — that's how VAD_RMS_FLOOR gets set.
        `now` is injectable for the same reason it is on tick()."""
        now = time.monotonic() if now is None else now
        with self._lock:
            state, since = self._state, now - self._state_since
            face_absent = (now - self._face_absent_since) if self._face_absent_since else 0.0
            status_override, speaking_override = self._project_status()
            no_speech_left = (max(0.0, SESSION_NO_SPEECH_S - since)
                              if state == STATE_LISTEN_WAIT else 0.0)
            no_face_left = (max(0.0, SESSION_NO_FACE_S - face_absent)
                            if state == STATE_LISTEN_WAIT and self._face_absent_since else 0.0)
            out = {
                "sess_state": state,
                "sess_state_s": round(since, 2),
                "sess_ready": self.ready,
                "sess_enabled": self.enabled,
                "sess_epoch": self._epoch,
                # Who Kai believes it is talking to (ai/identity.py), "" until they say so. Reads
                # from the assistant rather than being mirrored here, so there is one owner and the
                # dashboard cannot show a name the prompt no longer carries.
                "sess_person": self._voice.person_name or "",
                "sess_wake_ok": self._mic.wake.ready,
                "sess_wake_error": self._mic.wake.unavailable or "",
                # Which tier actually won. The first thing to read when someone reports "Kai stopped
                # hearing me" — tier 2 or 3 winning does not mean tier 2 or 3 is *good*.
                "sess_wake_engine": self._mic.wake.engine,
                "sess_wake_kind": self._mic.wake.kind,
                "sess_wake_frame": self._mic.wake.frame_length,
                "sess_wake_score": round(self._mic.wake.last_score, 3),
                # The APPLIED sensitivity, so the dashboard slider reflects what the engine is really
                # using rather than what was last requested.
                "sess_wake_sensitivity": round(self._mic.wake.sensitivity, 3),
                "sess_wake_tried": "; ".join(f"{k}={v}" for k, v in self._mic.wake.tiers.items()),
                # Whisper tier only; all zero on the frame tiers.
                "sess_scan_checks": self._scans,
                "sess_scan_matches": self._scan_matches,
                "sess_scan_skip_cooldown": self._scan_skipped["cooldown"],
                "sess_scan_skip_short": self._scan_skipped["short"],
                "sess_scan_skip_long": self._scan_skipped["long"],
                "sess_scan_skip_no_model": self._scan_skipped["no_model"],
                "sess_scan_last_ms": self._scan_last_ms,
                "sess_scan_last_text": self._scan_last_text,
                "sess_scan_ready_in_s": round(max(0.0, self._scan_ready_at - now), 1),
                "sess_wake_count": self._wakes,
                "sess_wake_rejected_busy": self._rejected["busy"],
                "sess_wake_rejected_speaking": self._rejected["speaking"],
                "sess_wake_rejected_refractory": self._rejected["refractory"],
                "sess_wake_rejected_not_ready": self._rejected["not_ready"],
                "sess_rms": round(self._mic.last_rms, 1),
                # The LIVE floor off the gate, not the config constant: this pair is the mic-tuning
                # display, and reporting the startup default while the gate used a dashboard-set value
                # would make the one number an operator relies on a lie.
                "sess_rms_floor": self._gate.rms_floor,
                # The measured room, and what the floors actually became after adapting to it. With
                # WAKE_AMBIENT_ADAPT off, or before the first window, these read 0.0 and equal
                # sess_rms_floor respectively. sess_rms_ambient is the number to look at when the
                # wake word works at your desk and not in the venue.
                "sess_rms_ambient": round(self._gate.ambient, 1),
                "sess_rms_floor_live": round(self._gate.open_floor, 1),
                "sess_rms_hold_live": round(self._gate.hold_floor, 1),
                "sess_vad_onsets": self._gate.onsets,
                "sess_vad_available": self._gate.vad_available,
                "sess_discarded_short": self._discarded_short,
                "sess_mic_muted": state in _SPEECH_STATES,
                "sess_muted_blocks": self._mic.muted_blocks,
                "sess_mic_live": self._mic.live,
                "sess_mic_kind": self._mic.kind,
                "sess_mic_hotswaps": self._hotswaps,
                "sess_mic_hotplug_watching": self._hotplug.active,
                "sess_mic_reopens": self._mic.reopens,
                "sess_audio_overflows": self._mic.overflows,
                "sess_blocks_dropped": self._mic.dropped_blocks,
                "sess_capture_s": round(self._mic.capture_seconds, 2),
                "sess_capture_truncated": self._mic.capture_truncated,
                "sess_face_present": self._face_state(now),
                "sess_face_ever_seen": self._face_ever_seen,
                "sess_face_absent_s": round(face_absent, 1),
                "sess_no_speech_left_s": round(no_speech_left, 1),
                "sess_no_face_left_s": round(no_face_left, 1),
                "sess_busy_s": round(since, 1) if state == STATE_BUSY else 0.0,
                "sess_turns": self._turns,
                "sess_end_reason": self._end_reason,
                "sess_stale_results": self._stale_results,
                # The latency breakdown. sess_last_turn_ms is what sess_last_llm_ms used to hold
                # (STT+RAG+LLM under an LLM label); sess_last_llm_ms is now the LLM alone, and
                # sess_last_stt_ms is populated for the first time — it was hardcoded 0 before.
                "sess_last_stt_ms": self._timings["stt_ms"],
                "sess_last_llm_ms": int(self._stage_ms.get("llm_ms", 0)),
                "sess_last_turn_ms": self._timings["llm_ms"],
                "sess_last_rag_ms": int(self._stage_ms.get("rag_ms", 0)),
                # Ollama's own counters: a big prompt number means the KV cache prefix is being
                # invalidated, a low tok/s means the model landed on CPU.
                "sess_last_llm_prompt_ms": int(self._stage_ms.get("llm_prompt_ms", 0)),
                "sess_last_llm_gen_ms": int(self._stage_ms.get("llm_gen_ms", 0)),
                "sess_last_llm_tok_s": float(self._stage_ms.get("llm_tok_s", 0.0)),
                "sess_last_tts_synth_ms": int(self._stage_ms.get("tts_synth_ms", 0)),
                # The number a person actually feels: end of their sentence -> first sound back.
                "sess_last_first_audio_ms": int(self._stage_ms.get("first_audio_ms", 0)),
            }
        # Outside the lock: both of these take locks of their own (the assistant's, the recorder's),
        # and nesting those inside the session lock is the lock-order inversion that _process_block
        # already goes out of its way to avoid.
        #
        # sess_asr_rms is the distance readout. It is the level of the audio Whisper actually
        # decoded, so unlike sess_rms it is per-utterance rather than per-block, and reading well
        # under ASR_NORMALIZE_TARGET_RMS with sess_asr_gain pinned at ASR_NORMALIZE_MAX_GAIN means
        # the speaker was further away than level correction can reach.
        try:
            levels = self._voice.input_levels()
            out["sess_asr_rms"] = round(levels["rms"]["turn"], 5)
            out["sess_asr_gain"] = round(levels["gain"]["turn"], 2)
            out["sess_asr_rms_scan"] = round(levels["rms"]["scan"], 5)
        except (AttributeError, TypeError, KeyError):
            pass
        rec = self._recorder.status()
        out["sess_debug_capture"] = rec["enabled"]
        out["sess_debug_clips"] = rec["written"]
        out["sess_debug_mb"] = rec["mb"]
        out["sess_debug_skipped"] = rec["skipped"]
        out["sess_debug_error"] = rec["error"]
        if status_override is not None:
            out["voice_status"] = status_override
        if speaking_override is not None:
            out["voice_speaking"] = speaking_override
        return out

    def _face_state(self, now: float) -> str:
        if self._presence is None:
            return "unknown"
        try:
            visible, _since, is_fresh = self._presence(now)
        except Exception as exc:
            self._note_presence_error(exc, now)
            return "unknown"
        if not is_fresh:
            return "unknown"
        return "yes" if visible else "no"

    def _note_presence_error(self, exc: BaseException, now: float) -> None:
        """Say that the presence feed is broken, at most once a minute.

        Failing open is right — a camera hiccup must never end a conversation — but doing it
        silently is not: a snapshot callable that raises every time is indistinguishable from a
        camera that simply cannot see anyone, and sess_face_present reports "unknown" for both.
        Rate-limited because this is reached from the 20 Hz tick and from every /params snapshot.
        """
        detail = f"{type(exc).__name__}: {exc}"
        if detail == self._presence_error and now - self._presence_error_t < PRESENCE_ERROR_LOG_S:
            return
        self._presence_error, self._presence_error_t = detail, now
        self._log(f"WARNING: presence feed failed ({detail}) — treating presence as unknown")

    def _log(self, message: str) -> None:
        # flush=True because stdout is block-buffered into /tmp/face-servo.log under the autostart,
        # and a transition you only see minutes later is no use while tuning.
        print(f"[session] {message}", flush=True)

    def _heartbeat(self, now: float) -> None:
        """A periodic line while a session is live, mirroring face_track's `[control] N Hz`."""
        # Scan states are excluded too: every nearby sentence would otherwise produce 2 Hz of
        # heartbeat lines. The one-line-per-check log is the right granularity.
        if self._state in (STATE_IDLE, STATE_DISABLED) or self._state in _SCAN_STATES:
            return
        if now - self._last_heartbeat < 2.0:
            return
        self._last_heartbeat = now
        print(f"[session] {self._state} {now - self._state_since:4.1f}s "
              f"rms={self._mic.last_rms:6.0f} face={self._face_state(now)} "
              f"turns={self._turns}", flush=True)
