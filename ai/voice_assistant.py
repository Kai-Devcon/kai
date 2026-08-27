"""
Voice assistant pipeline: mic capture -> faster-whisper STT -> Ollama LLM -> Piper TTS + jaw.

Call VoiceAssistant.start_recording() / stop_recording() from Flask handlers.
Call VoiceAssistant.get_status() each dashboard SSE tick to read live state.

Capture has two modes. By default this class owns nothing: attach_mic() hands it a shared,
always-open stream (ai/session.py) and capture becomes arm/harvest calls against that. With no mic
attached it falls back to opening its own sd.InputStream per turn — the original push-to-talk path,
kept as the rollback for MIC_LEGACY_CAPTURE. Either way start_recording()/stop_recording() keep the
same names, guards and return shapes, so the Flask routes and the dashboard don't care which is live.

Anything asynchronous (STT, LLM, synth, playback) carries the turn `epoch` it was started under and
is dropped on mismatch. That one mechanism is what stops an abandoned reply from speaking into a
session that has already ended, and what stops a cleared history from being repopulated by a
request that was already in flight when it was cleared.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from ai import delivery
from ai import rag
from ai import tts
from ai.audio import normalize_for_asr
from ai.identity import extract_name
# Device plumbing lives in ai/mic_device.py — this class is one of its two consumers, not its owner.
# Re-exported (rather than reached through the module) because ensure_input_resolved() and
# start_recording() call these by bare name, which is what lets a test patch them here.
from ai.mic_device import (
    MicChoice, apply_i2s_route, free_i2s_device, resolve_input_device, resume_pulse_source,
)
# Three layers this class uses but does not own. Re-exported for the same reason as mic_device's
# entry points: _call_ollama, ensure_llm_warm, _speak and _transcribe call them by bare name, so
# they resolve through this module's globals — which is what a test patches.
from ai.llm import (
    _log_llm_timings, _ollama_request, build_chat_messages, load_persona, log_model_placement,
)
from ai.speak_envelope import _speak_segments, _speak_segments_for_duration, speaking_openness_at
from ai.transcript import _best_allowed_language, transcript_rejection

# ── Tuning ────────────────────────────────────────────────────────────────────
# All tunable knobs (audio, Whisper, Ollama, history, jaw-speaking envelope) live in
# config/voice.py; re-imported so the names stay module-level for internal use and so
# existing `from ai.voice_assistant import ...` callers keep working.
from config.voice import (
    SAMPLE_RATE, CHANNELS,
    WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE, WHISPER_LANGUAGE, WHISPER_LANGUAGES,
    WHISPER_CPU_THREADS, WHISPER_BEAM_SIZE, WHISPER_INITIAL_PROMPT,
    ASR_NORMALIZE, ASR_NORMALIZE_MAX_GAIN, ASR_NORMALIZE_SCAN,
    RAG_CONTEXT_PLACEMENT, MAX_HISTORY_TURNS,
    IDENTITY_CAPTURE, IDENTITY_PROMPT,
    SPEAK_TRIM_SILENCE, TTS_PLAYBACK_LEAD_S,
)
from config.wake import (
    CAPTURE_HARD_CAP_S, TTS_MAX_SPOKEN_CHARS, TTS_TAIL_MUTE_S,
    WAKE_WHISPER_SCAN_LANGUAGE, WAKE_WHISPER_SCAN_MODEL,
)


NO_SPEECH_RESPONSE = "(didn't catch that — try again)"

STATUS_IDLE         = "idle"
STATUS_RECORDING    = "recording"
STATUS_TRANSCRIBING = "transcribing"
STATUS_THINKING     = "thinking"
STATUS_DONE         = "done"
STATUS_ERROR        = "error"

# ── Jaw "speaking" pantomime ────────────────────────────────────────────────────
# Kai has no audio (yet), so when a reply is produced we drive the jaw servo for a
# window sized to how long that text would take to say aloud. The mouth opens once
# per sentence: it ramps open at the start of the sentence, holds open while the
# sentence is "spoken", then closes at the end, with a short closed pause between
# sentences. face_track.py reads speaking_openness() each frame and maps the 0..1
# result onto the jaw servo (overriding the human-mouth mirror during a reply).
# The SPEAK_* envelope values are imported from config/voice.py (see the import block above).


class VoiceAssistant:
    """Owns mic capture, STT, and LLM state. Independent of Flask/face_track globals."""

    def __init__(self) -> None:
        self._lock       = threading.Lock()
        self._status     = STATUS_IDLE
        self._transcript = ""
        self._response   = ""
        self._error      = ""
        self._history: list[dict] = []
        # Who Kai is talking to, if they said so (ai/identity.py). Session-scoped: cleared by
        # reset_history() alongside the rolling history and the sticky RAG topic, because all three
        # answer the same question — what may the NEXT person inherit? Nothing. See docs/tickets/S12.
        self._person_name: str | None = None
        self._audio_chunks: list[np.ndarray] = []
        self._audio_samples = 0        # running total, so the legacy buffer can be bounded
        self._stream: sd.InputStream | None = None
        # Shared always-open capture (ai/session.py). None = own a stream per turn instead.
        self._mic = None
        # Turn generation. Every async result carries the epoch it began under; a mismatch means a
        # newer turn or a session reset happened meanwhile, and the result is discarded.
        self._epoch = 0
        self._tts_active = False       # True from before synthesis starts until playback ends
        # Which utterance owns _tts_active. Bumped by _begin_speech for every new line, and checked
        # by each worker before it clears the flag — because _tts_active is ONE boolean shared by
        # every speech path (reply, ack, canned, filler) and its workers finish out of order.
        # Observed 2026-08-09: a 7.5 s filler opener was still playing when the 2.8 s reply started,
        # and the opener's worker then cleared the flag out from under the reply. speech_in_flight()
        # went False mid-answer, which dropped SPEAKING straight to COOLDOWN and told the filler
        # loop that nothing was playing — so it started more lines on top. Everything talked at once.
        self._speech_gen = 0
        self._last_language = ""       # last Whisper language label; read by the filler bank
        self._gate_until = 0.0         # monotonic deadline covering the sink drain + amp settle
        # Monotonic time the audio now playing is expected to END, or None when nothing is playing
        # or its length is unknown (a pantomime, or a WAV whose header wouldn't read). Published so
        # ai/session.py can size STATE_SPEAKING's deadline to the reply actually being spoken —
        # without it the session only has SESSION_SPEAK_MAX_UNKNOWN_S, armed before Piper even
        # starts, which cut every long reply off mid-sentence. Distinct from _gate_until, which is
        # this plus the settle tail and answers a different question (may the mic listen yet).
        self._audio_ends_at: float | None = None
        self._capture_device: int | None = None
        self._capture_rate = SAMPLE_RATE
        self._capture_channels = CHANNELS
        self._capture_channel  = 0
        self._capture_dtype    = "int16"
        self._capture_is_i2s   = False
        self._capture_card     = ""   # ALSA card, set only when the resolved device is raw hw:
        self._device_resolved = False
        self._whisper_model = None
        self._scan_model = None        # tiny model, wake-phrase spotting only
        self._speak_start: float | None = None
        self._speak_segments: tuple[tuple[float, float], ...] = ()
        # Monotonic count of COMPLETED turns that produced something to show. Published as
        # voice_turn_id and used by the dashboard to decide when to append a chat bubble.
        #
        # It exists because inferring that from `voice_status` reaching "done" is not safe: that field
        # is driven by both this class and ai/session.py's projection, so any momentary gap between
        # them reads as a brand-new completed turn and re-posts the previous exchange verbatim. Three
        # separate races produced exactly that bug. A counter that only ever moves when a turn really
        # finished cannot be faked by a gap.
        self._turn_id = 0
        # Per-stage latency for the LAST completed turn, in milliseconds. Measured HERE, next to the
        # stages themselves, because this is the only place that can see them apart: ai/session.py
        # only observes "turn started" and "turn finished", which is why its sess_last_llm_ms was
        # really STT+RAG+LLM under an LLM label, and its stt_ms was never written at all.
        #
        # llm_ms is the wall time of the POST; llm_prompt_ms/llm_gen_ms/llm_tok_s come from Ollama's
        # own counters (see _log_llm_timings) and are what separate a slow prompt from slow
        # generation. first_audio_ms is the one a person actually feels: utterance handed over ->
        # first sample sent to the speaker.
        self._stage_ms = {
            "stt_ms": 0, "rag_ms": 0, "llm_ms": 0,
            "llm_prompt_ms": 0, "llm_gen_ms": 0, "llm_load_ms": 0,
            "llm_prompt_tokens": 0, "llm_gen_tokens": 0, "llm_tok_s": 0.0,
            "tts_synth_ms": 0, "first_audio_ms": 0,
        }
        # ASR input level diagnostics, kept per path because they answer different questions: the
        # turn entry says how far away the person Kai is talking to was, the scan entry says the
        # same for whoever the wake tier last overheard. See ASR_NORMALIZE in config/voice.py.
        self._norm_gain = {"turn": 1.0, "scan": 1.0}
        self._norm_rms  = {"turn": 0.0, "scan": 0.0}

    def input_levels(self) -> dict:
        """Copy of the last measured ASR input RMS and applied gain, per path. Any thread.

        This is the distance readout. A turn RMS well under ASR_NORMALIZE_TARGET_RMS with the gain
        pinned at ASR_NORMALIZE_MAX_GAIN means the speaker is further away than level correction can
        cover, and the next lever is the mic, not a constant."""
        with self._lock:
            return {"gain": dict(self._norm_gain), "rms": dict(self._norm_rms)}

    def stage_timings(self) -> dict:
        """Copy of the last turn's per-stage latency (see _stage_ms). Safe to call from any thread."""
        with self._lock:
            return dict(self._stage_ms)

    # ── Whisper model (lazy singleton, can be pre-warmed at startup) ──────────

    def ensure_model_loaded(self) -> None:
        if self._whisper_model is None:
            from faster_whisper import WhisperModel
            kwargs = {}
            if WHISPER_CPU_THREADS:
                # Left at ctranslate2's default this uses every core, which starves the servo
                # control loop and the tracking loop — see config/voice.py.
                kwargs["cpu_threads"] = WHISPER_CPU_THREADS
            self._whisper_model = WhisperModel(
                WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE, **kwargs
            )

    def ensure_scan_model_loaded(self) -> None:
        """Load the small, fast model used ONLY for wake-phrase spotting.

        Separate from the turn model on purpose: spotting two known words is a much easier job than
        transcribing a question, and doing it with "small" costs seconds per check — enough to make
        the whisper wake tier unusable. Falls back to the turn model when they're configured the same,
        so nothing is loaded twice."""
        if not WAKE_WHISPER_SCAN_MODEL or WAKE_WHISPER_SCAN_MODEL == WHISPER_MODEL:
            self.ensure_model_loaded()
            return
        if self._scan_model is None:
            from faster_whisper import WhisperModel
            kwargs = {}
            if WHISPER_CPU_THREADS:
                kwargs["cpu_threads"] = WHISPER_CPU_THREADS
            self._scan_model = WhisperModel(
                WAKE_WHISPER_SCAN_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE,
                **kwargs
            )
            print(f"[voice_assistant] wake-scan model loaded: {WAKE_WHISPER_SCAN_MODEL}"
                  f"/{WHISPER_COMPUTE}", flush=True)

    @property
    def stt_ready(self) -> bool:
        """True once the turn model is loaded, so a wake can be answered rather than looking like a
        hang. ai/session.py's `ready` folds this in; before this existed it read _whisper_model
        directly, which is the kind of reach-through that makes a stub in a test guess at internals.
        """
        return self._whisper_model is not None

    @property
    def scan_ready(self) -> bool:
        """True when wake-phrase spotting can run without paying a model load inside the check."""
        return self._scan_model is not None or self._whisper_model is not None

    def ensure_input_resolved(self) -> None:
        """Probe for a live mic once (the probe takes real time); safe to call repeatedly. First
        applies the I2S capture route and suspends pulseaudio so the raw INMP441 device is live and
        openable before it's probed; if we don't end up on the I2S device, pulse is handed back."""
        if not self._device_resolved:
            apply_i2s_route()   # best-effort; no-op / graceful fallback when the APE card is absent
            free_i2s_device()   # release the card from pulse so the raw hw probe can open at 48 kHz
            choice = resolve_input_device()
            # Every card back to pulse EXCEPT the one about to be opened raw — on a build with no
            # module-suspend-on-idle, pulse re-grabs a resumed card immediately and the open then
            # fails with "Device unavailable". A USB mic on hw:<n>,0 is as exclusive as the I2S one,
            # which is why this asks for the card rather than for is_i2s. See resume_pulse_sources().
            resume_pulse_source(keep_card=choice.card)
            with self._lock:
                self._capture_device   = choice.device
                self._capture_rate     = choice.rate
                self._capture_channels = choice.channels
                self._capture_channel  = choice.take_channel
                self._capture_dtype    = choice.dtype
                self._capture_is_i2s   = choice.is_i2s
                self._capture_card     = choice.card
                self._device_resolved  = True

    def ensure_llm_warm(self) -> None:
        """Best-effort: force Ollama to load the model now, off the push-to-talk hot path.
        Uses load_persona() (not the raw default) so a broken/missing persona.txt surfaces
        in the startup log, not on the user's first spoken query.

        This matters for more than latency on this box. Ollama chooses CPU-vs-GPU placement at LOAD
        time, from whatever memory is free at that moment, and OLLAMA_KEEP_ALIVE pins the choice for
        the rest of the run. Warming here — while memory is still fresh — is what gets the model onto
        the GPU. If it fails, the first spoken query loads the model later under whatever pressure
        exists by then, and can be stuck on CPU (measured: 9.6 tok/s on CPU vs 19.2 on GPU)."""
        try:
            data = _ollama_request(build_chat_messages(load_persona(), [], "hello"))
        except RuntimeError as exc:
            # Logged rather than swallowed: a real request would surface the same error to the user
            # eventually, but silence also hides the CPU-vs-GPU consequence described above.
            print(f"[voice_assistant] WARNING: LLM warm-up failed ({exc}) — the first reply will pay "
                  f"the model load, and may end up on CPU instead of GPU", flush=True)
            return
        _log_llm_timings(data, label="warmup")
        # Now that the placement is decided and pinned, say out loud which way it went. This is the
        # whole point of warming here rather than on the first query, and until now the outcome was
        # invisible — a partial offload just showed up later as "Kai is slow today".
        log_model_placement()

    # ── Shared capture + turn epoch ─────────────────────────────────────────

    def attach_mic(self, mic) -> None:
        """Hand over an always-open capture stream to record from, instead of opening one per turn.

        `mic` needs arm_utterance(preroll: bool) -> bool and harvest_utterance() -> (audio, rate).
        The raw I2S hw device admits exactly one opener, so once a session owns the stream this class
        must never open its own — see start_recording()."""
        with self._lock:
            self._mic = mic

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def bump_epoch(self) -> int:
        """Invalidate every result still in flight and return the new epoch."""
        with self._lock:
            self._epoch += 1
            return self._epoch

    def _epoch_ok(self, epoch: int | None) -> bool:
        """True if `epoch` is still current. None means "unversioned" (legacy push-to-talk, say())
        and always passes, so the old call paths behave exactly as before."""
        if epoch is None:
            return True
        with self._lock:
            return self._epoch == epoch

    def speech_in_flight(self) -> bool:
        """True from just before synthesis begins until playback has finished or been cut.

        This is the "is Kai still saying something" signal. tts.is_playing() alone is not it: there is
        a 0.5-1.5 s Piper run before any playback process exists, and a caller that watched only
        is_playing() would conclude the reply was already over and move on."""
        with self._lock:
            return self._tts_active

    def last_language(self) -> str:
        """The language Whisper labelled the most recent transcription, or "" before the first one.

        Exists for the filler bank (ai/filler.pick_lang), which has to choose a language BEFORE the
        current turn's transcript is necessarily ready — the opener fires ~0.5 s into a turn, and
        STT may still be running. So it deliberately reports the PREVIOUS utterance's language: in a
        conversation that is almost always the same one, and being one turn stale is far cheaper
        than either blocking on STT or defaulting every opener to Tagalog.

        Note this can only ever be a member of config/voice.WHISPER_LANGUAGES ("en", "tl") — the
        detector has no label outside that set, which is why pick_lang reaches the Bisaya bank by
        its own route rather than by waiting for a "ceb" that never arrives."""
        with self._lock:
            return self._last_language

    def mic_muted(self, now: float | None = None) -> bool:
        """True while Kai's own audio could reach the mic. The audio thread drops blocks on this.

        Deliberately NOT derived from voice_speaking: that is the jaw envelope, which can be a silent
        text-timed pantomime, and which is only set AFTER synthesis finishes — so it says nothing
        about the Piper run that precedes playback. _tts_active covers exactly that gap."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._tts_active or now < self._gate_until:
                return True
        return tts.is_playing() or tts.quiet_since(now) < TTS_TAIL_MUTE_S

    # ── Public API ──────────────────────────────────────────────────────────

    def start_recording(self) -> dict:
        with self._lock:
            if self._status not in (STATUS_IDLE, STATUS_DONE, STATUS_ERROR):
                return {"error": f"cannot start recording while {self._status}"}
            self._audio_chunks = []
            self._transcript = ""
            self._response   = ""
            self._error      = ""
            self._status     = STATUS_RECORDING
            self._audio_samples = 0
            self._speak_start = None   # stop the jaw if Kai was still 'speaking' a previous reply
            mic = self._mic
        tts.stop()                     # and cut off its audio so it doesn't bleed into the recording

        if mic is not None:
            # A session owns the stream. Opening a second one on the same raw hw device is the
            # failure this branch exists to prevent — it surfaces as intermittent silent capture.
            if mic.arm_utterance(preroll=True):
                return {"status": "ok"}
            with self._lock:
                self._status = STATUS_ERROR
                self._error  = "Could not arm the microphone"
            return {"error": self._error}

        self.ensure_input_resolved()
        with self._lock:
            device, capture_rate = self._capture_device, self._capture_rate
            channels, dtype = self._capture_channels, self._capture_dtype
            is_i2s, card = self._capture_is_i2s, self._capture_card
        if card:
            # Re-assert for ANY raw hw device, not just the I2S one: pulse re-grabbing the card
            # between resolve and open is exactly what this guards, and it does that to a USB card
            # just as readily. `is_i2s` is left bound because the rest of the method still asks it.
            free_i2s_device()
        try:
            stream = sd.InputStream(
                samplerate=capture_rate, channels=channels, dtype=dtype,
                device=device, callback=self._on_audio_chunk,
            )
            stream.start()
        except Exception as exc:
            with self._lock:
                self._status = STATUS_ERROR
                self._error  = f"Could not open microphone: {exc}"
            return {"error": self._error}
        with self._lock:
            self._stream = stream
        return {"status": "ok"}

    def stop_recording(self) -> dict:
        with self._lock:
            if self._status != STATUS_RECORDING:
                return {"error": f"cannot stop recording while {self._status}"}
            stream = self._stream
            self._stream = None
            mic = self._mic
            self._status = STATUS_TRANSCRIBING

        if mic is not None:
            audio, rate = mic.harvest_utterance()
        else:
            if stream is not None:
                stream.stop()
                stream.close()
            with self._lock:
                chunks = self._audio_chunks
                self._audio_chunks = []
                self._audio_samples = 0
                rate = self._capture_rate
            audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, CHANNELS), dtype="int16")

        threading.Thread(target=self._process, args=(audio,), kwargs={"rate": rate},
                         daemon=True).start()
        return {"status": "ok"}

    def process_utterance(self, audio: np.ndarray, rate: int, epoch: int | None = None,
                          on_done=None) -> dict:
        """Run one already-captured utterance through STT -> LLM -> speech on a worker thread.

        The hands-free entry point, and the same code path stop_recording() takes — so a
        wake-word turn and a button turn cannot drift apart. `on_done(epoch, outcome)` fires when the
        worker finishes, with outcome one of "done" / "empty" / "error", which is how the session
        learns a turn ended without polling status."""
        with self._lock:
            if self._status not in (STATUS_IDLE, STATUS_DONE, STATUS_ERROR, STATUS_RECORDING):
                return {"error": f"busy: {self._status}"}
            self._transcript = ""
            self._response   = ""
            self._error      = ""
            self._status     = STATUS_TRANSCRIBING
        threading.Thread(target=self._process, args=(audio,),
                         kwargs={"rate": rate, "epoch": epoch, "on_done": on_done},
                         daemon=True, name="kai-turn").start()
        return {"status": "ok"}

    def get_status(self) -> dict:
        now = time.monotonic()
        with self._lock:
            speaking = speaking_openness_at(now, self._speak_start, self._speak_segments) is not None
            return {
                "voice_status":     self._status,
                "voice_transcript": self._transcript,
                "voice_response":   self._response,
                "voice_error":      self._error,
                "voice_speaking":   speaking,
                "voice_turn_id":    self._turn_id,
            }

    def speaking_openness(self, now: float | None = None) -> float | None:
        """Jaw openness 0..SPEAK_AMP while Kai is 'speaking' its reply, else None.
        Called each frame by face_track.py to animate the jaw servo."""
        now = time.monotonic() if now is None else now
        with self._lock:
            start, segments = self._speak_start, self._speak_segments
        return speaking_openness_at(now, start, segments)

    def clear_turn_status(self) -> None:
        """Retire a finished turn's status back to IDLE, keeping its transcript/response for display.

        Needed because the dashboard appends a chat bubble on the *transition into* "done"
        (web/frontend/dashboard.html). A hands-free session projects "recording" while listening, so
        when it ends and the projection stops overriding, voice_status falls back to this stale "done"
        — the dashboard reads that as a brand-new completed turn and posts the same question and answer
        a second time. Clearing the status removes the false edge; the bubbles already shown stay."""
        with self._lock:
            if self._status in (STATUS_DONE, STATUS_ERROR):
                self._status = STATUS_IDLE

    def reset_history(self) -> None:
        """Forget the conversation. Bumps the epoch too, so a reply already in flight can't append
        itself afterwards and hand the next person one turn of the last person's conversation."""
        with self._lock:
            self._history = []
            self._person_name = None
            self._epoch += 1
        # The retrieval side keeps its own one-line memory (the sticky DEVCON topic, see
        # config/rag.py STICKY_TURNS). It has to be forgotten with the history, or the next
        # person's first question inherits the last person's subject.
        rag.reset_topic()

    @property
    def person_name(self) -> str | None:
        """Who Kai currently believes it is talking to, for /params. None until they say so."""
        with self._lock:
            return self._person_name

    def note_identity(self, text: str, epoch: int | None = None) -> str | None:
        """Pin a name if `text` offers one. Returns the name in force after the call.

        Called BEFORE the LLM, so a name lands on the turn that offered it — "I'm Jhondel" gets
        answered by name rather than a turn late, which is the whole difference between this reading
        as recognition and reading as a delay.

        Epoch-guarded for the same reason the history append is: a session that ended while STT was
        running must not have its name applied to the next person's conversation. First name offered
        wins — a later "I'm just looking" cannot overwrite it, and re-extracting every turn would
        make the system prompt vary, which is exactly what IDENTITY_PROMPT's placement avoids."""
        if not IDENTITY_CAPTURE:
            return None
        if not self._epoch_ok(epoch):
            return None
        with self._lock:
            if self._person_name is not None:
                return self._person_name
        name = extract_name(text)
        if name is None:
            return None
        with self._lock:
            # Re-check under the lock: reset_history() may have run since _epoch_ok passed.
            if self._epoch_ok_locked(epoch) and self._person_name is None:
                self._person_name = name
                print(f"[identity] talking to {name!r}", flush=True)
            return self._person_name

    def _epoch_ok_locked(self, epoch: int | None) -> bool:
        """_epoch_ok for callers already holding the lock — self._lock is a plain Lock, not an
        RLock, so calling _epoch_ok() from inside the critical section would deadlock."""
        return epoch is None or self._epoch == epoch

    def say(self, text: str, use_llm: bool = True, epoch: int | None = None,
            on_done=None) -> dict:
        """Trigger a spoken reply (jaw animation) from text, with NO microphone involved.
        If use_llm (default) `text` is sent to Ollama as a prompt and the model's reply
        drives the mouth; otherwise `text` is spoken verbatim. Sets the same speak window
        face_track reads each frame, so the jaw animates identically to a voice turn — but
        completely decoupled from the audio-capture path (which can abort the process; see
        start_recording). The LLM call runs on a background thread so the caller returns fast.

        `epoch`/`on_done` mirror process_utterance and are what make a one-breath hands-free turn
        possible: the whisper wake tier already holds the transcript, so running the turn from text
        avoids transcribing the same audio twice. Both default to None → exactly today's behaviour,
        so the /voice/say route is unaffected."""
        text = (text or "").strip()
        if not text:
            return {"error": "no text provided"}
        with self._lock:
            if epoch is not None and self._epoch != epoch:
                return {"error": "stale"}
            if self._status not in (STATUS_IDLE, STATUS_DONE, STATUS_ERROR):
                return {"error": f"busy: {self._status}"}
            self._transcript = text if use_llm else ""
            self._response   = ""
            self._error      = ""
            self._status     = STATUS_THINKING if use_llm else STATUS_DONE

        def _run() -> None:
            outcome = "error"
            try:
                # The one-breath hands-free turn arrives here, not in _process() — the whisper wake
                # tier already holds the transcript, so "Hey Kai, my name is Jhondel" said without a
                # pause never touches the mic-turn path. Capturing in only one of the two was a real
                # gap: the same sentence pinned a name or did not, purely on whether the speaker drew
                # breath. use_llm=False is the verbatim /voice/say route, which is Kai reading a line
                # out, not somebody introducing themselves.
                if use_llm:
                    self.note_identity(text, epoch=epoch)
                reply = self._call_ollama(text) if use_llm else text
                if not self._epoch_ok(epoch):
                    # A cold model load takes ~50 s; the session can easily have ended meanwhile.
                    outcome = "stale"
                    return
                with self._lock:
                    if use_llm:
                        self._history.append({"role": "user", "content": text})
                        self._history.append({"role": "assistant", "content": reply})
                        self._history = self._history[-(MAX_HISTORY_TURNS * 2):]
                    self._response = reply
                    self._status   = STATUS_DONE
                    self._turn_id += 1
                outcome = "done"
                self._speak(reply, epoch=epoch)
            except Exception as exc:
                if self._epoch_ok(epoch):
                    with self._lock:
                        self._status = STATUS_ERROR
                        self._error  = str(exc)
                else:
                    outcome = "stale"
            finally:
                if outcome == "stale":
                    # Releasing the status is what makes a stale turn survivable. say() refuses
                    # while _status is mid-flight, so a turn abandoned here would leave THINKING
                    # latched forever: every later reply is rejected with "busy: thinking", and Kai
                    # degrades to ack-only — replies synthesized, never spoken, nothing logged.
                    # Only THINKING is cleared, so a newer turn that has already moved the status
                    # on is left strictly alone.
                    with self._lock:
                        if self._status == STATUS_THINKING:
                            self._status = STATUS_IDLE
                if on_done is not None:
                    try:
                        on_done(epoch, outcome)
                    except Exception as exc:
                        print(f"[voice_assistant] WARNING: say callback failed ({exc})")

        if use_llm:
            threading.Thread(target=_run, daemon=True).start()
        else:
            _run()   # verbatim: no network call, set the speak window immediately
        return {"status": "ok"}

    # ── Internal ────────────────────────────────────────────────────────────

    def _on_audio_chunk(self, indata, frames, time_info, status) -> None:
        # Keep only the live channel so everything downstream stays mono (N, 1): for the stereo
        # INMP441 that's the left slot; for a mono USB/default mic the slice is the whole buffer.
        # .copy() is required — PortAudio reuses indata's buffer after the callback returns.
        col = self._capture_channel
        chunk = indata[:, col:col + 1].copy()
        with self._lock:
            self._audio_chunks.append(chunk)
            self._audio_samples += len(chunk)
            # Nothing auto-stops a push-to-talk recording, so a stuck button or a dashboard tab that
            # went away grows this list until the process dies. Drop the oldest audio past the cap.
            cap = int(CAPTURE_HARD_CAP_S * max(1, self._capture_rate))
            while self._audio_samples > cap and len(self._audio_chunks) > 1:
                self._audio_samples -= len(self._audio_chunks.pop(0))

    def _speak(self, reply: str, epoch: int | None = None, turn_t0: float | None = None) -> None:
        """Drive the jaw 'speak window' and, when TTS is available, play `reply` aloud through the
        speaker. Non-blocking: synthesis + playback run on a daemon thread so no caller is held for
        the audio's length (notably the Flask verbatim say() path, which runs on the request thread).
        The jaw window is armed by _arm_jaw_for_wav() right before playback, which is what makes the
        mouth and the sound line up — see there for the two offsets that takes. Any TTS failure
        (disabled, engine/model missing, bad synth) falls back to the text-timed pantomime so the
        mouth still moves. MUST be called OUTSIDE self._lock — it acquires the lock itself.

        NOTE: `reply` (with emoji) is what the UI shows; the SPOKEN text has emoji/symbols stripped
        (tts.clean_for_speech) and is then delivery-shaped (ai/delivery.shape — breaths, an occasional
        opener) so it is not read out flat. The UI text is never shaped; the jaw is timed to the
        spoken text, which is why the shaping happens before the segments are built.

        `epoch` versions the whole worker. Synthesis takes 0.5-1.5 s and playback can take many
        seconds, so a session can end mid-flight; without the checks below this reply would be spoken
        into whatever came next. None keeps the old unversioned behaviour.

        `turn_t0` is the monotonic time the turn began, passed only from _process. It exists purely
        to record first_audio_ms — how long the person waited between finishing their sentence and
        hearing anything back, which is the number that actually describes "Kai feels slow". None
        (the say()/canned paths, which have no turn behind them) just skips the measurement."""
        # Clamp only what gets SPOKEN — the dashboard still shows the whole reply. Kai is deaf while
        # talking (barge-in is off), so an over-long reply is an over-long deaf spell.
        # Shape BEFORE the clamp so the added commas/opener count against the spoken-character
        # budget rather than sneaking past it, and so the clamp still cuts at a sentence end.
        spoken = tts.clamp_for_speech(
            delivery.shape(tts.clean_for_speech(reply)), TTS_MAX_SPOKEN_CHARS)
        jaw_text = spoken or reply           # if a reply is emoji-only, still mime to the original

        def _pantomime() -> None:
            with self._lock:
                self._speak_start, self._speak_segments = _speak_segments(jaw_text, time.monotonic())

        if not tts.enabled():
            _pantomime()
            return

        def _worker() -> None:
            tts.stop()                       # cut off any previous reply still playing
            try:
                synth_t0 = time.monotonic()
                wav = tts.synthesize(spoken, length_scale=delivery.length_scale(spoken))
                if turn_t0 is not None:
                    with self._lock:
                        self._stage_ms["tts_synth_ms"] = int((time.monotonic() - synth_t0) * 1000)
                if not self._epoch_ok(epoch):
                    # Silent by design, but indistinguishable in the log from a working speaker —
                    # say so, or a stale epoch reads as "the speaker is broken".
                    print(f"[tts] reply dropped after synthesis: epoch {epoch} is stale "
                          f"(now {self.epoch})", flush=True)
                    return                   # abandoned while Piper ran — say nothing at all
                if wav is None:              # synth failed — animate the jaw anyway, just silently
                    _pantomime()
                    return
                duration = tts.wav_duration(wav)
                if not self._epoch_ok(epoch):
                    print(f"[tts] reply dropped before playback: epoch {epoch} is stale "
                          f"(now {self.epoch})", flush=True)
                    return
                self._arm_jaw_for_wav(wav, jaw_text, duration)
                if turn_t0 is not None:
                    with self._lock:
                        self._stage_ms["first_audio_ms"] = int((time.monotonic() - turn_t0) * 1000)
                if turn_t0 is not None:
                    t = self.stage_timings()
                    # One line with the whole turn broken out, so the dead air is attributable at a
                    # glance instead of being one undifferentiated "llm_ms" number.
                    print(f"[turn] {t['first_audio_ms']}ms to first audio = "
                          f"stt {t['stt_ms']} + rag {t['rag_ms']} + llm {t['llm_ms']} "
                          f"(prompt {t['llm_prompt_ms']}, gen {t['llm_gen_ms']}) "
                          f"+ synth {t['tts_synth_ms']}", flush=True)
                tts.play(wav)                # blocks this worker thread only, until audio ends/cut
            finally:
                self._end_speech(gen)

        # Claimed BEFORE the worker starts, so the gate covers the synthesis window too — that's the
        # stretch voice_speaking can't see, since the jaw window only opens after synth returns.
        gen = self._begin_speech()
        threading.Thread(target=_worker, daemon=True, name="kai-tts").start()

    def _arm_jaw_for_wav(self, wav, jaw_text: str, duration: float) -> None:
        """Point the jaw at the SOUND in `wav`, and hold the mic shut until that sound has stopped.

        Called immediately before tts.play() on every path that plays a file — the reply, the ack,
        the canned failures, every filler line — so all of them line up the same way. MUST be called
        OUTSIDE self._lock. `duration` is the file length the caller already read.

        Two offsets, both measured on the robot 2026-08-12 and both documented in config/voice.py
        (TTS_PLAYBACK_LEAD_S and SPEAK_TRIM_SILENCE). Both used to be zero, and their sum is why
        "Hey Kai" was answered by a jaw that moved before the "Yes?" did:

          * play() does not make sound; it spawns paplay, and the first sample is not audible for
            another ~0.2 s while the process connects and the buffer fills. So the schedule starts
            at now + TTS_PLAYBACK_LEAD_S rather than now. speaking_openness_at() already returns
            None ahead of `start`, so the jaw simply stays shut through that window.
          * The FILE is longer than the SPEECH — Piper pads silence at both ends. So the schedule
            covers the audible span within the file, not the whole file.

        The mic gate is deliberately NOT trimmed the same way: what it protects against is Kai
        hearing itself, and the speaker is busy for the whole file (plus the lead, which is exactly
        the reason it now sums the two — the gate used to open a fifth of a second early)."""
        now = time.monotonic()
        audio_starts = now + TTS_PLAYBACK_LEAD_S
        speech_from, speech_to = 0.0, duration
        if SPEAK_TRIM_SILENCE and duration > 0:
            lo, hi = tts.wav_speech_span(wav)
            if hi > lo:                  # a silent or unreadable scan falls back to the whole file
                speech_from, speech_to = lo, hi
        with self._lock:
            if duration > 0:
                self._speak_start, self._speak_segments = _speak_segments_for_duration(
                    jaw_text, audio_starts + speech_from, speech_to - speech_from)
                self._audio_ends_at = audio_starts + duration
                self._gate_until = self._audio_ends_at + TTS_TAIL_MUTE_S
            else:                        # unknown length — best-effort text-timed window
                self._speak_start, self._speak_segments = _speak_segments(jaw_text, audio_starts)

    def _begin_speech(self) -> int:
        """Claim the speaker for a new utterance and return that utterance's generation token.

        Two jobs, both of which exist because ONE boolean and ONE paplay handle are shared by every
        speech path — reply, ack, canned and filler — whose workers finish out of order:

        1. Cut whatever is still playing. tts.play() does not replace the current process, it just
           starts a second one, so without this a filler that outlives its turn keeps talking
           straight over the answer it was covering. The session docstring always claimed the
           arriving reply cut the filler; nothing ever did, because the reply starts speaking inside
           the turn worker BEFORE on_done reaches _enter_speaking, so the session has no seam late
           enough to stop it and early enough to matter. Here is that seam.

        2. Hand back a token so only the NEWEST utterance may clear _tts_active. A worker that
           finishes late must not report silence on behalf of a line that is still going.

        3. Retire the outgoing line's JAW schedule and its expected audio end, both of which
           describe audio that (1) is about to kill. Nothing else clears them — the schedule is a
           plain (start, segments) pair that face_track.py reads every frame, and the only reset in
           the class is start_recording()'s, which covers push-to-talk and nothing else. So a
           filler cut mid-word by an arriving reply went on miming the rest of its sentence in
           silence, for up to FILLER_MAX_LINE_S, right through the 0.5-1.5 s Piper run before the
           reply had a window of its own. Clearing here rather than at each caller is what keeps
           the fifth speech path from re-introducing it.
        """
        with self._lock:
            self._speech_gen += 1
            self._tts_active = True
            self._speak_start, self._speak_segments = None, ()
            # Load-bearing for the session's deadline: _enter_speaking arms it from on_done, which
            # fires BEFORE this line's worker knows a duration. Leaving the previous line's (now
            # past) end time in place would make the session cut this one the instant it started.
            self._audio_ends_at = None
            gen = self._speech_gen
        # Outside the lock: stop() waits on a process, and the worker it cuts takes _lock in its
        # finally. Holding _lock across that is a deadlock.
        tts.stop()
        return gen

    def _end_speech(self, gen: int) -> None:
        """Release the speaker, but only if no newer utterance has claimed it since."""
        with self._lock:
            if gen == self._speech_gen:
                self._tts_active = False
                # The jaw schedule is deliberately NOT cleared here: it ran exactly as long as the
                # audio did, so it has already expired on its own, and speaking_openness_at()
                # returns None past its end. Only the end TIME is retired, so a stale one can never
                # be read as this reply's.
                self._audio_ends_at = None

    def audio_ends_at(self) -> float | None:
        """Monotonic time the audio now playing should finish, or None if that isn't known.

        None covers three real cases and they all mean the same thing to a caller: nothing is
        playing, the WAV's header wouldn't read, or this is a silent text-timed pantomime. It is
        also None for the whole synthesis window, since no length exists until Piper has run.

        ai/session.py's STATE_SPEAKING deadline is the consumer — see _speaking_deadline() there
        for why a fixed cap was not good enough."""
        with self._lock:
            return self._audio_ends_at

    def stop_speech(self) -> None:
        """Cut Kai off mid-word: kill the audio AND retire the jaw.

        tts.stop() alone only kills the subprocess. The jaw runs off a schedule that no longer has
        anything to do with whether sound is coming out, so every site that cuts audio has to
        retire that too or the mouth keeps mouthing a sentence nobody can hear. Paired here so the
        two cannot drift apart; ai/session.py calls this everywhere it used to call tts.stop()."""
        with self._lock:
            self._speak_start, self._speak_segments = None, ()
            self._audio_ends_at = None
        # Outside the lock, for the reason _begin_speech documents: stop() waits on a process whose
        # worker takes _lock in its finally.
        tts.stop()

    def speak_wav(self, wav: Path, jaw_text: str, epoch: int | None = None) -> None:
        """Speak an already-synthesized WAV (see tts.prewarm_canned) with the jaw synced to it.

        Used for the wake acknowledgement, where live synthesis would put 0.5-1.5 s of dead air
        between "Hey Kai" and "Yes?". Deliberately does NOT touch _status or _response: routing the
        ack through say() would make the dashboard post a "Kai: Yes?" chat bubble on every wake.

        Public because ai/session.py is its caller — every canned line, every filler line and the
        ack all arrive here. Anything spoken that is NOT a turn goes through this method or
        speak_text(), never through say()."""
        def _worker() -> None:
            try:
                duration = tts.wav_duration(wav)
                if not self._epoch_ok(epoch):
                    return
                self._arm_jaw_for_wav(wav, jaw_text, duration)
                tts.play(wav)
            finally:
                self._end_speech(gen)

        gen = self._begin_speech()
        threading.Thread(target=_worker, daemon=True, name="kai-tts-canned").start()

    def speak_text(self, text: str, epoch: int | None = None) -> None:
        """Say `text` aloud with the jaw synced to it, without touching turn status.

        The fallback sibling of speak_wav(), for when a canned line was never synthesized and
        ai/session.py has to fall back to live synthesis. Same contract: no _status, no _response,
        so no chat bubble. Do not use it for a reply — that is what say() and process_utterance()
        are for, and they are what the dashboard's transcript is built from."""
        self._speak(text, epoch=epoch)

    def transcribe_async(self, audio: np.ndarray, rate: int, on_done, token=None,
                         log_language: bool = True) -> None:
        """Transcribe on a worker thread and hand back the text. STT only.

        No LLM, no speech, and deliberately **no writes to _status/_transcript/_response**. The
        whisper wake tier needs a transcript and nothing else — writing turn state here would make
        the dashboard post a chat bubble for a sentence nobody addressed to Kai.

        Threading lives here, next to the model, so the session never spawns an STT thread and tests
        can fake this the way process_utterance is already faked. Calls
        `on_done(token, text, error)` exactly once, and never raises to the thread."""
        def _worker() -> None:
            text, error = "", ""
            try:
                text = self._transcribe(audio, rate=rate, log_language=log_language, scan=True)
            except Exception as exc:
                error = str(exc)
            try:
                on_done(token, text, error)
            except Exception as exc:   # a broken callback must not kill the thread silently
                print(f"[voice_assistant] WARNING: transcribe callback failed ({exc})")

        threading.Thread(target=_worker, daemon=True, name="kai-wakescan").start()

    def _process(self, audio: np.ndarray, rate: int | None = None, epoch: int | None = None,
                 on_done=None) -> None:
        """One turn: STT -> LLM -> speech. Runs on its own thread; never raises to the caller.

        `outcome` reported through on_done is "done" (a real reply), "empty" (nothing transcribed) or
        "error". A stale epoch reports "stale" and touches no shared state at all — the session has
        moved on, and writing _status/_response here would stomp on the turn that replaced this one."""
        outcome = "error"
        # Start of the turn as the person experiences it: everything from here to the first sample
        # out of the speaker is dead air. Threaded down to _speak so first_audio_ms measures the
        # whole pipeline (STT + RAG + LLM + synthesis) rather than any one stage of it.
        turn_t0 = time.monotonic()
        try:
            transcript = self._transcribe(audio, rate=rate)
            stt_ms = int((time.monotonic() - turn_t0) * 1000)
            if not self._epoch_ok(epoch):
                outcome = "stale"
                return
            with self._lock:
                # Committed only past the epoch check, like every other write here: a stale turn
                # must touch no shared state, and that includes the timings the dashboard reads.
                self._stage_ms["stt_ms"] = stt_ms
                self._transcript = transcript

            if not transcript.strip():
                outcome = "empty"
                with self._lock:
                    self._response = NO_SPEECH_RESPONSE
                    self._status   = STATUS_DONE
                    self._turn_id += 1
                # The session speaks its own cached line here, so it isn't voiced twice.
                if on_done is None:
                    self._speak(NO_SPEECH_RESPONSE, epoch=epoch)
                return

            with self._lock:
                self._status = STATUS_THINKING
            # Before the LLM call, so a name offered this turn is answered this turn.
            self.note_identity(transcript, epoch=epoch)
            reply = self._call_ollama(transcript)

            if not self._epoch_ok(epoch):
                # A cold model load takes ~50 s; plenty of time for the session to have ended. The
                # guard also stops this append from resurrecting a history that was just cleared.
                outcome = "stale"
                return
            with self._lock:
                self._history.append({"role": "user", "content": transcript})
                self._history.append({"role": "assistant", "content": reply})
                self._history = self._history[-(MAX_HISTORY_TURNS * 2):]
                self._response = reply
                self._status   = STATUS_DONE
                self._turn_id += 1
            outcome = "done"
            self._speak(reply, epoch=epoch, turn_t0=turn_t0)
        except Exception as exc:
            if self._epoch_ok(epoch):
                with self._lock:
                    self._status = STATUS_ERROR
                    self._error  = str(exc)
            else:
                outcome = "stale"
        finally:
            if on_done is not None:
                try:
                    on_done(epoch, outcome)
                except Exception as exc:   # a broken callback must not take the turn thread down
                    print(f"[voice_assistant] WARNING: turn callback failed ({exc})")

    def _transcribe(self, audio: np.ndarray, rate: int | None = None,
                    log_language: bool = True, scan: bool = False) -> str:
        """Transcribe int16 audio captured at `rate` (default: this instance's resolved capture rate,
        which is what the legacy per-turn stream path uses). The shared always-open stream already
        emits 16 kHz and passes rate explicitly, so the resample below is skipped there.

        `log_language=False` for the wake-phrase scan — otherwise the detected-language line prints
        for every overheard sentence in the room and buries everything useful in the log.

        `scan=True` uses the small fast spotting model instead of the turn model. Not an optimisation:
        at "small" a check costs ~3 s per 1.6 s of audio, which made the whole tier time out."""
        if scan:
            self.ensure_scan_model_loaded()
            model = self._scan_model or self._whisper_model
        else:
            self.ensure_model_loaded()
            model = self._whisper_model
        if audio.size == 0:
            return ""
        rate = self._capture_rate if rate is None else rate
        samples = audio.astype(np.float32).reshape(-1) / 32768.0
        if rate != SAMPLE_RATE:
            g = math.gcd(SAMPLE_RATE, rate)
            samples = resample_poly(samples, SAMPLE_RATE // g, rate // g)
        # Level, not noise: a talker across the room arrives several times quieter than the audio
        # every constant here was tuned against, and Whisper decodes quiet input worse for no reason
        # other than the level. See ASR_NORMALIZE in config/voice.py. AFTER the resample so the
        # measured RMS is of the 16 kHz signal the decoder actually sees.
        if ASR_NORMALIZE and (ASR_NORMALIZE_SCAN or not scan):
            samples, gain, level = normalize_for_asr(samples)
            with self._lock:
                self._norm_gain["scan" if scan else "turn"] = gain
                self._norm_rms["scan" if scan else "turn"] = level
            # Logged only on the turn path and only when it did something: the scan runs on every
            # nearby utterance, so logging it would bury the log the way log_language would.
            if log_language and gain > 1.05:
                print(f"[voice_assistant] input level {level:.4f} rms — normalised x{gain:.1f}"
                      + (" (at the gain ceiling — the speaker is too far or too quiet)"
                         if gain >= ASR_NORMALIZE_MAX_GAIN - 1e-6 else ""), flush=True)
        # vad_filter drops non-speech stretches — extra defense against Whisper
        # hallucinating filler text (e.g. "You") on silence/near-silence audio.
        # The scan pins the language: auto-detect on a weak model and a 1-3 s clip produces confident
        # nonsense in random languages (see WAKE_WHISPER_SCAN_LANGUAGE).
        language = WAKE_WHISPER_SCAN_LANGUAGE if scan else WHISPER_LANGUAGE
        # Turn transcribes only — see WHISPER_INITIAL_PROMPT for why the wake scan stays unbiased.
        prompt = None if scan else (WHISPER_INITIAL_PROMPT or None)
        # list(), not a generator: the sanity gate below needs each segment's avg_logprob and
        # no_speech_prob, and joining the text would have already consumed them.
        segments, info = model.transcribe(samples, language=language, vad_filter=True,
                                          beam_size=WHISPER_BEAM_SIZE, initial_prompt=prompt)
        segs = list(segments)
        text = " ".join(seg.text.strip() for seg in segs).strip()

        # Auto-detect wandered outside the languages Kai is actually spoken to. Redo the pass, forced
        # to the most likely allowed language — the first result was decoded as e.g. Welsh and is not
        # worth keeping. Reusing the first pass when it lands on an allowed language is what keeps
        # this free in the normal case.
        if language is None and WHISPER_LANGUAGES and info is not None:
            allowed = tuple(WHISPER_LANGUAGES)
            if info.language not in allowed:
                forced = _best_allowed_language(info, allowed)
                if log_language:
                    print(f"[voice_assistant] detected '{info.language}' "
                          f"({info.language_probability:.2f}) — not in {allowed}; "
                          f"re-transcribing as '{forced}'")
                segments, info = model.transcribe(samples, language=forced, vad_filter=True,
                                                  beam_size=WHISPER_BEAM_SIZE,
                                                  initial_prompt=prompt)
                segs = list(segments)
                text = " ".join(seg.text.strip() for seg in segs).strip()

        # The label said an allowed language; that is not the same as the OUTPUT being one. Checked
        # AFTER any forced re-transcribe, so a rejection means both passes failed, not just the first.
        # Discarded rather than repaired: unintelligible audio does not become intelligible on a
        # second look, and "Sorry, I didn't catch that" is the honest answer. Feeding it onward means
        # Kai confidently answers a question nobody asked.
        reason = transcript_rejection(text, segs)
        if reason:
            # Always logged, even with log_language off (the wake scan): a discard that leaves no
            # trace is indistinguishable from a broken mic, which is exactly the class of bug that
            # took a hardware investigation to find on 2026-08-07.
            print(f"[voice_assistant] discarded transcript — {reason}: {text[:60]!r}", flush=True)
            return ""

        # Recorded only for a transcript that SURVIVED the rejection check above — a discarded
        # transcript's language label came from audio Kai decided was unintelligible, and letting
        # that steer the next turn's filler would propagate the noise.
        if info is not None and not scan and getattr(info, "language", None):
            with self._lock:
                self._last_language = info.language

        if log_language and language is None and info is not None:
            # Guarded: this is only a log line, and letting it raise would turn a perfectly good
            # transcript into a failed turn inside _process's except.
            print(f"[voice_assistant] detected language: {info.language} ({info.language_probability:.2f})")
        return text

    def _call_ollama(self, text: str) -> str:
        with self._lock:
            history = list(self._history)
            person_name = self._person_name
        persona = load_persona()
        # The one fact about the SPEAKER the prompt carries. System position is deliberate and is
        # the opposite call to the RAG context below: this string is constant for the rest of the
        # session, so it invalidates Ollama's cached prefix once, at the turn it is learned, rather
        # than on every turn. See IDENTITY_PROMPT in config/voice.py.
        if person_name:
            persona = f"{persona}\n\n{IDENTITY_PROMPT.format(name=person_name)}"
        # The previous user turn, so a follow-up that only says "it" still retrieves its subject.
        previous_user_text = next((m["content"] for m in reversed(history) if m["role"] == "user"),
                                  None)
        rag_t0 = time.monotonic()
        context = rag.retrieve_context(text, previous_user_text=previous_user_text)
        rag_ms = int((time.monotonic() - rag_t0) * 1000)

        # WHERE the context goes decides whether Ollama can reuse its KV cache between turns. It
        # caches the longest common PREFIX of the prompt, and the retrieved context changes every
        # turn — so putting it in the system message (the original behaviour, kept as the "system"
        # placement) parks a per-turn-varying block at the very front and forces the persona AND all
        # MAX_HISTORY_TURNS of history to be re-evaluated on every single turn. Prepending it to the
        # user turn instead leaves that whole prefix byte-identical, so only the new context and
        # question are evaluated. See RAG_CONTEXT_PLACEMENT in config/voice.py for the revert.
        #
        # Safe because self._history stores the RAW transcript for user turns (see _process), never
        # the injected context — so what got prepended this turn does not perturb the prefix next
        # turn, which is the property the whole optimisation rests on.
        if context and RAG_CONTEXT_PLACEMENT == "system":
            system_prompt, user_content = f"{persona}\n\n{context}", text
        elif context:
            system_prompt, user_content = persona, f"{context}\n\n{text}"
        else:
            system_prompt, user_content = persona, text

        messages = build_chat_messages(system_prompt, history, user_content)
        llm_t0 = time.monotonic()
        data = _ollama_request(messages)
        llm_ms = int((time.monotonic() - llm_t0) * 1000)
        timings = _log_llm_timings(data, label="turn")
        with self._lock:
            self._stage_ms.update(timings)
            self._stage_ms["rag_ms"] = rag_ms
            self._stage_ms["llm_ms"] = llm_ms
        return data["message"]["content"]
