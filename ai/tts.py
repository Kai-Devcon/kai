"""
Text-to-speech for Kai: synthesize a reply with Piper and play it through the USB speaker.

Deliberately thin — everything is a subprocess (Piper CLI + paplay), so importing this module
never pulls in a heavy runtime and a missing/broken engine degrades gracefully (synthesize()
returns None, and ai/voice_assistant.py falls back to the silent jaw pantomime). All tunables live
in config/voice.py (TTS_*). Not thread-managed beyond a single "current playback" handle: at most
one reply plays at a time, and stop() cuts it off when a new turn starts.

stop() cancels BOTH stages — the Piper synth subprocess and the paplay playback. That matters for
hands-free (ai/session.py): a turn can be abandoned while its synth is still running, and without
cancelling it the stale worker would go on to play a reply belonging to a session that has already
ended. is_playing()/quiet_since() expose when Kai's own audio could still be reaching the mic, which
is what the session's self-hearing gate is built on.
"""

from __future__ import annotations

import array
import os
import re
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

from config.voice import (
    SPEAK_SILENCE_PEAK, SPEAK_SILENCE_STEP_S,
    TTS_ENGINE, TTS_PIPER_CMD, TTS_VOICE_MODEL,
    TTS_SINK, TTS_OUTPUT_DIR, TTS_XDG_RUNTIME, TTS_LATENCY_MSEC,
    TTS_PIPER_NORMALIZE, TTS_PIPER_TIMEOUT_S,
    TTS_POST_PROCESS, TTS_POST_SOX, TTS_POST_CHANNELS, TTS_POST_EFFECTS,
    TTS_POST_HIGHPASS, TTS_POST_ROOM,
    TTS_ASSERT_CARD_PROFILE, TTS_CARD, TTS_CARD_PROFILE, TTS_PACTL_TIMEOUT_S,
)
# TTS_ENABLED / TTS_VOLUME / TTS_LENGTH_SCALE deliberately NOT imported: they are dashboard-settable,
# so they are read live via settings.get() at each use. settings.py takes its defaults from them.
import settings

# Project root (…/kai) — TTS_VOICE_MODEL is stored relative to it so the path is stable under
# scripts/run.sh and the @reboot autostart regardless of the process's cwd.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def voice_model_path() -> Path:
    """Absolute path to the configured voice .onnx (TTS_VOICE_MODEL, resolved against the project
    root when relative)."""
    p = Path(TTS_VOICE_MODEL)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def enabled() -> bool:
    """True only if TTS is on, the engine is one we support, and the voice model is actually present
    — so callers can cheaply decide between real speech and the silent pantomime without catching
    exceptions.

    The on/off flag is read live from settings (dashboard-settable); TTS_ENABLED in config/voice.py is
    the default it starts from. Turning it off falls back to the existing silent jaw pantomime.
    """
    return (bool(settings.get("tts_enabled")) and TTS_ENGINE == "piper"
            and voice_model_path().is_file())


# Emoji & decorative-symbol ranges to strip before synthesis — espeak-ng (Piper's phonemizer) would
# otherwise voice them literally (e.g. "😀" -> "grinning face"). Deliberately excludes General
# Punctuation (U+2000–206F) so smart quotes, dashes and the ellipsis stay speakable.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # emoji planes: pictographs, emoticons, transport, supplemental, cards
    "\U00002600-\U000026FF"   # miscellaneous symbols   ☀ ★ ☎ ⚡ ♻
    "\U00002700-\U000027BF"   # dingbats                ✂ ✅ ✈ ❤ ✨
    "\U00002B00-\U00002BFF"   # misc symbols & arrows   ⭐ ⬅ ⬆
    "\U00002300-\U000023FF"   # misc technical          ⌚ ⌛ ⏰ ⏳ ▶
    "\U00002190-\U000021FF"   # arrows                  ← → ↔
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0000200D"              # zero-width joiner (emoji sequences)
    "]+"
)


# Markdown emphasis, which gemma still emits now and then despite persona.txt asking for plain
# conversational text. Piper's phonemizer voices these as words rather than skipping them —
# measured on en_US-hfc_female-medium: "hello *world*" runs 1.11s longer than "hello world",
# "**world**" 1.90s longer, and a real reply ("* I don't know that. *") 1.50s longer. That is
# both nonsense to listen to and dead air Kai cannot hear through, since barge-in is off.
# Underscores measured silent (-0.08s, i.e. noise), so they are left alone.
_MARKDOWN_RE = re.compile(r"[*`]+")

# Bullet and heading markers, stripped only at the start of a line so hyphenated words
# ("push-to-talk") and mid-sentence dashes survive.
_LIST_MARKER_RE = re.compile(r"^[ \t]*(?:[-–—•]|#{1,6})[ \t]+", re.MULTILINE)


def clean_for_speech(text: str) -> str:
    """Strip emoji/decorative symbols and markdown so they aren't read aloud, and collapse the
    whitespace left behind. Only the SPOKEN text is cleaned — callers keep the original
    (emoji-bearing) text for the UI. Returns '' if nothing speakable remains."""
    cleaned = _EMOJI_RE.sub("", text or "")
    cleaned = _LIST_MARKER_RE.sub("", cleaned)   # before the whitespace collapse: needs line starts
    cleaned = _MARKDOWN_RE.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def clamp_for_speech(text: str, max_chars: int) -> str:
    """Trim `text` to at most `max_chars`, preferring to stop at a sentence end.

    Only the SPOKEN text is clamped — the UI keeps the whole reply. This exists because Kai cannot
    hear anything while he is talking (no echo cancellation, so voice barge-in is off), which makes a
    runaway reply a proportionally long stretch of deafness. Non-positive max_chars disables it."""
    text = text or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = text[:max_chars]
    cut = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if cut >= max_chars // 2:          # a sentence break we can use without losing most of the reply
        return head[:cut + 1]
    cut = head.rfind(" ")              # otherwise at least don't cut mid-word
    return (head[:cut] if cut > 0 else head).rstrip() + "…"


# One retry when playback fails outright (see play()). Small enough not to delay a turn noticeably,
# and the only thing standing between a PulseAudio hiccup and a reply nobody ever hears.
_RETRY_ON_FAILURE = True
_RETRY_DELAY_S = 0.4

# synthesize_to_duration() fitting bounds. The tolerance is what stops a needless second Piper run
# for a line that already lands close enough; the scale clamps keep the correction inside the range
# where Piper still sounds like speech — well past 3x a short interjection becomes a drawn-out smear,
# and the phonemizer gets unstable at the extremes.
_DURATION_FIT_TOLERANCE_S = 0.05
_DURATION_FIT_PASSES = 3
_MIN_LENGTH_SCALE = 0.5
_MAX_LENGTH_SCALE = 4.0

_OUTPUT_WAV = Path(TTS_OUTPUT_DIR) / "kai_tts.wav"
# Piper writes here first; _post_process() reads it and produces the louder/stereo _OUTPUT_WAV.
# Kept separate so a failed post-process can fall back to this raw file untouched.
_RAW_WAV = Path(TTS_OUTPUT_DIR) / "kai_tts_raw.wav"


# ── In-flight subprocess handles ──────────────────────────────────────────────────────────────
# At most one synth and one playback at a time. _proc_lock guards both handles plus _last_end, so
# stop() — called from the mic/turn/session thread — can cancel work started on a speak worker
# thread. Cancelling the SYNTH matters as much as the playback: a turn abandoned mid-Piper would
# otherwise still reach play() afterwards and speak into a session that has already ended.
_proc_lock = threading.Lock()
_synth_proc: subprocess.Popen | None = None
_current_proc: subprocess.Popen | None = None
_last_end: float = 0.0   # time.monotonic() when playback last stopped; 0.0 = nothing has played yet

# Set once the card profile has been asserted for this process, so the pactl call is paid on the
# first reply rather than on every one. play()'s failure path re-asserts regardless of this flag:
# the flag means "we have asked once", not "the card is still where we put it".
_profile_applied = False


def apply_output_profile() -> bool:
    """Force the output card onto TTS_CARD_PROFILE, so TTS_SINK exists and audio leaves on the
    ANALOG jack.

    PulseAudio flips this dongle to its digital (S/PDIF) profile unprompted, which deletes the
    analog sink and makes every reply inaudible while raising nothing — see the TTS_ASSERT_CARD_PROFILE
    note in config/voice.py. Called once before the first playback and again before play()'s retry.

    Setting the profile a card is already on is a no-op that still exits 0, so this is safe to call
    repeatedly. Best-effort and never raises: a missing pactl or card logs and returns False, and
    playback proceeds exactly as it did before. Returns True only if the profile applied."""
    if not TTS_ASSERT_CARD_PROFILE:
        return False
    try:
        subprocess.run(
            ["pactl", "set-card-profile", TTS_CARD, TTS_CARD_PROFILE],
            check=True, capture_output=True, text=True, timeout=TTS_PACTL_TIMEOUT_S,
            # Same reason play() forces it: pactl needs this to find the Pulse socket under the
            # @reboot cron autostart / an SSH session, not just an interactive login.
            env={**os.environ, "XDG_RUNTIME_DIR": TTS_XDG_RUNTIME},
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"[tts] WARNING: could not set output card profile "
              f"({TTS_CARD} -> {TTS_CARD_PROFILE}: {exc}) — if the card is on its digital "
              f"(S/PDIF) profile, replies will be inaudible on the analog jack", flush=True)
        return False
    return True


def _sox_chain() -> list[str]:
    """The sox effects, in the one order that makes acoustic sense — and the order is the whole point,
    which is why this is a function and not a single config list:

      TTS_POST_HIGHPASS   before the compand. It removes energy this speaker cannot reproduce, so it
                          has to go first: cutting it afterwards would be after the compressor has
                          already spent its headroom reacting to it.
      TTS_POST_EFFECTS    the loudness chain (compand + `gain -n -1`), untouched.
      TTS_POST_ROOM       after the compand — compress the dry signal, then put it in a space, the
                          order a person mixing this would use.
      a final `gain -n -1` only when a room is configured, because reverb adds energy after the
      chain's own normalise and would otherwise leave the peak somewhere above -1 dB.

    Either addition being empty yields exactly the previous command line, so the whole feature reverts
    from config alone."""
    room = list(TTS_POST_ROOM)
    return [*TTS_POST_HIGHPASS, *TTS_POST_EFFECTS, *room,
            *(["gain", "-n", "-1"] if room else [])]


def _post_process(src: Path, dst: Path) -> Path:
    """Run sox to duplicate `src` (Piper's quiet mono WAV) to TTS_POST_CHANNELS and apply the effect
    chain from _sox_chain(), writing `dst`. Returns `dst` on success, else `src` unchanged
    (best-effort: a missing/broken sox must never cost us the reply — speech still plays, just
    quieter/mono/dry). No-op passthrough when TTS_POST_PROCESS is off.

    Note that fallback is louder-consequence than it used to be: TTS_PIPER_NORMALIZE = False means the
    raw Piper WAV is also ~10 dB quieter (see _run_piper)."""
    if not TTS_POST_PROCESS:
        return src
    cmd = [TTS_POST_SOX, str(src), "-c", str(TTS_POST_CHANNELS), str(dst), *_sox_chain()]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except OSError as exc:
        print(f"[tts] WARNING: sox post-process unavailable ({exc}) — using raw Piper audio")
        return src
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        detail = err[-1] if err else f"exit {proc.returncode}"
        print(f"[tts] WARNING: sox post-process failed ({detail}) — using raw Piper audio")
        return src
    return dst


def _run_piper(text: str, dst: Path, length_scale: float | None = None) -> bool:
    """Run Piper to write `text` as a raw WAV at `dst`. True on success, False on any failure
    (logged) or on deliberate cancellation by stop(). Text is fed on stdin; Piper's stderr (a benign
    onnxruntime GPU-discovery warning on the Jetson) is captured so it never clutters our logs.
    The Popen handle is published under _proc_lock so stop() can kill a synth mid-flight.

    `length_scale` overrides the dashboard's tts_length_scale for this one synthesis — used to
    stretch a single cached line to a target duration (see synthesize_to_duration). None keeps the
    live setting, which is what every ordinary reply uses."""
    global _synth_proc
    model = voice_model_path()
    if not model.is_file():
        print(f"[tts] WARNING: voice model not found ({model}) — skipping speech")
        return False
    cmd = [
        *TTS_PIPER_CMD,
        "-m", str(model),
        # Read live per synthesis, so a rate change from the dashboard applies to the very next thing
        # Kai says. config/voice.py supplies the default.
        # Volume is deliberately NOT set here — see play(). Synthesising at 1.0 keeps the raw audio
        # clip-free and gives the post-processing chain a consistent level to work from.
        "--length-scale", str(settings.get("tts_length_scale") if length_scale is None
                              else length_scale),
        # The other three prosody parameters, all live for the same reason. --sentence-silence is the
        # one that changes how Kai sounds: its default is 0, so before 2026-08-10 a four-sentence reply
        # ran together without a single breath. See config/voice.py for the measurements behind all
        # three, and for why the two noise values default to the voice's own.
        #
        # FLAG SPELLINGS VERIFIED against the pinned piper-tts 1.4.2 on this robot
        # (`python3 -m piper --help`, 2026-08-10): --sentence-silence, --noise-scale and
        # --noise-w-scale, the last also accepting --noise-w / --noise_w as aliases. Do not take a
        # spelling from a blog post or a different version — Piper rejects an unknown flag by exiting
        # non-zero, which _run_piper reports as a failed synthesis, which means EVERY REPLY IS SILENT.
        # That failure looks nothing like a flag problem in the log, so re-check --help after any
        # piper-tts upgrade.
        "--sentence-silence", str(settings.get("tts_sentence_silence")),
        "--noise-scale", str(settings.get("tts_noise_scale")),
        "--noise-w-scale", str(settings.get("tts_noise_w")),
        "-f", str(dst),
    ]
    # Only skip Piper's per-sentence normalisation when sox is going to normalise the whole reply
    # instead. Without that pairing the raw output is ~10 dB quieter and nothing puts the level back.
    if not TTS_PIPER_NORMALIZE and TTS_POST_PROCESS:
        cmd.insert(-2, "--no-normalize")
    try:
        with _proc_lock:
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            _synth_proc = proc
    except OSError as exc:
        print(f"[tts] WARNING: could not run Piper ({exc}) — skipping speech")
        return False
    try:
        _, stderr = proc.communicate(input=text.encode("utf-8"), timeout=TTS_PIPER_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        # No caller-side deadline reaches here — the background warm/rewarm threads
        # (ai/session.py's _warm_all/_prewarm_bank/_rewarm_when_quiet) call _run_piper with nothing
        # watching them. Without this, a wedged Piper blocks the calling thread and leaks its child
        # process forever (A5, 2026-09-17).
        proc.kill()
        proc.communicate()   # reap: drain the pipes so the killed child doesn't zombie
        print(f"[tts] WARNING: Piper synthesis timed out after {TTS_PIPER_TIMEOUT_S}s "
              f"— killed and skipping speech")
        return False
    except OSError as exc:   # e.g. broken pipe if the synth died as we fed it stdin
        print(f"[tts] WARNING: Piper synthesis failed ({exc}) — skipping speech")
        return False
    finally:
        with _proc_lock:
            if _synth_proc is proc:
                _synth_proc = None
    if proc.returncode is not None and proc.returncode < 0:
        return False   # terminated by stop(): a newer turn took over. Not an error, don't log.
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        err = (stderr or b"").decode("utf-8", "replace").strip().splitlines()
        detail = err[-1] if err else f"exit {proc.returncode}"
        print(f"[tts] WARNING: Piper synthesis failed ({detail}) — skipping speech")
        return False
    return True


def synthesize(text: str, length_scale: float | None = None) -> Path | None:
    """Synthesize `text` to a WAV with Piper, apply the loudness/stereo post-process, and return the
    path to play (or None on any failure, logged). Piper writes the raw WAV to
    TTS_OUTPUT_DIR/kai_tts_raw.wav, which _post_process() turns into TTS_OUTPUT_DIR/kai_tts.wav.

    Both paths are FIXED and shared by every reply, so the returned WAV is only valid until the next
    synthesize() call. Anything meant to be cached and replayed must use synthesize_to().

    `length_scale` overrides the dashboard's rate for this one reply — ai/voice_assistant passes the
    per-reply tempo jitter from ai/delivery.length_scale here. None keeps the live setting."""
    text = clean_for_speech(text)   # never let emoji/symbols reach the phonemizer
    if not text:
        return None
    if not _run_piper(text, _RAW_WAV, length_scale=length_scale):
        return None
    return _post_process(_RAW_WAV, _OUTPUT_WAV)


def synthesize_to(text: str, dest: Path, length_scale: float | None = None) -> Path | None:
    """Like synthesize(), but the result lands at `dest` and stays there — for lines that are
    synthesized once and replayed many times (see prewarm_canned). The raw Piper output goes to a
    sibling `*_raw.wav`; if the loudness post-process is off or fails, that raw file is moved onto
    `dest` so callers always get one stable path back. Returns None on failure (logged)."""
    text = clean_for_speech(text)
    if not text:
        return None
    dest = Path(dest)
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[tts] WARNING: could not create {dest.parent} ({exc}) — skipping speech")
        return None
    raw = dest.with_name(f"{dest.stem}_raw{dest.suffix}")
    if not _run_piper(text, raw, length_scale=length_scale):
        return None
    if _post_process(raw, dest) != dest:
        try:
            raw.replace(dest)   # post-process fell back to the raw file — still hand back `dest`
        except OSError as exc:
            print(f"[tts] WARNING: could not move raw synth to {dest} ({exc})")
            return raw
    return dest


def synthesize_to_duration(text: str, dest: Path, target_s: float) -> Path | None:
    """Synthesize `text` to `dest`, stretched (or compressed) to last about `target_s` seconds.

    Synthesize, measure, correct the length scale by target/actual, repeat. Piper's --length-scale is
    close to linear in output duration but not exactly — and the trailing "..." pause does not scale
    with the phonemes — so one correction lands near but not on the target. Two or three iterations
    converge to within the tolerance. MEASURED rather than a hardcoded scale on purpose: the same text
    runs to a different length in a different voice, and the canned lines are re-synthesized whenever
    the voice changes (session.reprewarm_canned), so a fixed number would quietly drift.

    Only ever called for the handful of cached lines at startup, so the extra Piper runs cost nothing
    a caller waits on. Keeps the best attempt so far if a later pass fails — a slightly-wrong length
    beats no sound at all."""
    base = float(settings.get("tts_length_scale"))
    best = synthesize_to(text, dest, length_scale=base)
    if best is None or target_s <= 0:
        return best
    scale = base
    for _ in range(_DURATION_FIT_PASSES):
        actual = wav_duration(best)
        if actual <= 0 or abs(actual - target_s) <= _DURATION_FIT_TOLERANCE_S:
            break
        # Clamped: a very short line against a long target asks for a scale that turns speech into a
        # smear, and Piper gets unstable at the extremes.
        nxt = min(max(scale * (target_s / actual), _MIN_LENGTH_SCALE), _MAX_LENGTH_SCALE)
        if abs(nxt - scale) < 1e-3:
            break        # clamped or already converged — another identical pass buys nothing
        fitted = synthesize_to(text, dest, length_scale=nxt)
        if fitted is None:
            print(f"[tts] WARNING: could not stretch {text[:20]!r} to {target_s:.2f}s "
                  f"— keeping {actual:.2f}s")
            break
        best, scale = fitted, nxt
    print(f"[tts] fitted {text[:20]!r} to {wav_duration(best):.2f}s "
          f"(target {target_s:.2f}s, length-scale {scale:.2f})", flush=True)
    return best


def prewarm_canned(lines: dict[str, str], out_dir: str | Path,
                   targets: dict[str, float] | None = None) -> dict[str, Path]:
    """Synthesize each {key: text} once up front and return {key: wav} for the ones that worked.

    Meant for a startup thread. Playing a cached WAV is effectively free, whereas synthesizing the
    wake acknowledgement on demand would put 0.5-1.5 s of dead air between "Hey Kai" and "Yes?" —
    which is most of what makes hands-free feel broken. Best-effort per line: a failure just omits
    that key, and the caller falls back to live synthesis.

    `targets` gives {key: seconds} for lines that must come out a specific length rather than
    however long the voice happens to say them (the "Hmm..." filler is sized to the pause it fills).
    """
    out: dict[str, Path] = {}
    if not enabled():
        return out
    out_dir = Path(out_dir)
    targets = targets or {}
    for key, text in lines.items():
        dest = out_dir / f"kai_canned_{key}.wav"
        target = targets.get(key)
        wav = (synthesize_to_duration(text, dest, target) if target
               else synthesize_to(text, dest))
        if wav is None:
            print(f"[tts] WARNING: could not pre-synthesize canned line {key!r}")
        else:
            out[key] = wav
    return out


def wav_duration(path: Path) -> float:
    """Length of a WAV in seconds from its header (frames / frame-rate). 0.0 if it can't be read —
    the caller uses this to size the jaw window, so a bad read just yields an empty window."""
    try:
        with wave.open(str(path), "rb") as w:
            rate = w.getframerate()
            return w.getnframes() / rate if rate else 0.0
    except (OSError, wave.Error) as exc:
        print(f"[tts] WARNING: could not read WAV duration ({exc})")
        return 0.0


def wav_speech_span(path: Path) -> tuple[float, float]:
    """(start, end) of the AUDIBLE span inside a WAV, in seconds from the start of the file.

    Piper pads leading and trailing silence into everything it writes, and TTS_POST_ROOM's reverb
    decays into the trailing pad rather than extending the file — so the file is reliably longer
    than the speech, by 0.29 s of the 0.82 s ack WAV (measured 2026-08-12). wav_duration() is the
    file; this is the sound inside it, and it is what the jaw should be timed to. Callers that care
    how long the SPEAKER is busy (the filler length caps, the self-hearing gate) still want
    wav_duration.

    Returns an EMPTY span (0.0, 0.0) for a WAV that cannot be read, and the whole file for one that
    is silent throughout; the caller treats both as "no trim" and falls back to the file length, so
    a bad scan can only ever cost the trim. Deliberately silent about a failed read, unlike
    wav_duration: every caller reads the duration off the same file first, so anything really wrong
    with it has already been reported once and a second warning per playback is just noise.
    """
    try:
        with wave.open(str(path), "rb") as w:
            rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
            frames = w.readframes(w.getnframes())
    except (OSError, wave.Error):
        return 0.0, 0.0
    # 16-bit only, which is everything in this pipeline (Piper writes it and TTS_POST_SOX keeps it).
    # Anything else falls back to "no trim" rather than being misread as silence.
    if not rate or not channels or width != 2:
        return 0.0, 0.0

    samples = array.array("h")
    samples.frombytes(frames[:len(frames) - len(frames) % 2])
    if sys.byteorder == "big":
        samples.byteswap()             # WAV is little-endian; array uses the host's order
    per_block = max(1, int(rate * SPEAK_SILENCE_STEP_S)) * channels
    duration = (len(samples) // channels) / rate
    blocks = range(0, max(0, len(samples) - per_block + 1), per_block)

    # PEAK, not RMS: max()/min() over an array slice runs in C, and this sits on the speech path
    # immediately before playback — a per-sample loop in Python would add to the very latency the
    # trim exists to hide. Piper's padding is digital silence (exactly 0), so the two measures pick
    # the same edges; only the threshold differs. (audioop.rms would have been the obvious tool and
    # is what this was written with first — audioop was REMOVED from the stdlib in Python 3.13, and
    # the suite runs on 3.14 on the Windows box.)
    def _loud(i: int) -> bool:
        block = samples[i:i + per_block]
        return max(max(block), -min(block)) > SPEAK_SILENCE_PEAK

    first = next((i for i in blocks if _loud(i)), None)
    if first is None:
        return 0.0, duration           # silent throughout — mime the whole file
    last = next((i for i in reversed(blocks) if _loud(i)), first)
    return (first / (rate * channels),
            min(duration, (last + per_block) / (rate * channels)))


def play(path: Path) -> None:
    """Play a WAV through the configured PulseAudio sink and BLOCK until it finishes (or stop() cuts
    it off). XDG_RUNTIME_DIR is forced so paplay finds the Pulse socket under the @reboot cron
    autostart / SSH, not just an interactive login. Best-effort: playback failure only logs."""
    global _current_proc, _last_end, _profile_applied
    # Make sure TTS_SINK actually exists before the first reply of this process. Cheap (one pactl,
    # a no-op when the card is already right) and it front-loads the failure: without it the first
    # thing a digital-profile flip costs is a whole spoken reply.
    if not _profile_applied:
        _profile_applied = True   # set first: one warning per process, not one per reply
        apply_output_profile()
    env = {**os.environ, "XDG_RUNTIME_DIR": TTS_XDG_RUNTIME}  # inherit PATH/Pulse vars, override XDG
    cmd = ["paplay", f"--device={TTS_SINK}"]
    if TTS_LATENCY_MSEC:
        # Without this, Pulse's default buffer on this box is 2 s, so paplay returns long before the
        # sound stops — and "playback finished" is what the self-hearing gate keys off.
        cmd.append(f"--latency-msec={int(TTS_LATENCY_MSEC)}")
    # Volume is applied HERE, at playback, not at synthesis. Piper's --volume scales the raw audio,
    # which TTS_POST_EFFECTS then normalises straight back out (it ends in `gain -n -1`, i.e. peak to
    # -1 dBFS) — measured: the post-processed peak came out identical at volume 0.4 and 1.6, while 1.6
    # clipped the raw synthesis at full scale. Applying it to the sink input keeps the knob monotonic
    # and clip-free whether or not post-processing is on. PA_VOLUME_NORM is 65536 = 1.0.
    cmd.append(f"--volume={int(round(max(0.0, settings.get('tts_volume')) * 65536))}")
    cmd.append(str(path))
    try:
        with _proc_lock:
            _current_proc = subprocess.Popen(
                cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            proc = _current_proc
    except OSError as exc:
        print(f"[tts] WARNING: could not start playback ({exc})")
        return
    proc.wait()
    # A paplay that fails (missing/renamed sink, no Pulse socket) exits non-zero and is otherwise
    # completely inaudible AND invisible — which is exactly how Kai ends up miming replies in
    # silence with a clean log. Negative returncode is stop() cutting a reply short: not an error.
    # stderr is read AFTER wait() rather than via communicate() so the "newer handle" contract in
    # tests/test_tts.py still holds; paplay's stderr is a line or two, far under the pipe buffer.
    if proc.returncode is not None and proc.returncode > 0:
        try:
            stderr = proc.stderr.read() if proc.stderr is not None else b""
        except (OSError, ValueError):
            stderr = b""
        err = (stderr or b"").decode("utf-8", "replace").strip().splitlines()
        detail = err[-1] if err else f"exit {proc.returncode}"
        print(f"[tts] WARNING: playback failed ({detail}) — device={TTS_SINK}", flush=True)
        # PulseAudio is not always up when Kai wants to talk — it respawns on login/restart, and a
        # reply landing in that window dies with "Connection refused". One retry costs a few hundred
        # ms and turns a silently-skipped sentence into a spoken one. Only retried when this
        # playback is still the current one: if stop() has already handed over to a newer reply,
        # retrying would speak into a turn that has moved on.
        with _proc_lock:
            superseded = _current_proc is not proc
        if not superseded and _RETRY_ON_FAILURE:
            # Re-assert the card profile before retrying. A PulseAudio respawn is one reason
            # playback just failed; the card having flipped to its digital profile — which deletes
            # TTS_SINK outright — is another, and in that case an identical retry fails identically.
            # This is what turns the retry from "wait and hope" into an actual recovery.
            apply_output_profile()
            time.sleep(_RETRY_DELAY_S)
            try:
                with _proc_lock:
                    if _current_proc is not proc:
                        return          # a newer reply took over while we waited
                    _current_proc = subprocess.Popen(
                        cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    proc = _current_proc
            except OSError as exc:
                print(f"[tts] WARNING: playback retry could not start ({exc})")
                return
            proc.wait()
            if proc.returncode is not None and proc.returncode > 0:
                print(f"[tts] WARNING: playback retry also failed (exit {proc.returncode})",
                      flush=True)
    with _proc_lock:
        if _current_proc is proc:
            _current_proc = None
        _last_end = time.monotonic()


def is_playing() -> bool:
    """True while a playback subprocess is alive — ground truth for "Kai's audio is in the air".

    Deliberately NOT the same thing as the jaw's speaking window: that window can be a silent
    text-timed pantomime (synthesis disabled or failed), and it is only set AFTER synthesize()
    returns, so it says nothing about the Piper run that precedes playback."""
    with _proc_lock:
        proc = _current_proc
    return proc is not None and proc.poll() is None


def quiet_since(now: float | None = None) -> float:
    """Seconds since playback last stopped: 0.0 while still playing, inf if nothing has ever played.

    Callers hold the mic shut for a tail beyond this, because paplay exits once the WAV is in the
    PulseAudio sink buffer — which can be several hundred ms before the amp actually goes silent.
    That gap is how a robot ends up answering itself."""
    if is_playing():
        return 0.0
    with _proc_lock:
        last = _last_end
    if last <= 0.0:
        return float("inf")
    now = time.monotonic() if now is None else now
    return max(0.0, now - last)


def stop() -> None:
    """Cancel any in-flight speech — BOTH the Piper synth and the playback. Called when a new turn
    starts (so replies don't stack) and when a session ends (so an abandoned reply never reaches the
    speaker at all). No-op if nothing is running.

    Killing the synth is the half that is easy to forget: without it a worker cancelled mid-Piper
    still goes on to play(), and the audio lands in whatever session came next."""
    global _current_proc, _synth_proc, _last_end
    with _proc_lock:
        play_proc, _current_proc = _current_proc, None
        synth_proc, _synth_proc = _synth_proc, None
        if play_proc is not None:
            _last_end = time.monotonic()   # start the quiet-tail clock from the cut, not from exit
    for proc in (play_proc, synth_proc):
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
