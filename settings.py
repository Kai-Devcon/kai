"""Live, operator-settable knobs — the runtime overlay on top of config/*.py.

config/ holds hand-edited literals that need a restart (see config/README.md). This module holds the
handful of knobs the dashboard can change *while Kai is running*, persisted to
~/.config/kai/settings.json so they survive the next reboot. config/ stays the single source of truth
for DEFAULTS: every default below is read from a config import, so deleting settings.json restores
exactly the committed behaviour.

Same shape as vision/presence.py — a module-level lock plus module-level functions rather than a
class, so there is one instance per process for free, ai/ and vision/ can import it without dragging
in the tracking stack, and tests can patch PATH and call reset().

Reaching the running code happens two ways, and which one a knob uses is a judgement call recorded in
face_track.py's callback registration:

  PULL   settings.get(name) at the point of use. For values whose only effect is being read — the
         jaw/tracking gates, the TTS args, the camera mode. One dict lookup under an uncontended
         lock; unmeasurable even at the 15 Hz control loop.
  PUSH   on_change(name, fn). For values another object holds its own copy of, or where the change
         needs a side effect (ending a session, re-deriving a paired value, reopening an engine).
         Keeps the 20 ms audio path free of any settings lookup.

TWO RULES, because writers are Flask request threads and readers are the tracking loop, the control
loop, the session tick and the kai-audio worker:

  1. Never hold _lock while calling out — no callbacks, no file I/O, no print. It guards _values and
     nothing else, for the duration of a dict operation.
  2. Never call set_many() while holding another subsystem's lock. Callbacks take the session RLock
     (ai/session.py), so the reverse order deadlocks.

Every value is an immutable scalar (bool/float/str) — no dicts or lists. That is what lets get()
return without a defensive copy: a reader racing a writer sees the old value or the new one, never a
half-written one.

Nothing here raises on a bad file, an unwritable disk or a missing HOME. Settings must never be a
reason the robot fails to start; the same principle as ai/audio.py's wake tiers.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple

from config.thinking import THINKING_SOUNDS, THINKING_SWEEP
from config.voice import (
    DELIVERY_ENABLED, MIC_PREFERENCE, TTS_ENABLED, TTS_LENGTH_SCALE, TTS_NOISE_SCALE, TTS_NOISE_W,
    TTS_SENTENCE_SILENCE_S, TTS_VOLUME,
)
from config.wake import HANDS_FREE_ENABLED, VAD_RMS_FLOOR, WAKE_SENSITIVITIES

# Module constant, not a config knob: tests patch it. Kept out of config/ deliberately — this file's
# location is not something an operator retunes.
PATH = "~/.config/kai/settings.json"

_FLOAT_DP = 6   # round floats on the way in; keeps the JSON file and the SSE echo clean


class Spec(NamedTuple):
    """One knob's contract. `kind` drives coercion; lo/hi bound floats; choices bound strings."""
    kind:    str                      # "bool" | "float" | "choice"
    default: Any
    lo:      float | None = None
    hi:      float | None = None
    choices: tuple[str, ...] = ()


# The single source of truth for what is settable and what counts as valid. Defaults come from the
# config imports above, never from literals repeated here — see the module docstring.
_SPECS: dict[str, Spec] = {
    "camera_mode":      Spec("choice", "auto", choices=("auto", "off")),
    "hands_free":       Spec("bool",   HANDS_FREE_ENABLED),
    "servo_tracking":   Spec("bool",   True),
    "jaw_enabled":      Spec("bool",   True),
    "tts_enabled":      Spec("bool",   TTS_ENABLED),
    "tts_volume":       Spec("float",  TTS_VOLUME,       lo=0.0,  hi=2.0),
    "tts_length_scale": Spec("float",  TTS_LENGTH_SCALE, lo=0.5,  hi=2.0),
    # Piper's three prosody parameters. All live for the same reason tts_length_scale is: the only
    # instrument that can judge them is an ear, and an ear needs to flip back and forth while Kai is
    # talking. The upper bounds are not arbitrary — past ~1.2 the two noise parameters slur consonants
    # and wander off pitch, and a sentence pause past ~1 s reads as Kai having given up rather than as
    # a breath. All three re-warm the canned lines on change (face_track.py), because the filler bank
    # and the wake ack are pre-synthesised and would otherwise keep the old prosody.
    "tts_sentence_silence": Spec("float", TTS_SENTENCE_SILENCE_S, lo=0.0, hi=1.0),
    "tts_noise_scale":  Spec("float",  TTS_NOISE_SCALE,  lo=0.0,  hi=1.5),
    "tts_noise_w":      Spec("float",  TTS_NOISE_W,      lo=0.0,  hi=1.5),
    # Delivery shaping (ai/delivery.py) — breaths, opener, per-reply tempo jitter. PULL-read once per
    # reply, and live on purpose: it is a change only an ear can judge, so it has to be flippable
    # mid-conversation to A/B against the unshaped voice.
    "delivery_shaping": Spec("bool",   DELIVERY_ENABLED),
    # Which microphone to prefer. PULL-read by ai/mic_device._preference() on the resolve path only
    # — never on the audio path — so it needs no on_change wiring, but it also does not take effect
    # until the next resolve. That is what the dashboard's "find the microphone again" button is
    # for, and the two controls sit together there for exactly that reason.
    "mic_preference":   Spec("choice", MIC_PREFERENCE,
                             choices=("auto", "i2s", "usb", "analog", "pulse")),
    "vad_rms_floor":    Spec("float",  VAD_RMS_FLOOR,    lo=50.0, hi=5000.0),
    "wake_sensitivity": Spec("float",  WAKE_SENSITIVITIES[0] if WAKE_SENSITIVITIES else 0.5,
                             lo=0.0, hi=1.0),
    # The "thinking" expression. Both are PULL-read at the point of use (the control loop tick and the
    # BUSY tick branch), so they need no on_change wiring and each takes effect on the very next tick.
    # Two separate knobs on purpose: the head movement and the noise are independently revertible,
    # because they can disturb different things (servo current vs. the wake word).
    "thinking_sweep":   Spec("bool",   THINKING_SWEEP),
    "thinking_sounds":  Spec("bool",   THINKING_SOUNDS),
}

_lock      = threading.RLock()          # guards _values only
_save_lock = threading.Lock()           # serialises file writes, never held with _lock
_values: dict[str, Any] = {name: spec.default for name, spec in _SPECS.items()}
_rev = 0                                # bumped on every mutation; lets a superseded save skip
_persist_error = ""                     # non-empty once a write has failed; surfaced on /params
_loaded = False

# name -> [(callback, debounce_seconds, timer_or_None)]
_callbacks: dict[str, list[list[Any]]] = {}
_cb_lock = threading.Lock()


# ── Validation ────────────────────────────────────────────────────────────────

_TRUE  = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


def _coerce(name: str, value: Any, strict: bool) -> Any:
    """Turn an arbitrary JSON value into a valid value for `name`.

    strict=True (a dashboard POST): a value outside the spec is a caller error -> ValueError, so the
    route can 400 and the UI can say which control was wrong.
    strict=False (the settings file): never reject. Coerce and clamp what we can, so one stale or
    hand-edited entry cannot cost the operator every other setting in the file.
    """
    spec = _SPECS.get(name)
    if spec is None:
        raise ValueError(f"unknown setting: {name}")

    if spec.kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in _TRUE:
            return True
        if isinstance(value, str) and value.strip().lower() in _FALSE:
            return False
        if strict:
            raise ValueError(f"{name}: expected true or false, got {value!r}")
        return bool(spec.default)

    if spec.kind == "float":
        try:
            out = float(value)
        except (TypeError, ValueError):
            if strict:
                raise ValueError(f"{name}: expected a number, got {value!r}")
            return float(spec.default)
        if out != out or out in (float("inf"), float("-inf")):     # NaN / inf
            if strict:
                raise ValueError(f"{name}: expected a finite number, got {value!r}")
            return float(spec.default)
        lo, hi = spec.lo, spec.hi
        if lo is not None and hi is not None and not (lo <= out <= hi):
            if strict:
                raise ValueError(f"{name}: {out:g} is outside {lo:g}..{hi:g}")
            out = min(max(out, lo), hi)
        return round(out, _FLOAT_DP)

    if spec.kind == "choice":
        if isinstance(value, str) and value in spec.choices:
            return value
        if strict:
            raise ValueError(f"{name}: {value!r} is not one of {', '.join(spec.choices)}")
        return spec.default

    raise ValueError(f"{name}: unsupported kind {spec.kind!r}")   # unreachable; guards a typo above


# ── Reads ─────────────────────────────────────────────────────────────────────

def get(name: str) -> Any:
    """The live value. Hot path — called from the tracking and control loops."""
    with _lock:
        return _values[name]


def snapshot() -> dict[str, Any]:
    """Every knob -> live value. json.dumps-able (all scalars)."""
    with _lock:
        return dict(_values)


def defaults() -> dict[str, Any]:
    """Every knob -> its config/*.py default."""
    return {name: spec.default for name, spec in _SPECS.items()}


def describe() -> list[dict[str, Any]]:
    """Spec metadata for GET /settings — enough for a client to build a control, and for a human to
    see the valid range from curl."""
    out = []
    for name, spec in _SPECS.items():
        row: dict[str, Any] = {"name": name, "kind": spec.kind, "default": spec.default}
        if spec.kind == "float":
            row["min"], row["max"] = spec.lo, spec.hi
        if spec.kind == "choice":
            row["choices"] = list(spec.choices)
        out.append(row)
    return out


def persist_error() -> str:
    """Empty when healthy. Non-empty means changes are live but not being saved — the dashboard says
    so, because a setting that silently won't survive a reboot is worse than one that fails loudly."""
    with _lock:
        return _persist_error


# ── Writes ────────────────────────────────────────────────────────────────────

def set_many(updates: dict[str, Any], strict: bool = True, persist: bool = True) -> dict[str, Any]:
    """Validate every update, then apply them all, then save, then fire callbacks.

    All-or-nothing on purpose: a batch that half-applied would leave two knobs disagreeing (tts
    enabled at a volume that was rejected), and the dashboard sends coherent groups. Raises
    ValueError before touching anything if any key is unknown or any value invalid.

    Returns the stored value for every requested key, so the caller can echo what actually landed.
    """
    if not isinstance(updates, dict):
        raise ValueError("expected a JSON object of setting -> value")

    coerced = {name: _coerce(name, value, strict) for name, value in updates.items()}

    global _rev
    with _lock:
        changed = {n: v for n, v in coerced.items() if _values[n] != v}
        _values.update(coerced)
        _rev += 1
        rev = _rev
        applied = {n: _values[n] for n in coerced}

    if persist and changed:
        _save(rev)
    for name, value in changed.items():
        _fire(name, value)
    return applied


def set_one(name: str, value: Any, strict: bool = True) -> Any:
    """Convenience for a single knob. set_many is the primary path — see its docstring."""
    return set_many({name: value}, strict=strict)[name]


def reset() -> dict[str, Any]:
    """Back to the config/*.py defaults, and forget the overlay file.

    The first recovery path, and the one to reach for first: it gets an operator back to a known
    state after over-tuning without dropping the camera, the mic or a conversation in progress. The
    dashboard's restart button (POST /restart) is the heavier one, for the failures no value here
    can reach.
    """
    global _rev
    with _lock:
        before = dict(_values)
        _values.update(defaults())
        _rev += 1
        after = dict(_values)
    changed = {n: v for n, v in after.items() if before[n] != v}

    try:
        path = _path()
        if path is not None and path.exists():
            path.unlink()
        _clear_persist_error()
    except OSError as exc:
        _note_persist_error(f"could not remove {PATH} ({exc})")

    for name, value in changed.items():
        _fire(name, value)
    return after


# ── Change notification ───────────────────────────────────────────────────────

def on_change(name: str, fn: Callable[[Any], None], debounce: float = 0.0) -> None:
    """Call fn(new_value) whenever `name` actually changes.

    debounce > 0 collapses a burst into one trailing call — for subscribers whose work is expensive,
    like re-synthesising the canned wake replies when the TTS voice changes. A dragged slider fires
    dozens of POSTs; Piper should run once.
    """
    if name not in _SPECS:
        raise ValueError(f"unknown setting: {name}")
    with _cb_lock:
        _callbacks.setdefault(name, []).append([fn, debounce, None])


def _fire(name: str, value: Any) -> None:
    """Notify subscribers. Never called with _lock held (rule 1)."""
    with _cb_lock:
        entries = list(_callbacks.get(name, ()))
    for entry in entries:
        fn, debounce = entry[0], entry[1]      # entry[2] is the live timer, read under _cb_lock
        if debounce > 0:
            with _cb_lock:
                if entry[2] is not None:
                    entry[2].cancel()
                entry[2] = threading.Timer(debounce, _invoke, args=(name, fn, value))
                entry[2].daemon = True
                entry[2].start()
        else:
            _invoke(name, fn, value)


def _invoke(name: str, fn: Callable[[Any], None], value: Any) -> None:
    try:
        fn(value)
    except Exception as exc:
        # One broken subscriber must not abort the batch or lose the persist.
        print(f"[settings] ERROR in {name} callback: {type(exc).__name__}: {exc}", flush=True)


# ── Persistence ───────────────────────────────────────────────────────────────

def _path() -> Path | None:
    """The overlay path, or None if it cannot even be resolved.

    expanduser() raises RuntimeError when HOME is unset — which is exactly the @reboot cron
    environment this robot starts in (the same hazard ai/audio.py handles for the Porcupine key).
    """
    try:
        return Path(PATH).expanduser()
    except RuntimeError as exc:
        print(f"[settings] WARNING: cannot resolve {PATH} ({exc}) — using config/ defaults, "
              f"changes will not persist", flush=True)
        return None


def _note_persist_error(msg: str) -> None:
    global _persist_error
    with _lock:
        _persist_error = msg
    print(f"[settings] WARNING: {msg} — the change is live but will not survive a restart",
          flush=True)


def _clear_persist_error() -> None:
    global _persist_error
    with _lock:
        _persist_error = ""


def load() -> None:
    """Apply the overlay file over the defaults. Call once, first thing in run().

    Never raises: a corrupt, unreadable or nonsensical file costs you your overlay, not your robot.
    """
    global _loaded
    try:
        _load_inner()
    except Exception as exc:       # blanket on purpose — see the module docstring
        print(f"[settings] WARNING: could not load settings ({type(exc).__name__}: {exc}) — "
              f"using config/ defaults", flush=True)
    finally:
        _loaded = True


def _load_inner() -> None:
    path = _path()
    if path is None:
        return
    if not path.exists():
        print(f"[settings] no overlay at {path} — using config/ defaults", flush=True)
        return

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        # Deliberately NOT quarantined: the file may be perfectly good and the read merely transient
        # (EIO on a flaky SD card — this box has ext4 errors). Moving it aside would lose the
        # operator's settings for no reason.
        print(f"[settings] WARNING: cannot read {path} ({exc}) — using config/ defaults", flush=True)
        return

    try:
        raw = json.loads(text)
    except ValueError as exc:
        _quarantine(path, f"is not valid JSON ({exc})")
        return
    if not isinstance(raw, dict):
        _quarantine(path, f"is {type(raw).__name__}, expected a JSON object")
        return

    unknown = [k for k in raw if k not in _SPECS]
    wanted  = {k: v for k, v in raw.items() if k in _SPECS}
    # strict=False: clamp and coerce, never reject. persist=False: loading is not a change worth
    # rewriting the file for, and rewriting here would drop the very keys we just warned about.
    applied = set_many(wanted, strict=False, persist=False)
    if unknown:
        print(f"[settings] WARNING: ignoring unknown setting(s) in {path}: "
              f"{', '.join(sorted(unknown))} — leftover from an older version?", flush=True)
    changed = {n: v for n, v in applied.items() if v != _SPECS[n].default}
    print(f"[settings] loaded {len(applied)} setting(s) from {path}"
          + (f"; overriding {', '.join(f'{n}={v}' for n, v in sorted(changed.items()))}"
             if changed else " (all at defaults)"), flush=True)


def _quarantine(path: Path, why: str) -> None:
    """Move a bad file aside so the operator can inspect it and the next save is not fighting it."""
    print(f"[settings] WARNING: {path} {why} — using config/ defaults", flush=True)
    try:
        os.replace(str(path), str(path) + ".bad")
        print(f"[settings] moved it to {path}.bad", flush=True)
    except OSError as exc:
        print(f"[settings] (could not move it aside: {exc})", flush=True)


def _save(rev: int) -> None:
    """Write the non-default values atomically. Never raises.

    Only what differs from the default is written, so a future change to a config/*.py default
    propagates instead of being frozen forever by a stale overlay.
    """
    path = _path()
    if path is None:
        return
    with _save_lock:
        with _lock:
            if _rev != rev:
                return          # a newer writer is right behind us; its snapshot is a superset
            body = {n: v for n, v in _values.items() if v != _SPECS[n].default}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")     # same directory -> same filesystem
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(body, fh, indent=2, sort_keys=True)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except OSError as exc:
            _note_persist_error(f"could not persist to {path} ({exc})")
            return
        _clear_persist_error()


def _reset_for_tests() -> None:
    """Restore module state — values, callbacks, error, revision. Tests only.

    Module-level state is shared across tests in a process, exactly as vision/presence.reset() is.
    """
    global _rev, _persist_error, _loaded
    with _cb_lock:
        for entries in _callbacks.values():
            for entry in entries:
                if entry[2] is not None:
                    entry[2].cancel()
        _callbacks.clear()
    with _lock:
        _values.update(defaults())
        _rev = 0
        _persist_error = ""
        _loaded = False


def _wait_for_debounced(timeout: float = 5.0) -> None:
    """Block until every pending debounced callback has run. Tests only."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with _cb_lock:
            timers = [e[2] for entries in _callbacks.values() for e in entries
                      if e[2] is not None and e[2].is_alive()]
        if not timers:
            return
        for timer in timers:
            timer.join(max(0.0, deadline - time.monotonic()))
