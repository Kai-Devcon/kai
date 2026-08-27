"""Finding a microphone that actually captures signal, and getting the card ready to open it.

This is device plumbing, not conversation: ALSA mixer routes, PulseAudio suspend/resume, liveness
probes, and the arithmetic deciding which sample rates are usable. It lives on its own because two
different consumers need it and neither should own it — ai/voice_assistant.py resolves a device for
its legacy per-turn stream, and ai/session.py's MicStream resolves one for the shared always-open
stream. Until this module existed the session imported the plumbing FROM the assistant, which said
the wrong thing about which of them is the lower layer.

The hard-won parts, all documented at their call sites and in config/voice.py:

  * The INMP441 is captured raw on hw:APE with PulseAudio suspended, because pulse otherwise locks
    the card to 44100 and injects noise that garbles speech. A raw hw device admits exactly ONE
    opener, which is why capture is centralised in the first place.
  * Every capture rate offered is one the pipeline can resample with an integer ratio. Returning a
    rate that cannot be is not merely suboptimal — it raises at Decimator construction and takes the
    whole session down (2026-08-09).
  * A device that reads as silent and a device that refuses to open are different problems with
    different fixes, so both are logged, and only the former is retried.
  * PortAudio snapshots the ALSA device list at Pa_Initialize and never refreshes it, so nothing here
    can see a card plugged in after startup until refresh_devices() forces a re-scan — and that can
    only be done with no stream open.

A USB mic is a supported input and not merely the fallback the I2S mic degrades to: MIC_PREFERENCE
picks which kind is probed first, _profile_for() gives each kind its own channel/retry treatment, and
ai/mic_hotplug.py notices one being plugged in or pulled out so the switch does not need a restart.
What is NOT symmetric between them is the speaker: exactly one card must never be captured raw, and
_is_speaker_card() identifies it by ALSA card index rather than by name so that a USB mic sharing the
speaker dongle's product name is still usable.

Best-effort throughout: a missing amixer, a missing pactl or an absent APE card logs and falls back
to the USB/system-default mic rather than raising into startup.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from collections import namedtuple

import numpy as np
import sounddevice as sd

import settings
from config.voice import (
    CHANNELS, FALLBACK_CAPTURE_RATES, I2S_APPLY_ROUTE_ON_STARTUP, I2S_CAPTURE_CHANNELS,
    I2S_CAPTURE_RATE, I2S_MIC_NAME_HINTS, I2S_PROBE_RETRY_DELAY_S, I2S_PROBE_SILENT_RETRIES,
    I2S_PULSE_SOURCE, I2S_ROUTE_CARD, I2S_ROUTE_CONTROLS, I2S_SUSPEND_PULSE, I2S_TAKE_CHANNEL,
    LIVE_PROBE_DURATION_S, LIVE_PROBE_RMS_THRESHOLD, LIVE_PROBE_TIMEOUT_S, MIC_PREFERENCE,
    PULSE_SUSPEND_ALL_SOURCES, SAMPLE_RATE, SPEAKER_CARD_FROM_TTS_CARD, SPEAKER_CARD_NAME_HINTS,
    TTS_CARD, USB_MIC_NAME_HINTS, USB_PROBE_SILENT_RETRIES,
)
from config.wake import MIXER_TIMEOUT_S

# resolve_input_device() uses this to parse ALSA card ids so it can dedupe the many subdevice
# entries one card exposes. Card ids can be numeric ("hw:1,0") or named ("hw:APE,0"), so match
# everything up to the subdevice comma / closing paren (structural, not a tunable).
_HW_CARD_RE = re.compile(r"hw:([^,\)]+)")

# Serialises everything that changes PortAudio's own structure — refresh_devices() (which terminates
# and re-initialises the host API) and opening a stream. They run on different threads: the audio
# worker's stall watchdog reopens, and the dashboard's /audio/reresolve button resolves, and until
# this lock existed they could interleave. Tearing the host API down while another thread is inside
# Pa_OpenStream does not fail cleanly — it surfaces as "Illegal combination of I/O devices"
# [PaErrorCode -9993], an error about duplex devices on a stream that has no output side, which cost
# a full investigation on 2026-08-27 before it was recognised as a torn re-init rather than a device
# problem. Re-entrant because MicStream.reopen() holds it across refresh_devices() AND open().
pa_lock = threading.RLock()

# The resolved mic: which device index/rate to open, how many channels to capture, which channel
# holds the real audio (the INMP441 puts it only in the left slot), the sample dtype, whether it's
# the raw I2S device (which needs pulseaudio suspended before each open), and which bucket it came
# from.
#
# `kind` is not derivable from is_i2s — that flag answers "does this need pulse suspended", and it
# collapses "the USB mic the operator chose" and "whatever the system default turned out to be" into
# one False. Now that USB is a deliberate choice rather than a fallback, the dashboard has to be able
# to say which of the two Kai is actually on, so the bucket travels with the choice.
#
# `card` is the ALSA card id ("3", "APE") when the chosen device is a RAW hw: entry, and "" when it
# is a pulse-mediated one ("default", "pulse", or no device at all). It exists because is_i2s was
# being used as a stand-in for "we are about to open a raw device, so pulse must stay off this
# card", and that is not what is_i2s means: a USB mic resolves to hw:<n>,0 — just as raw, and just
# as exclusive — with is_i2s False. Handing pulse the card back before opening it is what made a
# perfectly healthy BY-PM700 fail every open with "Device unavailable" (2026-08-27).
MicChoice = namedtuple("MicChoice", "device rate channels take_channel dtype is_i2s kind card")
# is_i2s defaults False, kind "other", card "" (keeps non-i2s call sites terse)
MicChoice.__new__.__defaults__ = (False, "other", "")


def _alsa_card_of(name: str) -> str:
    """The ALSA card id in a PortAudio device name ("BY-PM700: USB Audio (hw:3,0)" -> "3"), or ""
    for the named plugin PCMs (default/pulse/sysdefault), which are not raw and not exclusive."""
    m = _HW_CARD_RE.search(name or "")
    return m.group(1) if m else ""

def _classify_device(name: str) -> str:
    """Bucket an input device by its name: 'i2s' (the preferred INMP441/APE mic), 'usb' (the
    fallback), or 'other'. Case-insensitive substring match; I2S is checked before USB."""
    lowered = (name or "").lower()
    if any(hint.lower() in lowered for hint in I2S_MIC_NAME_HINTS):
        return "i2s"
    if any(hint.lower() in lowered for hint in USB_MIC_NAME_HINTS):
        return "usb"
    return "other"


def _preference() -> str:
    """Which kind to probe first: "i2s", "usb", or "auto". Live-settable from the dashboard.

    PULL-read (settings.py's term) rather than pushed, because it is only consulted on the resolve
    path — startup, the dashboard button, a hot-plug — never on the 20 ms audio path. Falls back to
    the config default if settings is unreadable for any reason, and an unrecognised value reads as
    "auto": this is operator-writable, and a typo must not be able to change which mic Kai picks in a
    way nobody can explain.
    """
    try:
        pref = settings.get("mic_preference")
    except Exception:
        pref = MIC_PREFERENCE
    return pref if pref in ("i2s", "usb", "auto") else "auto"


# The speaker's ALSA card index, resolved from TTS_CARD via pactl. Three states, and they are
# different: an index string means "resolved, block this card"; "" means "asked pactl, it could not
# tell us" (tier 2 takes over); None means "not asked yet". Cached because it is a subprocess and
# _is_speaker_card runs once per candidate device per resolve — cleared by refresh_devices(), since
# a re-plug renumbers cards.
_speaker_card: str | None = None


def _speaker_alsa_card() -> str:
    """The ALSA card index the speaker sits on, or "" if it cannot be determined.

    Reads `pactl list cards` and finds the block whose `Name:` is TTS_CARD — the card this build
    already plays through — then returns its `alsa.card` property. That index is what appears as
    `hw:<N>` in the PortAudio device name, which is what makes the match exact rather than a guess at
    a product name (see SPEAKER_CARD_NAME_HINTS in config/voice.py for why that matters now).

    Best-effort like everything else in this module: no pactl, no pulse, or a TTS_CARD pulse has
    never heard of all return "", and the caller falls back to the name hints.
    """
    global _speaker_card
    if _speaker_card is not None:
        return _speaker_card
    _speaker_card = ""
    if not SPEAKER_CARD_FROM_TTS_CARD or not TTS_CARD:
        return _speaker_card
    try:
        out = subprocess.run(["pactl", "list", "cards"], check=True, capture_output=True,
                             text=True, timeout=MIXER_TIMEOUT_S).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"[mic] could not ask pactl which card the speaker is on ({exc}) — falling back to "
              f"matching SPEAKER_CARD_NAME_HINTS against the device name", flush=True)
        return _speaker_card
    # `pactl list cards` is blocks separated by "Card #N", each with a "Name:" line and a Properties
    # section. Walk it linearly rather than parsing properly: we need exactly one field out of one
    # block, and the format has been stable for a decade.
    in_card = False
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Card #"):
            in_card = False
        elif stripped.startswith("Name:"):
            in_card = stripped.split(":", 1)[1].strip() == TTS_CARD
        elif in_card and stripped.startswith("alsa.card ") and "=" in stripped:
            _speaker_card = stripped.split("=", 1)[1].strip().strip('"')
            print(f"[mic] speaker card {TTS_CARD} is ALSA card {_speaker_card} — input devices on "
                  f"hw:{_speaker_card} will not be captured", flush=True)
            break
    return _speaker_card


def _is_speaker_card(name: str) -> bool:
    """True if this input device sits on the same sound card as the speaker.

    Capturing there is not merely a bad choice of mic — it is raw ALSA capture on a card that
    tts.play() reconfigures with `pactl set-card-profile` before the first reply, and the process
    segfaulted on the robot doing exactly that (2026-08-11, at the startup greeting). See
    SPEAKER_CARD_NAME_HINTS in config/voice.py for the log excerpt and the trade.

    Two tiers, in this order and not the other one. If pactl can tell us which ALSA card the speaker
    is on, that answer is exact and it is the whole answer — a separate USB mic that happens to share
    the dongle's product name is on a different card and is NOT blocked, which is the point. Only
    when pactl cannot answer do we fall back to matching the name, which is coarse enough to block a
    real mic but errs in the safe direction."""
    card = _speaker_alsa_card()
    if card:
        m = _HW_CARD_RE.search(name or "")
        return bool(m) and m.group(1) == card
    lowered = (name or "").lower()
    return any(hint.lower() in lowered for hint in SPEAKER_CARD_NAME_HINTS if hint)


# One log line per device name, not one per resolve: resolve_input_device() runs again on every
# /audio/reresolve, and the reason a mic was skipped is a startup fact, not a per-attempt one.
_speaker_card_logged: set[str] = set()


def _log_speaker_card_skip(idx: int, name: str) -> None:
    if name in _speaker_card_logged:
        return
    _speaker_card_logged.add(name)
    print(f"[mic] skipping input device {idx} ({name!r}) — it is on the speaker's own card, and "
          f"capturing it raw while playback reconfigures that card segfaulted the process at the "
          f"startup greeting (see SPEAKER_CARD_NAME_HINTS in config/voice.py). Falling through to "
          f"the pulse-mediated default; if nothing else is live, this run has no mic.", flush=True)


def _capture_rates_for(kind: str, advertised: int) -> tuple[int, ...]:
    """Capture rates to try for a device, in order. Every one of them is usable by the pipeline.

    The I2S mic is pinned: the route runs I2S2 at a fixed clock and the advertised rate is a lie
    (the real APE device reports 44100 while the route runs at 48 kHz), so there is exactly one
    candidate and it comes from config.

    For everything else the filter is arithmetic. MicStream resamples with an integer-ratio
    decimator, so a rate that does not divide SAMPLE_RATE is not merely suboptimal — it fails at
    Decimator construction and takes the whole session down (see FALLBACK_CAPTURE_RATES in
    config/voice.py for the incident). The advertised rate is therefore offered ONLY when it is
    divisible, and it goes first when it is, because opening a device at its native rate avoids a
    driver-side resample. The rest of the list follows as fallbacks, and the liveness probe — which
    opens the device for real — is what decides which of them the hardware actually accepts.
    """
    if kind == "i2s" and I2S_CAPTURE_RATE:
        return (I2S_CAPTURE_RATE,)
    rates = [r for r in FALLBACK_CAPTURE_RATES if r > 0 and r % SAMPLE_RATE == 0]
    if advertised > 0 and advertised % SAMPLE_RATE == 0 and advertised not in rates:
        rates.insert(0, advertised)
    elif advertised in rates:
        rates.insert(0, rates.pop(rates.index(advertised)))
    return tuple(rates)


def _profile_for(kind: str) -> tuple[int, int, int]:
    """How to open and probe a device of this kind: (channels, take_channel, silent_retries).

    The INMP441 must be captured in stereo and read from its left slot — a mono open of that
    stereo-only device fails outright, and its right slot is silent. Everything else is mono channel
    0, which is not merely the same value by coincidence: passing I2S_TAKE_CHANNEL to a USB mic was
    harmless only because it happens to be 0, and it would silently read the wrong slot the day
    someone retunes it for a differently-wired I2S board.

    The retry counts differ for reasons documented at each constant: the I2S mic has a boot-race
    warm-up worth three extra reads, a USB mic has a hot-plug settling window worth one, and neither
    number should be spent on the other device."""
    if kind == "i2s":
        return I2S_CAPTURE_CHANNELS, I2S_TAKE_CHANNEL, I2S_PROBE_SILENT_RETRIES
    return CHANNELS, 0, USB_PROBE_SILENT_RETRIES


def _candidate_input_devices(devices: list[dict]) -> list[int]:
    """Distinct input-capable devices to probe, in preference order: the preferred kind first, then
    the other mic kind, then everything else — with the system default heading the 'other' bucket.
    Keeps one representative per underlying ALSA card (avoids probing 20+ duplicate subdevice entries
    some cards expose). When no I2S/USB device is present this collapses to 'default first, then
    cards in order' — the historical behavior.

    MIC_PREFERENCE only reorders. Every bucket is still probed, so preferring one mic can never leave
    Kai deaf when only the other one is plugged in — see _preference().

    Devices on the speaker's own card are dropped entirely rather than ranked last: they are not a
    worse mic, they are the one choice that can take the process down (_is_speaker_card)."""
    buckets: dict[str, list[int]] = {"i2s": [], "usb": [], "other": []}
    seen: set[int] = set()

    try:
        default_idx = sd.default.device[0]
    except Exception:
        default_idx = None
    # Seed 'other' with the system default so it leads the non-preferred devices. Checked against
    # the speaker's card too, because the default can point straight AT a hw device — the named
    # "default"/"pulse" entries do not match the hints and so are unaffected, which is the point:
    # going through pulse is the safe way to touch that card.
    default_name = ""
    if isinstance(default_idx, int) and 0 <= default_idx < len(devices):
        default_name = devices[default_idx].get("name", "") or ""
    if isinstance(default_idx, int) and default_idx >= 0 and not _is_speaker_card(default_name):
        buckets["other"].append(default_idx)
        seen.add(default_idx)

    seen_cards: set[str] = set()
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) <= 0 or idx in seen:
            continue
        if _is_speaker_card(dev.get("name", "")):
            _log_speaker_card_skip(idx, dev.get("name", ""))
            continue
        m = _HW_CARD_RE.search(dev.get("name", ""))
        if m:
            card = m.group(1)
            if card in seen_cards:
                continue
            seen_cards.add(card)
        buckets[_classify_device(dev.get("name", ""))].append(idx)
        seen.add(idx)
    order = {"i2s": ("i2s", "usb", "other"),
             "usb": ("usb", "i2s", "other")}.get(_preference(), ("i2s", "usb", "other"))
    return [idx for kind in order for idx in buckets[kind]]


def _probe_is_live(device: int, rate: int, channels: int = CHANNELS,
                   take_channel: int = 0, retries: int = 0) -> bool:
    """Record a brief burst and check for real signal — silent/disconnected inputs read as zero.

    `retries` re-reads a device that came back SILENT, up to that many extra times. It exists for
    the I2S mic, which can return exact digital silence on the first capture after its route is
    applied and read normally moments later (see I2S_PROBE_SILENT_RETRIES in config/voice.py).
    Silence is the only outcome worth retrying: a device that refuses to open, or that hangs, has
    given a definite answer, and re-asking costs a multiple of LIVE_PROBE_TIMEOUT_S on the session
    start path for no gain. Defaults to 0 so every other caller keeps the old single-read behavior.
    """
    for attempt in range(max(0, retries) + 1):
        if attempt:
            time.sleep(I2S_PROBE_RETRY_DELAY_S)
        live, silent = _probe_once(device, rate, channels, take_channel)
        if live:
            if attempt:
                print(f"[mic] device {device} came back on retry {attempt} — the first read was "
                      f"taken before the mic was delivering samples", flush=True)
            return True
        if not silent:
            return False        # refused to open, or hung: a definite answer, not a warm-up
    return False


def _probe_once(device: int, rate: int, channels: int = CHANNELS,
                take_channel: int = 0) -> tuple[bool, bool]:
    """One liveness read. Returns (live, was_silent) — the second flag separates "this device
    delivered nothing but zeros" from "this device would not give us samples at all", which is what
    lets the caller retry only the former.

    Captures `channels` channels (a mono open of the stereo-only INMP441 device fails outright)
    and measures RMS on `take_channel` only, so the mic's silent right channel can't dilute it."""
    result: dict = {}

    def _capture() -> None:
        try:
            rec = sd.rec(int(LIVE_PROBE_DURATION_S * rate), samplerate=rate,
                          channels=channels, dtype="int16", device=device)
            sd.wait()
        except Exception as exc:
            # RECORDED, not swallowed. This was a bare `return`, which made "the device refused to
            # open" indistinguishable from "the mic is silent" — both produced `i2s=False` with no
            # reason anywhere in the log. On 2026-08-07 that cost a full hardware investigation to
            # establish the mic was fine and startup contention had simply lost the probe. The
            # timeout path below already logs; this one has to as well.
            result["error"] = f"{type(exc).__name__}: {exc}"
            return                       # no "rec" key — reads as "not live" below
        result["rec"] = rec

    # Bounded on its own thread: sd.wait() has no timeout, and a device that opens but never
    # delivers frames blocks it forever rather than raising, so the except above cannot catch it.
    # See LIVE_PROBE_TIMEOUT_S — a hang here strands the whole session start.
    probe = threading.Thread(target=_capture, daemon=True, name="kai-mic-probe")
    probe.start()
    probe.join(LIVE_PROBE_TIMEOUT_S)
    if probe.is_alive():
        try:
            sd.stop()                    # abort the wedged stream so the thread can unwind
        except Exception:
            pass
        probe.join(1.0)
        print(f"[mic] WARNING: live probe on device {device} did not return within "
              f"{LIVE_PROBE_TIMEOUT_S:.0f}s — treating it as not live", flush=True)
        return False, False

    rec = result.get("rec")
    if rec is None:
        print(f"[mic] device {device} rejected the probe "
              f"({rate} Hz x{channels}): {result.get('error', 'unknown error')}", flush=True)
        return False, False
    if rec.ndim > 1 and rec.shape[1] > take_channel:
        rec = rec[:, take_channel]
    rms = float(np.sqrt(np.mean(rec.astype(np.float64) ** 2)))
    # Both outcomes logged, at one line per candidate device. "Read as silent" and "refused to open"
    # are different problems with different fixes (check the wiring vs. check what holds the card),
    # and telling them apart afterwards is only possible if the log said which happened.
    if rms <= LIVE_PROBE_RMS_THRESHOLD:
        print(f"[mic] device {device} read as silent "
              f"(rms={rms:.1f} <= {LIVE_PROBE_RMS_THRESHOLD})", flush=True)
        return False, True
    return True, False


def apply_i2s_route() -> bool:
    """Bring up the ALSA XBAR/I2S2 capture route the INMP441 needs (see mictest/RESULTS.md), so
    the mic works on every app start without the external i2s-mic-route.service or a manual SSH
    session. Runs the exact `amixer` control sequence from config. Best-effort and never raises:
    if `amixer` or the APE card is missing (dev box, or before the device-tree overlay loads) it
    logs and returns False, and resolve_input_device() falls back to the USB mic. Returns True
    only if the full route applied."""
    if not I2S_APPLY_ROUTE_ON_STARTUP:
        return False
    for name, value in I2S_ROUTE_CONTROLS:
        try:
            subprocess.run(
                ["amixer", "-c", I2S_ROUTE_CARD, "cset", f"name={name}", value],
                check=True, capture_output=True, text=True, timeout=MIXER_TIMEOUT_S,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            # The dominant failure is "no APE card / no amixer" — one control failing means the
            # rest will too, so stop after the first instead of emitting nine warnings.
            print(f"[voice_assistant] WARNING: could not apply I2S capture route "
                  f"('{name}' on card {I2S_ROUTE_CARD}: {exc}) — I2S mic may be unavailable; "
                  f"selection will fall back to the USB/default mic")
            return False
    print(f"[voice_assistant] applied I2S capture route on card {I2S_ROUTE_CARD}")
    return True


def _pactl_suspend(source: str, on: bool) -> None:
    """Suspend/resume a pulseaudio source via pactl. Best-effort; raises nothing.

    Bounded by a timeout because an unresponsive pulseaudio would otherwise block here forever, and
    this now runs on every mic open AND on every watchdog reopen — not just once per turn."""
    try:
        subprocess.run(["pactl", "suspend-source", source, "1" if on else "0"],
                       check=True, capture_output=True, text=True, timeout=MIXER_TIMEOUT_S)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"[voice_assistant] WARNING: pactl suspend-source {source} "
              f"{'1' if on else '0'} failed ({exc})")


def _pactl_source_names() -> list[str]:
    """Every pulseaudio capture source, monitors excluded. Empty if pactl/pulse is unavailable.

    Monitors are skipped deliberately: they are taps on an OUTPUT and hold no capture hardware, so
    suspending them would gain nothing and would break anything listening to what Kai plays.
    """
    try:
        out = subprocess.run(["pactl", "list", "short", "sources"], check=True,
                             capture_output=True, text=True, timeout=MIXER_TIMEOUT_S).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    names = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and not parts[1].endswith(".monitor"):
            names.append(parts[1])
    return names


def _pactl_source_cards() -> dict[str, str]:
    """Map each pulseaudio capture source to the ALSA card index it sits on. Empty if pactl/pulse is
    unavailable, which the caller reads as "cannot attribute" and treats as the cautious answer.

    Same linear walk over `pactl list sources` as _speaker_alsa_card() does over cards, and for the
    same reason: one property out of each block, in a format that has been stable for a decade."""
    try:
        out = subprocess.run(["pactl", "list", "sources"], check=True, capture_output=True,
                             text=True, timeout=MIXER_TIMEOUT_S).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    cards: dict[str, str] = {}
    name = ""
    for line in out.splitlines():
        stripped = line.strip()
        if stripped.startswith("Source #"):
            name = ""
        elif stripped.startswith("Name:"):
            name = stripped.split(":", 1)[1].strip()
        elif name and stripped.startswith("alsa.card ") and "=" in stripped:
            cards[name] = stripped.split("=", 1)[1].strip().strip('"')
    return cards


# Every source free_i2s_device() actually suspended, so resume_pulse_sources() can hand back exactly
# that set. Module-level for the same reason the stream is centralised: suspend and resume happen on
# different calls, in different objects, and the only correct resume list is the one the suspend
# built.
_suspended_sources: list[str] = []


def free_i2s_device() -> None:
    """Release the capture cards from pulseaudio so the app can open a raw hw device directly at its
    true rate. pulse otherwise locks the APE card to 44100 and injects noise that garbles speech
    (Whisper hears nothing). No-op if disabled or pactl/pulse is absent.

    Every source is released, not just the I2S one: a source pulse holds makes the liveness probe of
    that device block until it times out, and a timed-out probe reads as "not live" — which is enough
    to skip the real mic and fall back to a 44.1 kHz pulse device that cannot be resampled to 16 kHz.
    See PULSE_SUSPEND_ALL_SOURCES in config/voice.py.
    """
    # I2S_SUSPEND_PULSE stays the master switch: it is documented as the way to opt out entirely (e.g.
    # pulse is not installed), so nothing here may touch pactl when it is False.
    if not I2S_SUSPEND_PULSE:
        return
    _suspended_sources.clear()
    _pactl_suspend(I2S_PULSE_SOURCE, True)
    _suspended_sources.append(I2S_PULSE_SOURCE)
    if PULSE_SUSPEND_ALL_SOURCES:
        for src in _pactl_source_names():
            if src != I2S_PULSE_SOURCE:
                _pactl_suspend(src, True)
                _suspended_sources.append(src)


def resume_pulse_sources(keep_card: str = "") -> None:
    """Hand the cards back to pulseaudio after the probe, EXCEPT the one we are about to capture raw.

    Resumes everything free_i2s_device() suspended, not just I2S_PULSE_SOURCE. That asymmetry was
    invisible while USB was only ever a boot-time fallback onto a raw hw device — a suspended source
    does not stop a raw open, so nothing looked broken. It stops being invisible the moment a run is
    *meant* to end on the USB mic: PULSE_SUSPEND_ALL_SOURCES had muted that source too, and only the
    I2S one was ever given back, so pulse-mediated capture on the chosen mic stayed suspended for the
    life of the process. Idempotent, so calling it on a path that suspended nothing is a no-op.

    `keep_card` is the ALSA card of the device the caller is about to open RAW, and it is the whole
    reason this takes an argument. Resuming is not a courtesy to other software — on this build there
    is no module-suspend-on-idle, so pulse holds every card it owns open permanently, and handing it
    back a card means it re-grabs the hw device within milliseconds. Doing that between resolving the
    USB mic and opening it is what made a healthy BY-PM700 probe live at 48 kHz and then fail EVERY
    open with "Device unavailable" [PaErrorCode -9985], forever, on 2026-08-27. The card being opened
    raw therefore stays suspended; every other one is handed back, so nothing else is left muted.

    A source pulse cannot attribute to a card is left suspended whenever keep_card is set: guessing
    wrong in that direction costs a muted source somewhere else, and guessing wrong in the other one
    costs the microphone. When keep_card is "" (a pulse-mediated device, which is not exclusive)
    everything is handed back unconditionally — the historical behavior."""
    if not I2S_SUSPEND_PULSE:
        return
    source_cards = _pactl_source_cards() if keep_card else {}
    kept: list[str] = []
    for src in _suspended_sources or [I2S_PULSE_SOURCE]:
        if keep_card and source_cards.get(src, "") == keep_card:
            kept.append(src)
            continue
        if keep_card and src not in source_cards:
            kept.append(src)              # unattributable: cautious side, see docstring
            continue
        _pactl_suspend(src, False)
    _suspended_sources[:] = kept
    if kept:
        print(f"[mic] pulse stays suspended on the capture card (hw:{keep_card}): "
              f"{', '.join(kept)} — it would re-grab the device before we can open it", flush=True)


# The old name, kept because it is imported by ai/mic_stream.py, ai/voice_assistant.py and
# scripts/wake_test.py, and because "resume the pulse source" is still what those call sites mean.
resume_pulse_source = resume_pulse_sources


def refresh_devices() -> bool:
    """Make PortAudio re-read the ALSA device list. Returns True if it did.

    ⚠ MUST NOT be called while any stream is open. This tears the PortAudio host API down and builds
    it again, which invalidates every open stream — the one safe window is inside MicStream.reopen(),
    between closing the old stream and opening the new one, and that is the only caller.

    It exists because PortAudio snapshots the device list at Pa_Initialize and never refreshes it.
    Everything downstream of that snapshot — sd.query_devices(), the whole of resolve_input_device(),
    and therefore the dashboard's "find the microphone again" button — is blind to a card plugged in
    after startup, for the life of the process. That is not a hot-plug limitation we chose to accept;
    it is why the button could not find a newly plugged mic and why a restart used to be the only way.

    Also drops the speaker-card cache: a re-plug renumbers ALSA cards, so the index resolved before
    the change may name a different card after it."""
    global _speaker_card
    _speaker_card = None
    _speaker_card_logged.clear()
    try:
        with pa_lock:                    # never concurrently with a stream open — see pa_lock
            sd._terminate()
            sd._initialize()
    except Exception as exc:
        # Best-effort like the rest of this module. A failure here means the device list is stale,
        # not that the mic is gone — the caller goes on to open whatever the old list still offers.
        print(f"[mic] WARNING: could not re-scan audio devices ({type(exc).__name__}: {exc}) — "
              f"a mic plugged in since startup may still be invisible", flush=True)
        return False
    return True


def resolve_input_device() -> MicChoice:
    """Find a mic that actually captures signal, preferring the INMP441 I2S mic, then a USB mic,
    then the system default. Returns how to open it. NOTE: for the raw I2S device to probe live,
    pulseaudio must already be suspended (see free_i2s_device) — ensure_input_resolved does this."""
    try:
        devices = sd.query_devices()
    except Exception:
        return MicChoice(None, SAMPLE_RATE, CHANNELS, 0, "int16", False)
    for idx in _candidate_input_devices(devices):
        kind = _classify_device(devices[idx].get("name", ""))
        channels, take_channel, retries = _profile_for(kind)
        advertised = int(devices[idx].get("default_samplerate") or SAMPLE_RATE)
        # Every rate here is one MicStream can actually resample; a device that opens at none of
        # them is skipped rather than returned. Returning an unusable rate is what took the session
        # down on 2026-08-09 — see _capture_rates_for and FALLBACK_CAPTURE_RATES.
        # How many silent reads to forgive, and which channel holds the audio, both come from
        # _profile_for — they differ per device kind and the reasons are recorded there.
        for rate in _capture_rates_for(kind, advertised):
            if _probe_is_live(idx, rate, channels, take_channel, retries):
                print(f"[mic] selected {kind} mic: device {idx} "
                      f"({devices[idx].get('name', '?')}) at {rate} Hz", flush=True)
                return MicChoice(idx, rate, channels, take_channel, "int16", kind == "i2s", kind,
                                 _alsa_card_of(devices[idx].get("name", "")))
    print("[voice_assistant] WARNING: every candidate input device read as silent or refused every "
          "usable rate — falling back to system default mic (recordings may be empty)")
    return MicChoice(None, SAMPLE_RATE, CHANNELS, 0, "int16", False)
