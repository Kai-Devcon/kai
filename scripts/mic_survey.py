#!/usr/bin/env python3
"""Survey every capture device on this machine and say what Kai would do with each.

Answers the questions you cannot answer from a product photo or a listing, and the ones that have
cost this project real time:

  * What does the device ACTUALLY enumerate as? The name is the only thing mic selection can key
    on (ANALOG_MIC_NAME_HINTS, USB_MIC_NAME_HINTS), and adapter names are not predictable.
  * What rates can it REALLY open? `default_samplerate` is a hint, not a capability. On 2026-08-09 a
    working USB mic advertised 44100 — the one rate the pipeline cannot resample — while 48000 was
    available the whole time and nothing said so. That took the whole voice session down. This asks
    `arecord --dump-hw-params`, which is the honest answer.
  * Which ALSA card is the speaker on, and does that exclude this device? Raw capture on the card
    playback reconfigures segfaulted the process at the startup greeting (2026-08-11), so that card
    is dropped. With two identically-named USB dongles, WHICH one gets excluded is the question.
  * Does a pulse-mediated capture still work while Kai has every pulse source suspended? Kai
    suspends them before probing (free_i2s_device), so a device that only works through pulse may
    probe as silent for that reason alone. `--suspend-pulse` measures it instead of guessing.

READ-ONLY BY DEFAULT. Plain `mic_survey.py` opens no device and changes no system state; it only
reads what is already there. `--probe` opens devices for a fraction of a second each.
`--suspend-pulse` DOES change state (it is the point) and restores it in a finally block.

    python3 scripts/mic_survey.py                      # survey only, touches nothing
    python3 scripts/mic_survey.py --probe              # also open each candidate and measure RMS
    python3 scripts/mic_survey.py --probe --suspend-pulse   # ...as Kai actually sees it

Run it on the robot with everything plugged in. It prints one block per device plus a verdict.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.voice import (                                              # noqa: E402
    ANALOG_MIC_NAME_HINTS, FALLBACK_CAPTURE_RATES, I2S_MIC_NAME_HINTS, SAMPLE_RATE,
    SPEAKER_CARD_NAME_HINTS, TTS_CARD, TTS_SINK, USB_MIC_NAME_HINTS,
)

TIMEOUT_S = 6.0


def sh(*cmd: str) -> str:
    """Run a command and return stdout, or "" if it is unavailable. Never raises."""
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True,
                              timeout=TIMEOUT_S).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def hw_params(card: str) -> tuple[str, list[int]]:
    """`arecord --dump-hw-params` on a raw device: the rates the hardware really admits.

    This is the whole reason the script exists. Returns (raw text, parsed rates); an empty list
    means arecord could not tell us, NOT that the device has no rates."""
    out = sh("arecord", "-D", f"hw:{card},0", "--dump-hw-params", "-d", "1")
    if not out:
        # arecord writes hw_params to stderr on some builds, and exits non-zero either way.
        try:
            out = subprocess.run(["arecord", "-D", f"hw:{card},0", "--dump-hw-params", "-d", "1"],
                                 capture_output=True, text=True, timeout=TIMEOUT_S).stderr
        except (OSError, subprocess.TimeoutExpired):
            return "", []
    rates: list[int] = []
    m = re.search(r"^RATE:\s*(.+)$", out, re.M)
    if m:
        field = m.group(1)
        rates = [int(n) for n in re.findall(r"\d+", field)]
        # "[44100 48000]" is a list; "[8000 48000]" from a bracketed RANGE means anything between,
        # so say so rather than pretending the two endpoints are the only options.
        if "[" in field and len(rates) == 2 and rates[1] - rates[0] > 4000:
            rates = [r for r in (16000, 32000, 48000) if rates[0] <= r <= rates[1]]
    return out, rates


def pulse_card_index(name: str) -> str:
    """The ALSA card index behind a pulse card name, or "" — the same lookup ai/mic_device does."""
    out = sh("pactl", "list", "cards")
    in_card = False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Card #"):
            in_card = False
        elif s.startswith("Name:"):
            in_card = s.split(":", 1)[1].strip() == name
        elif in_card and s.startswith("alsa.card ") and "=" in s:
            return s.split("=", 1)[1].strip().strip('"')
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true",
                    help="open each candidate briefly and measure RMS (still changes no config)")
    ap.add_argument("--suspend-pulse", action="store_true",
                    help="replicate Kai's pulse suspend before probing, then restore it. CHANGES "
                         "SYSTEM STATE while it runs")
    args = ap.parse_args()

    import sounddevice as sd
    from ai import mic_device
    from ai.mic_device import (
        _classify_device, _is_speaker_card, _profile_for, _probe_is_live, _capture_rates_for,
    )

    print("=" * 78)
    print("PULSE / SPEAKER")
    print("=" * 78)
    sinks = sh("pactl", "list", "short", "sinks")
    cards = sh("pactl", "list", "short", "cards")
    srcs = sh("pactl", "list", "short", "sources")
    if not (sinks or cards):
        print("  pactl unavailable — pulse is not running, or not reachable from this shell.")
        print("  NOTE: mic selection then falls back to matching SPEAKER_CARD_NAME_HINTS")
        print(f"        {SPEAKER_CARD_NAME_HINTS} against the device NAME. Any device whose name")
        print("        matches is excluded — including a second, identically-named dongle.")
    print(f"  cards:\n    " + "\n    ".join(cards.strip().splitlines() or ["(none)"]))
    print(f"  sinks:\n    " + "\n    ".join(sinks.strip().splitlines() or ["(none)"]))
    print(f"  sources:\n    " + "\n    ".join(srcs.strip().splitlines() or ["(none)"]))

    print(f"\n  TTS_SINK configured : {TTS_SINK}")
    print(f"    exists?           : {'YES' if TTS_SINK in sinks else 'NO  <-- replies are silent'}")
    print(f"  TTS_CARD configured : {TTS_CARD}")
    print(f"    exists?           : {'YES' if TTS_CARD in cards else 'NO'}")
    speaker_card = pulse_card_index(TTS_CARD)
    print(f"    ALSA card index   : {speaker_card or '(unresolved -> name-matching fallback)'}")
    if speaker_card:
        print(f"    => input devices on hw:{speaker_card} will NOT be captured (2026-08-11 segfault)")

    dupes = [ln for ln in cards.splitlines() if "usb" in ln.lower()]
    if len(dupes) > 1:
        print("\n  ** MORE THAN ONE USB AUDIO CARD **  Confirm the one above is the dongle with the")
        print("     amp and speaker on it. These names are not guaranteed stable across reboots or")
        print("     USB ports, and if TTS_CARD names the wrong one, two things break at once:")
        print("     replies go to the wrong jack, AND the exclusion protects the wrong card.")

    print()
    print("=" * 78)
    print("CAPTURE DEVICES (as PortAudio sees them)")
    print("=" * 78)

    restore = False
    try:
        if args.suspend_pulse:
            print("  suspending every pulse capture source, exactly as MicStream.open() does...")
            mic_device.free_i2s_device()
            restore = True

        try:
            devices = sd.query_devices()
        except Exception as exc:
            print(f"  cannot query PortAudio: {exc}")
            return 1

        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) <= 0:
                continue
            name = dev.get("name", "")
            kind = _classify_device(name)
            excluded = _is_speaker_card(name)
            m = re.search(r"hw:([^,\)]+)", name)
            card = m.group(1) if m else ""
            advertised = int(dev.get("default_samplerate") or 0)

            print(f"\n  [{idx}] {name!r}")
            print(f"       card={card or '(not a hw: device — pulse-mediated)'}  "
                  f"in={dev.get('max_input_channels')}  out={dev.get('max_output_channels')}")
            print(f"       Kai calls this kind={kind}"
                  + ("   ** EXCLUDED: this is the speaker's card **" if excluded else ""))
            print(f"       advertises {advertised} Hz  (a hint, NOT a capability)")

            if card:
                _, real = hw_params(card)
                if real:
                    usable = [r for r in real if r % SAMPLE_RATE == 0]
                    print(f"       arecord says rates: {real}")
                    if usable:
                        print(f"       usable by Kai     : {usable}  (must divide into {SAMPLE_RATE})")
                    else:
                        print(f"       usable by Kai     : NONE  <-- UNUSABLE. No rate divides into")
                        print(f"                           {SAMPLE_RATE}; the decimator needs an integer")
                        print(f"                           ratio. This is the 2026-08-09 failure.")
                else:
                    print("       arecord could not report hw params (device busy, or no arecord)")

            if args.probe and not excluded:
                channels, take, _ = _profile_for(kind)
                for rate in _capture_rates_for(kind, advertised):
                    live = _probe_is_live(idx, rate, channels, take, 0)
                    print(f"       probe @ {rate:6} Hz x{channels} ch{take}: "
                          f"{'LIVE' if live else 'silent / would not open'}")
                    if live:
                        break
    finally:
        if restore:
            print("\n  restoring pulse sources...")
            mic_device.resume_pulse_sources()

    print()
    print("=" * 78)
    print("WHAT KAI WOULD PICK")
    print("=" * 78)
    print("  (the same call the robot makes; with --suspend-pulse above it has already been")
    print("   restored, so this line reflects pulse being UP)")
    try:
        choice = mic_device.resolve_input_device()
        print(f"  {choice}")
        if choice.device is None:
            print("  device=None means the pulse-mediated system default — Kai found no named mic.")
    except Exception as exc:
        print(f"  resolve_input_device() raised: {type(exc).__name__}: {exc}")

    print("\n  hints currently in play:")
    print(f"    I2S    : {I2S_MIC_NAME_HINTS}")
    print(f"    ANALOG : {ANALOG_MIC_NAME_HINTS}")
    print(f"    USB    : {USB_MIC_NAME_HINTS}")
    print(f"    SPEAKER: {SPEAKER_CARD_NAME_HINTS}  (name fallback only)")
    print(f"    rates  : {FALLBACK_CAPTURE_RATES}")
    print("\n  A device reported as kind=usb that you consider a 3.5mm adapter is only a LABEL")
    print("  miss — it still works. Add its name to ANALOG_MIC_NAME_HINTS in config/voice.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
