#!/usr/bin/env python3
"""On-device diagnostic for the hands-free audio path. Prints levels, VAD decisions and wake hits.

This is the tool for setting VAD_RMS_FLOOR and WAKE_SENSITIVITIES on a headless Jetson. Doing it by
watching face_track.py's log means fighting the tracking loop, the servo chatter and the dashboard at
the same time; here nothing else runs.

  # 1. What is the room's noise floor? Sit quiet, then with the fan and amp powered.
  python3 scripts/wake_test.py --seconds 20

  # 2. Does the wake word fire reliably? Say "Hey Kai" 20 times at 1 m, then at 3 m.
  python3 scripts/wake_test.py --seconds 120

  # 3. Does it false-accept? Leave a podcast playing and walk away.
  python3 scripts/wake_test.py --seconds 600 --quiet

  # 4. Is the resampled audio actually clean? Listen for periodic clicking.
  python3 scripts/wake_test.py --seconds 8 --save /tmp/mic_check.wav
  paplay /tmp/mic_check.wav

  # 5. Pin one tier of the fallback chain instead of taking whichever initializes first.
  python3 scripts/wake_test.py --engine openwakeword --seconds 120   # prints the live score
  python3 scripts/wake_test.py --engine whisper --seconds 120        # prints each transcript + match

Set VAD_RMS_FLOOR to roughly 3x the loudest idle RMS you see, then confirm speech reads at least 5x
the floor. IMPORTANT: this cannot run at the same time as face_track.py — the raw I2S hw device
admits exactly one opener, so stop the service first.
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.audio import Decimator, FrameAssembler, SpeechGate, WakeDetector, rms   # noqa: E402
from ai.voice_assistant import (                                                # noqa: E402
    apply_i2s_route, free_i2s_device, resolve_input_device, resume_pulse_source,
)
from ai.wake_phrase import match_wake_phrase                                    # noqa: E402
from config.voice import SAMPLE_RATE                                            # noqa: E402
from config.wake import (                                                       # noqa: E402
    CAPTURE_BLOCKSIZE, MIC_INPUT_GAIN, VAD_RMS_FLOOR, WAKE_REFRACTORY_S,
    WAKE_WHISPER_MAX_UTTERANCE_S, WAKE_WHISPER_MIN_UTTERANCE_S,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=20.0, help="how long to listen")
    ap.add_argument("--save", metavar="WAV", help="also write the 16 kHz mono stream here")
    ap.add_argument("--quiet", action="store_true",
                    help="only print wake hits and speech events, not the level line")
    ap.add_argument("--floor", type=float, default=VAD_RMS_FLOOR,
                    help=f"override VAD_RMS_FLOOR for this run (config: {VAD_RMS_FLOOR})")
    ap.add_argument("--no-wake", action="store_true", help="skip the wake engine, levels only")
    ap.add_argument("--engine", choices=("auto", "porcupine", "openwakeword", "whisper"),
                    default="auto",
                    help="pin one tier of the fallback chain instead of taking the first that "
                         "initializes (auto). 'whisper' is utterance-level, so this script reports "
                         "matches on the transcript rather than per-frame hits.")
    args = ap.parse_args()

    import sounddevice as sd   # imported late so --help works without the audio stack

    # Same sequence as VoiceAssistant.ensure_input_resolved(): set up the XBAR route, take the card
    # off pulse so the raw hw probe can open it at 48 kHz, and hand pulse back if we end up
    # elsewhere. resolve_input_device() never returns None — it falls back to device=None, meaning
    # "the system default", which is the case that usually reads as digital silence.
    apply_i2s_route()
    free_i2s_device()
    mic = resolve_input_device()
    if not mic.is_i2s:
        resume_pulse_source()
    print(f"[wake_test] mic: device={mic.device} kind={mic.kind} rate={mic.rate} "
          f"channels={mic.channels} take_channel={mic.take_channel} i2s={mic.is_i2s}", flush=True)
    if mic.device is None:
        print("[wake_test] WARNING: fell back to the system default mic — the INMP441 was not "
              "found or read silent. Check the APE card and the I2S route.", flush=True)

    # Only the raw I2S device is rate-locked; USB/default go through pulse's plughw, which
    # resamples for us, so we ask for 16 kHz directly and skip the decimator entirely.
    decim = Decimator(mic.rate, SAMPLE_RATE, gain=MIC_INPUT_GAIN) if mic.rate != SAMPLE_RATE else None
    if decim:
        print(f"[wake_test] decimating {mic.rate} -> {SAMPLE_RATE} (ratio {decim.ratio})", flush=True)

    wake = WakeDetector(force=None if args.engine == "auto" else args.engine)
    if not args.no_wake and not wake.open():
        print("[wake_test] continuing without a wake engine — levels and VAD only", flush=True)
    # AFTER open(): the winning tier decides the frame size (512 for Porcupine, 1280 for
    # openWakeWord). Sizing this beforehand is the bug that made openWakeWord silently never fire.
    wake_frames = FrameAssembler(max(1, wake.frame_length))
    scan_tier = wake.kind == "utterance"
    if scan_tier:
        print(f"[wake_test] engine '{wake.engine}' is utterance-level: each speech segment is "
              f"transcribed and matched, so expect ~0.4-1.0s per check", flush=True)
    gate = SpeechGate(rate=SAMPLE_RATE, rms_floor=args.floor)
    if not gate.vad_available:
        print("[wake_test] WARNING: webrtcvad not installed — speech detection is the RMS floor "
              "alone. Run: pip3 install webrtcvad || pip3 install webrtcvad-wheels", flush=True)

    print(f"[wake_test] floor={args.floor:.0f}  listening for {args.seconds:.0f}s  "
          f"(ctrl-c to stop)\n", flush=True)

    saved: list[np.ndarray] = []
    capture: list[np.ndarray] = []      # utterance tier only
    stats = {"blocks": 0, "wakes": 0, "onsets": 0, "hangovers": 0, "overflows": 0,
             "checks": 0, "matches": 0, "skipped": 0}
    whisper_model = [None]             # lazily loaded, only for --engine whisper

    def _run_scan_check(audio, spoken_s, elapsed, stats) -> None:
        """Transcribe one candidate utterance and report whether the wake phrase was in it.

        Deliberately synchronous: this is a tuning tool, and interleaving would make the printed
        timings meaningless. It does mean blocks are dropped during the check — that is fine here and
        is exactly why the real session does this on a worker thread."""
        if spoken_s < WAKE_WHISPER_MIN_UTTERANCE_S or spoken_s > WAKE_WHISPER_MAX_UTTERANCE_S:
            stats["skipped"] += 1
            print(f"  [{elapsed:6.2f}s] skipped ({spoken_s:.2f}s outside "
                  f"{WAKE_WHISPER_MIN_UTTERANCE_S}-{WAKE_WHISPER_MAX_UTTERANCE_S}s)", flush=True)
            return
        if whisper_model[0] is None:
            from faster_whisper import WhisperModel
            from config.voice import WHISPER_COMPUTE, WHISPER_CPU_THREADS, WHISPER_DEVICE, WHISPER_MODEL
            print(f"  loading whisper ({WHISPER_MODEL}/{WHISPER_COMPUTE})…", flush=True)
            kw = {"cpu_threads": WHISPER_CPU_THREADS} if WHISPER_CPU_THREADS else {}
            whisper_model[0] = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE,
                                            compute_type=WHISPER_COMPUTE, **kw)
        t0 = time.monotonic()
        segments, _info = whisper_model[0].transcribe(
            audio.astype(np.float32) / 32768.0, language=None, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        ms = int((time.monotonic() - t0) * 1000)
        stats["checks"] += 1
        match = match_wake_phrase(text)
        if match is None:
            print(f"  [{elapsed:6.2f}s] check: {spoken_s:.1f}s -> {text!r} ({ms}ms) no match",
                  flush=True)
        else:
            stats["matches"] += 1
            print(f"  [{elapsed:6.2f}s] check: {spoken_s:.1f}s -> {text!r} ({ms}ms) "
                  f"MATCH {match.score:.2f} cmd={match.command!r}", flush=True)
    peak_rms, idle_peak = 0.0, 0.0
    last_wake_t = -WAKE_REFRACTORY_S
    last_print = 0.0
    start = time.monotonic()

    def on_block(indata, frames, time_info, status) -> None:
        nonlocal peak_rms, idle_peak, last_wake_t, last_print
        if status and status.input_overflow:
            stats["overflows"] += 1
        now = time.monotonic()
        stats["blocks"] += 1

        mono = indata[:, mic.take_channel:mic.take_channel + 1].copy().ravel()
        pcm = decim.feed(mono) if decim else mono.astype(np.int16)
        if args.save:
            saved.append(pcm)

        level = rms(pcm)
        peak_rms = max(peak_rms, level)

        if wake.frame_ready:
            for frame in wake_frames.push(pcm):
                hit = wake.process(frame)
                if hit and now - last_wake_t >= WAKE_REFRACTORY_S:
                    last_wake_t = now
                    stats["wakes"] += 1
                    score = f" score={wake.last_score:.2f}" if wake.last_score else ""
                    print(f"  [{now - start:6.2f}s] *** WAKE *** "
                          f"(#{stats['wakes']} rms={level:.0f}{score})", flush=True)

        if scan_tier:
            capture.append(pcm)

        event = gate.update(pcm, now)
        if event == "onset":
            stats["onsets"] += 1
            if scan_tier:
                del capture[:-int(0.5 * SAMPLE_RATE // len(pcm) + 1)]   # keep a short pre-roll
            print(f"  [{now - start:6.2f}s] speech onset (rms={gate.last_rms:.0f})", flush=True)
        elif event == "hangover":
            stats["hangovers"] += 1
            spoken = gate.speech_duration(now)
            print(f"  [{now - start:6.2f}s] speech end   (duration={spoken:.2f}s)", flush=True)
            if scan_tier:
                _run_scan_check(np.concatenate(capture) if capture else np.zeros(0, np.int16),
                                spoken, now - start, stats)
                capture.clear()
        if gate.state == SpeechGate.IDLE:
            idle_peak = max(idle_peak, level)

        if not args.quiet and now - last_print >= 0.5:
            last_print = now
            bar = "#" * min(40, int(level / max(1.0, args.floor) * 10))
            print(f"  [{now - start:6.2f}s] rms={level:7.0f} "
                  f"{'SPEECH' if gate.state == SpeechGate.SPEECH else '      '} |{bar}", flush=True)

    try:
        with sd.InputStream(samplerate=mic.rate, channels=mic.channels, dtype=mic.dtype,
                            device=mic.device, blocksize=CAPTURE_BLOCKSIZE, callback=on_block):
            deadline = start + args.seconds
            while time.monotonic() < deadline:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[wake_test] interrupted", flush=True)
    except Exception as exc:
        print(f"[wake_test] could not open the microphone: {exc}", flush=True)
        return 1
    finally:
        wake.close()

    elapsed = time.monotonic() - start
    print(f"\n[wake_test] engine: {wake.engine or 'none'} ({wake.kind or 'n/a'}, "
          f"{wake.frame_length} samples)")
    if wake.tiers:
        for name, why in wake.tiers.items():
            print(f"[wake_test]   {name:14s} {why}")
    print(f"[wake_test] {elapsed:.1f}s, {stats['blocks']} blocks, "
          f"{stats['overflows']} input overflows")
    print(f"[wake_test] wake hits: {stats['wakes']}   speech onsets: {stats['onsets']}   "
          f"turns ended: {stats['hangovers']}")
    if scan_tier:
        print(f"[wake_test] whisper checks: {stats['checks']}   matches: {stats['matches']}   "
              f"skipped (outside the length band): {stats['skipped']}")
    print(f"[wake_test] peak rms: {peak_rms:.0f}   peak while idle: {idle_peak:.0f}   "
          f"floor: {args.floor:.0f}")
    if idle_peak >= args.floor:
        print(f"[wake_test] ^ the idle peak is AT OR ABOVE the floor — raise VAD_RMS_FLOOR to "
              f"about {idle_peak * 3:.0f} or the VAD will trip on the room itself")
    elif peak_rms < args.floor * 5:
        print("[wake_test] ^ speech never reached 5x the floor — either speak up, raise "
              "MIC_INPUT_GAIN, or lower the floor")
    else:
        print(f"[wake_test] ^ headroom looks healthy "
              f"(speech peaks at {peak_rms / max(1.0, args.floor):.1f}x the floor)")
    if stats["overflows"]:
        print("[wake_test] ^ input overflows mean the callback couldn't keep up — raise "
              "CAPTURE_BLOCKSIZE")

    if args.save and saved:
        pcm = np.concatenate(saved)
        with wave.open(args.save, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())
        print(f"[wake_test] wrote {len(pcm) / SAMPLE_RATE:.1f}s to {args.save} — play it back and "
              f"listen for periodic clicking, which would mean the resampler lost its state")
    return 0


if __name__ == "__main__":
    sys.exit(main())
