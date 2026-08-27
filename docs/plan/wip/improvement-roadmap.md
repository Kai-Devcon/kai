# Improvement Roadmap

The system evolved through three deliberate phases from v1 (EMA + dead zone) to v2 (PD controller, auto-sleep, pan-tilt). This section documents each phase, what it changed, and why.

### Phase 1 — PD Controller (replaces EMA + dead zone)

**v1 problem:** EMA smoothing delays the signal uniformly regardless of how fast the face moves. The dead zone eliminates jitter but causes the servo to snap abruptly when the threshold is crossed.

**v2 solution:** PD (Proportional-Derivative) controller.

```
target_angle  = nose.x × 180
error         = target_angle − current_angle
correction    = Kp × error + Kd × (error − prev_error)
current_angle = clamp(current_angle + correction, 0, 180)
```

- **P term** (Kp): correction proportional to how far off the servo is — larger offset → faster catch-up
- **D term** (Kd): correction proportional to rate of change — oscillation produces alternating ±d_error → D term cancels it naturally, no dead zone needed

The D term replaces the dead zone: instead of gating commands, it dampens micro-oscillations so small noise produces near-zero net correction, while real motion produces consistent d_error that reinforces the P term.

### Phase 2 — Auto-Sleep + Confidence Gate

**Problem:** When no face is visible the servo holds its last position indefinitely and keeps receiving stale commands. Distant or partially-visible faces produce noisy landmark estimates.

**Auto-sleep:** `SLEEP_AFTER = 3.0s`. No face detected for 3 seconds → servo returns to 90° and stops sending serial. PD state resets. On face reappearance, tracking resumes cleanly from center.

**Confidence gate:** `MIN_FACE_AREA = 0.04`. MediaPipe can detect faces at any distance but landmark accuracy degrades with distance. Bounding box area (normalized, 0–1) is a simple proxy for face confidence. Faces smaller than 4% of frame area are skipped.

### Phase 3 — Pan-Tilt (Y-axis)

**Hardware:** Second SG90 on Arduino Pin 10. Pan-tilt bracket mounts both servos at 90° to each other.

**Serial protocol change:** `"90\n"` → `"pan,tilt\n"`. Arduino sketch handles both formats (backward compatible).

**PD controller:** Independent `PDAxis` instance for tilt. Tracks `nose.y` (0.0=top, 1.0=bottom). `--flip-y` flag inverts tilt direction. If `--tilt` is not set, tilt defaults to 90° and Arduino tilt servo stays centered.

**Enable:**
```bash
python3 -u face_track.py --network <ip> --no-display --flip --tilt
```

### Next Steps (Planned)

> **DONE (2026-07-28).** The section below is kept for the reasoning; both halves shipped. The
> INMP441 I²S MEMS mic and the USB-dongle → PAM8403 → speaker output are wired and configured, the
> jaw is synced to real WAV duration instead of a time-estimated envelope, and Kai is hands-free
> via the "Hey Kai" wake word. See the hands-free changelog entry near the top.
>
> Two details below have since moved. `resolve_input_device()` lives in `ai/mic_device.py`, not
> `ai/voice_assistant.py`. And the prediction that the pipeline would "adopt the new input with
> little/no code change" did not hold: the advertised sample rate turned out to be a hint rather
> than a capability, and a device offering only 44.1 kHz cannot be resampled to 16 kHz by an
> integer-ratio decimator at all — see `rnd/challenges.md`. Both mics are supported side by side
> now, and swapping between them no longer needs a restart (2026-08-27).

**Onboard audio — embedded mic + speaker (cleaner enclosure).** Replace the external USB
mic with an **embedded microphone**, and add an **on-board 3W–5W speaker**. The goal is a
clean, integrated look for Kai — no dongles or cables hanging off the enclosure.

- **Mic:** swap the USB capture device for a built-in mic (e.g. I²S MEMS or an analog
  electret into the audio input). `resolve_input_device()` in `ai/voice_assistant.py`
  already probes candidates for live signal, so the pipeline should adopt the new input
  with little/no code change — just re-verify with
  `python3 -c "import sounddevice as sd; print(sd.query_devices())"`.
- **Speaker:** add a 3W–5W speaker so Kai can actually talk out loud. This unblocks the
  TTS/speaker output phase noted above (currently text-only) — the jaw "speaking" pantomime
  in `voice_assistant.py` can then be synced to real audio playback instead of a
  time-estimated envelope.
- **Why:** removes external USB peripherals for a tidier, self-contained build and gives Kai
  a real voice.
