# Challenges & How They Were Overcome

### 1. Jetson GPIO is capped at 3.3V — servo needs 5V

**Problem:** All 40 GPIO pins on the Jetson Orin Nano output 3.3V logic. The SG90 servo tested required ~5V signal to respond. At 3.3V, the servo electronics marginally detected the signal but the motor lacked torque to actually move.

**Attempted first:** Direct 3.3V PWM from Jetson Pin 33. Verified with sysfs (`/sys/class/pwm/pwmchip2/pwm0`) that the signal was correct (1500µs duty, 50Hz). Confirmed signal was present via Arduino acting as a reader — it read ~535µs instead of the expected 1500µs. This short reading caused the servo to drive to minimum position and buzz.

**Root cause of the 535µs reading:** The Arduino's digital HIGH threshold is ~3.0V. The 3.3V signal was only marginally above threshold, causing early false-low edge detection in `pulseIn()`.

**Final solution:** Arduino as transparent serial bridge. Jetson sends angle values (`"90\n"`) over USB serial. Arduino receives and calls `sg90.write(angle)` — native 5V PWM. No signal level translation needed.

---

### 2. Arduino not detected on Jetson (`/dev/ttyUSB0` missing)

**Problem:** The CH340 USB-serial chip on the Arduino clone uses the `ch341` kernel module, which is **not compiled into NVIDIA's Jetson tegra kernel** (5.15.148-tegra).

**Attempted first:** `sudo modprobe ch341` → "Module not found". The module simply doesn't exist in the Jetson kernel image.

**Also blocking:** `brltty` (Braille accessibility daemon) had a udev rule (`/etc/udev/rules.d/85-brltty.rules`) that claimed the CH340 USB device before `ch341` could bind it.

**Solution:**
1. Remove brltty: `sudo apt remove -y brltty`
2. Download `ch341.c` from the matching Linux 5.15 kernel source
3. Build `ch341.ko` against Jetson kernel headers at `/usr/src/linux-headers-5.15.148-tegra-ubuntu22.04_aarch64/`
4. Load: `sudo modprobe usbserial && sudo insmod ch341.ko`
5. Manual bind if needed: `echo "1-2.x:1.0" | sudo tee /sys/bus/usb/drivers/ch341/bind`
6. Persistent auto-bind via udev rule in `install_udev.sh`

---

### 3. Compiling Arduino sketches without Arduino IDE on Jetson

**Problem:** The Jetson has `avr-gcc` and `avrdude` installed (from the `arduino` package) but the Arduino IDE is unavailable or impractical. Building `.ino` files manually has many pitfalls.

**Issues encountered during manual build:**
- `avr-gcc-ar`: LTO (Link Time Optimization) plugin error → removed `-flto` entirely
- `Arduino.h` not found in sketch → prepend `#include <Arduino.h>` before compilation
- `.S` assembly files (`wiring_pulse.S`) not compiled → added `.S` compilation loop
- Duplicate symbol `wiring_pulse.o` from both `.c` and `.S` → renamed: `core_*.o` for C/C++, `core_*_asm.o` for `.S`
- `pulseIn` undefined reference → linking `core.a` archive had resolution issues → switched to linking all `.o` files directly

**Solution:** `ch341_build/build_servo_serial.sh` — a self-contained build script that compiles Arduino core + Servo library + sketch and uploads via avrdude, all from the Jetson CLI.

---

### 4. Faulty servo

**Problem:** The first SG90 servo showed puzzling behavior — it would buzz and resist movement but not rotate. This consumed significant debugging time suspecting wiring, voltage, or code issues.

**Diagnosis process:**
- Direct Arduino 5V sweep → servo moved ✓ (hardware path works)
- Arduino serial → servo still not moving ✗
- Manual push: servo felt stiff and buzzed when forced → motor was energized
- The servo was holding a position but completely ignoring angle changes

**Root cause:** The servo was internally faulty. The control circuit was partially working (could hold/power the motor) but the potentiometer or gearing was damaged — it could not actually drive rotation.

**Key lesson:** Test the physical servo with a simple standalone sweep sketch (`servo_standalone.ino`) before debugging any software. If it doesn't move during standalone sweep, it's hardware.

---

### 5. MediaPipe on Jetson ARM64 (aarch64)

**Problem:** Standard `mediapipe` pip package does not always install cleanly on Jetson's ARM64 architecture.

**Finding:** `mediapipe 0.10.18` installs and runs correctly on the Jetson Orin Nano via pip. The `XNNPACK` delegate is used automatically (CPU-optimized). Processing 320×240 frames achieves **35–45fps** in practice (with full LOFI face param capture including solvePnP head pose; ~40–50fps without).

**Optimization:** `refine_landmarks=False` skips iris and detailed mesh refinement — not needed for nose tip tracking — and meaningfully reduces per-frame CPU time.

---

### 6. Servo jitter from face detection noise

**Problem:** MediaPipe's nose tip landmark fluctuates by ~±2° even when the face is perfectly still (natural detection noise). With a small dead zone (1°) and high EMA alpha (0.7), every frame produced a new servo command, causing continuous micro-twitching.

**Solution:** Tuned two parameters together:
- `EMA_ALPHA = 0.3` — heavy low-pass filter; requires several frames of consistent signal to change the output
- `DEAD_ZONE_DEG = 8` — only dispatch a serial command when the smoothed angle changes by more than 8°

This eliminates jitter when the face is stationary while still tracking deliberate head movements.

### 7. "The mic is not being detected" — a working mic, rejected on arithmetic

**Problem:** After a boot on 2026-08-09 Kai was deaf. `sess_state` sat on `disabled`, `sess_mic_live` was `false`, `sess_wake_tried` was empty, and push-to-talk answered *"didn't catch that"*. The log:

```
[mic] device 5 read as silent (rms=0.0 <= 5.0)
[mic] resolved device=0 rate=44100 ch=1 i2s=False — opening stream…
[mic] ERROR: cannot resample 44100 Hz: decimation needs an integer ratio, got 44100 -> 16000
[face_track] WARNING: shared capture unavailable after 14 attempts — falling back to per-turn
```

Both microphones were fine the entire time. `arecord` on the raw I2S device returned a strong signal, and the USB dongle probed live on the first try.

**Two faults, and the second is what made it total:**

1. The INMP441 read as exact digital silence on the boot probe. It is a warm-up race, not a fault — the same device on the same route reads rms 124–435 when probed a second later, and `arecord` never saw silence at all. But one bad 0.3 s read condemned the preferred mic for the whole life of the process.
2. The USB fallback was **unusable by construction**. `resolve_input_device()` returned the rate ALSA advertises (44100), and `MicStream` resamples with an integer-ratio decimator, so `Decimator(44100 → 16000)` raised, `MicStream.open()` returned `False`, and `ConversationSession.start()` returned `False`. No capture stream at all — which is why a *silent* mic presented as *no* mic.

The advertised rate was never a capability. `arecord -D hw:0,0 --dump-hw-params` reports `S16_LE mono, RATE: [44100 48000]`: 16 kHz cannot be opened on that dongle and 44100 cannot be resampled. **48000 was available and usable the whole time**, and nothing in the advertised rate said so.

**Solution:**
- `FALLBACK_CAPTURE_RATES` (`config/voice.py`) — non-I2S devices are only ever offered rates that divide into `SAMPLE_RATE`, and the liveness probe, which opens the device for real, decides which one the hardware accepts. A device that opens at none of them is skipped rather than returned; returning it is the failure above.
- `I2S_PROBE_SILENT_RETRIES` — a *silent* I2S read is retried a few times before the mic is written off. Only silence is retried; a device that refuses to open or hangs has given a definite answer, and re-asking would multiply `LIVE_PROBE_TIMEOUT_S` on the session start path.

**The lesson worth keeping:** `default_samplerate` is a hint, not a capability, and the only honest test of a capture device is opening it. The rate the driver advertised was the one rate the pipeline could not use.

**Since then (2026-08-27).** The word "fallback" in fault 2 has stopped being accurate — a USB mic is now a supported input in its own right, with `MIC_PREFERENCE` deciding which kind is probed first and hot-plug switching between them without a restart. Two things this entry did not surface came out of that work. The guard that stops Kai capturing the speaker's own card matched on the device *name*, and `"USB Audio Device"` is what most cheap USB mics enumerate as, so on some hardware the fallback was not merely unusable — it was discarded before it was ever probed, and silently. And the last log line above (`falling back to per-turn`) is a dead end that no amount of retrying escapes: once startup gives up there is no tick loop, so nothing was watching for the mic to be plugged in, which is the state a deaf robot spends all its time in. Both are fixed; the arithmetic lesson above is unchanged and is still the load-bearing one.
