# FAQ

**Q: Why not just use the Jetson's 5V pins to power the servo signal directly?**
The Jetson 40-pin header's 5V pins (Pin 2, Pin 4) are power rails only — they cannot be used as GPIO signal outputs. GPIO pins are separate and are all 3.3V.

**Q: Why do we need the Arduino? Can't the Jetson drive the servo directly?**

> **TL;DR:** The Jetson outputs 3.3V max. The SG90 is unreliable below ~4.8V power and the signal detection is marginal at 3.3V. The Arduino outputs 5V and handles all PWM timing — the Jetson just sends a number over USB.

The Jetson Orin Nano's 40-pin GPIO outputs **3.3V maximum** on every pin — this is a hard silicon limit confirmed by NVIDIA's official documentation. The SG90 servo's rated operating voltage is **4.8V–6.0V** (per datasheet). At 3.3V two problems occur:

1. **Marginal signal detection** — the ATmega328 on the Arduino has a digital HIGH threshold of **3.0V** (= 0.6 × VCC at 5V per datasheet). A 3.3V Jetson signal only gives 0.3V of noise margin, which caused false-low edge detection during R&D. A correct 1500µs PWM signal from the Jetson was read as ~535µs by `pulseIn()`.
2. **Unreliable across servo units** — some SG90s accept a 3.3V signal when powered at 5V; others don't. The behavior is unit-dependent and not guaranteed by the datasheet. The unit tested in this R&D did not respond reliably.

Alternatives if you want to avoid the Arduino:

| Option | Notes |
|--------|-------|
| **Arduino Uno (current)** | USB-powered, 5V GPIO, `Servo` library handles PWM timing. Jetson just sends an angle over serial — no PWM code needed. |
| **Logic level shifter (BSS138)** | Converts 3.3V → 5V signal. Cheap (~$1), no extra microcontroller. But Jetson still needs to generate PWM via `lgpio`/sysfs — more complex, CPU-dependent. |
| **5V-tolerant servo** | Some servos work with 3.3V signal if powered at 5V. Unit-dependent — not guaranteed for SG90. |
| **Raspberry Pi** | GPIO is also 3.3V — same problem. |
| **ESP32** | GPIO is also 3.3V — same problem. |

The Arduino was chosen because it was already available and eliminates PWM complexity entirely.

**Q: Why does face_track.py not import from face-detection-movements?**
`face-detection-movements` has Bluetooth (BLE) dependencies (`bleak`, `dbus-fast`, `bluez-peripheral`) that are complex to install and not needed here. `face_track.py` is self-contained — it reimplements only the camera and network receiver code it needs (~60 lines).

**Q: The servo moves but in the wrong direction. How do I fix it?**
Add `--flip` for pan, `--flip-y` for tilt. These invert the nose coordinate before mapping to angle.

**Q: What happens when no face is detected?**
The control thread enters its *hold* branch: it stops sending commands entirely and keeps the PD synced to the position it is holding, so re-acquiring a face glides from where the head really is instead of snapping. Because nothing is being sent, the firmware's 4 s idle-detach relaxes the servos on its own — they stop buzzing and stop drawing current. There is no "return to 90°" any more; the head stays where it last saw someone. A single dropped detection does not trigger any of this: the target is only told the face is gone after `SERVO_ABSENCE_FRAMES`, because MediaPipe flickers at the `MIN_FACE_AREA` boundary and a flip in and out of hold used to cost a visible twitch.

**Q: How do I enable Y-axis (tilt) tracking?**
Everything above the firmware is already there — `--tilt`, the EMA, the PD target, the wire protocol's second field and the dashboard reading. What is missing is the firmware: `servo_serial.ino` declares `TILT_PIN = 10` and never attaches a servo to it, so a tilt field is parsed and discarded. Wiring a second SG90 to pin 10 is therefore not enough; the sketch needs the attach, the limits and the idle-detach handling too. Before doing it, read the shared-rail note in [hardware.md](hardware.md) — a second servo changes the current budget that `SEND_INTERVAL` and `PAN_MAX_STEP` were set against. See [ticket R10](tickets/R10-tilt-axis-plumbed-without-hardware.md).

**Q: The Arduino isn't detected after replug.**
Run `sudo bash hardware/fix_usb.sh` from the project root. For permanent auto-binding: `sudo bash hardware/install_udev.sh` (run once). In normal operation you should not need either — `servo/servo.py` loads the driver on demand when the port is missing, and `scripts/run.sh` does the same before launching.

**Q: How do I verify the Arduino is receiving commands?**
```bash
python3 -m servo.servo_serial --sweep
```
Watch the servo, not the console: the firmware is **fire-and-forget and no longer echoes**. It used to reply `OK:<angle>` per command; that was removed so a write costs no round-trip, which is also why `send()` returns whether it *wrote*, not whether the Arduino acted. If the sweep moves the horn, the link is good.

**Q: Why can't Kai hear me while it is talking?**
Because there is no acoustic echo cancellation, so voice barge-in is deliberately off — the mic is gated shut for the whole time Kai's own audio could reach it, plus a settle tail after playback ends. That is what stops the robot answering itself. It is also why replies are length-capped (`TTS_MAX_SPOKEN_CHARS`, `OLLAMA_NUM_PREDICT`): a long reply is a proportionally long deaf spell. The dashboard's mic button always takes precedence and *can* interrupt a reply.

**Q: Can I use a USB microphone, or a 3.5mm one, instead of the built-in one?**
Yes to both, and you usually do not have to do anything: plug it in and Kai switches to it within a
few seconds. Every mic is probed on every resolve and the first one that captures real signal wins.
A 3.5mm mic needs a **separate USB→3.5mm adapter** — the Jetson has no usable analog input of its
own, and the mic jack on the dongle driving the speaker is deliberately not usable (see the caveat
below). Read the three buying gotchas in `docs/hardware.md` first; the common one is that a
44.1 kHz-only adapter cannot be used at all.
Pick deliberately with **Prefer** on the dashboard's Microphone card (`MIC_PREFERENCE` in
`config/voice.py`), then press *Find the microphone again* — that setting only takes effect the next
time a mic is resolved, which is what the button does. It reorders and never restricts: preferring
one mic still falls back to the others if it is not live, so the setting cannot leave Kai deaf.
`sess_mic_kind` on `/params` says which one he is actually on — `i2s`, `usb`, `analog` or `other`.

One caveat worth knowing before you buy: the USB dongle that drives the speaker is *also* an input
device, and capturing it raw while playback reconfigures the same card segfaulted the process once
(2026-08-11), so that card is excluded. It is identified by ALSA card index, not by name, so a
separate USB mic reporting the same generic "USB Audio Device" name is fine. Where `pactl` cannot be
reached to make that identification, the exclusion falls back to matching the name — and on that
path a mic with that name is skipped. `docs/hardware.md` has the detail.

**Q: Kai is not picking up the microphone I just plugged in. What now?**
Check `curl -s localhost:8081/params | jq .sess_mic_hotplug_watching` first. If it is `false`, the
hot-plug watcher could not read `/proc/asound/cards` and mics are only found at startup and by the
dashboard button — press *Find the microphone again*. If it is `true`, a swap waits for a quiet
moment, so it will not happen mid-turn; give it a few seconds after Kai stops talking. If it still
does not take, the log names the reason per device, and the two reasons want opposite fixes:
`read as silent` means check the wiring, `rejected the probe` means check what else is holding the
card. Note that a capture rate that is not a whole multiple of 16 kHz cannot be used at all — a
44.1 kHz-only device is skipped by design, not by accident.

**Q: How do I change Kai's voice, or how it speaks?**
The voice model is one line — `TTS_VOICE_MODEL` in `config/voice.py`; several Piper voices are already downloaded in `voices/` and are interchangeable by editing that line. Volume and speaking rate are live in the ⚙ Settings tab. What Kai *says* comes from `ai/persona.txt`, which is re-read on every turn, so edits apply to the next reply with no restart. Before reaching for a different engine, read [docs/plan/completed/expressive-voice-plan.md](plan/completed/expressive-voice-plan.md) — 29 voices across 7 families were measured and rejected the same way, and the conclusion was that the remaining lever is delivery, not timbre.

**Q: Can the laptop camera server handle multiple Jetson clients?**
No — `laptop_camera.py` accepts one connection at a time. After disconnect it waits for the next client.
