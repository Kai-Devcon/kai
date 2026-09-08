# Hardware

## Core

| Component | Notes |
|-----------|-------|
| NVIDIA Jetson Orin Nano | Any variant; tested on 8GB Super. 8 GB is the binding constraint — see [memory-budget.md](memory-budget.md) |
| Arduino Uno | CH340 clone works; needs the ch341 kernel module on the Jetson (not in the tegra kernel) |
| SG90 9g Micro Servo × 1 | **Pan** axis, Arduino pin 9. Orange = signal, Red = VCC, Brown = GND |
| SG90 9g Micro Servo × 1 | **Jaw**, Arduino pin 6. Driven by speech, and by mouth-mirroring when idle |
| USB-A to USB-B cable | Arduino to Jetson |
| Jumper wires (female-female) | 3 wires per servo to the Arduino |

## Camera — one of

| Component | Notes |
|-----------|-------|
| CSI camera (IMX219 or similar) | The ribbon camera. Probed via `nvarguscamerasrc`; see the CSI bind note in [operating.md](operating.md) |
| USB webcam | Genuinely hot-pluggable, unlike CSI |
| Laptop on the same network | Runs `vision/laptop_camera.py`, streams JPEG over TCP 8485 |

Kai runs with **none** of these attached — "no camera" is a reported state, not a failure.

## Voice

| Component | Notes |
|-----------|-------|
| INMP441 I2S MEMS microphone | The built-in mic. Captured raw on ALSA card `APE` at 48 kHz with PulseAudio suspended — see `config/voice.py` |
| USB audio dongle (C-Media) | Output DAC. Named as a PulseAudio sink in `TTS_SINK`, and its card profile is asserted on every start because Pulse flips it to S/PDIF unprompted |
| PAM8403 amplifier + speaker | Driven from the dongle's analog jack |
| USB microphone *(optional)* | A supported input, not just a fallback. Plug one in and Kai switches to it within seconds; unplug it and he goes back to the I2S mic |
| 3.5mm microphone *(optional)* | Also supported. On a **separate** USB→3.5mm adapter it is the `analog` kind; on the **same** dongle that drives the speaker it is the `pulse` kind and needs `PULSE_CAPTURE_ENABLED` |

### Which microphone Kai uses

Every mic is probed on every resolve and the first one that captures real signal wins.
`MIC_PREFERENCE` (`config/voice.py`, live-settable on the dashboard) decides which *kind* is tried
first — `auto`, `i2s`, `usb` or `analog`. It only reorders: preferring one mic still falls through to
the others when it is not live, so the setting can never leave Kai deaf. `auto` is I2S, then USB,
then analog, then the system default.

Plugging a mic in or pulling one out is noticed on its own. `ai/mic_hotplug.py` watches
`/proc/asound/cards` and, once the card set has settled, the session re-resolves at its next quiet
moment — never mid-turn, so a swap cannot cut someone off mid-sentence. On a machine without that
file the watcher turns itself off and the dashboard button is the way to re-resolve.

> **The USB mic and the speaker dongle can share a name.** Capturing on the speaker's own card
> segfaulted the process at the startup greeting (2026-08-11) and is blocked, but the block used to
> be a name-substring match on `"usb audio device"` — which is also what a cheap USB mic typically
> enumerates as. The card is now identified by resolving `TTS_CARD` to its ALSA card index via
> `pactl`, so the two are never confused. Where `pactl` cannot answer, the name rule still applies
> and a USB mic with that name will be skipped.

### Using a 3.5mm microphone

There is no native analog capture path on this board — the Jetson's own analog input "enumerates as
a normal input device but isn't wired to anything, so it only captures digital silence"
(`config/voice.py`). A 3.5mm mic therefore reaches Kai through a **USB→3.5mm audio adapter**, which
as far as ALSA is concerned is simply another USB sound card. Do *not* use the mic jack on the
C-Media dongle that drives the speaker: that card is excluded from capture because capturing it raw
while playback reconfigures it segfaulted the process (see below).

Because an adapter is a USB sound card, telling it apart from a native USB mic is a **name match**
and nothing more — `ANALOG_MIC_NAME_HINTS` in `config/voice.py`. The defaults are plausible, not
measured. Check what yours actually reports and add it:

```bash
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

Getting that list wrong is cosmetic: an unmatched adapter classifies as `usb` and still works
exactly as before. What it costs is the dashboard's ability to tell two USB inputs apart, and
`MIC_PREFERENCE`'s ability to choose between them.

#### One dongle for both, or two dongles?

A dongle with two jacks — headphone out and mic in — can serve both, but the two arrangements are
not equally simple:

| | Card layout | What it needs |
|---|---|---|
| **Two dongles** (mic on one, speaker on the other) | separate cards | nothing; works as shipped. Mic is the `analog` kind |
| **One dongle** (both jacks) | one shared card | `PULSE_CAPTURE_ENABLED`, and the mic becomes the `pulse` kind |

One dongle is **one ALSA card**, and the card — not the jack — is the unit of configuration:

```
card N  ──┬── playback  hw:N,0    sink:   alsa_output.usb-<product>-00.analog-stereo
          └── capture   hw:N,0    source: alsa_input.usb-<product>-00.mono-fallback
```

`pactl set-card-profile`, which `tts.play()` asserts before the first reply, acts on the whole card.
That re-opened the card's ALSA devices underneath a live **raw** capture stream on 2026-08-11 and the
process took SIGSEGV at the startup greeting. So **raw capture on the speaker's card is refused, and
always will be.** The `pulse` kind routes around it: PulseAudio serialises access to the card, so a
pulse-mediated stream has no raw device for a profile change to pull out from under it.

To use one dongle for both, three config values must all name that dongle — `TTS_SINK`, `TTS_CARD`
and `PULSE_CAPTURE_SOURCE` — plus `PULSE_CAPTURE_ENABLED = True`. `scripts/mic_survey.py` prints the
real names and says whether each resolves. Two upsides worth knowing: pulse resamples, so **a
44.1 kHz-only dongle is perfectly usable through this route** even though it is unusable raw; and
the sample-rate gotcha below stops applying.

The downside is physical, not fixable in software: one card puts the mic and speaker on a common
ground as well as a common chassis, so playback bleed into the mic gets *worse*. There is no echo
cancellation either way, so barge-in stays off.

Three ways a **separate** adapter fails that no amount of software can see, worth knowing
**before you buy**:

- **The adapter is 44.1 kHz only.** Kai cannot use it at all — `FALLBACK_CAPTURE_RATES` only offers
  rates that divide evenly into 16 kHz, because an integer-ratio decimator cannot resample 44100 and
  returning that rate took the whole voice session down once (2026-08-09). Check with
  `arecord -D hw:<N>,0 --dump-hw-params`; you want 48000 in the `RATE:` list. This is the single
  most likely way a cheap adapter disappoints.
- **The mic needs plug-in power and the adapter does not supply it.** Most 3.5mm computer mics are
  electret and need bias voltage on the ring. A dynamic mic will read as near-silence on any
  adapter.
- **The plug is 4-pole TRRS, not 3-pole TRS.** A phone headset plugged into a dedicated mic jack
  puts the mic on the wrong contact. Presents as silence.

All three look identical from the log — `device N read as silent` — which is why they are listed
here rather than left to be diagnosed.

> **The I2S wiring** is in §4 of `docs/buildsheet/Kai_Build_Sheet_hardware_revE.docx` — the INMP441
> to 40-pin header mapping, including the two that bite: VDD on **Pin 1 (3.3 V, not 5 V)** and L/R
> tied to **Pin 9 (GND)**, which is what puts the audio in the left slot. `config/voice.py`
> documents the software side: the XBAR/I2S2 route applied by `apply_i2s_route()`, the 48 kHz clock,
> and stereo capture with real audio only in the left channel.

## Optional / not currently wired

| Component | Notes |
|-----------|-------|
| SG90 9g Micro Servo × 1 | **Tilt** axis, pin 10. The `--tilt` flag, the wire protocol and the dashboard field all exist, but the firmware declares `TILT_PIN` and never attaches a servo to it — see [ticket R10](tickets/R10-tilt-axis-plumbed-without-hardware.md) |
| Pan-tilt bracket | Only needed if the tilt axis is actually wired |

> **Servo quality matters.** A faulty servo (internally broken) can appear to respond but won't move or will behave erratically. If the servo buzzes but doesn't rotate during a standalone sweep test, replace it before debugging software.

> **Servos share the Arduino's USB 5V rail.** That rail is why `SEND_INTERVAL` (10 Hz pan) and
> `PAN_MAX_STEP` (8°/command) are what they are: faster or larger commands raise average current
> and can brown out the CH340, which drops the USB link mid-track. `config/servo.py` records the
> measurement. The real fix for faster motion is a separate servo supply, not a config change.

---

## Wiring


### As built — pan + jaw

| Servo wire | Pan servo | Jaw servo |
|------------|-----------|-----------|
| **Orange** (signal) | **Pin 9** | **Pin 6** |
| **Red** (power) | **5V** pin | **5V** pin (shared rail) |
| **Brown** (ground) | **GND** pin | **GND** pin (shared) |

The Arduino is powered entirely by its USB connection to the Jetson. Its 5V pin outputs USB power directly to the servos.

```
Arduino board
┌────────────────────────────────┐
│  Pin 9  ──── Pan  Orange       │
│  Pin 6  ──── Jaw  Orange       │
│  Pin 10 ──── (tilt, not wired) │
│  5V     ──── Pan Red + Jaw Red │
│  GND    ──── Pan + Jaw Brown   │
│  USB ←── Jetson USB port       │
└────────────────────────────────┘
```

Both servos are written by the same serial link but on **different channels and different rates**:
pan goes through the `"pan,tilt"` line at 10 Hz, the jaw through `"J<angle>"` at 20 Hz. That split
exists so speech animation stays smooth without raising the pan send rate into the brownout region.

### Pan only

Wire pin 9 as above and omit the jaw. Kai runs unchanged — the jaw channel simply commands nothing,
and `--jaw` (which is a hard AND with the `jaw_enabled` setting) is left off.

> **Do not connect the servo to the Jetson 40-pin header for signal.** 3.3V is insufficient. Power (5V from Pin 2) is fine for the Red wire IF you also share ground through the Arduino, but using Arduino 5V pin is simpler and safer.
