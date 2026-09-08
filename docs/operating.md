# Operating Kai

### face_track.py — main script

```
usage: face_track.py [-h] [--camera N] [--network HOST] [--network-port PORT]
                     [--port PATH] [--flip] [--flip-y] [--tilt] [--jaw]
                     [--lofi] [--no-display] [--no-servo] [--no-camera]
                     [--rotate {0,90,180,270}] [--wake] [--no-hands-free]

  Camera
  --camera N        CSI sensor-id / V4L2 index (default: 0)
  --network HOST    Laptop TCP camera IP — used if no local camera is found
  --network-port N  TCP port (default: 8485)
  --rotate DEG      Rotate the live feed clockwise (0/90/180/270)
  --no-camera       Run with no camera at all, deliberately. Locks the
                    dashboard's camera control off with the reason shown

  Servos
  --port PATH       Arduino serial port (default: /dev/ttyUSB0)
  --flip            Invert pan direction (X-axis)
  --flip-y          Invert tilt direction (Y-axis)
  --tilt            Enable the tilt servo on pin 10 (no hardware today — see R10)
  --jaw             Enable the jaw servo on pin 6. A hard AND with the
                    'jaw_enabled' setting: no setting can conjure a servo
                    that is not wired
  --no-servo        Run with no serial link, deliberately

  Voice
  --wake            Hands-free: one always-open mic, "Hey Kai", VAD turn-taking
  --no-hands-free   With --wake: keep the shared stream but leave push-to-talk
                    as the only way in

  Other
  --no-display      Headless — no OpenCV window (required over SSH and cron)
  --lofi            Emit the 19-digit LOFI face-param string each tick
```

**Example — as the robot actually runs (see `scripts/autostart.sh`):**
```bash
python3 -u face_track.py --no-display --flip --jaw --wake
```

**Example — dashboard and voice only, no camera:**
```bash
python3 -u face_track.py --no-display --no-camera --wake
```

**Example — laptop as the camera:**
```bash
python3 -u face_track.py --network 192.168.1.x --no-display --flip
```

### laptop_camera.py — camera server (run on the laptop)

```
usage: laptop_camera.py [--port PORT] [--camera CAMERA]

  --port N      TCP port to serve on (default: 8485)
  --camera N    Webcam index (default: 0)
```

### Console output

Every subsystem prefixes its own lines, so `grep` on the tag is the fastest way to read a log.

```
[settings] loaded 3 setting(s) from /home/devconph/.config/kai/settings.json
[face_track] flip=True  tilt=False  lofi=False  ema=0.5  infer=15fps
[servo] Centered at 90°
[camera] CSI camera (sensor 0)
[camera] live camera acquired: csi
[wake] engine: porcupine (frame, 512 samples) — skipped none
[mic] open: device=5 (i2s) 48000 Hz x2 -> 16000 Hz
[llm] gemma2:2b fully on GPU (2374 MB VRAM)
[control] 14.9 Hz  face=True
[face_track] pan=97° sent | 15fps | yaw=52 pitch=48 roll=1 mouth=12 leye=61 reye=60 dist=44 smile=3 emotion=neutral
[face_track] NO FACE — pan=97°
[session] idle -> ack
[session] speech onset (rms=1180)
[turn] 4820ms to first audio = stt 2380 + rag 140 + llm 1180 (prompt 410, gen 760) + synth 1120
```

| Tag | What it tells you |
|---|---|
| `[face_track]` | The inference loop. `sent`/`hold` is whether the control thread's last command reached the wire; `fps` is inference throughput, not camera rate |
| `[control]` | The servo thread's real rate. Edge-triggered on face presence, plus a 30 s heartbeat. If this sags well below 15 Hz, something is holding the GIL |
| `[camera]` | Acquire / release / probe failures, logged only when the reason changes |
| `[session]` | State transitions and turn boundaries — the first place to look for a wake that did not land |
| `[mic]` / `[wake]` | Which device and which wake tier actually won. Also why a candidate mic was rejected — `read as silent` and `rejected the probe` are different problems — and any hot-plug swap |
| `[turn]` | The latency breakdown, one line per reply. This is how "Kai feels slow" gets attributed |
| `[llm]` | Ollama placement and per-request timings. A partial GPU offload is called out here |

Two log cadences are deliberately slow: `NO FACE` repeats at most every 30 s and `[control]`
heartbeats every 30 s. Both used to print once a second — measured at 58% of a 1.5-hour log — which
buried everything worth reading. Transitions always print regardless.

---

## Configuration & Tuning


### Live settings (⚙ Settings tab)

Twelve knobs are adjustable while Kai is running, from the dashboard's **⚙ Settings** tab on
`http://<jetson>:8081` — camera mode, hands-free wake word, wake sensitivity, mic noise floor, speak
replies, volume, speaking rate, natural delivery, follow faces, move the jaw, thinking sweep and
thinking sounds. Every one applies the instant you change it — none of them needs a restart. The
authoritative list is `_SPECS` in `settings.py`; `curl localhost:8081/settings` prints it with the
valid range for each.

They persist in `~/.config/kai/settings.json` (an overlay on the `config/*.py` defaults — never
committed). Delete that file, or press **Restore defaults**, to get exactly the committed behaviour
back. Also reachable over ssh:

```bash
curl localhost:8081/settings                      # values, defaults, valid ranges
curl -X POST localhost:8081/settings \
     -H 'content-type: application/json' \
     -d '{"tts_volume": 1.4, "vad_rms_floor": 800}'
curl -X POST localhost:8081/settings/reset
curl -X POST localhost:8081/camera/probe          # look for a camera right now
```

**Restart Kai** — at the bottom of the same tab, below the knobs, is the one control that is not a
setting. It stops `face_track.py` the same way `SIGTERM` does (mic, Porcupine, serial port and camera
all released properly) and exits `7`, which `scripts/autostart.sh` treats as "relaunch now, and don't
count it as a crash". Kai is back in roughly half a minute and the dashboard reconnects on its own.

It exists for the failures no knob above can reach — a capture device that has wedged, a wake engine
that stopped hearing, an Ollama that came back after Kai gave up on it — and it takes two taps, so a
stray touch on a kiosk cannot stop the robot mid-demo.

```bash
curl -X POST localhost:8081/restart               # {"status":"ok","supervised":true,...}
```

`supervised` in that reply is the honest part: it is true only when `KAI_SUPERVISED=1` is set, which
`scripts/autostart.sh` exports and nothing else does. Started by hand (`scripts/run.sh`, or
`python3 face_track.py` over ssh) it reads false, the button says so in red, and the click is a
shutdown rather than a restart — start it again from the Jetson.

`--no-camera` and `--no-hands-free` still win over the stored setting: they declare what this machine's
hardware situation is for the run, and the dashboard shows the control disabled with the reason.

### The recovery ladder — cheapest first

Three controls, stacked in the order to try them. **The order on screen is the order to use.** Each
one costs more than the one above it, and reaching for the bottom of the ladder first is how a
five-second problem becomes ninety seconds of dead robot.

| | Control | Cost | Use it for |
|---|---|---|---|
| 1 | 🎙 Find the microphone again | ~2 s, nothing interrupted | Kai cannot hear; `sess_mic_live` is false |
| 2 | ⟳ Restart Kai | ~30 s, conversation lost | Anything a setting cannot reach |
| 3 | ⏻ Reboot the Jetson | ~90 s, whole board down | Wedged kernel audio, nvargus, GPU fragmentation |

**1. Find the microphone again** (`POST /audio/reresolve`) re-runs the whole discovery path — I2S
route, pulse release, device probe — and reopens the stream, without touching the camera, the
servos or the conversation. It exists because both mics fail in ways a *later* look fixes, and
nothing short of a restart used to take that later look: across three boots the INMP441 read silent
on one, timed out on the next and worked on a third, with `arecord` finding real audio every time.

The existing watchdog cannot cover this. It reopens a stream that *died*, and is gated on
`state != disabled` — but a mic that never came up leaves the session in exactly `disabled`, so the
one case needing a second look was the one case it skipped.

```bash
curl -X POST localhost:8081/audio/reresolve
# {"ok":true,"device":5,"rate":48000,"kind":"i2s","is_i2s":true,"live":true,...}
```

The reply says *which* mic it landed on, because "it worked" is not the whole answer — Kai on the
USB mic when he should be on the I2S mic is a different situation with a different next step.

### Switching microphones

Kai takes three kinds of microphone — the built-in INMP441 (`i2s`), a USB mic (`usb`), and a 3.5mm
mic on a USB adapter (`analog`) — and you rarely have to do anything: plugging one in or pulling one
out is noticed within a few seconds and applied at the next quiet moment. A swap never interrupts a
turn in progress — if you plug a mic in while Kai is listening or replying, the change waits for the
turn to finish.

To choose deliberately, set **Prefer** on the Microphone card (or `POST /settings` with
`mic_preference` of `auto`, `i2s`, `usb` or `analog`) and press *Find the microphone again*. Unlike
every knob in the Settings panel, this one does not apply the instant it changes — a preference is
only read while a mic is being resolved — which is why it sits next to the button that resolves one.

It reorders, it does not restrict. Preferring any one mic still falls back to the others when it is
not live, so no setting here can leave Kai deaf. Two things to check when a swap does not go the way
you expected:

```bash
curl -s localhost:8081/params | jq '{kind:.sess_mic_kind, watching:.sess_mic_hotplug_watching,
                                     swaps:.sess_mic_hotswaps, live:.sess_mic_live}'
```

- `watching: false` means the hot-plug watcher is off — it could not read `/proc/asound/cards`, so
  mics are only picked up at startup and by the button.
- `kind` not matching your preference means the preferred mic read as silent or refused to open, and
  selection fell through. The log says which, per device: `read as silent` and `rejected the probe`
  are different problems (check the wiring vs. check what is holding the card).
- `kind: usb` when you expected `analog` is only a labelling miss — the adapter works, its name just
  is not in `ANALOG_MIC_NAME_HINTS`. Add it (`docs/hardware.md` has the one-liner that prints the
  real name). A 3.5mm adapter reading silent is more likely one of the three hardware gotchas listed
  there — a 44.1 kHz-only adapter, a mic needing plug-in power, or a TRRS plug — than anything in
  the software.

**If the log selects a mic and then fails to open it,** read the PortAudio error code — the two you
are likely to see mean different things and neither is a broken microphone:

- `Device unavailable [PaErrorCode -9985]` — something else holds the raw ALSA device, and on this
  build that is almost always pulseaudio. Kai suspends every capture source before probing and then
  hands back every card *except* the one it is about to open (you will see `[mic] pulse stays
  suspended on the capture card (hw:N)`); if that line names the wrong card, or is missing when the
  chosen device is a `hw:` one, pulse re-grabs the mic in the gap. `pactl list short sources` shows
  who owns what.
- `Illegal combination of I/O devices [PaErrorCode -9993]` — despite the wording, not a device
  problem at all. It is PortAudio being re-initialised (`refresh_devices()`) on one thread while
  another is opening a stream. `pa_lock` in `ai/mic_device.py` serialises those, so seeing this
  again means a new caller reached `sd._terminate()` without taking it.

**3. Reboot the Jetson** (`POST /system/reboot`) is **off by default** and needs two deliberate
steps to switch on — see `REBOOT_ENABLED` in `config/tracking.py`, which explains why. In short:
this dashboard has **no authentication at all** (Flask binds `0.0.0.0`), so unlike every other
control here, enabling it hands an unauthenticated LAN endpoint the ability to take Kai off the air.
It needs `REBOOT_ENABLED = True` *and* a sudoers line scoped to exactly `/usr/bin/systemctl reboot`
— never `NOPASSWD: ALL`. Consider running the rootfs `e2fsck` first; making reboots one click away
on a filesystem that already has known ext4 errors is how a demo robot becomes an unbootable one.

Requests carry a `{"confirm": "reboot"}` body, and the endpoint checks `sudo -l` *before* claiming
success — a misconfigured sudoers gets an error you can act on rather than a button that reports
"ok" and does nothing.

**Restarts now have a deadline.** The graceful shutdown is still the default, but if it has not
finished within `RESTART_FORCE_AFTER_S` (`app/lifecycle.py`, 12 s) the process exits by force with the same code, so the
supervisor still brings Kai back. This is not hypothetical: on 2026-08-09 a `POST /restart` replied
`{"status":"ok"}` and the process never exited — it was wedged inside mic resolution, so the
teardown it depends on never completed. Same pid forty minutes later, nothing in the log. From the
dashboard that is indistinguishable from a restart that worked, which is the worst way for a
recovery control to fail.

### Starting without a camera

Kai starts and stays up with no camera attached — the dashboard, voice assistant, wake word and servos
do not need one, and the ⚙ Settings tab shows **NO CAMERA** with the actual reason (e.g. `no
/dev/video* device`). A background supervisor keeps checking, so **plugging a camera in later brings it
up live, with no restart**. Set Camera to `off` to stop using one deliberately.

The same applies to the Arduino: if the serial port is missing, Kai runs without servos rather than
failing to start. Under the cron `@reboot` launcher there is no supervisor to retry, so a startup crash
would mean a dead robot until the next reboot.

**CSI is probed only once, at boot.** If that probe fails — a cable reconnected while the board was
off, or a camera plugged in after boot — the sensor stays invisible for the whole boot, because
creating `/dev/video0` needs a driver bind and that needs root. A `@reboot` helper in **root's**
crontab retries the bind when no capture device appeared:

```bash
sudo bash scripts/camera_bind_retry.sh --install   # once; @reboot in root's crontab
sudo bash scripts/camera_bind_retry.sh --now       # or run it by hand any time
tail -f /tmp/camera-bind.log
```

When it succeeds, `face_track.py`'s supervisor picks the camera up within ~5 s — no restart. It is
deliberately separate from `autostart.sh`, which runs unprivileged: the robot process never needs root.
USB webcams are unaffected by any of this; they genuinely hot-plug.

**Diagnosing a camera that will not come up:**

```bash
sudo bash scripts/camera_diag.sh
```

Reports in plain language whether a sensor actually *answers* on the CSI bus, and ends with a verdict
that separates a software mismatch from a cable/module fault. Worth knowing why it exists: a CSI sensor
is only powered for ~10 ms while the driver probes it, so an ordinary `i2cdetect` sweep (50–100 ms)
misses the window and shows an empty bus whether or not a camera is attached. This probes single
addresses inside that window while cycling the rail, and checks the addresses used by other common
modules so "it is not actually an IMX219" surfaces instead of hiding. Run it after every hardware swap.

### Restart-only constants

Everything not in the ⚙ Settings tab is a hand-edited literal in [`config/`](../config/), one file
per subsystem, and takes effect on the next start. **Each one is annotated in place with the
measurement that set it** — read the comment before changing the number, because most of these
values are the answer to a specific failure and not a preference.

| File | Covers |
|---|---|
| `config/tracking.py` | Inference and control rates, EMA, pan/tilt scale, jaw range, log cadence, web port |
| `config/servo.py` | Serial port and baud, send-rate gates, angle limits, deadbands |
| `config/camera.py` | Capture and processing resolution, probe budgets, retry backoff, stall detection |
| `config/voice.py` | Whisper, Ollama, Piper, delivery shaping, the mic device and I2S route |
| `config/wake.py` | Wake tiers, VAD floors, session timers, capture geometry, debug capture |
| `config/rag.py` | Chunking, thresholds, the failsafe chain, the gazetteer |
| `config/filler.py`, `config/thinking.py`, `config/gesture.py` | Filler bank, thinking sweep and sounds, gesture thresholds |

The tracking values most often reached for:

```python
INFERENCE_FPS  = 15    # MediaPipe rate. Raising it STARVES the control thread — see the
                       # GIL note in config/tracking.py before touching this
CONTROL_FPS    = 15    # servo control thread; current-sensitive, see config/servo.py
PAN_KP         = 0.20  # proportional gain: higher = faster tracking, may overshoot
PAN_KD         = 0.25  # derivative gain: higher = more damping, less jitter
PAN_SCALE      = 120   # servo degrees swept across the full nose-X range
PAN_MAX_STEP   = 8     # hard slew cap per command — bounds SG90 current spikes
EMA_ALPHA      = 0.50  # input smoothing before the PD
MIN_FACE_AREA  = 0.04  # landmark-spread gate; raise to ignore distant faces
PROCESS_WIDTH  = 320   # MediaPipe input (config/camera.py); lower = faster
PROCESS_HEIGHT = 240
SEND_INTERVAL     = 0.10   # 10 Hz pan/tilt serial gate  (config/servo.py)
JAW_SEND_INTERVAL = 0.05   # 20 Hz jaw-only channel
```

| Goal | Change |
|------|--------|
| Less jitter / more natural | Decrease `EMA_ALPHA` (try 0.20) |
| Faster input response | Increase `EMA_ALPHA` (try 0.60) |
| Faster catch-up | Increase `PAN_KP` (try 0.25–0.30) |
| Dampen overshoot | Increase `PAN_KD` relative to `PAN_KP` |
| Bigger head movements | Increase `PAN_SCALE` (try 140–160) |
| Less CPU load | Decrease `PROCESS_WIDTH/HEIGHT` (try 160×120) |
| Ignore far-away faces | Increase `MIN_FACE_AREA` (try 0.08) |
| Wider jaw travel | Raise `JAW_OPEN` toward `SERVO_MAX` (170) |

There is **no sleep timeout any more.** On no face the control thread enters its *hold* branch: it
sends nothing at all, which lets the firmware's 4 s idle-detach relax the servos on its own, and it
keeps the PD synced to the held position so re-acquiring glides instead of snapping. The old
"return to 90° after 3 s" behaviour is gone, and `SLEEP_AFTER` no longer exists.
