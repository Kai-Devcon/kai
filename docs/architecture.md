# Architecture

Kai is one process with several threads. Two independent pipelines run inside it — **seeing and
moving**, and **hearing and answering** — sharing only the jaw servo and the dashboard.

## The tracking pipeline

Perception and actuation run at deliberately different rates. The inference loop publishes a
*target*; a separate control thread drives the servo toward it. See `config/tracking.py` for the
GIL measurement that fixed both rates.

```
┌────────────────────────────────────────────────┐
│                Jetson Orin Nano                 │
│                                                 │
│  Camera: CSI · USB (V4L2) · TCP · video file ·  │
│          none  (vision/camera.py)               │
│       │  read on its own thread, newest frame   │
│       ▼  only — the loop never blocks on it     │
│  resize 320×240  (PROCESS_WIDTH/HEIGHT)         │
│       │                                         │
│       ▼   decimated to INFERENCE_FPS = 15       │
│  MediaPipe FaceMesh                             │
│  → nose tip landmark (index 1)                  │
│  → landmark spread gate (MIN_FACE_AREA = 0.04)  │
│       │                                         │
│       ▼                                         │
│  EMA smoothing (EMA_ALPHA = 0.50)               │
│  target_pan = (x - 0.5) × PAN_SCALE + 90        │
│       │                                         │
│       ▼   TrackingTarget.set()  ── thread seam ─┼──┐
│  publish target + presence + web frame          │  │
└─────────────────────────────────────────────────┘  │
                                                     │
┌─────────────────────────────────────────────────┐  │
│  servo-control thread  @ CONTROL_FPS = 15       │◀─┘
│  (app/control_loop.py)                          │
│                                                 │
│  PD: correction = 0.20×err + 0.25×d_err         │
│  slew clamp: |Δ| ≤ PAN_MAX_STEP = 8°/command    │
│  + thinking sweep while the LLM is working      │
│  hold on no-face / stale target — sends nothing, │
│  so the firmware's idle-detach relaxes the servo │
│       │                                         │
│       ▼  serial, gated to 10 Hz (SEND_INTERVAL) │
│  /dev/ttyUSB0  "pan,tilt\n"                     │
└──────────────┬──────────────────────────────────┘
               │ USB
               ▼
┌───────────────────────────────────────┐
│  Arduino Uno  (servo_serial.ino)      │
│                                       │
│  "pan[,tilt[,jaw]]\n"  → pan + jaw    │
│  "J<angle>\n"          → jaw only     │
│  "G:<code>\n"          → gesture ack  │
│                                       │
│  Pin 9  → pan  (5V PWM)               │
│  Pin 6  → jaw  (5V PWM)               │
│  Pin 10 → tilt — declared, NOT driven │
│  detach after 4 s idle (no buzz/draw) │
└──────────┬────────────────────────────┘
           ▼
     SG90 servos
```

The jaw is **not** driven from the control thread. It has its own 20 Hz channel (`J<angle>`) written
from the main loop, so the mouth keeps animating mid-sentence even if the camera stalls — and so
speech pantomime never contends with pan/tilt for the 10 Hz gate.

## The conversation pipeline

One always-open capture stream, fanned out to several consumers. Nothing here needs the camera.

```
INMP441 I2S mic (raw hw, PulseAudio suspended)  — or a USB / 3.5mm mic; ai/mic_device.py picks
   │  one opener only — everything shares this stream
   ▼
ai/mic_stream.py   PortAudio callback: slice, copy, enqueue (never blocks)
   │
   ▼  worker thread
mute gate → decimate 48k→16k → high-pass
   │
   ├──▶ wake chain: porcupine → openWakeWord → whisper   (ai/audio.py)
   ├──▶ VAD / SpeechGate: onset, hangover, ambient adapt
   └──▶ utterance buffer (+ pre-roll)
   │
   ▼  ai/session.py — the state machine and every timer
idle → ack → cooldown → listen_wait → listen_speech → busy → speaking
   │
   ▼
faster-whisper (base, int8, CPU)  →  ai/rag.py retrieval over documents/
   │                                    dense + lexical rescue + failsafe chain
   ▼
Ollama gemma2:2b  (localhost, non-streaming)
   │
   ▼
ai/delivery.py shaping → Piper → sox → paplay
   │                                      │
   └──▶ jaw envelope fitted to the real ──┘
        audio length (ai/speak_envelope.py)
```

## Threads

| Thread | Rate | Owns |
|---|---|---|
| main (`face_track.run`) | frame-driven | MediaPipe, the jaw channel, dashboard publishing |
| `servo-control` | 15 Hz | pan PD, slew clamp, thinking sweep |
| camera reader | camera-driven | newest-frame slot, swap queue |
| `kai-camera` | ~5 s | probe / hot-swap / release the live camera |
| `kai-audio` | ~30 blocks/s | resample, wake, VAD, capture fan-out |
| `kai-session` | 20 Hz | conversation timers, filler, session end |
| `kai-turn` / `kai-tts` | per turn | STT → LLM, then synthesis and playback |
| Flask | per request | dashboard routes, `/params` SSE, `/video` MJPEG |

**Why Arduino?**
The Jetson Orin Nano's GPIO is capped at **3.3V** on all pins — no exceptions, no workarounds in hardware. The SG90 servo requires a **~5V signal** to respond. The Arduino, powered by USB, outputs 5V PWM on its digital pins. The Arduino is completely passive — it just converts the serial angle command into a 5V PWM pulse. All tracking logic stays on the Jetson.

**Camera modes** — probed in this order, and any of them can appear or vanish at runtime:
- **CSI** (`nvarguscamerasrc`) — the Jetson ribbon camera
- **Local USB** — plug a webcam directly into the Jetson
- **Network TCP** — run `vision/laptop_camera.py` on a laptop; the Jetson receives JPEG frames on port 8485
- **Video file** — uploaded through the dashboard, for testing with no camera at all
- **None** — a first-class state that reports *why*, not a startup failure

---

## File Reference


```
kai/
├── face_track.py          Entry point: CLI, wiring, and the MediaPipe tracking loop
│
├── app/                   The robot application
│   ├── lifecycle.py       Instance lock, signal handlers, restart/reboot, exit codes
│   ├── camera_supervisor.py  Probe / hot-swap / release the live camera, and say why
│   └── control_loop.py    Servo control thread: pan PD, slew clamp, thinking sweep
│
├── web/                   The dashboard — the only operator interface Kai has
│   ├── server.py          Flask app: every route, built around injected collaborators
│   ├── state.py           What the dashboard is currently being told, behind one lock
│   └── frontend/          dashboard.html, guide.html
│
├── ai/                    Hearing, thinking, speaking
│   ├── session.py         Hands-free conversation: the state machine and every timer
│   ├── mic_stream.py      The process's ONE open capture stream, fanned out to consumers
│   ├── mic_device.py      Which mic to open: ALSA route, pulse suspend, liveness probe
│   ├── mic_hotplug.py     Notices a mic plugged in or pulled out, so swapping needs no restart
│   ├── audio.py           Resampler, framing, pre-roll, capture buffer, wake chain, VAD
│   ├── audio_debug.py     Optional on-disk corpus of what Kai actually heard
│   ├── voice_assistant.py One turn: STT -> LLM -> TTS + jaw, with turn epochs
│   ├── llm.py             Ollama client: persona, prompt, request, timing breakdown
│   ├── transcript.py      Gates deciding whether a transcript is worth acting on
│   ├── speak_envelope.py  The jaw open/close schedule (text-timed or audio-fitted)
│   ├── tts.py             Piper synthesis and playback, cancellable, with cached lines
│   ├── delivery.py        Breaths, an occasional opener, per-reply tempo jitter (spoken text only)
│   ├── filler.py          Which filler line to play, and when (pure; config/filler.py has the words)
│   ├── wake_phrase.py     Fuzzy "hey kai" matching over a transcript (pure stdlib)
│   ├── rag.py             Retrieval: chunking, embeddings, ranking, the failsafe chain
│   ├── query_alias.py     Fuzzy "DEVCON" + gazetteer matching on the query (pure stdlib)
│   ├── index_documents.py Run manually to (re)build the RAG index from documents/
│   └── persona.txt        Kai's editable personality — edit freely, no restart needed
│
├── vision/                Eyes
│   ├── camera.py          CSI / V4L2 / network / video-file / no-camera sources
│   ├── face_params.py     Landmarks -> FaceParams (LOFI format), emotion classification
│   ├── controller.py      EMA filter, PD axis, the thread-safe tracking target
│   ├── gesture.py         Nod / shake / approach / retreat / mouth detection
│   ├── presence.py        Three-valued "is anybody there", written by the inference loop
│   └── laptop_camera.py   TCP camera server — copy to a laptop and run it there
│
├── servo/                 Neck and jaw
│   ├── servo.py           The Arduino serial link (reconnecting, rate-gated, thread-safe)
│   ├── servo_serial.py    Manual servo control / sweep test
│   └── servo_diag.py      Slow diagnostic sweep (position verification)
│
├── config/                Every hand-tuned constant, one file per subsystem (see config/README.md)
├── settings.py            The live overlay: knobs the dashboard can change while running
│
├── documents/             Drop .txt/.md/.pdf here, then run python3 -m ai.index_documents
├── docs/                  Reference, plans, tickets and the R&D write-ups (see docs/README.md)
├── scripts/               autostart.sh, diagnostics, benchmarks (wake_test.py first)
├── hardware/              CH340 driver build, USB bind, udev rule, sudoers helper
├── tests/                 One file per module; no hardware needed for any of them
├── arduino/
│   ├── servo_serial/servo_serial.ino       Serial-controlled servo (active sketch)
│   └── servo_standalone/servo_standalone.ino  Standalone sweep (hardware test)
├── CHANGELOG.md           What changed and when, newest first
└── README.md              Project overview (this tree lives in docs/architecture.md)
```
