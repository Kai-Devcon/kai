# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Kai is a face-tracking, talking companion robot running on an NVIDIA Jetson Orin Nano — one Python
process, no cloud, no API keys on the conversation path. An Arduino Uno is a passive 5V PWM bridge
for the servos (Jetson GPIO is 3.3V-only); all logic lives in Python on the Jetson.

Two independent pipelines share only the jaw servo and the dashboard: **seeing and moving** (camera
→ MediaPipe → PD controller → servo thread → Arduino) and **hearing and answering** (mic → wake word
→ VAD → Whisper → RAG → Ollama → Piper TTS, with the jaw synced to the real audio length). See
[docs/architecture.md](docs/architecture.md) for the full data-flow diagrams, thread table, and a
file-by-file reference — read it before touching either pipeline.

## Commands

```bash
python -m pytest -q                      # full suite: 1173 tests, ~33s, no hardware/network/models
python -m pytest tests/test_session.py   # one file
python -m pytest tests/test_session.py::test_name   # one test
python3 face_track.py --network 192.168.1.x --no-display --flip --wake   # run it (needs a Jetson or fakes)
python3 -m ai.index_documents           # rebuild the RAG index after editing documents/
```

There is no lint/format/type-check tooling configured in this repo (no ruff/black/mypy config) — don't
invent one. Tests are the correctness gate; there's no hardware, network, or model dependency in the
suite — audio, vision, and serial are all driven through fakes and injected clocks, so tests must stay
that way when adding new ones.

## Architecture

```
face_track.py     entry point: CLI, inference loop, startup ordering
app/               process concerns: control loop, camera supervisor, lifecycle
ai/                capture, wake, STT, RAG, LLM, TTS, the conversation state machine
vision/            camera sources, face params, PD controller, gestures, presence
servo/             the Arduino serial link
web/               the dashboard: Flask routes and published state
config/            every tunable constant, one file per subsystem
settings.py        the live, dashboard-settable overlay on top of config/
arduino/           firmware (servo_serial.ino is the active sketch)
documents/         the corpus Kai answers from (+ the built RAG index)
```

Key design points that aren't obvious from any single file:

- **Two rates, one seam.** Inference runs at `INFERENCE_FPS` (15) and publishes a target via
  `vision/controller.py`'s thread-safe `TrackingTarget`; `app/control_loop.py` drives the servo at
  `CONTROL_FPS` (15) independently. Raising `INFERENCE_FPS` starves the control thread via the GIL —
  see the measurement note in `config/tracking.py` before changing either.
- **The jaw is not part of pan/tilt.** It has its own 20 Hz serial channel (`J<angle>`) written from
  the main loop, so mouth animation never contends with the 10 Hz pan/tilt serial gate and keeps
  moving even if the camera stalls.
- **Everything degrades, nothing crashes.** No camera, no servo, no mic, no Flask, no wake engine —
  each is a reported state the robot keeps running through. Camera and mic hot-swap live via
  supervisor threads (`app/camera_supervisor.py`, `ai/mic_hotplug.py`); new hardware-optional code
  should follow this pattern rather than fail startup.
- **One capture stream, fanned out.** `ai/mic_stream.py` owns the single open audio stream; wake
  detection, VAD, and the utterance buffer all consume the same fan-out — never open a second stream.
- **`config/` is the source of truth for restart-only tunables**, one file per subsystem, each
  literal annotated with the measurement that set it (read the comment before changing a number — most
  encode a specific past failure, not a preference). A subset is mirrored as live dashboard knobs
  through `settings.py`, persisted as an overlay in `~/.config/kai/settings.json` (never committed).
  See [config/README.md](config/README.md) for which knobs are live vs. restart-only.
- **The dashboard has no authentication** (Flask binds `0.0.0.0`) — a deliberate trade for a home LAN
  demo, tracked as a known exposure in `docs/tickets/S7-unauthenticated-dev-server-dashboard.md`. Don't
  add real secrets or trust boundaries around it without addressing that first.

## Engineering process

- Known issues and planned work are tracked as one file per finding in
  [docs/tickets/](docs/tickets/) (tiered by impact vs. effort) and design docs in
  [docs/plan/](docs/plan/) (`completed/` vs `wip/`). Check there before assuming something is
  undiscovered, and check [docs/tickets/README.md](docs/tickets/README.md)'s framing notes — its
  cross-ticket dependency section says which tickets block or ease each other.
- A landed ticket keeps its file and gains either a `> **Status: FIXED**` banner at the top (naming
  the branch and change, preferred) or a `## Resolution` section — it is not deleted.
- [docs/memory-budget.md](docs/memory-budget.md) documents what's resident in the Jetson's shared 8GB
  — read it before changing the LLM model or `OLLAMA_NUM_CTX`.
