# config/

Central home for Kai's **tunable** settings — one file per subsystem:

| File          | Tunes                                                             | Used by |
|---------------|------------------------------------------------------------------|---------|
| `servo.py`    | serial port/baud, send rates, servo angle limits, deadband       | `servo/` |
| `tracking.py` | inference fps, EMA/PD smoothing, jaw range, slew cap, web port    | `face_track.py`, `app/`, `web/server.py` |
| `gesture.py`  | nod / shake / proximity / mouth thresholds and windows           | `vision/gesture.py` |
| `camera.py`   | capture + processing resolution, network/stream ports            | `vision/` |
| `voice.py`    | Whisper + Ollama models, context size, jaw-speaking envelope, which mic and how it is found | `ai/voice_assistant.py`, `ai/llm.py`, `ai/transcript.py`, `ai/speak_envelope.py`, `ai/mic_device.py`, `ai/mic_hotplug.py` |
| `wake.py`     | wake engine chain, always-open capture, VAD turn-end, session timeouts, phrase matching | `ai/audio.py`, `ai/mic_stream.py`, `ai/session.py`, `ai/wake_phrase.py` |
| `rag.py`      | embedding model, chunking, top-k, similarity threshold, fuzzy "DEVCON" matching, entity gazetteer, lexical fallback, DEVCON failsafe chain | `ai/rag.py`, `ai/query_alias.py`, `ai/index_documents.py` |
| `thinking.py` | the "thinking" expression — pan sweep amplitude/period, the "hmm" delay | `app/control_loop.py`, `ai/session.py` |
| `filler.py`   | the filler bank — 20 openers + 20 stalls in tl/ceb/en, and the 2 s dead-air ceiling | `ai/filler.py`, `ai/session.py` |

`voice.py` covers a turn once the audio exists — plus *which* microphone the audio comes from, since
everything that decides that (`MIC_PREFERENCE`, the mic name hints, the speaker-card exclusion, the
hot-plug watcher) is more useful read in one place than split by which module consumes it.
`wake.py` covers everything that decides *when* Kai listens. The knobs most likely to need tuning on real hardware are `VAD_RMS_FLOOR`,
`WAKE_SENSITIVITIES`, and `TTS_TAIL_MUTE_S` — raise that last one first if Kai ever answers himself,
since `paplay` exits before the amp actually goes quiet. **The first two are now adjustable from the
dashboard while Kai is running** (see below), so `scripts/wake_test.py` is no longer the only way to
set the noise floor.

## Two kinds of setting

**These files hold the DEFAULTS, and changing one needs a restart.** Sixteen of them can also be changed
live from the dashboard (fifteen in the ⚙ Settings tab, plus the mic preference on the Microphone card), which stores an overlay in `~/.config/kai/settings.json`
(never committed, never written here — see `settings.py`).

| Dashboard knob | Default lives in | Applies |
|---|---|---|
| Camera (auto / off) | — (`auto`) | immediately; `auto` also picks up a camera plugged in later |
| Hands-free wake word | `wake.py` `HANDS_FREE_ENABLED` | immediately |
| Wake sensitivity | `wake.py` `WAKE_SENSITIVITIES[0]` | immediately (reloads the engine for the Porcupine tier) |
| Mic noise floor | `wake.py` `VAD_RMS_FLOOR` | immediately |
| Preferred microphone (`auto`/`i2s`/`usb`/`analog`) | `voice.py` `MIC_PREFERENCE` | **next time a mic is resolved** — the Microphone card's button, a hot-plug, or a restart. The one knob that is not immediate, which is why it lives on that card rather than in the Settings tab |
| Speak replies | `voice.py` `TTS_ENABLED` | next reply |
| Volume / Speaking rate | `voice.py` `TTS_VOLUME`, `TTS_LENGTH_SCALE` | next reply, and re-records the cached "Yes?" |
| Pause between sentences | `voice.py` `TTS_SENTENCE_SILENCE_S` | next reply, and re-records every cached line. 0 restores the pre-2026-08-10 sound, where sentences ran together with no breath |
| Tone / Rhythm variation | `voice.py` `TTS_NOISE_SCALE`, `TTS_NOISE_W` | next reply, and re-records every cached line. Both ship at the voice model's own values because moving them measured *less* than the run-to-run noise floor — see the CHANGELOG entry before spending time on them |
| Natural delivery | `voice.py` `DELIVERY_ENABLED` | next reply (breaths, tempo jitter, the occasional opener — `ai/delivery.py`) |
| Follow faces / Move the jaw | — (both on) | immediately |
| Think out loud | `thinking.py` `THINKING_SOUNDS` | next turn (gates the filler bank and the "hmm"; both are pre-synthesised at startup) |
| Sweep while thinking | `thinking.py` `THINKING_SWEEP` | immediately, on the next control tick; also needs Follow faces on |

Delete `~/.config/kai/settings.json`, or press **Restore defaults** in the dashboard, and you are back
to exactly the values in these files. Only knobs that differ from their default are stored, so editing
a default here still propagates.

Everything else is restart-only, by design: there is deliberately no half-applied "restart required"
category in the dashboard. `INFERENCE_FPS`, `CONTROL_FPS`, the MediaPipe confidences and the capture
resolution are baked into constructors or bounded by hardware, and `tracking.py` documents why.

## How to change a restart-only setting

1. Edit the value in the relevant file (comments explain what each one does and why).
2. Restart the affected process (e.g. `face_track.py`, or the cron `@reboot` service).

Values are plain Python literals — no parsing, no restart-safe hot-reload; a restart picks
them up. Each source module re-imports these names, so this package is the single source of
truth: change it here and every consumer sees it. (The sixteen live knobs are the exception: their
consumers read them through `settings.py`, which takes its defaults from here.)

## What is NOT here

Constants coupled to code correctness are intentionally left in their modules, because
changing them would break behavior rather than tune it: MediaPipe landmark indices and the
3D pose model (`vision/face_params.py`), the TCP frame-header size (`vision/camera.py`),
dashboard status strings (`ai/voice_assistant.py`), and file paths derived from `__file__`.
