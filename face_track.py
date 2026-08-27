#!/usr/bin/env python3
"""
Face tracking servo — Jetson Orin Nano

Usage:
  python3 face_track.py --network 192.168.1.x --no-display --flip
  python3 face_track.py --network 192.168.1.x --no-display --flip --tilt
  python3 face_track.py --network 192.168.1.x --no-display --flip --lofi
"""

from __future__ import annotations

import argparse
import os
import socket
import threading
import time

import cv2
import mediapipe as mp
import numpy as np

from app               import control_loop, lifecycle
from app.camera_supervisor import CameraSupervisor
from web.server         import Dashboard
from web.state          import DashboardState
from vision.camera      import CameraThread, NullCamera
from vision.controller  import EMAFilter, TrackingTarget
from vision.face_params import FaceParams, NOSE_TIP, PROCESS_W, PROCESS_H, compute_face_params, classify_emotion
from vision.gesture     import GestureDetector
from vision            import presence
from servo.servo        import ServoSerial
from ai.voice_assistant import VoiceAssistant
from ai.session        import ConversationSession
from ai import rag, tts
import settings


class _NullServo:
    """Drop-in ServoSerial replacement when --no-servo is set, or when the serial port is missing."""
    last_pan = 90; last_tilt = 90; last_jaw = 90
    def send(self, pan, tilt, jaw=None): return False
    def send_jaw(self, angle): return False
    def send_gesture(self, name): return False
    def center(self): pass
    def close(self): pass


# ── Tuning ────────────────────────────────────────────────────────────────────
# Tunable knobs live in config/tracking.py; re-imported here so the names stay module-level.
from config.tracking import (
    INFERENCE_FPS, NO_FRAME_WAIT, EMA_ALPHA, PAN_SCALE, TILT_SCALE, MIN_FACE_AREA,
    JAW_CLOSED, JAW_OPEN, JAW_EMA_ALPHA, JAW_DEADBAND, EMA_RESET_FRAMES,
    WEB_PUBLISH_INTERVAL, FACE_MIN_DETECTION_CONF, FACE_MIN_TRACKING_CONF,
    WEB_PORT, UPLOAD_DIR, SERVO_ABSENCE_FRAMES, NO_FACE_LOG_INTERVAL_S,
)
from config.wake import SESSION_START_ATTEMPTS, SESSION_START_BACKOFF_S

INFERENCE_INTERVAL = 1.0 / INFERENCE_FPS   # derived from INFERENCE_FPS

os.makedirs(UPLOAD_DIR, exist_ok=True)

# The live camera's owner: probing, hot-swap, release, and the state the dashboard reads. Built at
# import time (the routes need something to talk to before run() has parsed the CLI) and given the
# CLI facts by configure() in run().
_camera = CameraSupervisor()

# ── Voice ─────────────────────────────────────────────────────────────────────
# Module scope, NOT behind a flask check: neither of these touches Flask, and constructing them
# there would mean "no flask installed -> no wake word", with nothing saying so. The dashboard
# routes are only a second way to reach them; the wake word is the first.
_voice   = VoiceAssistant()
_session = ConversationSession(_voice, presence=presence.snapshot)

# ── Dashboard ─────────────────────────────────────────────────────────────────
# The published state, and the Flask app built around it. _flask_app is None when flask is not
# installed, which is the headless case: the robot still runs and the wake word still works,
# nothing is served.
_web       = DashboardState()
_dashboard = Dashboard(voice=_voice, session=_session, camera=_camera, state=_web)
_flask_app = _dashboard.create_app()


# ── Servo availability ────────────────────────────────────────────────────────
# Set once by _open_servo; the dashboard shows it so an unplugged Arduino is visible rather than
# just silently motionless.
_servo_state = {"ok": True, "reason": ""}


def _open_servo(port: str):
    """Open the servo serial link, or fall back to _NullServo.

    ServoSerial's constructor opens the port and raises if it is missing — and this runs during
    startup, where a raise takes the dashboard, MediaPipe, the control thread and the wake word down
    with it. Under cron @reboot there is no supervisor to restart us, so an unplugged CH340 would mean
    a dead robot until the next reboot. Once running, ServoSerial already reconnects on its own.
    """
    try:
        return ServoSerial(port)
    except Exception as exc:
        _servo_state.update(ok=False, reason=f"{type(exc).__name__}: {exc}")
        print(f"[servo] WARNING: serial port unavailable ({exc}) — running without servos; "
              f"the dashboard, voice and wake word are unaffected", flush=True)
        return _NullServo()


# ── Run helpers ───────────────────────────────────────────────────────────────

def _start_web_server(resolve_mic: bool = True) -> None:
    thread = threading.Thread(
        target=lambda: _flask_app.run(host='0.0.0.0', port=WEB_PORT, debug=False,
                                      use_reloader=False, threaded=True),
        daemon=True,
    )
    thread.start()
    threading.Thread(target=_voice.ensure_model_loaded, daemon=True).start()
    # The wake-scan model too, or the very first "Hey Kai" pays its load inside the check and times out.
    threading.Thread(target=_voice.ensure_scan_model_loaded, daemon=True).start()
    if resolve_mic:
        # Skipped when the session owns capture: the raw I2S hw device admits exactly ONE opener, and
        # probing it here while MicStream is opening it deadlocks both — which also freezes the
        # tracking loop, since the mic is resolved before the main loop starts.
        threading.Thread(target=_voice.ensure_input_resolved, daemon=True).start()
    threading.Thread(target=_voice.ensure_llm_warm, daemon=True).start()
    threading.Thread(target=rag.load_index, daemon=True).start()
    threading.Thread(target=rag.ensure_model_loaded, daemon=True).start()
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        local_ip = "localhost"
    print(f"[face_track] Web dashboard: http://localhost:{WEB_PORT}  (network: http://{local_ip}:{WEB_PORT})")


def _gate_face(results) -> tuple[bool, object]:
    """Check confidence gate. Returns (valid, landmarks_or_None)."""
    if not results.multi_face_landmarks:
        return False, None
    lm = results.multi_face_landmarks[0].landmark
    xs = np.fromiter((l.x for l in lm), dtype=np.float32, count=len(lm))
    ys = np.fromiter((l.y for l in lm), dtype=np.float32, count=len(lm))
    if (xs.max() - xs.min()) * (ys.max() - ys.min()) < MIN_FACE_AREA:
        return False, None
    return True, lm


def _jaw_on(args: argparse.Namespace) -> bool:
    """Is the jaw servo both present and wanted?

    --jaw stays a hard AND: like --no-servo it declares what is physically wired, and no setting can
    conjure a jaw servo that isn't there. The setting only decides whether to drive one that is.
    """
    return bool(args.jaw) and bool(settings.get("jaw_enabled"))


def _compute_targets(lm, fp: FaceParams,
                     ema_x: EMAFilter, ema_y: EMAFilter,
                     args: argparse.Namespace,
                     jaw_on: bool) -> tuple[float, float, int | None]:
    """EMA smooth → target pan/tilt angles and jaw."""
    raw_x = 1.0 - lm[NOSE_TIP].x if args.flip   else lm[NOSE_TIP].x
    raw_y = 1.0 - lm[NOSE_TIP].y if args.flip_y else lm[NOSE_TIP].y
    x     = ema_x.update(raw_x)
    y     = ema_y.update(raw_y)
    target_pan  = max(0.0, min(180.0, (x - 0.5) * PAN_SCALE  + 90.0))
    target_tilt = max(0.0, min(180.0, (y - 0.5) * TILT_SCALE + 90.0)) if args.tilt else 90.0
    jaw  = int(JAW_CLOSED + (JAW_OPEN - JAW_CLOSED) * fp.mouth / 99.0) if jaw_on else None
    return target_pan, target_tilt, jaw


def _speaking_jaw(openness: float) -> int:
    """Map speaking openness (0..1 from the voice assistant) onto a jaw servo angle,
    using the same closed/open range as the human-mouth mirror."""
    return int(round(JAW_CLOSED + (JAW_OPEN - JAW_CLOSED) * openness))


def _annotate_frame(frame, lm, pan: int, tilt: int, jaw: int | None,
                    args: argparse.Namespace) -> None:
    h, w = frame.shape[:2]
    cx, cy = int(lm[NOSE_TIP].x * w), int(lm[NOSE_TIP].y * h)
    cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)
    parts = [f"pan:{pan}"]
    if args.tilt: parts.append(f"tilt:{tilt}")
    if args.jaw:  parts.append(f"jaw:{jaw}")
    cv2.putText(frame, "  ".join(parts), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def _log_face(pan: int, tilt: int, jaw: int | None, fp: FaceParams,
              fps: float, sent: bool, args: argparse.Namespace) -> None:
    if args.lofi:
        print(fp.to_lofi_string())
        return
    status = "sent" if sent else "hold"
    tail   = (f"yaw={fp.yaw} pitch={fp.pitch} roll={fp.roll} "
               f"mouth={fp.mouth} leye={fp.left_eye} reye={fp.right_eye} "
               f"dist={fp.distance} smile={fp.smile_kiss} emotion={classify_emotion(fp)}")
    if args.tilt:
        print(f"[face_track] pan={pan}° tilt={tilt}° {status} | {fps:.0f}fps | {tail}")
    else:
        print(f"[face_track] pan={pan}° {status} | {fps:.0f}fps | {tail}")


def _make_face_data(fp: FaceParams | None, gesture: str | None, frame_t: float) -> dict:
    if fp is None:
        return {"face_visible": 0, "x": 0, "y": 0, "distance": 0,
                "yaw": 0, "pitch": 0, "roll": 0, "mouth": 0,
                "left_eye": 0, "right_eye": 0, "smile_kiss": 0,
                "gesture": "", "gesture_ts": 0, "emotion": ""}
    p = fp.clamp()
    return {"face_visible": p.face_visible, "x": p.x, "y": p.y,
            "distance": p.distance, "yaw": p.yaw, "pitch": p.pitch,
            "roll": p.roll, "mouth": p.mouth, "left_eye": p.left_eye,
            "right_eye": p.right_eye, "smile_kiss": p.smile_kiss,
            "gesture": gesture or "", "gesture_ts": frame_t if gesture else 0,
            "emotion": classify_emotion(fp)}


def _publish_web(frame, fp: FaceParams | None, gesture: str | None,
                 frame_t: float, last_web_t: float) -> float:
    """Store the latest raw frame + face params if WEB_PUBLISH_INTERVAL has elapsed. JPEG encoding
    happens lazily in the /video generator (only while a client is connected), not here.

    Frame-dependent state only — pan/tilt/jaw and cam_source moved to _publish_status, which runs
    even when there are no frames at all."""
    now = time.monotonic()
    if _flask_app is None or now - last_web_t < WEB_PUBLISH_INTERVAL:
        return last_web_t
    face_data = _make_face_data(fp, gesture, frame_t)
    # Effective feed rate = inter-publish interval (steady ~25 fps while streaming),
    # which is meaningful even though inference is decimated below the camera rate.
    fps_val   = int(1.0 / (now - last_web_t)) if last_web_t else 0
    _web.publish_frame(frame, {**face_data, "fps": fps_val}, now)
    return now


def _publish_status(cam_thread, servo, last_status_t: float) -> float:
    """Publish the frame-INDEPENDENT half of the dashboard state.

    Called every loop iteration, including the no-frame path, so the dashboard can be honest on a
    robot with no camera: cam_source/cam_reason explain the situation, and pan/tilt/jaw keep updating
    because the jaw genuinely animates from the voice assistant with no camera at all.

    Gated at the same 25 Hz as _publish_web, so the ~200 Hz no-frame loop does not rebuild this dict
    on every pass.
    """
    now = time.monotonic()
    if _flask_app is None or now - last_status_t < WEB_PUBLISH_INTERVAL:
        return last_status_t

    cam    = _camera.snapshot()
    source = cam_thread.source_name
    status = {
        "pan": servo.last_pan, "tilt": servo.last_tilt, "jaw": servo.last_jaw,
        "cam_source":      source,
        "cam_reason":      cam["reason"] if source == "none" else "",
        "cam_mode":        cam["mode"],
        "cam_mode_locked": cam["locked"],
        "cam_retry_in_s":  max(0.0, round(cam["next_probe_at"] - now, 1)) if cam["next_probe_at"] else 0.0,
        "servo_ok":        _servo_state["ok"],
        "servo_reason":    _servo_state["reason"],
        "set_error":       settings.persist_error(),
    }
    status.update({f"set_{name}": value for name, value in settings.snapshot().items()})

    # Built outside the state's lock (settings and the supervisor have their own); publish_status
    # holds it for the assignment only, and it also expires the face data once frames stop.
    _web.publish_status(status, now)
    return now


def _register_settings_callbacks() -> None:
    """Subscribe the knobs that cannot simply be read at the point of use.

    Everything else (camera_mode, servo_tracking, jaw_enabled, and all six TTS values) is PULLED via
    settings.get() where it is used, which needs no wiring at all. The ones here either live inside an
    object that holds its own copy, or need a side effect beyond storing a number.
    """
    settings.on_change("hands_free",     lambda v: _session.set_hands_free(v))
    settings.on_change("vad_rms_floor",  lambda v: _session.set_rms_floor(v))
    settings.on_change("wake_sensitivity", lambda v: _session.set_wake_sensitivity(v))
    # Debounced: the cached wake-ack WAVs are re-synthesised through Piper (~1s each), and a dragged
    # slider fires many changes. Without this, "Yes?" would keep the OLD voice forever while every
    # other reply used the new one.
    settings.on_change("tts_volume",       lambda v: _session.reprewarm_canned(), debounce=1.5)
    settings.on_change("tts_length_scale", lambda v: _session.reprewarm_canned(), debounce=1.5)
    # The prosody parameters are the same case: the filler bank, the wake ack and the greeting are all
    # pre-synthesised, so without these a sentence-pause or noise change would apply to live replies
    # only and the cached lines would audibly disagree with them mid-conversation.
    settings.on_change("tts_sentence_silence", lambda v: _session.reprewarm_canned(), debounce=1.5)
    settings.on_change("tts_noise_scale",  lambda v: _session.reprewarm_canned(), debounce=1.5)
    settings.on_change("tts_noise_w",      lambda v: _session.reprewarm_canned(), debounce=1.5)

    # Apply the persisted values ONCE at startup. Callbacks only fire on change, and the gate and the
    # wake detector were constructed from the config defaults before settings.load() ran — so a value
    # restored from settings.json would otherwise sit in the overlay doing nothing until someone
    # nudged it. (hands_free is applied via _session.enabled below; calling set_hands_free() here
    # would open the wake engine before the session has started.)
    _session.set_rms_floor(settings.get("vad_rms_floor"))
    _session.set_wake_sensitivity(settings.get("wake_sensitivity"))


# ── Main tracking loop ────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    # Startup order matters: everything camera-independent comes up FIRST, and the camera is the last
    # thing attempted. Opening the camera here used to be line 1, so a missing ribbon took the
    # dashboard, MediaPipe, the servo control thread and the wake word down with it — none of which
    # need a camera. Nothing between here and the supervisor below can raise on absent hardware.
    settings.load()
    # Before the dashboard can be served, so a --no-camera run never briefly shows the camera control
    # as changeable.
    _camera.configure(args.camera, args.network, args.network_port, args.no_camera)

    servo       = _NullServo() if args.no_servo else _open_servo(args.port)
    camera      = NullCamera("starting up")
    cam_thread  = CameraThread(camera, _camera.swap_queue).start()  # reads frames off the loop
    _camera.attach_thread(cam_thread)                       # so the supervisor can watch frame health
    print(f"[face_track] flip={args.flip}  tilt={args.tilt}  lofi={args.lofi}  "
          f"ema={EMA_ALPHA}  infer={INFERENCE_FPS}fps")

    if _flask_app is not None:
        _start_web_server(resolve_mic=not args.wake)

    mp_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=FACE_MIN_DETECTION_CONF,
        min_tracking_confidence=FACE_MIN_TRACKING_CONF,
    )
    ema_x    = EMAFilter(EMA_ALPHA)
    ema_y    = EMAFilter(EMA_ALPHA)
    ema_jaw  = EMAFilter(JAW_EMA_ALPHA)
    last_jaw_cmd  = JAW_CLOSED
    no_face_frames = 0
    detector      = GestureDetector(inference_fps=INFERENCE_FPS)

    # Pan/tilt is driven by a dedicated control thread at CONTROL_FPS (PD + slew clamp live there);
    # this inference loop only publishes the latest target angles into `target`.
    target         = TrackingTarget()
    stop_evt       = threading.Event()
    control_thread = threading.Thread(
        target=control_loop.run, args=(servo, target, _voice, stop_evt),
        daemon=True, name="servo-control")
    control_thread.start()

    # Hands-free listening. Started here rather than in _start_web_server() so it comes up with
    # --no-camera and with flask absent — the wake word is not a dashboard feature.
    #
    # On its OWN thread, deliberately. Opening the raw I2S hw device can block indefinitely when
    # something else still holds it (a previous face_track that hasn't fully exited, or pulse
    # re-grabbing the card), and face tracking must never wait on the microphone to start.
    if args.wake:
        # --no-hands-free is a hard override (it declares the intent for this run); the setting decides
        # otherwise, so the dashboard toggle survives a restart.
        _session.enabled = settings.get("hands_free") and not args.no_hands_free

        def _start_session() -> None:
            # Retry: losing the race for the single-opener capture device is usually transient (a
            # previous face_track still exiting, pulse re-grabbing the card), but a single attempt
            # turns that into a deaf robot for the whole run.
            #
            # BACKOFF, not a flat interval. The failure this exists for is startup contention, and
            # the old flat 5 x 5 s window was shorter than the thing it was racing — a 26 s Ollama
            # model load. Retries that all land inside the storm cannot win. See
            # SESSION_START_BACKOFF_S in config/wake.py for the measurement.
            for attempt in range(1, SESSION_START_ATTEMPTS + 1):
                if _session.start():
                    print(f"[face_track] voice capture: shared always-open stream "
                          f"({'hands-free' if _session.enabled else 'push-to-talk only'})"
                          + (f" — recovered on attempt {attempt}" if attempt > 1 else ""),
                          flush=True)
                    return
                if attempt < SESSION_START_ATTEMPTS:
                    delay = SESSION_START_BACKOFF_S[min(attempt - 1,
                                                        len(SESSION_START_BACKOFF_S) - 1)]
                    print(f"[face_track] shared capture unavailable (attempt {attempt}/"
                          f"{SESSION_START_ATTEMPTS}) — retrying in {delay:.0f}s"
                          f" [{_session.mic_error() or 'no reason reported'}]",
                          flush=True)
                    time.sleep(delay)
            print("[face_track] WARNING: shared capture unavailable after "
                  f"{SESSION_START_ATTEMPTS} attempts — falling back to "
                  "per-turn push-to-talk recording", flush=True)
            _voice.attach_mic(None)          # let the legacy per-turn path open its own stream
            threading.Thread(target=_voice.ensure_input_resolved, daemon=True).start()
            # Then keep watching. The retries above cover startup contention — something else still
            # holding the single-opener device — but not "there is no microphone attached", and that
            # is the state someone is most likely to fix by plugging one in. The tick loop's
            # hot-plug check cannot help here: start() never succeeded, so there is no tick loop.
            # This thread has finished its work either way, so watching costs nothing new, and
            # start() re-attaches the shared stream to the assistant if a mic does turn up.
            _session.watch_for_a_mic(stop_evt)

        threading.Thread(target=_start_session, daemon=True, name="kai-session-start").start()
    else:
        print("[face_track] hands-free disabled (--wake not set) — push-to-talk only", flush=True)

    _register_settings_callbacks()

    # LAST, and on its own thread: the only camera-dependent step. It probes, hot-swaps a real camera
    # in when one appears, and releases it when camera_mode goes to "off".
    cam_thread_sup = threading.Thread(target=_camera.run, args=(stop_evt,),
                                      daemon=True, name="kai-camera")
    cam_thread_sup.start()

    show        = not args.no_display
    last_log_t  = 0.0
    last_web_t  = 0.0
    last_status_t = 0.0
    last_infer_t = 0.0
    prev_jaw_on  = _jaw_on(args)
    last_fp: FaceParams | None = None    # persists between inference frames for the web feed
    frame_times: list[float] = []

    try:
        while True:
            frame = cam_thread.latest()   # non-blocking; None when no new frame yet
            # Drive the speaking jaw every iteration, independent of frames — the mouth must
            # keep animating mid-sentence even if the camera stalls. Uses the fast jaw-only
            # channel (20 Hz), decoupled from the 10 Hz pan/tilt gate.
            # No longer gated on flask: _voice is constructed at module scope now, and the wake
            # word can drive a reply with no dashboard running at all.
            jaw_on = _jaw_on(args)
            if prev_jaw_on and not jaw_on:
                servo.send_jaw(JAW_CLOSED)   # switched off mid-word: close it, don't freeze it open
            prev_jaw_on = jaw_on
            speak_open = _voice.speaking_openness() if jaw_on else None
            if speak_open is not None:
                servo.send_jaw(_speaking_jaw(speak_open))
            # Before the no-frame bail-out: with no camera this is the ONLY thing publishing state,
            # and it is what lets the dashboard say "no camera, and here's why" instead of showing a
            # frozen LIVE badge.
            last_status_t = _publish_status(cam_thread, servo, last_status_t)
            if frame is None:
                # Block for the next frame instead of polling for it. Wakes immediately when one is
                # stored, and otherwise after NO_FRAME_WAIT — which is what keeps the jaw animating
                # and _publish_status honest on a robot with no camera at all. See config/tracking.py
                # for why the bound is the publish interval and not the jaw interval.
                cam_thread.wait_for_frame(NO_FRAME_WAIT)
                continue

            live_rotate = args.rotate and cam_thread.source_name != "video_file"
            if live_rotate == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif live_rotate == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif live_rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            frame_t = time.monotonic()
            publish_gesture: str | None = None

            # Inference decimation: run MediaPipe only every INFERENCE_INTERVAL. Between
            # inferences we still publish the raw frame (smooth video) and coast the servo.
            if frame_t - last_infer_t >= INFERENCE_INTERVAL:
                last_infer_t = frame_t
                pw, ph   = (PROCESS_H, PROCESS_W) if live_rotate in (90, 270) else (PROCESS_W, PROCESS_H)
                small    = cv2.resize(frame, (pw, ph))
                results  = mp_mesh.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                face_valid, lm = _gate_face(results)

                if face_valid:
                    no_face_frames = 0
                    presence.mark(True, frame_t)   # ai/session.py reads this for session end
                    will_log = frame_t - last_log_t >= 0.5
                    # solvePnP (yaw/pitch/roll) only feeds logging + the dashboard, so skip it
                    # unless we're about to log or someone is watching the video feed.
                    fp = compute_face_params(lm, compute_pose=(will_log or _web.has_video_client()))
                    gesture = detector.update(fp, frame_t)
                    if gesture:
                        servo.send_gesture(gesture)
                    publish_gesture = gesture

                    target_pan, target_tilt, mirror_jaw = _compute_targets(lm, fp, ema_x, ema_y,
                                                                           args, jaw_on)
                    # Hand pan/tilt to the control thread (it runs the PD + slew clamp at CONTROL_FPS
                    # and glides continuously between these inference ticks).
                    target.set(target_pan, target_tilt, True)

                    # Jaw stays on this thread via the 'J' fast channel. Speaking pantomime (driven
                    # at the top of the loop) overrides the mouth-mirror during a reply.
                    if speak_open is not None:
                        jaw_for_log = servo.last_jaw
                    elif mirror_jaw is not None:
                        jaw_s = int(round(ema_jaw.update(float(mirror_jaw))))
                        if abs(jaw_s - last_jaw_cmd) >= JAW_DEADBAND:
                            last_jaw_cmd = jaw_s
                        servo.send_jaw(last_jaw_cmd)
                        jaw_for_log = last_jaw_cmd
                    else:
                        jaw_for_log = None

                    frame_times.append(time.monotonic() - frame_t)
                    # Annotate/log the control thread's actual commanded angles (servo.last_*).
                    _annotate_frame(frame, lm, servo.last_pan, servo.last_tilt, jaw_for_log, args)
                    last_fp = fp
                    if will_log:
                        fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
                        frame_times.clear()
                        _log_face(servo.last_pan, servo.last_tilt, jaw_for_log, fp, fps, True, args)
                        last_log_t = frame_t
                else:
                    no_face_frames += 1
                    # presence gets the truth every frame — the session-end rules need real absence.
                    presence.mark(False, frame_t)
                    # The SERVO target does not: one dropped detection is not someone leaving. Flipping
                    # the control loop into its hold branch resets the PD, which used to cost a visible
                    # head jump on every MediaPipe flicker. Below the grace we simply publish nothing,
                    # so the loop keeps gliding toward the last good target. SERVO_ABSENCE_FRAMES stays
                    # well under CONTROL_STALE_TIMEOUT, so a real absence is never missed.
                    if no_face_frames >= SERVO_ABSENCE_FRAMES:
                        target.set(servo.last_pan, servo.last_tilt, False)   # control thread holds
                    if speak_open is None and jaw_on:
                        servo.send_jaw(JAW_CLOSED)   # rest closed; send_jaw's deadband self-quiesces
                    if no_face_frames >= EMA_RESET_FRAMES:
                        ema_x.reset()
                        ema_y.reset()
                    if args.lofi:
                        # A machine-readable stream, not a human log — unchanged 1 Hz cadence.
                        if frame_t - last_log_t >= 1.0:
                            print(FaceParams.no_face().to_lofi_string())
                            last_log_t = frame_t
                    elif no_face_frames == 1 or frame_t - last_log_t >= NO_FACE_LOG_INTERVAL_S:
                        # The transition is the news; the rest is a heartbeat. NO_FACE_LOG_INTERVAL_S
                        # explains why this is no longer once per second forever.
                        print(f"[face_track] NO FACE — pan={servo.last_pan}°")
                        last_log_t = frame_t
                    cv2.putText(frame, "NO FACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    last_fp = None

            last_web_t = _publish_web(frame, last_fp, publish_gesture, frame_t, last_web_t)

            if show:
                cv2.imshow("face_track", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        # FIRST, before anything else is torn down. tts.stop() kills both the Piper synth and the
        # paplay playback; those are child PROCESSES, not threads, so the interpreter exiting does
        # not take them with it — they get re-parented and keep playing. scripts/autostart.sh brings
        # the replacement up within seconds, so without this the new process starts talking over
        # audio the old one left in the air, with its own mute gate wide open because it knows
        # nothing about that sound. Doing it first also means the 1-2 s of teardown below happens in
        # silence rather than under a half-spoken reply.
        # NOTE: lifecycle.arm_restart_deadline()'s os._exit path deliberately skips this whole
        # block; scripts/autostart.sh's wait_for_capture_device is the backstop there.
        tts.stop()
        stop_evt.set()
        control_thread.join(timeout=1.0)   # stop the control thread before closing the serial
        # Before cam_thread.close(), or the supervisor could hot-swap a camera into a closing thread.
        cam_thread_sup.join(timeout=1.0)
        # Before the servo, so the mic and Porcupine are released even if the serial close hangs.
        # Porcupine holds native memory — leaking it across restarts is a real leak.
        _session.stop()
        mp_mesh.close()
        cam_thread.close()
        servo.close()
        if show:
            cv2.destroyAllWindows()
        print("[face_track] Stopped.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    from vision.camera import NETWORK_PORT
    parser = argparse.ArgumentParser(description="Face tracking servo — Jetson Orin Nano")
    parser.add_argument("--camera",       type=int, default=0)
    parser.add_argument("--network",      metavar="HOST", default=None)
    parser.add_argument("--network-port", type=int, default=NETWORK_PORT)
    parser.add_argument("--port",         default="/dev/ttyUSB0")
    parser.add_argument("--flip",         action="store_true", help="Invert pan direction")
    parser.add_argument("--flip-y",       action="store_true", help="Invert tilt direction")
    parser.add_argument("--tilt",         action="store_true", help="Enable Y-axis tilt servo (Pin 10)")
    parser.add_argument("--jaw",          action="store_true", help="Enable jaw servo driven by mouth openness (Pin 6)")
    parser.add_argument("--lofi",         action="store_true", help="Output LOFI param string each tick (19-digit format)")
    parser.add_argument("--no-display",   action="store_true")
    parser.add_argument("--no-servo",     action="store_true", help="Skip servo hardware (camera + face detection only)")
    parser.add_argument("--no-camera",    action="store_true", help="Skip camera hardware (dashboard + voice assistant only)")
    parser.add_argument("--rotate",       type=int, default=0, choices=[0, 90, 180, 270],
                                          help="Rotate camera feed clockwise (degrees)")
    parser.add_argument("--wake",         action="store_true",
                                          help="Hands-free: one always-open mic, 'Hey Kai' wake word, "
                                               "VAD turn-taking (see config/wake.py)")
    parser.add_argument("--no-hands-free", action="store_true",
                                          help="With --wake: use the shared always-open stream but "
                                               "keep push-to-talk as the only way in")
    if not lifecycle.claim_single_instance():
        print(f"[face_track] another instance already holds {lifecycle.LOCK_PATH} — refusing to "
              "start (stop the running one first: pkill -f face_track.py)", flush=True)
        raise SystemExit(lifecycle.EXIT_ALREADY_RUNNING)
    lifecycle.install_signal_handlers()
    run(parser.parse_args())
    # After run()'s `finally` — everything is released, so the replacement can have the hardware.
    if lifecycle.restart_requested.is_set():
        print(f"[face_track] exiting {lifecycle.EXIT_RESTART} for a supervised restart", flush=True)
        raise SystemExit(lifecycle.EXIT_RESTART)


if __name__ == "__main__":
    main()
