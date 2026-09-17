# R12 — `send_gesture()` blocks the main tracking loop for the same ~2s a USB reconnect takes

| | |
|---|---|
| **Tier** | 1 |
| **Severity** | Medium |
| **Effort** | Small |
| **Confidence** | High |
| **Lens** | Robotics |

## Location

- `face_track.py` — `run()`'s main loop, `servo.send_gesture(gesture)`
- `servo/servo.py` — `send_gesture()`, `_write()`, `_reconnect()`

## Problem

`send_gesture()` takes `self._lock` with a **blocking** acquire and calls `_write()` from inside it.
When the CH340 link has dropped, `_write()`'s failure path re-opens the port: `_reconnect()` sleeps
2 s so the Arduino can reboot after the DTR toggle, and that sleep runs while `send_gesture()` is
still holding the lock. Every other write path already accounts for a lock held this long —

- `send()` runs on the dedicated 15 Hz `servo-control` thread, and R1 already tickets the control
  thread stalling for up to `RECONNECT_INTERVAL` + the DTR wait.
- `send_jaw()` deliberately uses a **non-blocking** acquire for exactly this reason: "if the serial
  lock is held (e.g. a ~2 s USB reconnect on the control thread) we skip this jaw frame rather than
  stall the whole loop" (the comment in `servo/servo.py`).

`send_gesture()` is the one caller that does neither. It is invoked from `face_track.py`'s **main**
tracking loop (`GestureDetector.update()` → `if gesture: servo.send_gesture(gesture)`), which is the
same thread that publishes `/params` and `/video`, drives the jaw's independent fast channel, and
schedules the next MediaPipe inference. A gesture landing while the CH340 is flapping — the exact
condition R1's own log evidence and R4's brownout note both describe as a real, observed state on
this hardware — freezes all four of those for the ~2 s the reconnect takes, not just the pan/tilt
control loop R1 describes.

## Why it matters

The architecture's stated guarantee is that the jaw's fast channel and the dashboard keep working
"even if the camera stalls" (`docs/architecture.md`) — the design explicitly decouples them from
servo trouble via `send_jaw()`'s non-blocking pattern. `send_gesture()` reintroduces the coupling it
was built to avoid: a nod or shake detected during a serial brownout freezes the video feed, the
`/params` SSE stream, and the jaw pantomime for as long as the reconnect takes, on a thread nothing
else is watching (`R8`, no main-loop watchdog, is still open). The dropped frames this costs are
invisible in the log — nothing here logs a stall, only the eventual `[servo] reconnected on …` line.

## Suggested approach

Give `send_gesture()` the same non-blocking pattern `send_jaw()` already uses: `self._lock.acquire(blocking=False)`,
and drop the gesture (log a counter, the way `MicStream` counts `dropped_blocks`) if the link is
mid-reconnect. A missed gesture ack is a smaller loss than a frozen video feed and dashboard, and it
matches the precedent already set for the jaw channel. Update `tests/test_servo.py` alongside
`R11`/`R1`'s coverage of the reconnect path.

## Acceptance criteria

- [ ] `send_gesture()` uses a non-blocking lock acquire and returns without sending when the lock is
      held, instead of blocking the caller.
- [ ] A counter or log line records a dropped gesture the way `MicStream.dropped_blocks` records a
      dropped audio block, so the failure is diagnosable over SSH.
- [ ] A unit test holds the serial lock (as R1's tests presumably will for the reconnect path) and
      asserts `send_gesture()` returns immediately rather than blocking.
- [ ] **Needs the robot**: trigger a CH340 flap during a detected nod/shake and confirm `/params`
      and `/video` keep updating through the reconnect window.
