# A6 — Background TTS warm threads are never stopped on shutdown, reopening R7 for the filler bank

> **Status: FIXED**, 2026-09-17. `ConversationSession` now has `_warm_stop` (a `threading.Event`,
> set in `stop()` before `tts.stop()`) and `_warm_threads` (every `kai-ack-warm`/`kai-ack-rewarm`
> thread it spawns). `_quiet_for_synth()` returns `False` once `_warm_stop` is set, and every sleep
> in `_warm_all`/`_speak_greeting`/`_warm_one`/`_prewarm_bank`/`_rewarm_when_quiet` waits on the
> event instead of `time.sleep()`, with an early-return check before each place that would otherwise
> start a new Piper run. `stop()` joins the warm threads with a `WARM_JOIN_TIMEOUT_S` (2.0 s) timeout
> and logs if one does not stop in time. Test:
> `tests/test_session.py::TestStop::test_stop_signals_and_joins_a_live_warm_thread_before_it_can_synthesize_again`
> starts a fake long-running warm thread, calls `stop()`, and asserts it is joined and no further
> `tts.prewarm_canned` call happens. The on-hardware restart-cycling criterion is still open.

| | |
|---|---|
| **Tier** | 2 |
| **Severity** | Medium |
| **Effort** | Medium |
| **Confidence** | Medium |
| **Lens** | AI |

## Location

- `ai/session.py` — `start()` (spawns `kai-ack-warm`), `reprewarm_canned()` (spawns `kai-ack-rewarm`),
  `_warm_all()`, `_prewarm_bank()`, `_rewarm_when_quiet()`
- `ai/session.py` — `stop()`
- `face_track.py` — `run()`'s `finally`

## Problem

`R7` fixed the orderly-shutdown case for the **reply-speaking** path: `run()`'s `finally` calls
`tts.stop()` first, and `ConversationSession.stop()` calls it again after joining the tick thread and
before releasing the mic. Both calls terminate whatever `_synth_proc`/`_current_proc` currently point
to at that instant.

Neither call **stops the two background warm threads** — `kai-ack-warm` and `kai-ack-rewarm`. They
are never given a stop event, never joined, and never told shutdown is happening. `_prewarm_bank()`'s
own docstring says it "runs for MINUTES" by design, polling `_quiet_for_synth()` and sleeping
`BANK_SYNTH_GAP_S` between attempts — so at any given moment during a long-running robot, one of
these threads is plausibly awake, mid-loop, checking whether it is safe to start a Piper run.

Shutdown proceeds like this: `tts.stop()` (finally, first line) → `stop_evt.set()` →
`control_thread.join(1.0)` → `cam_thread_sup.join(1.0)` → `_session.stop()` (which does its own
`tts.stop()` after joining the tick thread). That is up to ~2 s of orderly teardown during which
nothing is playing — which is exactly the condition `_quiet_for_synth()` is watching for. A warm
thread woken in that window sees quiet, as designed, and starts a **new** Piper run. If that new run
is still in `communicate()` when `_session.stop()`'s `tts.stop()` fires, it is caught (a second
`tts.stop()` call catches it — this is why R7's own fix calls it twice). But nothing calls `tts.stop()`
a third time after that, and the warm thread is never joined or signalled — it loops straight back to
`_quiet_for_synth()`, sees quiet again (the process is now in its final few teardown steps:
`mp_mesh.close()`, `cam_thread.close()`, `servo.close()`), and can start **another** Piper run in the
final window before the interpreter actually exits. A daemon thread blocked inside
`Popen.communicate()` at the moment the process exits does not get a chance to clean up its child —
the child is re-parented exactly as R7 originally described, on a path R7's own regression tests
(scoped to `ConversationSession.stop()`'s ordering) do not exercise.

## Why it matters

This is R7's exact failure mode — a `paplay`/`piper` child surviving process exit and talking over
the replacement process's greeting — reopened through a code path that has no watchdog and was not
in scope when R7 was written and tested. It is intermittent by nature (it needs a warm thread to be
mid-loop at the moment of shutdown), which is exactly the kind of bug the two on-hardware criteria
R7 already left deferred (`pgrep -af 'paplay|piper'` after `SIGTERM`) would not reliably catch unless
run enough times to land inside the race window.

## Suggested approach

Give `ConversationSession` a `_warm_stop` event, set in `stop()` before the `tts.stop()` call there.
Thread it through every sleep in `_warm_all`, `_prewarm_bank`, `_rewarm_when_quiet` and `_warm_one`
(`stop_evt.wait(...)` in place of `time.sleep(...)`, and an early-return check at the top of each
retry loop), and have `_quiet_for_synth()` return `False` once it is set — the same "never start a
new Piper run" contract the reply-speaking path already gets from `tts.stop()`, applied to the warm
path's own gating function instead of relying on an external kill. `stop()` should then join these
threads too (with a short timeout, logged if it expires, the same shape as `RESTART_FORCE_AFTER_S`)
so the "orderly shutdown finished" log line means what it says.

## Acceptance criteria

- [x] `ConversationSession` tracks its warm/rewarm threads and signals them to stop before or
      alongside `tts.stop()` in `stop()`.
- [x] `_quiet_for_synth()` (or an equivalent check reached by every warm call site) returns `False`
      once shutdown has been signalled, so no new Piper run can start after that point.
- [x] A test starts a fake long-running warm cycle, calls `stop()`, and asserts no further
      `tts.prewarm_canned`/`_run_piper` call happens afterward.
- [ ] **Needs the robot**: restart the robot several times while the filler bank is still warming
      (within the first few minutes of a fresh boot) and confirm no overlapping/orphaned audio.

## Cross-ticket note

Related to `R7` (same failure family, different code path) and `A5` (a `_run_piper` timeout narrows
but does not close this — see A5's cross-ticket note).
