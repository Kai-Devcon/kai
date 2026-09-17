# A5 — `_run_piper`'s `communicate()` has no timeout, so a wedged Piper leaks a thread forever

> **Status: FIXED**, 2026-09-17. `ai/tts.py`'s `_run_piper()` now passes `timeout=TTS_PIPER_TIMEOUT_S`
> (15.0, `config/voice.py`) to `communicate()`, catches `subprocess.TimeoutExpired`, kills the
> process, drains the pipes with a second `communicate()`, and returns `False` and logs exactly like
> every other synthesis failure. Test: `tests/test_tts.py::TestRunPiper::test_hung_piper_is_killed_and_reaped_within_the_call`
> stubs a `Popen` whose `communicate()` raises `TimeoutExpired` and asserts `_run_piper` returns
> `False` without hanging. No caller changes needed, as predicted.

| | |
|---|---|
| **Tier** | 2 |
| **Severity** | Medium |
| **Effort** | Small |
| **Confidence** | High |
| **Lens** | AI |

## Location

- `ai/tts.py` — `_run_piper()`, `proc.communicate(input=text.encode("utf-8"))`

## Problem

`_run_piper()` spawns Piper and blocks on `proc.communicate()` with no `timeout=`. On a live turn
this is bounded from the outside: `ai/session.py`'s `_enter_speaking()` arms `SESSION_SPEAK_MAX_UNKNOWN_S`
(20 s) or `WAKE_ACK_MAX_S` as a fallback deadline, and the 20 Hz tick calls `_cut_speech()` →
`tts.stop()` past it, which `terminate()`s whatever `_synth_proc` currently holds. That backstop
only exists for the two states it was built for (`STATE_ACK`, `STATE_SPEAKING`).

The **background warm threads** — `kai-ack-warm` (`_warm_all` → `_prewarm_canned`/`_speak_greeting`/
`_prewarm_bank`) and `kai-ack-rewarm` (`_rewarm_when_quiet`) — call the same `_run_piper()` with
nothing watching them. There is no deadline anywhere in `ConversationSession` for a synth started
outside a conversation turn. If Piper genuinely wedges (not crashes — a stall: contended CPU under
the GIL pressure `config/tracking.py` already documents as this board's bottleneck, a corrupted
`.onnx`/espeak-ng data read, or simply memory pressure at the edge of the 2.0–2.3 GB headroom
`docs/memory-budget.md` measures), the calling thread blocks in `communicate()` indefinitely:

- `_prewarm_canned()` never returns, so if this happens during `_warm_all()`'s first phase, the
  startup greeting (`_speak_greeting()`, which runs *after* `_prewarm_canned()` in the same thread)
  **never plays**, silently — no error, no log line past "resolving input device…"-style startup
  noise, just an absent greeting on a still-otherwise-healthy robot.
- The thread itself never terminates. It is daemon, so it does not stop the process from exiting,
  but for the remaining lifetime of that run it holds one OS thread and one Piper child process that
  nothing can ever kill — `tts.stop()` can still find it via `_synth_proc` (good), but nothing ever
  calls `tts.stop()` on a background warm hang, because nothing is timing it.

## Why it matters

This is exactly the class of failure `R8` (no main-loop watchdog) and `R1`/`R3` (blocking calls with
no bound) already catalogue elsewhere in the tree — a wedge with no deadline attached — but it sits
in a part of the AI pipeline with zero eval or monitoring coverage (`A2`). A hung bank-prewarm thread
is invisible on `/params` (nothing publishes "warm thread progress" beyond the pass-completion log
line, which simply never arrives) and costs nothing until a listener notices the filler bank never
grew past its startup size or the wake ack never got faster than the fallback synth latency.

## Suggested approach

Add a `timeout=` to `proc.communicate()` (`TTS_PIPER_TIMEOUT_S` in `config/voice.py`, sized above the
slowest observed synth — the bank-warming note in the CHANGELOG already measures individual line
timings). Catch `subprocess.TimeoutExpired`, `proc.kill()`, drain the pipes, and return `False` the
same way every other `_run_piper` failure does. This closes the leak at its source rather than
relying on a caller-side deadline that only two of `_run_piper`'s several callers have.

## Acceptance criteria

- [x] `_run_piper()` passes a bounded `timeout=` to `communicate()`.
- [x] `subprocess.TimeoutExpired` is caught, the process is killed (not just the handle dropped), and
      the function returns `False` and logs, exactly like every other synthesis failure.
- [x] A test simulates a hang (a stub `Popen` whose `communicate()` raises `TimeoutExpired`) and
      asserts `_run_piper` returns `False` within the test's control rather than hanging the test
      thread.
- [x] The background warm call sites (`_warm_all`, `_prewarm_bank`, `_rewarm_when_quiet`) need no
      changes — they already treat a `False`/missing-key result as "try again later" or "skip."

## Cross-ticket note

See `A6` — bounding this call narrows the worst case of that ticket's shutdown race but does not
close it: a Piper run that completes normally (not a hang) can still be spawned after shutdown has
begun, because nothing stops the warm threads from starting one. Both should land.
