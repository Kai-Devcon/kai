# Engineering tickets — codebase review, 2026-08-10

Every finding from the two-lens codebase review (robotics engineering + software engineering),
converted into an implementation-ready ticket. One file per finding; nothing merged, nothing
dropped.

Tickets are grouped by **Tier**, which ranks them by severity × inverse effort — high-impact,
low-effort work first. Within a tier they appear in the review's own priority order. Each ticket
carries the review's Severity / Effort / Confidence / Lens verbatim.

**Five of these have been implemented** — R9, R7, S9 and S1 on 2026-08-09 (PRs #5–#8), and S12 on
2026-08-10 (PR #9). Everything else is still a specification, not a change record. A landed ticket
keeps its file rather than being deleted, so the spec and what actually shipped stay side by side.

How a landed ticket records itself, both forms being in use:

- a `> **Status: FIXED**` banner at the top of the file, naming the branch and the one-line change,
  with the acceptance criteria checked off in place — R9, R7, S9, S1;
- a `## Resolution` section at the bottom — S12.

The banner is the better of the two and is what new work should use: it is the first thing a reader
sees, and it puts the outcome next to the spec instead of a scroll away. Either way, **check the
acceptance criteria boxes** — S12's are still all unchecked despite the ticket having landed, which
is why the table below cites its Resolution section rather than its checklist.

> **This index went stale once, and it is worth knowing how.** Between 2026-08-09 and 2026-08-11 it
> still read "one of these has been implemented — S12" while four Tier 1 tickets had already merged.
> Each of the four PRs updated *its own ticket file* correctly and none of them updated this table,
> so every individual ticket was honest and the summary was not. Local `main` was also 13 commits
> behind `origin/main`, so `git log` on a fresh checkout agreed with the wrong version. If you are
> about to pick up a ticket, `git fetch` and check the ticket's own file before trusting this page.

**ID prefixes:** `R` = robotics engineering lens, `S` = software engineering lens, `A` = AI
engineering lens (added 2026-08-12, when that third lens was first run — see §Review context).
`S11` was a grouped "minor correctness and hygiene" finding and is split into `S11a`–`S11d` for
tracking; the four share no code and can land in any order. `R12`, `A5` and `A6` were added
2026-09-17 by a stability-focused pass across all three lenses (see §Review context). `A7` was
raised the same day as a direct follow-up to landing `A3` — not from either review pass — because
`A3`'s warning-only mitigation turned out not to fit how this robot is actually run; see `A7`'s own
file for why. `A8`, `A9` and `A10` were all found the same day by A2's OWN baseline run — a live
production bug (`A8`) and two reply-quality gaps (`A9`, `A10`) that no prior review pass could have
found, because nothing before `A2` ever looked at a generated reply.

**Not everything here came from the review.** `S12`–`S14` are feature specifications raised on
2026-08-10 from a separate question — what would make Kai's *conversation* land, given the camera is
not part of the answer. They are written to the same format and carry the same contract, but their
Severity reads as "enhancement, not a defect": nothing about the current behaviour is broken, it is
absent. Do not read them as review findings.

---

## Tier 1 — high impact, small effort

Do these first. All twelve are small, self-contained, and individually revertible. **Seven have
landed; five remain open** — `A4` and `A3` were added by the 2026-08-12 AI lens, and `A3` and its
follow-up `A7` both landed 2026-09-17. (`A2`, `A8`, `A9`, `A10` are Tier 2 — see that table.)

| ID | Ticket | Summary |
|---|---|---|
| **S12** ✅ | [Kai never learns who it is talking to](S12-no-identity-within-a-session.md) | **Landed 2026-08-10**, PR #9. A name offered in speech survived only `MAX_HISTORY_TURNS = 6` and was then evicted; nothing pinned it. See its `## Resolution`; its checklist was never ticked. |
| **R9** ✅ | [`PDAxis.update` truncates instead of rounding](R9-pdaxis-truncates-instead-of-rounding.md) | **Landed 2026-08-09**, PR #5, `fix/pd-axis-rounding`. Returns `int(round(self.current))`. The write-up's "asymmetric about 90°" claim was corrected while fixing it — angles are clamped positive, so truncation was a *uniform* downward bias. One on-hardware check deferred. |
| **R7** ✅ | [TTS subprocesses outlive the process](R7-tts-subprocesses-outlive-process.md) | **Landed 2026-08-09**, PR #6, `fix/tts-outlives-shutdown`. `tts.stop()` is now the first statement of `run()`'s `finally`. **Two on-hardware checks deferred.** |
| **S9** ✅ | [Fail-open blanket excepts swallow bugs silently](S9-blanket-excepts-swallow-bugs.md) | **Landed 2026-08-09**, PR #7, `fix/rag-silent-failures`. Rate-limited `_note_error()` in `ai/rag.py`; the fail-open return values are bit-identical. All criteria met. |
| **S1** ✅ | [The session RLock is held across disk I/O on two paths](S1-session-lock-held-across-disk-io.md) | **Landed 2026-08-09**, PR #8, `fix/session-lock-disk-io`. `tick()` hands both capture timeouts back to the caller to finish off the lock. All criteria met. |
| **A3** ✅ | [The prompt is never checked against `OLLAMA_NUM_CTX`](A3-prompt-never-checked-against-num-ctx.md) | **Landed 2026-09-17.** `sess_last_llm_prompt_tokens`/`sess_last_llm_gen_tokens`/`sess_llm_num_ctx` now on `/params`; an edge-triggered warning fires on crossing `OLLAMA_CTX_WARN_FRACTION`, and a separate line on hitting `OLLAMA_NUM_PREDICT` exactly. Its "no clamping" criterion was revised by A7 — see below. |
| **A7** ✅ | [Auto-trim the oldest history to fit the context budget](A7-auto-trim-history-to-fit-context-budget.md) | **Landed 2026-09-17.** `build_chat_messages()` now drops the oldest history pair at a time to fit `OLLAMA_NUM_CTX - OLLAMA_NUM_PREDICT`, deterministically — a standalone robot has nobody tailing the log A3's warning prints to. RAG retrieval (`TOP_K`, `CHUNK_SIZE_CHARS`) is untouched. |
| **A4** | [Only DEVCON facts are grounded](A4-only-devcon-facts-are-grounded.md) | The persona's grounding rule covers DEVCON and nothing else, so the date, the weather and the venue are answered confidently by a 2B model with no clock and no network. |
| **S8** | [`app/camera_supervisor.py` has no tests](S8-camera-supervisor-untested.md) | The only substantial untested module — and the one deciding whether the robot believes it has a camera. |
| **R4** | [Firmware clamps to 0–180 while the host clamps to 10–170](R4-firmware-servo-limits-mismatch.md) | A corrupted serial line drives the pan servo into its mechanical stop; `toInt()` turns garbage into 0°. |
| **S2** | [`/params` rebuilds the whole snapshot at 20 Hz, per client](S2-params-sse-snapshot-per-client.md) | Per-tab work on the same `RLock` the 30 Hz audio worker needs; the data can't change faster than the 25 Hz publishers. |
| **R2** | [200 Hz idle spin in the main loop](R2-idle-spin-loop-gil-contention.md) | The no-frame path burns the GIL 200×/s — the documented bottleneck for the control thread's cadence. |
| **R12** | [`send_gesture()` blocks the main tracking loop on a servo reconnect](R12-send-gesture-blocks-main-loop-on-reconnect.md) | The one serial write path with neither a dedicated thread (`send()`) nor a non-blocking acquire (`send_jaw()`) — a nod detected mid-CH340-flap freezes `/params`, `/video` and the jaw for ~2 s. |

**Four acceptance criteria across the landed tickets are deferred, all needing the robot**, and none
of them has been run:

- **R7** — after `SIGTERM` during a spoken reply, no `paplay` or `piper` survives; and a dashboard
  `POST /restart` mid-reply produces silence rather than two voices. These are the two that matter:
  R7's whole subject is a failure that only exists across a real process death.
- **R9** — the mirrored left/right sweep. Weak evidence by the ticket's own admission, since the
  corrected characterisation means there is no asymmetry to see, only a half-degree shift.
- **A6** — restart the robot several times while the filler bank is still warming (within the first
  few minutes of a fresh boot) and confirm no overlapping/orphaned audio. Same failure family as R7's
  two deferred criteria — see A6's cross-ticket note — so worth running together.

The unit suites are green for all four; what is missing is the on-hardware confirmation. Anyone with
the robot in front of them can close these in about ten minutes — see [operating.md](../operating.md).

## Tier 2 — high impact, medium effort

| ID | Ticket | Summary |
|---|---|---|
| **R1** | [Serial reconnect blocks the servo control thread for seconds](R1-serial-reconnect-blocks-control-thread.md) | Up to ~3.5 s of `sleep` + `sudo` under the serial lock, on the 15 Hz control thread — firing exactly when the robot is already misbehaving. |
| **R3** | [Firmware serial read can block the loop for a full second](R3-firmware-blocking-serial-read.md) | `readStringUntil`'s 1000 ms timeout re-creates the freeze-then-lurch the non-blocking LED ack was built to remove. |
| **A1** ✅ | [Ollama re-decides the model's placement on every turn](A1-per-turn-model-reload-defeats-kv-prefix.md) | **Investigated 2026-09-17.** `/api/ps` showed no placement change across three live turns despite `MODEL RELOADED` firing on all three — the log line is not trustworthy on Ollama 0.24.0. Deeper finding: `prompt_eval_count` doesn't shrink turn-over-turn either, so there is no KV prefix to invalidate in the first place. No fix exists on this box; `sess_last_llm_load_ms` now on `/params`. |
| **R6** | [Startup thundering herd](R6-startup-warm-thundering-herd.md) | Six concurrent warm-ups defeat the reason `ensure_llm_warm` runs at startup: Ollama pins its GPU/CPU split from free memory at that instant. |
| **A2** ✅ | [Nothing measures what Kai actually says](A2-no-evaluation-of-generated-replies.md) | **Landed 2026-09-17.** `scripts/reply_eval.py` — grounded/no_invent/language/refuse checks, `--repeats` for a pass rate, `--history` for multi-turn scripts. First run found a live bug (**A8**) and two new findings (**A9**, **A10**) before it finished measuring anything else. |
| **A8** ✅ | [An incomplete fastembed cache silently breaks dense retrieval](A8-fastembed-cache-can-silently-corrupt-across-boot.md) | **Fixed 2026-09-17** (this boot only — root cause still open). Found while running A2's harness for the first time: dense RAG retrieval had been silently disabled all day, falling back to lexical-only matching with nothing but a per-turn log line to show it. |
| **A9** ✅ | [Replies often exceed the four-sentence cap](A9-replies-often-exceed-the-four-sentence-cap.md) | **Fixed for the reproduced case, 2026-09-17.** One worked example in `persona.txt` took the enumeration case from 0/3 to 3/3 on the length check. General (non-list) overrun elsewhere is a separate, still-open tendency. |
| **A10** ✅ | [A Tagalog question often gets an English reply](A10-tagalog-questions-often-get-english-replies.md) | **Fixed, with a known side effect, 2026-09-17.** Retrieval fix (`ai/query_alias.py`) + two `persona.txt` examples took both Tagalog cases to 3/3 correct-language, correct-content — but the same examples now intermittently pull an English question into a Tagalog reply. |
| **S3** | [RAG retrieval is a Python loop over float64 vectors, from a JSON index](S3-rag-retrieval-python-loop-float64.md) | Per-chunk `cosine_similarity` with repeated norms, doubled memory, and embeddings parsed from decimal text at every startup. |
| **S4** | [TTS is module-global state with fixed shared output paths](S4-tts-global-state-shared-wav-paths.md) | Two filenames shared by every reply; the mitigation surface across three modules now exceeds the fix. Prerequisite for R5. |
| **S7** | [Flask dev server, unauthenticated, on 0.0.0.0](S7-unauthenticated-dev-server-dashboard.md) | Anyone on the venue network can silence, blind, restart or puppet the robot; unbounded streaming threads on a dev server. |
| **R8** | [No liveness watchdog on the inference loop itself](R8-no-main-loop-watchdog.md) | Every other subsystem is watched. A wedged MediaPipe leaves a convincingly-alive, half-working robot with nothing reporting it. |
| **S13** | [A conversation is forgotten the instant the session ends](S13-no-continuity-across-the-wake-gap.md) | 25 s of thinking discards the sticky RAG topic, so the same follow-up that resolved a moment ago degrades to "I'm not sure". |
| **S14** | [Kai only ever speaks when spoken to, and its sessions die silently](S14-kai-has-no-conversational-initiative.md) | The persona offers "gusto mo marinig?" but the state machine cannot act on an unanswered question; `no_speech` ends the conversation without a word. |
| **R11** | [The control thread's loop body has no test](R11-control-loop-body-untested.md) | The sweep maths is swept exhaustively; the hold branch, the anchor return and the slew clamp's feedback into the PD — each added to fix an observed fault — are not exercised at all. |
| **A5** ✅ | [`_run_piper`'s `communicate()` has no timeout](A5-piper-communicate-has-no-timeout.md) | **Landed 2026-09-17.** `TTS_PIPER_TIMEOUT_S` (15s) bounds `communicate()`; a timeout kills the process, drains the pipes, and returns `False` like every other synthesis failure. |
| **A6** ✅ | [Background TTS warm threads are never stopped on shutdown](A6-warm-threads-not-stopped-on-shutdown.md) | **Landed 2026-09-17.** `_warm_stop` is signalled before `tts.stop()` in `stop()`, every warm-path sleep waits on it instead of `time.sleep()`, and `stop()` joins the warm/rewarm threads with a logged timeout. **One on-hardware check deferred.** |

## Tier 3 — high impact, large effort

Plan these; don't squeeze them in. **R5** and **S6** are related — R5 removes much of the reason S6's
complexity exists, so sequencing matters.

| ID | Ticket | Summary |
|---|---|---|
| **R5** | [First-audio latency is serialised end to end](R5-serialised-first-audio-latency.md) | Non-streaming LLM + whole-reply synthesis; the filler bank exists to mask this rather than shorten it. Largest user-visible win available. |
| **S6** | [`ConversationSession` is a god object](S6-conversation-session-god-object.md) | 1587 lines: FSM, timers, wake tier, filler policy, warm lifecycle, status projection, mic recovery. The recorded bugs cluster in the extractable parts. |
| **S5** | [`face_track.py` constructs the whole robot at import time](S5-face-track-import-time-construction.md) | Import has filesystem side effects and builds most of the object graph before the CLI is parsed; forces two-phase construction outward. |

## Tier 4 — cleanup

| ID | Ticket | Summary |
|---|---|---|
| **R10** | [The tilt axis is plumbed everywhere but has no hardware](R10-tilt-axis-plumbed-without-hardware.md) | CLI flag, EMA, PD target, wire format, dashboard field and tests for an axis the firmware never attaches. Decide: wire it or collapse it. |
| **S11a** | [`has_video_client()` reads a shared counter without the lock](S11a-has-video-client-unlocked-read.md) | Benign under CPython, but it breaks the one-lock-guards-all contract the class is built on. |
| **S11b** | [`_publish_web`'s `fps` can never report the real rate](S11b-publish-web-fps-mislabelled.md) | Derived from the publish interval, so it pins at 25 and relates to neither the camera's 30 nor inference's 15. |
| **S11c** | [Dead and stray code](S11c-dead-and-stray-code.md) | Unused `open_camera()`, an ambiguous `autostart.sh.new` beside the live boot script, and a diagnostic reaching into `_force_send()`. |
| **S11d** | [`load_persona()` re-reads `persona.txt` on every LLM call](S11d-persona-reread-per-call.md) | Live reload is the intended feature; the silent mid-conversation drift and KV-prefix invalidation are not documented. |
| **S10** | [Dependency and environment fragility](S10-dependency-environment-fragility.md) | The lock file records the rebuild but the rebuild has never been executed; plus a process-global `subprocess` patch to satisfy pvporcupine's CPU allow-list. |

---

## Cross-ticket dependencies

Worth reading before scheduling — several of these are cheaper or safer in a particular order.

- **S4 → R5.** Streaming synthesis produces several WAVs per reply, which is impossible against the
  two fixed filenames. Do S4 step 1 (per-utterance paths) first.
- **R3 → R4.** R3 replaces the Arduino `String` parser with a `char` buffer; R4's field validation is
  much easier to write against that buffer. If both are scheduled, do R3 first and fold R4 in.
- **R5 → S6.** R5 may make a large fraction of the filler bank dead code. Refactoring code that is
  about to shrink is wasted motion.
- **S5 → S6.** Removing `face_track`'s import-time construction makes S6's test setup simpler —
  session tests stop needing to import `face_track` at all.
- **S8 → R8.** S8 builds the camera supervisor's test harness; R8 puts the loop watchdog on that same
  supervisor thread and can reuse it.
- **S4 → S6.** If S4 lands first, `VoiceWarmer`'s `_quiet_for_synth` gate reduces from a correctness
  requirement to CPU pacing.
- **S13 → S14.** A sign-off has to know whether the conversation can be resumed — "balik ka ha" is
  right when S13 has landed and misleading when it has not.
- **S12 → S14.** A nudge and a sign-off both read far better with a name in them, and S12 is the
  cheaper half. If both are scheduled, do S12 first and write the banks with the name slot in mind.
- **R5 ‖ S12–S14.** Independent. R5 shortens the wait before Kai speaks; these three change what it
  says and when. Neither blocks the other, and S12 is small enough to land while R5 is still being
  planned.
- **S6 → S13, S14.** Both add state and a deadline to `ConversationSession`, which S6 already calls
  a god object at 1587 lines. Landing them first makes S6 bigger; landing S6 first is a large
  prerequisite for two medium tickets. Prefer taking them first and folding the new state into S6's
  extraction plan rather than growing the class twice.
- **S10 ↔ S11c.** The rebuild rehearsal is the moment the ambiguous `autostart.sh.new` becomes a real
  trap. Resolve S11c before the rehearsal, or resolve it during.
- **A1 → R6.** Both are about Ollama's placement decision. R6 spends effort making the *startup*
  decision reliable, on the premise that `keep_alive = -1` then pins it for the run. A1 is the
  measurement showing placement is reported as re-decided on every turn. Take A1 first: if the
  premise does not hold, R6 needs restating before it is worth doing.
- **A1 → S12's open acceptance criteria.** `IDENTITY_PROMPT`'s comment explicitly defers its "one
  prefix invalidation" claim until the per-turn reload is understood. A1 is what makes that claim
  measurable, and S12's Resolution is where the answer belongs.
- **A2 → A4, S12, S13, S14, S11d.** All five change what Kai says, and each currently has to fall
  back on a manual demo for its acceptance. A2 is the harness that gives them a before/after number.
  It is also the cheapest way to answer R5's last criterion — whether the filler bank becomes dead
  code — since that is a question about what a listener hears.
- **A3 → S13, S14.** Both add prompt state. A3 says the prompt already crowds `OLLAMA_NUM_CTX` on a
  RAG turn and the eviction is silent, and `docs/memory-budget.md` fixes the ceiling at 2048. Publish
  the number first, or the two tickets spend a budget nobody is counting.
- **A3 ‖ A1.** Independent, but they share one line of work: both want `_stage_ms` fields that are
  measured and never projected onto `/params` (`llm_load_ms` for A1, the token counts for A3). If
  both are scheduled, do that projection once.
- **R11 → R10.** R10 edits the loop's `send()` call, and R11's harness is what would pin that change
  to observed behaviour rather than to the behaviour that replaced it. Landing R11 first costs
  nothing and gives R10 a regression test to change. The same argument applied to **R9**, which
  landed on 2026-08-09 without it — R9's own deferred criterion is the mirrored sweep, on hardware,
  which is exactly the check R11 would have made cheap.
- **R11 ‖ S8.** Same shape, same reason, no shared code: S8 builds the camera supervisor's harness,
  R11 the control loop's. S8 stays the prerequisite for R8; R11 does not feed a watchdog.
- **A5 → A6.** A timeout on `_run_piper` bounds how long a hung background synth can block a warm
  thread, which narrows A6's shutdown race but does not close it — a Piper run that completes
  normally can still be spawned after shutdown has begun. Land A5 first regardless: it is the
  smaller, more clearly-scoped fix and removes one whole class of "why is this thread still alive"
  reports before A6's shutdown-signalling work is judged against a quieter baseline.
- **A6 ↔ R7.** Not a dependency so much as a shared root: R7's own fix calls `tts.stop()` twice
  during shutdown specifically because a second wave of audio could start in the gap between them.
  A6 is the same shape of gap, one level up, in code R7's tests do not reach. Anyone re-opening R7's
  deferred on-hardware criteria should run enough restart cycles to have a chance of also hitting A6.
- **R12 ‖ R1.** Independent fixes to the same failure family (a serial write blocking its caller
  during a USB reconnect) on two different threads — R1 is the control thread, R12 is the main
  tracking loop via `send_gesture()`. Same test fixture should cover both once either lands.

## Review context

Health ratings from the 2026-08-10 two-lens review, for reference when judging whether a change
helped:

| Dimension | Rating |
|---|---|
| Efficiency | 6 / 10 |
| Performance | 5 / 10 |
| Stability | 8 / 10 |

**A third lens was run on 2026-08-12** — AI engineering, alongside a re-run of the other two. Its
ratings sit beside the ones above, not in place of them; the two sets measure different things and
both are worth keeping.

| Lens (2026-08-12) | Overall | Subscores |
|---|---|---|
| Robotics | 7 / 10 | real-time integrity 6 · control quality 8 · hardware failure handling 7 · physical safety margins 6 |
| AI | 7 / 10 | latency & memory budget 6 · prompt & context engineering 8 · retrieval quality 8 · evaluation & iteration 5 |
| Software | 7 / 10 | structure & coupling 7 · correctness & concurrency 8 · test effectiveness 7 · operability 7 |

That review agreed with all three 2026-08-10 numbers and added five tickets: `A1`–`A4` and `R11`.
Its own framing note is that the AI subsystem is the one place where this codebase's
comment-as-measurement discipline stops short of the thing being changed — retrieval is measured
carefully and the generated reply is not measured at all, which is `A2`.

**A stability-focused pass across all three lenses ran 2026-09-17**, at the request of a review
framed specifically around crash/hang/leak/degrade behaviour over long uptime rather than a general
health check. It read the full tree again rather than sampling, and treated the 27 open findings
above as established rather than re-deriving them — every one it touched is cited by ID rather than
restated. It added three new tickets (`R12`, `A5`, `A6`), all in the same family as `R7`/`R1`: a
blocking call or an un-signalled background thread with no deadline attached, which is this
codebase's one recurring defect shape. It ran the suite (1447 passed, 2710 subtests, ~62 s on the
dev box) and hit the same leaked-Piper-thread flake the baseline already names, unprompted — a
`kai-ack-rewarm` thread hung inside `Popen.communicate()` with no timeout, which is `A5` reproducing
itself in the test run that was checking for it.

| Lens (2026-09-17, stability framing) | Overall | Subscores |
|---|---|---|
| Robotics | 7 / 10 | real-time integrity 6 · control quality 8 · hardware failure handling 7 · physical safety margins 7 |
| AI | 6 / 10 | latency & memory budget 6 · prompt & context engineering 8 · retrieval quality 8 · evaluation & iteration 4 |
| Software | 8 / 10 | structure & coupling 7 · correctness & concurrency 8 · test effectiveness 7 · operability 8 |

These numbers are a stability-weighted re-reading of the same three lenses, not a strict update —
the AI overall is lower than 2026-08-12 mainly because this pass weighted "evaluation & iteration"
more heavily (a subsystem with no eval harness gets a materially different score when the framing
question is "would you notice this degrading" rather than "is retrieval well engineered"), and the
Software overall is higher mainly because this pass audited resource/lifecycle discipline
specifically (bounded buffers, atomic settings writes, properly-drained camera swaps) and found it
consistently good — see the full review's "What is already right" for the specific evidence.

Two framing notes carried over from the first review's conclusion:

1. Most of what these tickets describe is **policy**, not defect — blocking calls on real-time
   paths, serialised latency, shared globals. The defect density is genuinely low, and the
   comment-as-measurement discipline throughout the codebase is why.
2. **R5 and S6 are the same story told twice.** A large fraction of the session's complexity exists
   to paper over the latency R5 would remove. Fixing the latency first makes the refactor smaller.
