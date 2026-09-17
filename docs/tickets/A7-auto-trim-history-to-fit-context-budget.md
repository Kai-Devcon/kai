# A7 — Auto-trim the oldest history to fit the context budget, instead of only warning

> **Status: FIXED**, 2026-09-17. `ai/llm.build_chat_messages()` gained a `budget_tokens` parameter:
> when given, it drops the oldest history PAIR at a time (never the system prompt, never the new
> turn) until the estimated prompt fits, using a ~4-chars/token estimate. `ai/voice_assistant.py`'s
> `_call_ollama()` now passes `budget_tokens = OLLAMA_NUM_CTX - OLLAMA_NUM_PREDICT`. Tests in
> `tests/test_llm.py::TestBuildChatMessagesBudget` cover: staying under budget is a no-op, the
> oldest pair goes first, the system prompt and new turn are never dropped even at an impossible
> budget, and a log line fires only when trimming actually happens.

| | |
|---|---|
| **Tier** | 1 |
| **Severity** | Medium |
| **Effort** | Small |
| **Confidence** | High |
| **Lens** | AI |

## Location

- `ai/llm.py` — `build_chat_messages()`, new `_estimate_tokens()` / `_CHARS_PER_TOKEN_ESTIMATE`
- `ai/voice_assistant.py` — `_call_ollama()`, the `budget_tokens` call site
- `config/voice.py` — `OLLAMA_CTX_WARN_FRACTION`'s comment, updated to describe the new split

## Problem

A3 added an edge-triggered warning when a turn's prompt crosses `OLLAMA_CTX_WARN_FRACTION` of
`OLLAMA_NUM_CTX`, and deliberately did **not** auto-trim anything — its acceptance criteria say so
explicitly, on the reasoning that clamping `TOP_K`/history at runtime would make two identical
questions retrieve differently, which is a tuning judgement that belongs in `config/`, not a
heuristic.

That reasoning holds for `TOP_K` and retrieval, but misses something about the deployment: **Kai
runs standalone at a booth.** The A3 warning prints to `/tmp/face-servo.log`, and nobody is SSHed in
tailing that log while the robot is actually being used — which is the normal, intended way to run
it. A warning that only a developer doing a live debugging session would ever see is not a mitigation
for the end-user-facing failure (a follow-up question landing on evicted history, answered
confidently and wrong) — it is an instrument for a developer, not a fix for a robot working the room
alone.

The failure this replaces (Ollama's own silent truncation once `OLLAMA_NUM_CTX` is exceeded) is also
worse than a controlled trim would be: it happens somewhere inside Ollama's own context-shift logic,
not deterministically "oldest history first" at the message boundary, which is what produced the
original "Kai forgets the opening question, and answers about it confidently anyway" bug
`MAX_HISTORY_TURNS` was raised to fix once already — A3's own measurement shows the same shape can
still occur within the 6-turn cap once a RAG turn's FACTS block is large enough.

## Why it matters

Doing our own trimming, oldest-history-first, is:

- **Deterministic** — the same conversation trims the same way every time, unlike whatever Ollama's
  internal context handling does once the hard ceiling is hit.
- **Observable after the fact** — logged only when it actually happens, same as every other
  edge-triggered warning in this codebase.
- **Narrow** — it only ever removes ROLLING HISTORY, never the persona or the retrieved RAG context.
  It does not touch `TOP_K` or `CHUNK_SIZE_CHARS`, so it does not have the "two identical questions
  retrieve differently" problem A3 was avoiding — retrieval is unchanged; only how much *back and
  forth* survives into the prompt can shrink, and only when the budget is genuinely tight.

The real tradeoff, stated plainly: once this can fire, two otherwise-identical conversations *can*
behave differently depending on how long the session ran and how large its RAG context was — a long
DEVCON-heavy conversation may lose an early exchange that a short one would have kept. That is a real
behaviour change, not just instrumentation, and it is accepted here because the alternative — silent
context rot nobody at the booth can see happening — is worse for a robot meant to run unattended.

## What this deliberately still does NOT do

- No trimming of the RAG context (`TOP_K`, `CHUNK_SIZE_CHARS`) or the persona. Those stay A3's
  "tuning judgement, made once, in `config/`" territory.
- No trimming when `budget_tokens` is not passed — every other `build_chat_messages()` caller
  (the warm-up ping in `ensure_llm_warm`, and every existing test) is unaffected.
- The `OLLAMA_CTX_WARN_FRACTION` warning stays. It is now the signal for the case trimming cannot
  fix: persona + RAG context alone already over budget with zero history left to drop.

## Acceptance criteria

- [x] `build_chat_messages()` accepts an optional `budget_tokens` and drops the oldest history pair
      at a time (never the system prompt, never the new user turn) until the estimated prompt fits,
      or history is empty.
- [x] `_call_ollama()` passes `OLLAMA_NUM_CTX - OLLAMA_NUM_PREDICT` as the budget.
- [x] A log line fires only when trimming actually removes something (edge case, not every turn).
- [x] Every existing caller/test that does not pass `budget_tokens` is unaffected (verified: the
      full suite passes unchanged aside from the new tests this ticket added).
- [x] `config/voice.py`'s `OLLAMA_CTX_WARN_FRACTION` comment is updated to describe the split
      between "trim (automatic, history only)" and "warn (the case trimming can't fix)".

## Cross-ticket note

Supersedes A3's "no clamping" criterion for history specifically — RAG retrieval (`TOP_K`,
`CHUNK_SIZE_CHARS`) is untouched and that half of A3's reasoning still stands. Raised from a
conversation about A3 after landing it: the warning alone doesn't help an unattended, standalone
deployment, which is the normal way this robot runs.
