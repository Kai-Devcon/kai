# A1 — Ollama re-decides the model's placement on every turn, so no KV prefix survives

> **Status: INVESTIGATED, 2026-09-17.** This was an investigation ticket with a small code change
> at the end, and the code change has landed: `ai/session.get_status()` publishes
> `sess_last_llm_load_ms` (from the `llm_load_ms` `_stage_ms` already carried), so a reload is
> visible on `/params` instead of only `/tmp/face-servo.log`.
>
> The investigation is also done, and the finding is written into `config/voice.py` beside
> `IDENTITY_PROMPT` with the date. Summary: `/api/ps` sampled between three consecutive live turns
> showed the runner's `expires_at`/`size_vram` completely unchanged despite `MODEL RELOADED` firing
> on every one of them — so the reload line does not mean what it says on Ollama 0.24.0. That is
> good news for nothing, though: `prompt_eval_count` does not shrink turn-over-turn either (868 tok,
> then 908 tok on the next turn, growing with history) — so there is no KV prefix surviving between
> `/api/chat` calls on this Ollama version for the reload line to have been invalidating in the
> first place. The three prefix-preserving decisions (`RAG_CONTEXT_PLACEMENT="user"`,
> `IDENTITY_PROMPT`'s system-slot placement, raw-transcript history) are kept as-is per the ticket's
> own instruction — they cost nothing and become correct if `/api/chat` ever starts reusing a
> prefix across calls. **No fix exists on this box for the underlying "no prefix reuse" behaviour**;
> it did not require restating R6 (see the cross-ticket note below), since the destabilizing
> half of A1's premise — placement changing mid-session — was not observed either.
>
> Not done: a plain (non-RAG) chat turn's prompt-token count was not captured (a `/voice/wake`
> follow-up was rejected by the session's own idle-state gate, unrelated to this ticket), and the
> before/after `first_audio_ms` comparison does not apply since no fix landed to compare against.

| | |
|---|---|
| **Tier** | 2 |
| **Severity** | Medium-High |
| **Effort** | Medium |
| **Confidence** | High |
| **Lens** | AI |

## Location

- `ai/llm.py` — `_log_llm_timings()`, the `MODEL RELOADED: {n}ms — placement was re-decided` branch
  on a non-zero `load_duration`
- `ai/llm.py` — `_ollama_request()`, `"keep_alive": OLLAMA_KEEP_ALIVE`
- `config/voice.py` — `OLLAMA_KEEP_ALIVE = -1` and its comment; `RAG_CONTEXT_PLACEMENT = "user"`;
  `IDENTITY_PROMPT`, whose comment carries the measurement
- `ai/voice_assistant.py` — `_call_ollama()`, the placement branch and the note that `_history`
  stores the raw transcript "which is the property the whole optimisation rests on"

## Problem

`config/voice.py`'s `IDENTITY_PROMPT` comment records the measurement, dated 2026-08-10:

> Every `[llm] turn:` line in /tmp/face-servo.log is preceded by `MODEL RELOADED: ~200-360ms —
> placement was re-decided`, on every turn, so there is no surviving KV prefix between turns for
> anything to invalidate.

That line only prints when Ollama returns a non-zero `load_duration` for the request. It is firing on
every turn on a service running with `keep_alive = -1`, which is sent as a JSON number on every
request precisely so the runner is never torn down between turns.

Nobody has established why. The measurement was taken in passing, while checking something else
(whether pinning a name costs a prefix invalidation), and it was recorded as a reason that *other*
measurement could not be completed. It has not been followed up, and there is no ticket for it.

## Why it matters

Three separate things are downstream of it, and two of them are design decisions that currently buy
nothing.

**The latency.** 200–360 ms is added to every turn's `first_audio_ms` budget, on top of the
serialised pipeline **R5** is about. It is invisible in `sess_last_llm_ms` (which is the wall time of
the POST, reload included) and `llm_load_ms` is measured in `_stage_ms` but never published on
`/params` — so from the dashboard this is indistinguishable from a slow prompt.

**The prefix work.** Three places in two modules carry complexity to keep Ollama's cached prefix
byte-identical between turns:

- `RAG_CONTEXT_PLACEMENT = "user"`, chosen over the original `"system"` explicitly so "the persona
  AND all `MAX_HISTORY_TURNS` of history" are not re-evaluated every turn;
- `IDENTITY_PROMPT` placed in the system slot, the *opposite* call, for the same reason;
- `_history` storing the raw transcript rather than the injected context, called out in
  `_call_ollama` as "the property the whole optimisation rests on".

If the runner is reloaded every turn there is no prefix, and all three are invariants being
maintained for a benefit that is not being delivered. That is not an argument for removing them — it
is an argument for knowing which it is, because the comments currently assert a benefit that the
robot's own log contradicts.

**The placement.** `log_model_placement()` and **R6** are both built on the premise that Ollama
decides the CPU/GPU split once, at load, and `keep_alive = -1` pins it for the life of the service.
"Placement was re-decided" on every turn says that premise may not hold — and a partial offload
arriving mid-session is the documented ~2× generation slowdown, appearing without a restart to
explain it.

## Acceptance criteria

- [x] The measurement is reproduced from a current log: how many consecutive turns carry a non-zero
      `load_duration`, and the distribution of the value. One captured session is enough; record the
      date and the Ollama version.
- [x] It is established whether this is a genuine runner reload or Ollama reporting a non-zero
      `load_duration` for an already-resident runner. `GET /api/ps` sampled between consecutive turns
      is the discriminator — a changing `expires_at`, or a `size_vram` that moves, means a real
      reload; a stable entry across a reload-reporting turn means the field is being misread.
- [x] `prompt_eval_count` / `prompt_eval_duration` are read across a short conversation with a fixed
      persona. If a prefix is surviving, the second and later turns evaluate materially fewer prompt
      tokens than the first; if every turn evaluates the whole prompt, the prefix is gone. This is
      the direct test and it needs no Ollama internals.
- [x] `llm_load_ms` is published on `/params` as `sess_last_llm_load_ms`, so a reload is visible from
      the dashboard rather than only from `/tmp/face-servo.log`. It is already measured in
      `_stage_ms`; only the projection in `ai/session.get_status()` is missing.
- [x] The finding is written into `config/voice.py` beside `OLLAMA_KEEP_ALIVE`, with the date, in the
      same comment-as-measurement style as its neighbours — including the negative result if it turns
      out the field was being misread. (Landed beside `IDENTITY_PROMPT`, which is the comment that
      was actually deferring on this measurement — see the criterion below.)
- [ ] If the reload is real and has a fix (a `keep_alive` form the installed Ollama honours, a
      service-level setting, a version), it lands and the before/after `first_audio_ms` medians are
      recorded. **N/A** — the reload is not real (see above); nothing to fix here.
- [x] If it is real and has **no** fix on this box, the three prefix-preserving decisions above get a
      one-line note saying so, so a future reader does not re-derive an optimisation that is not
      running. Do not remove them: they cost nothing, and they become correct again the moment the
      reload stops. (The deeper finding is that there is no prefix regardless of the reload line —
      noted as such.)
- [x] `IDENTITY_PROMPT`'s comment is updated either way — it explicitly defers its own "one
      invalidation" claim until this is resolved, and that deferral is what this ticket closes.

## Suggested approach

This is an investigation with a small code change at the end, not a refactor. It needs the robot, and
it needs nothing else running that competes for GPU memory.

1. **Instrument first, cheaply.** Add `sess_last_llm_load_ms` to `ai/session.get_status()` from the
   `_stage_ms` value that already exists. One line, and it makes every later step observable over
   `/params` without a shell.
2. **Sample `/api/ps` between turns.** `log_model_placement()` already parses that endpoint and
   returns `{"size", "size_vram", "gpu_pct"}`. Call it from a small script (not from the turn path —
   `OLLAMA_PS_TIMEOUT_S` is short on purpose) once before and once after each of five `/voice/wake`
   turns, and diff.
3. **Read `prompt_eval_count` across a conversation.** `_log_llm_timings` already prints it. Five
   turns on one subject, then read the five prompt-token counts: a surviving prefix shows as a small
   count after the first turn, a lost one as the full prompt every time.
4. **Then decide.** The likely causes, in the order worth eliminating: the installed Ollama version's
   handling of `keep_alive` on `/api/chat`; another client or a systemd unit touching the model; VRAM
   pressure forcing an eviction (which `docs/memory-budget.md` says has ~2.0–2.3 GB of headroom
   against a 2.4 GB model, so this is not far-fetched); or the field simply being non-zero for a
   cached runner.

Sequence this **before R6**. R6 spends effort making the startup placement decision reliable; if
placement is re-decided on every turn regardless, R6's premise needs restating first.
