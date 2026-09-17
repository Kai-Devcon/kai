# A3 — The prompt is never checked against `OLLAMA_NUM_CTX`, and the overflow is silent

> **Status: FIXED**, 2026-09-17. `ai/session.get_status()` publishes `sess_last_llm_prompt_tokens`,
> `sess_last_llm_gen_tokens` and `sess_llm_num_ctx` on `/params`. `ai/llm._log_llm_timings()` warns
> once, edge-triggered (`OLLAMA_CTX_WARN_FRACTION = 0.85`, `config/voice.py`), when
> `prompt_eval_count + OLLAMA_NUM_PREDICT` crosses that fraction of `OLLAMA_NUM_CTX`, and separately
> (and unconditionally) when a reply hits `OLLAMA_NUM_PREDICT` exactly — both printed regardless of
> `OLLAMA_LOG_TIMINGS`, same as the `MODEL RELOADED` precedent. Tests in
> `tests/test_llm.py::TestLogLlmTimingsContextBudget` cover the threshold crossing, the no-re-warn
> case, re-arming after dropping back under budget, the num-predict line, and the mocked/no-timing
> case warning nothing. Purely observational throughout — nothing clamps `TOP_K`, history or the
> persona. A measured note (two live RAG turns, 868 then 908 tokens) landed beside `OLLAMA_NUM_CTX`;
> a plain non-RAG turn's count was not captured this pass (see A1's cross-ticket note).

| | |
|---|---|
| **Tier** | 1 |
| **Severity** | Medium |
| **Effort** | Small |
| **Confidence** | High |
| **Lens** | AI |

## Location

- `ai/llm.py` — `_log_llm_timings()` computes `llm_prompt_tokens` from Ollama's `prompt_eval_count`
  and puts it in the returned dict
- `ai/voice_assistant.py` — `_stage_ms` carries `llm_prompt_tokens` and `llm_gen_tokens`;
  `_call_ollama()` merges them in
- `ai/session.py` — `get_status()` publishes `sess_last_llm_prompt_ms`, `sess_last_llm_gen_ms`,
  `sess_last_llm_tok_s` … and **not** the token counts
- `config/voice.py` — `OLLAMA_NUM_CTX = 2048`, `MAX_HISTORY_TURNS = 6`, `OLLAMA_NUM_PREDICT = 160`
- `config/rag.py` — `TOP_K = 3`, `CHUNK_SIZE_CHARS = 800`, `PRIMER_MAX_CHUNKS`

## Problem

The number that says how full the context window is arrives on every single response, is parsed, is
stored, is printed to the log — and is then dropped before it reaches anything that could act on it.
`sess_last_llm_prompt_ms` is published; `llm_prompt_tokens` is not. Nothing anywhere compares it to
`OLLAMA_NUM_CTX`.

The budget is genuinely tight on a RAG turn. Rough sizing against the constants:

| part | size |
|---|---|
| `persona.txt` | ~1.9 kB, ~480 tok |
| FACTS block, `TOP_K = 3` × `CHUNK_SIZE_CHARS = 800` | ~2.4 kB, ~600 tok |
| `format_context()`'s trailing instructions | ~90 tok |
| `IDENTITY_PROMPT`, once a name is pinned | ~25 tok |
| rolling history, `MAX_HISTORY_TURNS = 6` pairs | up to several hundred tok |
| the question | tens |
| reserved for the reply, `OLLAMA_NUM_PREDICT` | 160 |

`config/voice.py` already records the shape of this from the other side: "a 6-turn session peaked at
959 tokens of the old 1024, i.e. 94% full, and Ollama silently drops history to fit (it preserves the
system prompt, so the documents survive and **the conversation is what rots**)." That measurement was
taken at `num_ctx = 1024` on chat turns. The window is 2048 now, but so is the load — a DEVCON-heavy
conversation adds a ~700-token FACTS block to every turn that the 1024-era measurement did not carry.

## Why it matters

The failure is silent and it looks like the model, not like the system.

Inputs: a visitor holds a conversation of six or more exchanges, most of them about DEVCON, so most
turns carry a full FACTS block. Observable behaviour: partway through, Kai stops being able to refer
back — "what was the first thing I asked you?" gets the wrong answer, a thread the visitor dropped
cannot be picked up — while `persona.txt` line 9 is actively telling the model it remembers the
conversation and `MAX_HISTORY_TURNS = 6` says six exchanges are being sent. Nothing is logged,
nothing on `/params` moves, and the natural conclusion is that a 2B model is just forgetful.

That is the same failure `MAX_HISTORY_TURNS` was raised from 3 to 6 to fix, arriving by a different
route. And it interacts with **S13**, which adds more state to the prompt across the wake gap, and
with **S12**, which already added a system-prompt line.

Two secondary points, both cheap to get at the same time:

- `llm_gen_tokens` hitting `OLLAMA_NUM_PREDICT = 160` means the reply was cut mid-word rather than at
  a sentence end — the ordering `config/voice.py` deliberately arranged so `TTS_MAX_SPOKEN_CHARS`
  normally binds first. There is currently no way to know how often the looser cap is the one that
  bound.
- `docs/memory-budget.md` says `num_ctx` may not be raised past 2048 on this hardware ("4096
  hard-crashes the llama runner"). So the ceiling is fixed, and the only levers are `TOP_K`,
  `CHUNK_SIZE_CHARS`, `MAX_HISTORY_TURNS` and the persona's length. Choosing between them needs the
  number.

## Acceptance criteria

- [x] `ai/session.get_status()` publishes `sess_last_llm_prompt_tokens` and
      `sess_last_llm_gen_tokens` from the `_stage_ms` values that already exist. No new measurement,
      no new call — the projection is the whole change.
- [x] `/params` also carries the ceiling (`sess_llm_num_ctx`), so the dashboard shows a fraction
      rather than a number an operator has to remember the denominator for.
- [x] `ai/llm._log_llm_timings()` prints one warning when
      `prompt_eval_count + OLLAMA_NUM_PREDICT` exceeds a configured fraction of `OLLAMA_NUM_CTX`,
      naming the likely eviction ("history is being dropped to fit"). The threshold is a constant in
      `config/voice.py` with a comment explaining the arithmetic above — `0.85` is a reasonable start
      and the comment should say it is a guess until measured.
- [x] The warning is rate-limited, or edge-triggered on crossing the threshold. This is the
      `NO_FACE` precedent: a per-turn warning on a long conversation is a log nobody reads.
- [x] A separate, quieter line when `eval_count` reaches `OLLAMA_NUM_PREDICT` exactly, since that is
      the "cut mid-word" case and it is a different fix from the one above.
- [x] Purely observational: no clamping, no automatic trimming of `TOP_K` or history. Deciding what
      to drop is a tuning judgement and it belongs in `config/`, made once, with a measurement — not
      in a runtime heuristic that would make two identical questions retrieve differently.
- [x] A short measured note lands in `config/voice.py` beside `OLLAMA_NUM_CTX`: observed prompt-token
      counts for a chat turn, a RAG turn, and a six-exchange RAG conversation, dated — the same shape
      as the 959-of-1024 figure already there, re-taken at 2048 with a FACTS block in play.
      (Two live RAG turns captured; the chat-turn and six-exchange numbers were not — noted as such
      in the comment rather than fabricated.)
- [x] `tests/test_llm.py` covers the warning threshold with a stubbed response body, including that a
      response with no timing fields (the mocked case `_log_llm_timings` already guards) warns
      nothing.

## Suggested approach

Three small edits, none of which touch the turn's control flow.

**1. Publish.** In `ai/session.get_status()`, beside the existing `sess_last_llm_*` block:

```
# sketch
"sess_last_llm_prompt_tokens": int(self._stage_ms.get("llm_prompt_tokens", 0)),
"sess_last_llm_gen_tokens":    int(self._stage_ms.get("llm_gen_tokens", 0)),
"sess_llm_num_ctx":            OLLAMA_NUM_CTX,
```

**2. Warn.** In `_log_llm_timings`, next to the existing `MODEL RELOADED` branch — which is the
precedent for "a rare, actionable event gets its own line rather than being buried in the common
case", and which stays printed even with `OLLAMA_LOG_TIMINGS` off. Give this one the same standing:
it is rare, it is actionable, and it is invisible otherwise.

**3. Measure.** Take the three numbers on the robot with `/voice/wake` turns and write them into
`config/voice.py`. That is the part that makes the threshold in step 2 stop being a guess, and it is
the part most likely to get skipped.

Sequence this **before S13**. S13 adds a resumable topic across the wake gap, which is more prompt;
doing it while nothing watches the budget is spending something nobody is counting.
