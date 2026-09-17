# A8 — An incomplete fastembed cache silently breaks dense RAG retrieval for a whole boot

> **Status: FIXED (on this boot)**, 2026-09-17. Found while building A2's eval harness and running
> it against the live robot for the first time — the harness never resembled a synthetic case, it
> hit a real, currently-active production bug. `rm -rf /tmp/fastembed_cache` followed by
> `python3 -c "from ai import rag; rag.ensure_model_loaded()"` forced a clean re-download (5 files,
> ~9s), then `python3 -m ai.index_documents` rebuilt the also-stale index. The live `face_track.py`
> process did **not** need a restart — `ensure_model_loaded()` retries on every call when
> `_embed_model` is `None`, so the very next RAG query after the cache was fixed succeeded. No fix
> landed for the ROOT CAUSE (why the download was incomplete, or how to detect it sooner) — see
> "Suggested approach" below for what is still open.

| | |
|---|---|
| **Tier** | 2 |
| **Severity** | High (silent, degrades every turn, no crash, no restart-visible symptom) |
| **Effort** | Small (the fix); Medium (a real startup guard) |
| **Confidence** | High — reproduced and fixed live |
| **Lens** | AI |

## Location

- `ai/rag.py` — `ensure_model_loaded()`, `embed_query()`
- `/tmp/fastembed_cache/` on the Jetson — the on-disk cache `fastembed`'s `TextEmbedding` reads

## Problem

This morning's boot (`/tmp/fastembed_cache` timestamped 05:02) left the embedding model's
HuggingFace-hub-style cache **incomplete**: the snapshot directory had `config.json` (706 bytes)
but never got `model_optimized.onnx` (the actual weights) or the tokenizer files. Every call to
`ai.rag.ensure_model_loaded()` since boot has been raising
`onnxruntime.capi.onnxruntime_pybind11_state.NoSuchFile`, caught by `embed_query()`'s `except
Exception`, which fails open and returns `None` — exactly as designed for a missing/broken
embedding model.

The consequence: **dense retrieval has been silently unavailable for the robot's entire uptime
today.** `retrieve_context()` fell through to its lexical-fallback layer for every single turn
(`[rag] dense retrieval empty; lexical fallback matched [...]`), which is a real safety net but a
strictly weaker retriever than the one `scripts/rag_accuracy.py` measures (30/31 = 97%). Nothing
crashed, nothing restarted, and the only visible sign was a `[rag] WARNING: embed_query failed
(...)` line buried in `/tmp/face-servo.log` on every turn — the same "nobody is tailing this log at
a booth" problem A7 was raised to fix for the context-budget warning, now showing up for retrieval
itself.

Separately, `documents/devcon_faq_rag.md` had also changed since the index was last built
(`[rag] WARNING: index is stale`), so even lexical retrieval was running against a stale index —
compounding, not caused by, the embedding failure.

## Why it matters

This is the same shape of failure the whole codebase's comment-as-measurement discipline exists to
prevent: a fail-open path (correct in isolation — `embed_query` must never crash a turn) with
nothing watching how long it has been failing open. `rag_accuracy.py`'s own recorded baseline
(97% answer-in-context accuracy) has been **untrue in practice for this entire boot**, because it
measures the dense path this bug had disabled. Anyone reading `/params` or the dashboard had no way
to know retrieval quality had silently degraded — `ai/rag.get_status()`'s `rag_errors`/
`rag_last_error` counters exist for exactly this and were not checked before this incident was
found by accident.

## Suggested approach

Not implemented this pass — flagging for a follow-up, since the immediate instance is fixed and
this ticket is already adjacent to A2's scope:

- A startup self-check: call `ensure_model_loaded()` and one `embed_query()` eagerly at boot
  (`face_track.py`'s existing pre-warm sequence is the natural place — see `ai/rag.py`'s own
  docstring, which already says to call `ensure_model_loaded()`/`load_index()` "once at startup"),
  and log loudly — not just the existing per-turn WARNING — if it fails, so a broken cache is a
  boot-time signal instead of a today-long silent degradation.
- Consider whether `rag_errors`/`rag_last_error` (`ai/rag.get_status()`) should be surfaced on
  `/params` if not already, so a dashboard glance would have caught this in seconds instead of it
  needing a `scripts/reply_eval.py` run to surface by accident.
- The root cause (why the download was interrupted — network timing at boot, a `HF_TOKEN` rate
  limit per the "unauthenticated requests" warning fastembed prints, or something else) was not
  investigated. If this recurs, that warning line is the first thing to check.

## Cross-ticket note

Found while establishing A2's baseline — the eval harness's very first real run against the robot
surfaced this before it could measure anything about reply quality, since every "grounded" case
looked like total amnesia with dense retrieval down. A2's baseline was re-taken after this fix.
