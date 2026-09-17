# A9 — Replies to enumeration-style questions often exceed persona.txt's 4-sentence cap

> **Status: FIXED (for the reproduced case)**, 2026-09-17. `ai/persona.txt`'s enumeration
> instruction now generalises beyond "a list of programs" to any list-shaped question, restates
> that the length cap applies TO the list rather than being suspended for it, and adds one worked
> example anchored to the exact failing question ("what tools were you built with?" — the only
> code change; `load_persona()` re-reads the file on every call, so this took effect with no
> restart). Verified against the live robot: `"what tools were used to build you?"` went from 0/3
> to 3/3 on the length check.
>
> **Not fully fixed — scope is narrower than the title suggests.** Re-testing two OTHER, non-
> enumeration cases from the same eval run (`"when did DEVCON start?"`, `"what is DEVCON?"`) still
> overran the cap 2 of 6 times. This fix targeted the specific reproduced failure (list-shaped
> questions) with a concrete worked example, which is what small models respond to reliably; it did
> not address gemma2:2b's more general tendency to occasionally run long on questions with no list
> in them at all. That is a bigger persona-tuning problem than a "quick fix" scope — left open below.

| | |
|---|---|
| **Tier** | 2 |
| **Severity** | Medium |
| **Effort** | Unclear — needs A/B on persona.txt wording, not a code change |
| **Confidence** | Medium (12 samples across one eval run) |
| **Lens** | AI |

## Location

- `ai/persona.txt` — "Most questions get two or three. Nothing you say ever runs past four
  sentences." and the enumeration guidance ("give the two or three that matter most and offer the
  rest")
- `scripts/reply_eval.py` — where this was found, via its length check

## Problem

A2's first real baseline run (2026-09-17, after fixing A8) found that the single most common
failure mode was not a wrong fact — it was length. For `"what tools were used to build you?"`, all
3 repeats were factually clean (no hallucination) but ran 6–8 sentences (338–676 characters),
listing every tool named in `documents/` (Jetson, Claude Code, MediaPipe, Whisper, Gemma 2, Piper,
Arduino) in one breath, rather than following persona.txt's own instruction to name "the two or
three that matter most and offer the rest."

This reproduced elsewhere at lower frequency: one of three "when did DEVCON start?" replies ran to
5 sentences by adding an unprompted excursion (PSIA, TechBar, expansion to other cities) the
question did not ask for.

## Why it matters

This is a real, previously-unmeasured gap between what persona.txt asks for and what gemma2:2b
reliably does — and it is a different bug from either a factual miss or an ungrounded invention
(see A2's own case-design rationale for why the three are checked separately). A reply that is
correct but doubles the expected length also:

- runs past `TTS_MAX_SPOKEN_CHARS` more often, which is what triggers a mid-sentence clamp cut by
  `tts.clamp_for_speech` — a different, worse-sounding failure than the model just stopping;
  and
- costs more Piper synthesis time (`ai/tts.py`) and more of `OLLAMA_NUM_PREDICT`'s budget for text
  a listener standing at a booth was not going to stay for.

## Suggested approach

The enumeration-specific half is done — see the status banner. Still open, for the general
(non-list) overrun:

- Re-run `scripts/reply_eval.py --repeats 10` against the full `CASES` list to size the general
  overrun rate now that the list-shaped case is fixed — 3 repeats is enough to notice it, not
  enough to size it, and a general persona rewrite should not be attempted without a number to
  check it against.
- A worked example fixed the reproduced case reliably; the same technique (one concrete example
  per failure SHAPE, not a general rule restated harder) is the more promising next step over
  further-tightening the existing four-sentence language, which was already explicit and still
  didn't bind on `"when did DEVCON start?"`'s unprompted excursion into PSIA/TechBar/expansion
  history nobody asked for.
- Check whether this correlates with `OLLAMA_NUM_PREDICT` headroom — a reply that would naturally
  run long has more room to do so before the token cap intervenes, which A3's `_log_llm_timings`
  warning would show as "did NOT hit OLLAMA_NUM_PREDICT" (confirming the model chose the length,
  not the cap).

## Cross-ticket note

Surfaced by A2's baseline run, the same run that found A10 (Tagalog reply consistency) and A8 (the
embedding-cache bug). All three came from the same single evaluation pass — this is what A2 was
for.
