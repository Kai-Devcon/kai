# A10 — A Tagalog question often gets an English (or barely-Tagalog) reply

> **Status: FIXED, with a known side effect**, 2026-09-17. Two independent causes, two fixes:
>
> **Retrieval** — `ai/rag.py`'s `_build_query` now calls
> `ai.query_alias.expand_tagalog_question_words()`, appending an English anchor (question word plus
> a few high-value FAQ content words like "nagtatag" → "founded") to the query before embedding,
> since `EMBED_MODEL` cannot read Tagalog at all. See `config/rag.py`'s `TAGALOG_QUESTION_WORDS`.
> Verified: both Tagalog phrasings of the founder question went from retrieving the wrong chunk
> entirely to retrieving Winston Damarillo correctly, 3/3.
>
> **Language choice** — `ai/persona.txt`'s language instruction gained two short worked examples
> (a fact-lookup shape and a number shape), the same technique that fixed A9: one general rule,
> restated, did not bind; a concrete example in the failing shape did. Verified: both Tagalog
> cases went from 0/3 and (previously) 1/3 to **3/3 Tagalog replies**, content-correct in both.
>
> **Known side effect, not fixed**: the SAME worked example now sometimes pulls an ENGLISH question
> into a Tagalog reply — `"who founded DEVCON?"` (asked in English) came back majority-Tagalog 2/3
> times in a follow-up check, echoing the example's exact phrasing regardless of the question's
> language. This is the mirror image of the original bug and was not present before this fix (or
> at least not observed at this rate). Traded a *reliable* Tagalog miss for an *intermittent*
> reverse mismatch — a net improvement (Tagalog speakers are now heard; English speakers
> occasionally get code-switched at, which is a smaller and much more common Filipino-English
> conversational pattern) but not a clean fix. `scripts/reply_eval.py`'s `language` case kind only
> checks "was a Tagalog question answered in Tagalog" — it has no case for the reverse constraint,
> so this side effect would keep escaping the harness. See "Suggested approach" for the fix.
>
> Separately observed and NOT fixed (folded into **A4**, not tracked here): the three "Ilang taon"
> replies gave inconsistent year counts (10, 12, 12, 13, 14, 15 across different runs) since Kai has
> no grounded sense of the current date and cannot compute an age from a founding year.

| | |
|---|---|
| **Tier** | 2 |
| **Severity** | Medium-High |
| **Effort** | Unclear — needs A/B on persona.txt wording or a Tagalog-specific instruction, not
  obviously a code fix |
| **Confidence** | Medium (6 samples across one eval run, but a clean pattern) |
| **Lens** | AI |

## Location

- `ai/persona.txt` — "Reply in the same language the person used, English or Tagalog."
- `scripts/reply_eval.py` — the `language` case kind, where this was found

## Problem

A2's first real baseline run (2026-09-17, after fixing A8) asked two Tagalog questions and checked
whether the reply came back majority-Tagalog by function-word count:

- `"Ilang taon na ang DEVCON?"` (how many years has DEVCON been around) — 1/3 replies were
  majority-Tagalog. One of the three ("Devcon is almost 10 years old now! Wow!") was **entirely in
  English**.
- `"Sino ang nagtatag ng DEVCON?"` (who founded DEVCON) — 0/3 replies were majority-Tagalog, and
  none of the three named the actual founder (Winston Damarillo), despite the plain-English phrasing
  of the same question scoring 3/3 correct in the same run.

## Why it matters

Two separate things are tangled here and worth separating before fixing either:

1. **Language choice.** persona.txt states the rule plainly and the model does not reliably follow
   it — a Tagalog speaker at a booth asking a Tagalog question can get answered in English, which
   reads as not having been understood even when the facts would have been right.
2. **Retrieval in Tagalog.** The founder question's content also came back wrong or evasive in
   Tagalog when the identical question in English is a clean 3/3 hit (see the `CASES` list in
   `scripts/reply_eval.py` and `scripts/rag_accuracy.py`'s own English-only needle set). This
   raises a real, distinct possibility: `embed_query()`'s embedding model
   (`qdrant/bge-small-en-v1.5`, an ENGLISH-tuned model per its name) may retrieve less reliably
   against an English-language knowledge base when the query itself is in Tagalog — a cross-lingual
   retrieval gap that would need fixing in `ai/rag.py`, not `persona.txt`, and that
   `scripts/rag_accuracy.py` cannot see because none of its 27 cases are phrased in Tagalog.

Both explanations are consistent with the data; this baseline run cannot tell them apart on its own.

## Suggested approach

Both fixes landed (see status banner). Still open:

- **Add the reverse-direction check to `scripts/reply_eval.py`**: an English "language" case kind
  (or a flag on the existing `Case`) asserting the reply is majority-English, so the side effect
  above shows up in the harness instead of needing to be noticed by hand. This is the highest-value
  next step — without it, a future persona edit could make the code-switching worse with nothing
  to catch it.
- **Re-balance the persona wording** once that check exists: the two Tagalog examples may need to
  be scoped more tightly ("only if the question itself was in Tagalog") rather than trusting the
  model to infer that boundary from two examples with no negative case shown. Do this WITH the
  reverse-check in hand, not before — otherwise there is no way to tell a real fix from another
  round of whack-a-mole.
- The discriminating experiment originally proposed here (Tagalog cases added to
  `rag_accuracy.py`'s own `CASES`, not just run ad hoc) is still worth doing properly as a
  permanent regression check for the retrieval half.
- Widen `scripts/reply_eval.py`'s Tagalog case count (currently 2) before trusting any rate more
  precisely than "clearly better, not perfect" — 3 samples per case is enough to notice a defect,
  not enough to size one exactly.

## Cross-ticket note

Surfaced by A2's baseline run, the same run that found A9 (length overrun) and A8 (the
embedding-cache bug). All three came from the same single evaluation pass — this is what A2 was
for.
