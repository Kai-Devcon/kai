#!/usr/bin/env python3
"""Measure retrieval against a fixed on-topic / off-topic split. Read-only.

This exists because of docs/plan/wip/known-issues.md P2: SIMILARITY_THRESHOLD never fires. Measured
over 16 queries, on-topic best-chunk scores landed at 0.572-0.843 and off-topic at 0.541-0.686 — the
two ranges OVERLAP, so no global cutoff separates them, and 0.45 sits below both. Every turn
therefore retrieves 3 chunks and carries a documents block, including "what's the weather like
today?".

The trap in that finding is that it looks like a number to raise. It is not: any threshold high
enough to reject 0.686 of noise also rejects the 0.572 on-topic queries. So this script is the
prerequisite for ANY change to the gate — it re-runs the same split so a proposed fix can be
compared like for like, instead of being judged by trying two questions and calling it better.

    python3 -m scripts.rag_eval              # score table + threshold sweep
    python3 -m scripts.rag_eval --context    # also show what retrieve_context() returns per query

Loads the existing index read-only. Touches no settings, no servos, no Ollama, no audio. Safe to
run on a live robot, though it will compete for CPU with the voice pipeline for ~20 seconds.

Two companions cover what this script doesn't: scripts/rag_accuracy.py asks whether the retrieved
block actually contains the answer (this script only scores the match, not the content), and
scripts/reply_eval.py goes one stage further and asks whether the MODEL then does anything true
with that block.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai import rag                                                  # noqa: E402
from config.rag import SIMILARITY_THRESHOLD, SOURCE_BOOST, TOP_K    # noqa: E402

# The 16-query split from the P2 measurement. Keep these EXACT if you are comparing to the numbers
# recorded in docs/plan/wip/known-issues.md — changing the queries changes the baseline and the
# comparison becomes meaningless. Add new ones at the end instead.
ON_TOPIC = [
    "what is DEVCON?",
    "when did DEVCON start?",
    "who founded DEVCON?",
    "how many chapters does DEVCON have?",
    "what is DEVCON Kids?",
    "tell me about Geeks on a Beach",
    "what is DEVCON CREST?",
    "when is your birthday?",
]

OFF_TOPIC = [
    "what's the weather like today?",
    "Kumusta ka na?",
    "what's your favourite colour?",
    "how do you cook adobo?",
    "tell me a joke",
    "what time is it?",
    "sing me a song",
    "who won the basketball game?",
]


def best_score(query: str) -> tuple[float, str]:
    """Top score and its source for one query, using the same rewrites retrieval uses.

    Deliberately reimplements only the SCORING half of rank_chunks — including SOURCE_BOOST,
    which is applied before the threshold — so a query that retrieves nothing still reports the
    score it would have needed to clear. rank_chunks() returns [] there and tells you nothing.
    """
    rewritten, _brand = rag._build_query(query, None)
    vec = rag.embed_query(rewritten)
    if vec is None or rag._INDEX_VECTORS is None:
        return 0.0, "-"
    import numpy as np
    q = np.array(vec, dtype=np.float64)
    scored = [(rag.cosine_similarity(q, v) + SOURCE_BOOST.get(c.get("source", ""), 0.0), c)
              for v, c in zip(rag._INDEX_VECTORS, rag._INDEX_CHUNKS)]
    if not scored:
        return 0.0, "-"
    score, chunk = max(scored, key=lambda pair: pair[0])
    return score, chunk.get("source", "unknown")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--context", action="store_true",
                    help="also print what retrieve_context() actually returns for each query")
    args = ap.parse_args()

    rag.ensure_model_loaded()
    rag.load_index()
    if not rag._INDEX_CHUNKS:
        print("No index. Run: python3 -m ai.index_documents", file=sys.stderr)
        return 1
    print(f"index: {len(rag._INDEX_CHUNKS)} chunks, "
          f"threshold {SIMILARITY_THRESHOLD}, top_k {TOP_K}\n")

    results: dict[str, list[tuple[str, float, str, bool]]] = {}
    for label, queries in (("on-topic", ON_TOPIC), ("off-topic", OFF_TOPIC)):
        rows = []
        for q in queries:
            score, source = best_score(q)
            # retrieve_context is the thing that actually matters: it includes the whole failsafe
            # chain (lexical, sticky, fallback threshold, primer), not just the dense score.
            rag.reset_topic()
            got_block = bool(rag.retrieve_context(q))
            rows.append((q, score, source, got_block))
        results[label] = rows

    for label in ("on-topic", "off-topic"):
        print(f"── {label} ──")
        print(f"{'query':<42} {'best':>6}  {'block?':<7} source")
        for q, score, source, got in results[label]:
            print(f"{q:<42} {score:6.3f}  {'YES' if got else 'no':<7} {source}")
        scores = [s for _q, s, _src, _g in results[label]]
        blocks = sum(1 for *_x, g in results[label] if g)
        print(f"{'':<42} {min(scores):6.3f} – {max(scores):.3f}   "
              f"{blocks}/{len(results[label])} carried a documents block\n")

    lo_on = min(s for _q, s, _src, _g in results["on-topic"])
    hi_off = max(s for _q, s, _src, _g in results["off-topic"])
    print("── separability ──")
    print(f"lowest on-topic  {lo_on:.3f}")
    print(f"highest off-topic {hi_off:.3f}")
    if lo_on > hi_off:
        print(f"SEPARABLE — a threshold anywhere in ({hi_off:.3f}, {lo_on:.3f}] splits them "
              f"cleanly. This is the outcome that would make a threshold change legitimate.")
    else:
        print(f"OVERLAPPING by {hi_off - lo_on:.3f} — NO global cutoff separates these two sets. "
              f"Raising SIMILARITY_THRESHOLD trades real answers for silence; it does not fix P2. "
              f"A fix has to use a signal other than the score (see docs/plan/wip/resolution-plan.md).")

    print("\n── threshold sweep (what each cutoff would cost) ──")
    print(f"{'cutoff':>7}  {'on-topic kept':>14}  {'off-topic rejected':>19}")
    for cutoff in [round(0.40 + 0.05 * i, 2) for i in range(9)]:
        kept = sum(1 for _q, s, _src, _g in results["on-topic"] if s >= cutoff)
        rejected = sum(1 for _q, s, _src, _g in results["off-topic"] if s < cutoff)
        marker = "  <- current" if abs(cutoff - SIMILARITY_THRESHOLD) < 1e-9 else ""
        print(f"{cutoff:>7.2f}  {kept:>10}/{len(ON_TOPIC)}  {rejected:>15}/{len(OFF_TOPIC)}{marker}")

    if args.context:
        print("\n── retrieved context ──")
        for label in ("on-topic", "off-topic"):
            for q, *_rest in results[label]:
                rag.reset_topic()
                block = rag.retrieve_context(q)
                print(f"\n[{label}] {q}\n{block or '(empty)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
