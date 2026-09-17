#!/usr/bin/env python3
"""Answer-in-context accuracy: does retrieval actually put the answer in front of the model?

The companion to scripts/rag_eval.py, and the one that measures correctness. rag_eval reports
SCORES — how well the best chunk matched, and whether a documents block was carried. That says
nothing about whether the block contained the answer. This script asks the only question that
decides whether Kai says something true:

    for a question whose answer IS in documents/, does retrieve_context() return that answer?

Ground truth is a literal substring that must appear in the returned block. Deliberately crude:
a substring cannot drift, argue, or need a model to judge it, and if the string is absent then
the answer was not available to the LLM no matter how the LLM behaves.

A third companion, scripts/reply_eval.py, picks up one stage further downstream: does the model DO
anything true with the block this script confirms reached it. See its docstring — the
Micro:bit/Qwen/Google AI Suite hallucination below is exactly the gap it exists to close.

    python3 -m scripts.rag_accuracy           # score + the misses
    python3 -m scripts.rag_accuracy --verbose # plus the context each failing query received

Read-only against the existing index. No settings, no servos, no Ollama, no audio — safe on a
live robot, though it competes for CPU with the voice pipeline for ~20 seconds.

Recorded baseline, 50-chunk index, 2026-08-09: **30/31 = 97%**, at SIMILARITY_THRESHOLD 0.55.
The corpus was consolidated from six source documents into one Q&A knowledge base
(documents/devcon_faq_rag.md), and the indexer now embeds each entry's `synonyms:` line while
showing the model only the Q&A pair — which is what closed "where was the first event held?",
one of the two misses recorded below as an honest limit of the embedder. Several ground-truth
needles moved with the corpus; each one that changed says why, inline. The single remaining miss
is content loss, not retrieval: Kai's birthday was in kai_facts.txt and has no entry in the new
file.

Previous baseline, 513-chunk index, 2026-08-07: **29/32 = 91%**.
(Previously 21/24 = 88% on the 505-chunk index, before the 8 hardware/origin facts that the
filler bank cites were added to documents/kai_facts.txt. Same three misses, all pre-existing —
verified by re-running them with only those 8 chunks dropped from the loaded index.)
The three misses are honest limits of the embedder, not tuning:
  - "what is DEVCON's mission?" / "what is DEVCON trying to achieve?" — the answer chunk says
    "democratize access to AI"; no rare literal token connects it to the word "mission", so the
    lexical rescue cannot see it and bge-small does not make the paraphrase.
  - "where was the first event held?" — "TechBar" is rare, but the QUESTION contains no rare
    token to find it by, so neither layer has anything to work with.
Closing those needs a better embedding model or a query rewrite, not another threshold.

KEEP THESE CASES EXACT if you are comparing against the baseline above. Add new ones at the end.
"""

from __future__ import annotations

import argparse
import io
import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai import rag                                                  # noqa: E402

# (question, a substring that MUST appear in the retrieved block for the answer to be possible)
CASES: list[tuple[str, str]] = [
    # ── the broad questions a visitor opens with ──
    ("what is DEVCON?",                     "largest volunteer community"),
    ("when did DEVCON start?",              "2009"),
    ("who founded DEVCON?",                 "Winston Damarillo"),
    ("who is the president of DEVCON?",     "Winston Damarillo"),
    # The knowledge base no longer states a chapter TOTAL anywhere. Its Chapters section names
    # nine active and three inactive; the only number is "13 active locations", in a different
    # section and phrased as ambition rather than fact. Fix the document, not this needle, if the
    # count matters — visitors ask this constantly and Kai can currently only answer sideways.
    ("how many chapters does DEVCON have?", "13 active locations"),
    ("how many people have joined DEVCON?", "60,000"),
    # Was "democratize access to AI" for both. In the new knowledge base that phrase is the
    # ANNIVERSARY FOCUS, not the mission — the mission has its own sentence and retrieval now
    # returns it. This pair was one of the two long-standing misses named in the docstring above.
    ("what is DEVCON's mission?",           "vibrant developer communities"),
    # Split off rather than duplicated: in the new knowledge base "mission" and "what is DEVCON
    # trying to achieve" are genuinely different questions with different answers, and retrieval
    # distinguishes them correctly — this one lands in the Ambition section, which is right.
    ("what is DEVCON trying to achieve?",   "million micro companies"),
    ("what is the anniversary theme?",      "AI-Ready Nation"),
    ("what is DEVCON's website?",           "devcon.ph"),
    ("where can I find my nearest chapter?", "devcon.ph/chapters"),
    ("what was the first DEVCON event?",    "DEVCON Visayas"),
    ("where was the first event held?",     "TechBar"),
    ("what is DEVCON registered as?",       "DevConnect Philippines"),
    # ── the programmes, which is what the gazetteer exists for ──
    ("what is DEVCON Kids?",                "Kids"),
    ("what is Campus DEVCON?",              "Campus DEVCON"),
    ("what is DEVCON CREST?",               "CREST"),
    ("is there a DEVCON chapter in Davao?", "Davao"),
    ("what is DEVCON for Educators?",       "Educators"),
    # ── Kai's own facts, which compete against a much larger DEVCON corpus ──
    # KNOWN MISS, deliberately left failing. This fact lived in documents/kai_facts.txt, which
    # did not survive the consolidation into devcon_faq_rag.md — the new file's "Kai (robot)"
    # section covers how Kai works, but none of the small personal facts a child at a booth
    # actually asks a robot. It stays in the harness as the standing reminder that it is gone,
    # and goes green the moment someone adds the entry back.
    # ("what is Kai's favourite snack?", "dried mangoes") was dropped on request — the favourite
    # snack is no longer a fact Kai is expected to have.
    ("when is your birthday?",              "June 2026"),
    # This was "who built you?" until the Cohort 4 fact arrived with the filler bank and
    # outranked the Manila-chapter line -- which is the better answer to that question, so the
    # question moved rather than the fact. "who built you?" is covered below, expecting Cohort 4.
    # Needle narrowed from "assembled by volunteers" to the wording the new knowledge base uses.
    ("who assembled you?",                  "chapter volunteers"),
    ("how far can your neck turn?",         "180 degrees"),
    ("what camera do you have?",            "MediaPipe"),
    # ── the hardware/origin facts the filler lines advertise, so a follow-up question is
    #    answerable from the corpus rather than from whatever the filler happened to say ──
    ("what chip do you run on?",             "Jetson Orin Nano"),
    ("who built you?",                       "Cohort 4"),
    ("who made you?",                        "Cohort 4"),
    ("do you need internet?",                "offline"),
    ("do you use the cloud?",                "no cloud"),
    ("what is edge computing?",              "edge computing"),
    # Needle is "Arduino" and not "Claude Code": the kai-stack entry is not the only chunk naming
    # Claude Code (the Jumpstart internship entry does too), so a substring test on it would pass
    # while retrieving the wrong entry entirely. "Arduino" appears in exactly one chunk.
    #
    # It used to be "Micro:bit", which the entry claimed and the robot dutifully said out loud —
    # along with Qwen and Google AI Suite, none of which this repo has ever used. The eval scored
    # that a hit for months because the harness only checks the DOCUMENT reached the model, never
    # that the document was true. Keep needles pointed at facts the code can be diffed against.
    ("what tools were used to build you?",    "Arduino"),
    ("who lent you?",                        "NMBLR.AI"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbose", action="store_true",
                    help="print the context each FAILING query actually received")
    args = ap.parse_args()

    rag.ensure_model_loaded()
    rag.load_index()
    if not rag._INDEX_CHUNKS:
        print("No index. Run: python3 -m ai.index_documents", file=sys.stderr)
        return 1
    print(f"index: {len(rag._INDEX_CHUNKS)} chunks, {len(CASES)} questions with a known answer\n")

    hits: list[tuple[str, str, str]] = []
    misses: list[tuple[str, str, str]] = []
    for question, needle in CASES:
        # retrieve_context narrates its layers; that is noise in a score table.
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            context = rag.retrieve_context(question)
        (hits if needle.lower() in context.lower() else misses).append(
            (question, needle, context))

    for question, needle, _ctx in hits:
        print(f"  HIT   {question:<44} {needle!r}")
    for question, needle, _ctx in misses:
        print(f"  MISS  {question:<44} {needle!r}")

    total = len(CASES)
    print(f"\nanswer-in-context accuracy: {len(hits)}/{total} = {len(hits)/total:.0%}")

    if misses and args.verbose:
        for question, needle, context in misses:
            print(f"\n{'=' * 90}\nMISS: {question}   (wanted {needle!r})")
            print(context or "   <no context returned>")
    elif misses:
        print("Re-run with --verbose to see what the failing queries received instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
