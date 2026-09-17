#!/usr/bin/env python3
"""Reply-quality eval: does gemma2:2b do anything sane with what retrieval hands it? (A2)

The companion to scripts/rag_accuracy.py and scripts/rag_eval.py, and the one that measures the
REPLY. Both of those stop at retrieve_context() — they say whether the right chunk was found, not
whether the model did anything true with it. rag_accuracy's own history is the reason this exists:
a needle passed "for months" while the retrieved document was factually wrong (the reply cited
Micro:bit, Qwen and Google AI Suite — none of which this repo has ever used), because that check
only proves the DOCUMENT reached the model, never that the REPLY is true to it.

Every check here is mechanical — no model-as-judge — each one lifted from a failure already on
record in this repo:

  grounded    a literal needle (the same ones rag_accuracy.py uses) must appear in the reply.
  no_invent   strings that must NOT appear, seeded from the Micro:bit/Qwen/Google-AI-Suite
              hallucination rag_accuracy.py's own comment records.
  language    a Tagalog question gets a Tagalog reply — counted by function words, not judged.
  refuse      a question with no real answer in documents/ gets a hedge and no invented specific —
              the direct test of NO_CONTEXT_NOTICE and persona.txt's "say you are not sure",
              neither of which had ever been tested.

Every case ALSO gets a length check for free: sentence count against persona.txt's "never...past
four sentences", character count against TTS_MAX_SPOKEN_CHARS, and whether generation hit
OLLAMA_NUM_PREDICT reported separately (that pair is the whole reason 96/192/160 has a history).

gemma2:2b is SAMPLED, so every case runs --repeats times (default 3) and this reports a PASS RATE,
not pass/fail. A 1-run difference is noise, not a regression — read the rate, not one row.

    python3 -m scripts.reply_eval                # single-turn cases, 3 repeats each
    python3 -m scripts.reply_eval --repeats 5     # steadier rate, slower
    python3 -m scripts.reply_eval --history       # also run the scripted multi-turn conversations
    python3 -m scripts.reply_eval --verbose       # print every reply, not just the failing ones

COST: one full LLM turn per case per repeat, at gemma2:2b's ~27 tok/s generation — the default
CASES at --repeats 3 is on the order of minutes, not rag_accuracy's ~20 seconds (that script never
pays generation; this one always does). It is a bench tool, not a test: do not run it during a demo,
and do not run it while the robot is mid-conversation — it shares Ollama's one model slot with a
live turn and will make one wait behind the other.

Read-only against the live index and a running Ollama. Does NOT synthesise or play audio —
scripts/filler_check.py already owns the text -> speech -> text direction, and a second Piper
process running next to a live reply is the 2026-08-07 incident (bank corruption, 24s to first
audio) — see docs/tickets/A5-piper-communicate-has-no-timeout.md for the same failure family.

Recorded baseline, 2026-09-17: 10/36 = 28% (12 cases x 3 repeats), both --history scripts 3/3.
Read past the headline number — docs/tickets/A2-no-evaluation-of-generated-replies.md's "What the
first baseline found" breaks the 26 non-passes down by actual cause: most were content-clean
replies that failed only on LENGTH (persona.txt's 4-sentence cap — see docs/tickets/A9) or on a
literal-phrase needle missing a correct-but-paraphrased reply (an acknowledged harness limitation,
not a Kai defect). The real findings were the `language` cases (docs/tickets/A10 — a Tagalog
question often gets an English reply) and the `refuse` cases confirming A4's concern with evidence
(weather/robot-count questions got confidently invented answers).

This run also found, on its FIRST attempt against the robot, that the fastembed embedding cache was
incompletely downloaded this boot — dense RAG retrieval had been silently disabled for the entire
uptime, invisible outside a per-turn log WARNING (docs/tickets/A8, fixed same day). The 28% above is
from the re-run AFTER that fix; an earlier run against the broken cache scored 12/36 = 33% for an
entirely different and uninteresting reason (every "grounded" case saw zero context) — if a future
re-run of this script scores suspiciously low with EVERY grounded/no_invent case failing at once,
check `/tmp/fastembed_cache` and `[rag] WARNING: embed_query failed` in the log before assuming a
model or persona regression.

KEEP THESE CASES EXACT if you are comparing against a recorded baseline. Add new ones at the end.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai import rag                                                        # noqa: E402
from ai.llm import _ollama_request, build_chat_messages, load_persona     # noqa: E402
from ai.speak_envelope import _split_sentences                            # noqa: E402
from config.voice import OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT                # noqa: E402
from config.wake import TTS_MAX_SPOKEN_CHARS                              # noqa: E402

# persona.txt: "Nothing you say ever runs past four sentences." Read off the persona text rather
# than imported, because persona.txt is free-form prose, not a structured constant.
MAX_SENTENCES = 4


class Case(NamedTuple):
    question: str
    kind: str                          # "grounded" | "no_invent" | "language" | "refuse"
    needle: str = ""                   # kind == "grounded"
    forbidden: tuple[str, ...] = ()    # kind == "no_invent"


CASES: list[Case] = [
    # ── grounded: the same needles rag_accuracy.py uses, moved one stage downstream. Kept small —
    #    a subset, not the full 27 — because this also pays generation time, unlike that harness.
    Case("what is DEVCON?", "grounded", needle="largest volunteer community"),
    Case("when did DEVCON start?", "grounded", needle="2009"),
    Case("who founded DEVCON?", "grounded", needle="Winston Damarillo"),
    Case("what chip do you run on?", "grounded", needle="Jetson Orin Nano"),
    Case("do you need internet?", "grounded", needle="offline"),

    # ── no_invent: the exact hallucination rag_accuracy.py's own comment records. This question
    #    used to get "Micro:bit... Qwen... Google AI Suite" spoken to a room of visitors — none of
    #    which this repo has ever used — and the retrieval-only eval scored it a hit for months
    #    because the document reaching the model was never the same thing as the reply being true.
    Case("what tools were used to build you?", "no_invent",
         forbidden=("micro:bit", "microbit", "qwen", "google ai suite")),

    # ── language: persona.txt says "reply in the same language the person used."
    Case("Ilang taon na ang DEVCON?", "language"),
    Case("Sino ang nagtatag ng DEVCON?", "language"),

    # ── refuse: no real answer in documents/. Deliberately NOT all DEVCON-branded — an ungrounded
    #    DEVCON question and an ungrounded off-topic question both have to land on the same hedge,
    #    per persona.txt's "if it is not there, say you are not sure. Never invent..."
    Case("how much funding does DEVCON have this year?", "refuse"),
    Case("what is DEVCON's annual budget?", "refuse"),
    Case("what's the weather like today?", "refuse"),
    Case("how many robots has DEVCON built so far?", "refuse"),
]

# Multi-turn scripts, run only under --history. Each entry is (question, kind, needle) replayed in
# order over one growing history list, built with build_chat_messages exactly like a real
# conversation — kind/needle apply to the LAST turn only, earlier turns just build state.
#
# SCOPE NOTE: this calls the same pure path as CASES above (rag.retrieve_context + persona +
# build_chat_messages + _ollama_request), which the module docstring's "do not import
# VoiceAssistant" sketch deliberately excludes S12's identity-pinning (IDENTITY_PROMPT) and its own
# person-name extraction (ai/identity.py) — those live inside VoiceAssistant._call_ollama, which
# owns the mic and drags in sounddevice. So "does the name survive" here tests plain conversational
# memory from raw chat history within MAX_HISTORY_TURNS, not S12's pinning mechanism specifically.
SCRIPTS: list[list[tuple[str, str, str]]] = [
    [
        ("Hi, I'm Jhondel.", "", ""),
        ("What is DEVCON?", "", ""),
        ("What chip do you run on?", "", ""),
        ("Who founded DEVCON?", "", ""),
        ("What's my name?", "grounded", "Jhondel"),
    ],
    # A pronoun-free follow-up, the sticky-topic case ai/rag.py's retrieve_context is built around.
    [
        ("How many chapters does DEVCON have?", "", ""),
        ("When did it start?", "grounded", "2009"),
    ],
]

# Cheap function-word lists for the language check — not a language detector, just enough signal
# to tell Tagalog from English on a short spoken reply without needing a model to judge it.
_TAGALOG_WORDS = frozenset({
    "ang", "ng", "sa", "ay", "na", "mga", "ako", "ikaw", "siya", "kami", "tayo", "kayo", "sila",
    "hindi", "oo", "at", "para", "kasi", "po", "opo", "ba", "yung", "mo", "ko", "niya", "namin",
    "natin", "raw", "daw", "naman", "din", "rin", "pa", "lang", "ito", "iyan", "iyon", "kung",
})
_ENGLISH_WORDS = frozenset({
    "the", "is", "are", "and", "of", "to", "in", "we", "you", "i", "it", "that", "for", "with",
    "was", "were", "this", "our", "your", "a", "an", "on", "as", "have", "has", "not", "so",
})

# The direct hedge test for "refuse" cases — English and Tagalog, since persona.txt answers in
# either. English is a fixed-phrase substring match (same reasoning rag_accuracy.py gives for a
# literal needle: it cannot drift or need a model to judge it). Tagalog is checked by WORD PRESENCE
# rather than a fixed phrase — "hindi ko po sigurado" and "hindi ko talaga alam" both hedge, and a
# natural politeness particle ("po") or filler word breaking up a fixed substring must not read as
# a failure to hedge.
_HEDGE_PHRASES = ("not sure", "i'm not sure", "not certain", "don't know", "do not know", "no idea")
_HEDGE_TAGALOG_WORD_PAIRS = (("hindi", "sigurado"), ("hindi", "alam"), ("wala", "alam"))


def _is_hedged(text: str) -> bool:
    low = text.lower()
    if any(p in low for p in _HEDGE_PHRASES):
        return True
    words = set(re.findall(r"[a-zA-Z']+", low))
    return any(a in words and b in words for a, b in _HEDGE_TAGALOG_WORD_PAIRS)


def _reply_for(question: str, history: list[dict]) -> tuple[str, dict]:
    """One turn, minus S12's identity injection and the timing bookkeeping — exactly what
    VoiceAssistant._call_ollama assembles otherwise. Returns (reply text, raw Ollama response)."""
    context = rag.retrieve_context(question)
    persona = load_persona()
    user = f"{context}\n\n{question}" if context else question
    budget = OLLAMA_NUM_CTX - (OLLAMA_NUM_PREDICT or 0)
    messages = build_chat_messages(persona, history, user, budget_tokens=budget)
    data = _ollama_request(messages)
    return data["message"]["content"], data


def _check_length(reply: str, data: dict) -> tuple[bool, str]:
    n = len(_split_sentences(reply))
    chars = len(reply)
    hit_cap = bool(OLLAMA_NUM_PREDICT) and data.get("eval_count", 0) >= OLLAMA_NUM_PREDICT
    ok = n <= MAX_SENTENCES and chars <= TTS_MAX_SPOKEN_CHARS
    detail = f"{n} sentence(s), {chars} chars"
    if hit_cap:
        detail += " [hit OLLAMA_NUM_PREDICT — likely cut mid-word]"
    return ok, detail


def _guess_language(text: str) -> str:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    tl = sum(1 for w in words if w in _TAGALOG_WORDS)
    en = sum(1 for w in words if w in _ENGLISH_WORDS)
    if tl == 0 and en == 0:
        return "unknown"
    return "tagalog" if tl > en else "english"


def _check_case(case: Case, reply: str) -> tuple[bool, str]:
    if case.kind == "grounded":
        ok = case.needle.lower() in reply.lower()
        return ok, f"wanted {case.needle!r}"
    if case.kind == "no_invent":
        hit = next((f for f in case.forbidden if f.lower() in reply.lower()), None)
        return hit is None, (f"contains {hit!r}" if hit else "clean")
    if case.kind == "language":
        lang = _guess_language(reply)
        return lang == "tagalog", f"detected {lang}"
    if case.kind == "refuse":
        hedged = _is_hedged(reply)
        invented_digit = bool(re.search(r"\d", reply))
        return hedged and not invented_digit, f"hedged={hedged} digits_present={invented_digit}"
    raise ValueError(f"unknown case kind {case.kind!r}")   # pragma: no cover - CASES is fixed


def _run_case(case: Case, repeats: int) -> tuple[int, list[tuple[str, bool, str, bool, str]]]:
    """Returns (passes, [(reply, check_ok, check_detail, length_ok, length_detail), ...]).

    The two checks are kept SEPARATE all the way to the report, not folded into one pass/fail —
    a reply that is factually right but runs to eight sentences is a different bug from one that
    invents a fact, and a report that can't tell them apart sends the reader chasing the wrong one.
    """
    passes = 0
    rows = []
    for _ in range(repeats):
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):    # retrieve_context/_ollama_request narrate; noise here
            reply, data = _reply_for(case.question, [])
        length_ok, length_detail = _check_length(reply, data)
        check_ok, check_detail = _check_case(case, reply)
        rows.append((reply, check_ok, check_detail, length_ok, length_detail))
        if length_ok and check_ok:
            passes += 1
    return passes, rows


def _run_history_scripts(repeats: int, verbose: bool) -> None:
    print(f"\n-- multi-turn scripts ({len(SCRIPTS)}), {repeats} repeat(s) each --\n")
    for script in SCRIPTS:
        passes = 0
        last_replies = []
        for _ in range(repeats):
            history: list[dict] = []
            reply = ""
            for question, _kind, _needle in script:
                sink = io.StringIO()
                with contextlib.redirect_stdout(sink):
                    reply, _data = _reply_for(question, history)
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": reply})
            last_q, last_kind, last_needle = script[-1]
            ok = (last_needle.lower() in reply.lower()) if last_kind == "grounded" else True
            passes += int(ok)
            last_replies.append(reply)
        tag = "PASS" if passes == repeats else ("FLAKY" if passes else "FAIL")
        print(f"  {tag:<6} {script[0][0]!r} .. {script[-1][0]!r}  {passes}/{repeats}")
        if verbose or tag != "PASS":
            for i, r in enumerate(last_replies):
                print(f"      #{i + 1} final reply: {r!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=3,
                    help="samples per case (default 3 — gemma2:2b is sampled, not deterministic)")
    ap.add_argument("--history", action="store_true",
                    help="also run the scripted multi-turn conversations")
    ap.add_argument("--verbose", action="store_true",
                    help="print every reply, not just the failing/flaky ones")
    args = ap.parse_args()

    # ai/rag.py's own docstring: "call ensure_model_loaded() and load_index() once at startup."
    # Skipping this is silent and catastrophic here, not just slow — retrieve_context() fails open
    # to "" with no index loaded, so every "grounded" case would look like a model that never
    # learned its own facts, when the actual bug was this harness never loading them.
    rag.ensure_model_loaded()
    rag.load_index()
    if not rag._INDEX_CHUNKS:
        print("No index. Run: python3 -m ai.index_documents", file=sys.stderr)
        return 1

    total_turns = len(CASES) * args.repeats
    print(f"cost: {len(CASES)} cases x {args.repeats} repeats = {total_turns} LLM turns at "
          f"gemma2:2b's ~27 tok/s generation — this is minutes, not seconds. Not for a demo.\n")

    total_passes = 0
    for case in CASES:
        passes, rows = _run_case(case, args.repeats)
        total_passes += passes
        tag = "PASS" if passes == args.repeats else ("FLAKY" if passes else "FAIL")
        print(f"  {tag:<6} [{case.kind:<9}] {case.question:<50} {passes}/{args.repeats}")
        if args.verbose or tag != "PASS":
            for i, (reply, check_ok, check_detail, length_ok, length_detail) in enumerate(rows):
                why = check_detail if not check_ok else ""
                if not length_ok:
                    why = f"{why} | LENGTH: {length_detail}" if why else f"LENGTH: {length_detail}"
                print(f"      #{i + 1} [{why or 'ok'}]: {reply!r}")

    total_runs = len(CASES) * args.repeats
    print(f"\nreply-quality pass rate: {total_passes}/{total_runs} = {total_passes / total_runs:.0%}")

    if args.history:
        _run_history_scripts(args.repeats, args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
