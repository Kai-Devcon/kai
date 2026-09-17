"""Fuzzy "DEVCON" spotting — rewrite however Whisper spelled the brand into the canonical form.

Called by ai/rag.py on the query, immediately before it is embedded.

Why: almost everything in documents/ is about DEVCON, and every chunk spells it "DEVCON". Whisper
does not. Real transcripts contain "defcon", "dev com", "debcon", "Devon", "de con", "dev khan".
bge-small-en-v1.5 embeds those as different words, so the query drifts away from the very chunks
that answer it, every score falls under SIMILARITY_THRESHOLD, and the most on-topic question Kai
can be asked retrieves nothing at all. Folding them back onto "DEVCON" is what makes the documents
reachable by ear rather than by spelling.

Scope: the RAG query only — never the transcript shown on /params, and never the text handed to
the LLM as the turn. A fuzzy guess is good enough to steer retrieval (a wrong one just retrieves
nothing, exactly as today) but not good enough to put words in the speaker's mouth.

Three matchers live here, all on the query side and all feeding ai/rag.py: canonicalize_devcon()
rewrites the brand, mentions_devcon() answers "is this turn on-topic?" for the failsafe chain, and
match_entities() does the same job for the program/chapter/person names in the gazetteer, which
are just as mishearable and carry no DEVCON token to fall back on.

Pure stdlib, and it reuses ai/wake_phrase.py's tokenizer rather than growing a second one. Same
reasoning as that module: everything here is string handling with all the bugs in the offsets and
the thresholds, so it must stay testable instantly with plain strings — no audio, no model, no
numpy in the way.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from ai.wake_phrase import Token, normalize_tokens
from config.rag import (
    DEVCON_BLOCKLIST, DEVCON_CANONICAL, DEVCON_MATCH_RATIO, DEVCON_SKELETON_CLASSES,
    DEVCON_SKELETON_DROP, DEVCON_SPELLINGS, GAZETTEER_MATCH_RATIO, GAZETTEER_MAX_TOKENS,
    TAGALOG_QUESTION_WORDS,
)

# Length window for a single token, from len("devcon") == 6. Everything real that Whisper produces
# is 5-8 characters ("decon", "devcon", "devcorn", "devconph"). The window is defense in depth
# behind the ratio, and mirrors _NAME_LEN_SLACK in wake_phrase.py: it is what keeps the short
# fragments out structurally, so "dev" and "con" can never fire on their own no matter how the
# threshold is later tuned.
_MIN_LEN = 5
_MAX_LEN = 8

# Two-token form ("dev con", "de con", "dev khan"). Joins may run one character longer than a single
# token to admit "devkhan" (7) and "dev conn" (8) without letting "devcon founder" merge.
_JOINED_MIN_LEN = 5
_JOINED_MAX_LEN = 9

# Both halves of a join must be too short to stand alone. Without this, any 2-3 letter word in front
# of a perfectly good "devcon" gets absorbed: "isdevcon" is 0.857 and "what is DEVCON Philippines?"
# came out as "what DEVCON Philippines?", "uy devcon ba" as "DEVCON ba". Requiring both halves to be
# fragments is what distinguishes a split rendering ("dev"+"con") from a real word next to a whole
# one ("is"+"devcon") — and it protects the trailing side too, keeping "devcon ph" intact.
_FRAGMENT_MAX_LEN = _MIN_LEN - 1


def _ratio(token: str) -> float:
    """Closeness of `token` to its nearest accepted spelling, 0..1. Exact hits short-circuit."""
    if token in DEVCON_SPELLINGS:
        return 1.0
    return max(SequenceMatcher(None, token, spelling).ratio() for spelling in DEVCON_SPELLINGS)


def _skeleton(token: str) -> str:
    """Sound-alike key: vowels dropped, consonants folded onto classes, doubles collapsed.

    "devcon", "davcan", "duvcun", "devkon", "tefgon" and "dev khan" all reduce to TFKN. This is
    the failsafe behind _ratio: difflib compares characters in ORDER, so a vowel shift costs it
    real score while costing the sound nothing. See DEVCON_SKELETON_CLASSES for the folding.
    """
    out: list[str] = []
    for ch in token:
        if ch in DEVCON_SKELETON_DROP:
            continue
        cls = DEVCON_SKELETON_CLASSES.get(ch)
        if cls is None:          # a digit or a letter with no class — keep it, it is a difference
            cls = ch
        if not out or out[-1] != cls:
            out.append(cls)
    return "".join(out)


# Built once from the accepted spellings, so adding a spelling widens both matchers at once.
_SKELETONS = frozenset(_skeleton(s) for s in DEVCON_SPELLINGS)


def _matches(token: str) -> bool:
    """Either matcher accepts. Callers must have applied the blocklist and length window first."""
    return _ratio(token) >= DEVCON_MATCH_RATIO or _skeleton(token) in _SKELETONS


def looks_like_devcon(token: str) -> bool:
    """True if this single normalized token is a rendering of "devcon". The length window and the
    blocklist are both checked before the matchers — they are cheaper and they are the part that
    stops real words, not the ratio."""
    if token in DEVCON_BLOCKLIST or not (_MIN_LEN <= len(token) <= _MAX_LEN):
        return False
    return _matches(token)


def _match_at(tokens: list[Token], i: int) -> int | None:
    """Index of the last token of a DEVCON span starting at `i`, or None.

    Single token first, two-token join second — and the join only fires between two fragments, see
    _FRAGMENT_MAX_LEN. Returning the END index (not a length) is what lets the caller resume after
    the span and keep the offsets it needs."""
    if looks_like_devcon(tokens[i].text):
        return i
    if i + 1 < len(tokens):
        first, second = tokens[i].text, tokens[i + 1].text
        joined = first + second
        if (first not in DEVCON_BLOCKLIST and second not in DEVCON_BLOCKLIST
                and len(first) <= _FRAGMENT_MAX_LEN and len(second) <= _FRAGMENT_MAX_LEN
                and _JOINED_MIN_LEN <= len(joined) <= _JOINED_MAX_LEN
                and _matches(joined)):
            return i + 1
    return None


def canonicalize_devcon(text: str) -> str:
    """Replace every fuzzy "DEVCON" span in `text` with DEVCON_CANONICAL, leaving the rest of the
    string byte-for-byte alone.

    Rebuilt by slicing the ORIGINAL text around the matched spans, never re-joined from the
    normalized tokens — that would hand the embedder casefolded, punctuation-stripped,
    accent-stripped text and quietly mangle anything numeric ("DEVCON 2026" -> "devcon 2026").

    Returns the input unchanged (same object) when nothing matches, which is the common case: this
    runs on every voice turn, most of which never mention the brand.
    """
    if not text:
        return text
    tokens = normalize_tokens(text)
    if not tokens:
        return text

    parts: list[str] = []
    cursor = 0
    i = 0
    while i < len(tokens):
        last = _match_at(tokens, i)
        if last is None:
            i += 1
            continue
        parts.append(text[cursor:tokens[i].start])
        parts.append(DEVCON_CANONICAL)
        cursor = tokens[last].end
        i = last + 1

    if not parts:
        return text
    parts.append(text[cursor:])
    return "".join(parts)


def expand_tagalog_question_words(text: str) -> str:
    """Append an English translation for every Tagalog question word found in `text` — "sino ang
    nagtatag ng DEVCON?" gains a trailing " who". Retrieval-only, additive, and idempotent-in-
    practice: see TAGALOG_QUESTION_WORDS for why (EMBED_MODEL cannot read Tagalog at all, and the
    question word carries most of a short query's meaning). Returns `text` unchanged when nothing
    matches, which is the common case — most turns are English or carry no question word at all.
    """
    if not text:
        return text
    hits: list[str] = []
    seen: set[str] = set()
    for tok in normalize_tokens(text):
        english = TAGALOG_QUESTION_WORDS.get(tok.text)
        if english and english not in seen:
            seen.add(english)
            hits.append(english)
    if not hits:
        return text
    return f"{text} {' '.join(hits)}"


def mentions_devcon(text: str) -> bool:
    """True if the brand is in `text` however it was spelled — the "is this turn on-topic?" test
    the failsafe chain in ai/rag.py gates on.

    Deliberately the same matcher canonicalize_devcon uses rather than a substring check: the
    whole point is that the token in the transcript is usually not the string "DEVCON".
    """
    tokens = normalize_tokens(text)
    i = 0
    while i < len(tokens):
        last = _match_at(tokens, i)
        if last is None:
            i += 1
        else:
            return True
    return False


# ── Entity gazetteer ───────────────────────────────────────────────────────────
# The names in documents/ that are NOT the brand: programs, chapters, people. Whisper mangles
# these too, and a mangled program name carries no DEVCON token at all, so nothing above fires on
# it. The list is built at index time and travels in the index — see ai/index_documents.py for
# how it is harvested and why the rare-word filter is what keeps ordinary headings out of it.

# Preparing an entity means normalizing and grouping it by token count; the query is only ever
# compared against entities of the same length. Cached because retrieve_context runs per voice
# turn against the same list — keyed by identity of the loaded list, not its contents.
_PREPARED_KEY: object = None
_PREPARED: dict[int, list[tuple[str, str]]] = {}


def _prepare(entities: list[str]) -> dict[int, list[tuple[str, str]]]:
    global _PREPARED_KEY, _PREPARED
    if _PREPARED_KEY is entities:
        return _PREPARED
    by_len: dict[int, list[tuple[str, str]]] = {}
    for entity in entities:
        parts = [t.text for t in normalize_tokens(entity)]
        if not parts or len(parts) > GAZETTEER_MAX_TOKENS:
            continue
        by_len.setdefault(len(parts), []).append((" ".join(parts), entity))
    _PREPARED_KEY, _PREPARED = entities, by_len
    return by_len


def match_entities(text: str, entities: list[str]) -> list[str]:
    """Canonical spellings of every gazetteer name `text` appears to mention, in index order.

    Fuzzy, at GAZETTEER_MATCH_RATIO — "geeks on the beach" and "gigs on a beach" both resolve to
    "Geeks on a Beach". Returns the CANONICAL string (not the matched span) because the caller
    appends it to the embedded query: the document's own spelling is the one that retrieves.
    """
    if not entities:
        return []
    by_len = _prepare(entities)
    if not by_len:
        return []
    words = [t.text for t in normalize_tokens(text)]
    if not words:
        return []

    hits: list[str] = []
    seen: set[str] = set()
    for n, candidates in by_len.items():
        if n > len(words):
            continue
        for start in range(len(words) - n + 1):
            gram = " ".join(words[start:start + n])
            for norm, canonical in candidates:
                if canonical in seen:
                    continue
                if gram == norm or SequenceMatcher(None, gram, norm).ratio() >= GAZETTEER_MATCH_RATIO:
                    seen.add(canonical)
                    hits.append(canonical)
    return hits
