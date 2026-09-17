"""
Retrieval-augmented generation over documents in documents/.

Call ensure_model_loaded() and load_index() once at startup (see face_track.py's pre-warm
threads). Call retrieve_context(query_text) from voice_assistant.py before each Ollama call.
index_documents.py builds documents/.rag_index.json — run it after adding/changing files.

Embeddings run entirely in-process via fastembed (ONNX/CPU), NOT through Ollama. Measured on
this Jetson: gemma3:4b and any Ollama-served embedding model (tried nomic-embed-text at 595MB,
even all-minilm at 76MB) cannot both stay resident — loading either one always evicts the
other, even with keep_alive=-1 on both. That turns every RAG-enabled voice turn into a
double reload (~7-13s + ~48-51s). fastembed sidesteps this entirely: it never touches
Ollama's model-loading/eviction logic, so gemma3:4b stays put.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

# ── Paths (derived from this file's location — not tunable config) ─────────────
DOCUMENTS_DIR = Path(__file__).parent.parent / "documents"
INDEX_PATH    = DOCUMENTS_DIR / ".rag_index.json"

# Tunable RAG knobs (embedding model, task prefixes, chunking, top-k, threshold) live in
# config/rag.py; re-imported here so the names stay module-level for internal use + callers.
from config.rag import (
    EMBED_MODEL, DOCUMENT_PREFIX, QUERY_PREFIX,
    CHUNK_SIZE_CHARS, CHUNK_OVERLAP_CHARS, TOP_K, SIMILARITY_THRESHOLD, SOURCE_BOOST,
    ANAPHORA_WORDS, DEVCON_CANONICAL, FALLBACK_THRESHOLD, FALLBACK_TOP_K,
    LEXICAL_MAX_DF_RATIO, LEXICAL_MIN_SCORE, LEXICAL_RESCUE_MAX_DF,
    LEXICAL_RESCUE_MIN_SCORE, LEXICAL_RESCUE_SLOTS, MMR_MAX_SIMILARITY,
    NO_CONTEXT_NOTICE, PRIMER_MAX_CHUNKS, PRIMER_MAX_IN_RANKING, PRIMER_SECTION, PRIMER_SOURCE,
    SECTION_BOOST, STICKY_TURNS,
)


def is_primer(chunk: dict) -> bool:
    """True for the broad "what is DEVCON" material the failsafe layer falls back on.

    Matched by section OR filename: the primer used to be its own file and is now a section of
    the one knowledge base. See PRIMER_SECTION — a filename-only test went silently false for
    every chunk the moment documents/ was consolidated, which disables a failsafe rather than
    breaking anything loudly."""
    return (chunk.get("section") == PRIMER_SECTION) or (chunk.get("source") == PRIMER_SOURCE)


def source_boost(chunk: dict) -> float:
    """The per-chunk score bonus from SECTION_BOOST / SOURCE_BOOST, larger of the two."""
    return max(SECTION_BOOST.get(chunk.get("section", ""), 0.0),
               SOURCE_BOOST.get(chunk.get("source", ""), 0.0))


def chunk_label(chunk: dict) -> str:
    """What a chunk is called in the prompt. The section ("Chapters") tells the model something;
    the filename stopped doing that when every chunk started coming from one file."""
    return chunk.get("section") or chunk.get("source") or "unknown"

# Fuzzy brand-name folding on the query side (see retrieve_context). Pure stdlib — importing it
# here does not pull the audio/whisper stack in, only wake_phrase.py's tokenizer.
from ai.query_alias import (
    canonicalize_devcon, expand_tagalog_question_words, match_entities, mentions_devcon,
)
from ai.wake_phrase import normalize_tokens

_embed_model = None
_INDEX_CHUNKS: list[dict] = []
_INDEX_VECTORS: np.ndarray | None = None
# Gazetteer of names harvested from the documents at index time (ai/index_documents.py).
_INDEX_ENTITIES: list[str] = []
# Lexical fallback tables, derived from _INDEX_CHUNKS at load: per-chunk token sets, and the
# inverse document frequency that decides which of a query's words actually identify anything.
_INDEX_TOKENS: list[frozenset[str]] = []
_INDEX_DF: dict[str, int] = {}
_INDEX_IDF: dict[str, float] = {}
# Turns remaining in which a brand-flagged subject may be re-applied to an unrelated-looking
# follow-up. Module state, like the index cache above; reset_topic() clears it with the session.
_sticky_turns = 0

# Failure accounting for the fail-open handlers below. The fail-open POLICY is right — a broken
# index must behave exactly as if RAG did not exist — but the SILENCE was not: retrieve_context()
# returning "" for a TypeError is indistinguishable from "no relevant documents", and "" is the
# dangerous state, the one where gemma2:2b answers DEVCON questions from pretraining. A whole
# retrieval regression could ship and present only as "Kai's answers got vaguer".
_ERROR_LOG_INTERVAL_S = 60.0
_error_count = 0
_last_error = ""
_last_error_t = 0.0


def _note_error(where: str, exc: BaseException) -> None:
    """Record a swallowed failure, and say so — at most once a minute per distinct error.

    Rate-limited for the reason config/tracking.py records about NO_FACE: an unconditional line
    here would run at turn rate against a persistent fault and bury everything worth reading. The
    counter is unconditional, so /params still shows the true total.
    """
    global _error_count, _last_error, _last_error_t
    _error_count += 1
    detail = f"{type(exc).__name__}: {exc}"
    now = time.monotonic()
    if detail != _last_error or now - _last_error_t >= _ERROR_LOG_INTERVAL_S:
        print(f"[rag] WARNING: {where} failed ({detail}) — answering without documents", flush=True)
        _last_error, _last_error_t = detail, now


def status() -> dict:
    """Retrieval health for /params. `rag_errors` is monotonic across the process."""
    return {"rag_errors": _error_count, "rag_last_error": _last_error}


def _reset_errors() -> None:
    """Tests only — module state is shared across a process, like vision/presence.reset()."""
    global _error_count, _last_error, _last_error_t
    _error_count, _last_error, _last_error_t = 0, "", 0.0


def ensure_model_loaded() -> None:
    """Lazy-load the embedding model singleton. Call once at startup to pre-warm."""
    global _embed_model
    if _embed_model is None:
        from fastembed import TextEmbedding
        _embed_model = TextEmbedding(model_name=EMBED_MODEL)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS, overlap: int = CHUNK_OVERLAP_CHARS) -> list[str]:
    """Fixed-size chunker with overlap. Single source of truth — used by index_documents.py too."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in-process via fastembed. Used by both embed_query (one text)
    and index_documents.py (many chunks)."""
    ensure_model_loaded()
    return [vec.tolist() for vec in _embed_model.embed(texts)]


def embed_query(text: str) -> list[float] | None:
    """Embed a live user query. Fails open (returns None) — must never crash a voice turn."""
    try:
        return embed_batch([QUERY_PREFIX + text])[0]
    except Exception as exc:
        _note_error("embed_query", exc)
        return None


def load_index() -> None:
    """Load the persisted index into the module cache. Missing/broken file -> empty cache,
    never raises. Idempotent — safe to call repeatedly (mirrors ensure_input_resolved)."""
    global _INDEX_CHUNKS, _INDEX_VECTORS, _INDEX_ENTITIES
    try:
        data = json.loads(INDEX_PATH.read_text())
        chunks = data.get("chunks", [])
        entities = data.get("entities", [])
        _warn_if_stale(data.get("sources"))
    except Exception as exc:
        # A missing index is the ordinary case on a fresh checkout, not a fault — it means
        # "nothing indexed yet". A malformed one is a real failure and has to be visible.
        if not isinstance(exc, FileNotFoundError):
            _note_error("load_index", exc)
        _INDEX_CHUNKS = []
        _INDEX_VECTORS = None
        _INDEX_ENTITIES = []
        _build_lexical_tables()
        return
    _INDEX_CHUNKS = chunks
    _INDEX_VECTORS = np.array([c["embedding"] for c in chunks], dtype=np.float64) if chunks else None
    # Pre-1.1 indexes have no "entities" key; an empty gazetteer simply disables that layer
    # rather than breaking retrieval, so an unreindexed box degrades instead of failing.
    _INDEX_ENTITIES = entities if isinstance(entities, list) else []
    _build_lexical_tables()


def _warn_if_stale(manifest: dict | None) -> None:
    """Say so, loudly, when documents/ no longer matches what was indexed.

    Silent staleness is the worst failure this module has: Kai keeps answering, fluently and
    confidently, from documents that have since been corrected — which is precisely how the
    chapter-count contradiction (known-issues P7) stayed live after the sources were fixed.
    A print at startup costs nothing and turns that into something someone can see.

    Advisory only, and never raises: a mismatch still loads the index and still answers. Indexes
    written before this key existed (no "sources") simply skip the check, like "entities" does.
    Size and mtime, not a hash — copying documents/ between machines rewrites mtimes, and a
    false "reindex me" is a cheap error while a slow startup is not."""
    if not isinstance(manifest, dict):
        return
    try:
        current = {p.name: [p.stat().st_size, int(p.stat().st_mtime)]
                   for p in DOCUMENTS_DIR.iterdir()
                   if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"}}
    except Exception:
        return
    added   = sorted(set(current) - set(manifest))
    removed = sorted(set(manifest) - set(current))
    changed = sorted(n for n in set(current) & set(manifest)
                     if list(current[n]) != list(manifest[n]))
    if not (added or removed or changed):
        return
    detail = ", ".join(f"{label} {names}" for label, names in
                       (("added", added), ("removed", removed), ("changed", changed)) if names)
    print(f"[rag] WARNING: index is stale — {detail}. "
          f"Run: python3 -m ai.index_documents")


def _build_lexical_tables() -> None:
    """Tokenize the cached chunks once, at load, for lexical_rank.

    Frequency rather than a stoplist: "what", "is" and "the" are excluded by their own document
    frequency, "Iligan" and "60,000" carry the weight. That is the entire point of this layer —
    it is strong exactly where the embedder is weak, on rare literal tokens — and nothing here
    needs maintaining when documents/ changes."""
    global _INDEX_TOKENS, _INDEX_DF, _INDEX_IDF
    _INDEX_TOKENS = [frozenset(t.text for t in normalize_tokens(c.get("text", "")))
                     for c in _INDEX_CHUNKS]
    df: dict[str, int] = {}
    for tokens in _INDEX_TOKENS:
        for token in tokens:
            df[token] = df.get(token, 0) + 1
    total = len(_INDEX_TOKENS)
    _INDEX_DF = df
    _INDEX_IDF = {token: math.log(total / (1 + count)) for token, count in df.items()} if total else {}


def lexical_rank(query: str, top_k: int = TOP_K, max_df: int | None = None,
                 min_score: float = LEXICAL_MIN_SCORE) -> list[dict]:
    """IDF-weighted token overlap over the cached chunks — the non-semantic half of retrieval.

    Score is the fraction of the query's IDF mass the chunk contains, so it is comparable across
    queries of different lengths and `min_score` means the same thing for all of them.
    A query made entirely of common words scores nothing against anything, which is correct:
    it has no rare token to be found by, and dense retrieval is the right tool for it.

    Two callers with deliberately different bars. The fallback layer runs only when dense found
    NOTHING and uses the loose defaults, because there any signal beats none. lexical_rescue runs
    on every turn and passes much stricter ones — see LEXICAL_RESCUE_MAX_DF for the measurement
    showing what the loose bar does when it is always on.
    """
    if not _INDEX_TOKENS or not _INDEX_IDF:
        return []
    ceiling = max(1, int(len(_INDEX_TOKENS) * LEXICAL_MAX_DF_RATIO))
    if max_df is not None:
        ceiling = min(ceiling, max_df)
    query_tokens = {t.text for t in normalize_tokens(query)}
    weights = {t: _INDEX_IDF[t] for t in query_tokens
               if 0 < _INDEX_DF.get(t, 0) <= ceiling and _INDEX_IDF.get(t, 0.0) > 0.0}
    total = sum(weights.values())
    if total <= 0:
        return []
    scored = []
    for tokens, chunk in zip(_INDEX_TOKENS, _INDEX_CHUNKS):
        hit = sum(w for t, w in weights.items() if t in tokens)
        if hit / total >= min_score:
            scored.append((hit / total, chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _score, chunk in scored[:top_k]]


def lexical_rescue(query: str, chunks: list[dict]) -> list[dict]:
    """Give one of the dense results' slots to a strong lexical hit dense missed.

    The measured failure this fixes: dense retrieval smooths away exactly the rare literal token
    that identifies the answering chunk. "what is DEVCON CREST?" missed the CREST section although
    "crest" appears in 1 chunk out of 505, because the query also looks like every other broad
    DEVCON question and the primer wins on overall shape. Running the lexical layer only when
    dense returns NOTHING meant it never saw these — dense had already returned three chunks.

    The displaced chunk is the weakest dense one, so the strongest dense results are never at
    risk. See LEXICAL_RESCUE_SLOTS for why this is one slot and not two, and
    LEXICAL_RESCUE_MAX_DF for why this layer cannot reuse the fallback's much looser bar."""
    if not chunks or LEXICAL_RESCUE_SLOTS <= 0:
        return chunks
    seen = {c.get("text") for c in chunks}
    extra = [c for c in lexical_rank(query, top_k=len(chunks) + LEXICAL_RESCUE_SLOTS,
                                     max_df=LEXICAL_RESCUE_MAX_DF,
                                     min_score=LEXICAL_RESCUE_MIN_SCORE)
             if c.get("text") not in seen][:LEXICAL_RESCUE_SLOTS]
    if not extra:
        return chunks
    # Labeled by section, not filename: with one knowledge base every rescue logged the same
    # string, which made the line useless exactly when it is the only evidence of what happened.
    print(f"[rag] lexical rescue: {[chunk_label(c) for c in extra]} "
          f"displacing the weakest dense chunk")
    return chunks[:max(0, len(chunks) - len(extra))] + extra


def primer_chunks() -> list[dict]:
    """The pinned fact sheet, or [] if it was never indexed. Last resort in retrieve_context."""
    return [c for c in _INDEX_CHUNKS if is_primer(c)][:PRIMER_MAX_CHUNKS]


def reset_topic() -> None:
    """Forget the sticky subject. Called with the conversation history — a new session must not
    inherit the last one's topic."""
    global _sticky_turns
    _sticky_turns = 0


def rank_chunks(query_vec: list[float], threshold: float = SIMILARITY_THRESHOLD, top_k: int = TOP_K) -> list[dict]:
    """Brute-force cosine similarity over the cached index — plenty fast at personal-document
    scale (a few thousand chunks at most), avoids adding faiss/chromadb as a new dependency."""
    if _INDEX_VECTORS is None or not _INDEX_CHUNKS:
        return []
    q = np.array(query_vec, dtype=np.float64)
    # SOURCE_BOOST is applied before the threshold, not after ranking: a boosted source has to be
    # able to enter the results at all, not just reorder the ones that already got in.
    scored = [(cosine_similarity(q, vec) + source_boost(chunk), chunk, vec)
              for vec, chunk in zip(_INDEX_VECTORS, _INDEX_CHUNKS)]
    scored = [row for row in scored if row[0] >= threshold]
    scored.sort(key=lambda row: row[0], reverse=True)

    # The primer competes, but capped — see PRIMER_MAX_IN_RANKING. Applied after the sort, so it
    # keeps the primer's BEST line and gives the remaining slots away, rather than dropping the
    # primer to wherever the top_k cut happens to land.
    out: list[dict] = []
    picked: list[np.ndarray] = []
    primers = 0
    for _score, chunk, vec in scored:
        if is_primer(chunk) and primers >= PRIMER_MAX_IN_RANKING:
            continue
        # Diversity: a chunk that restates one already picked spends a slot without adding a
        # fact. See MMR_MAX_SIMILARITY — documents/ repeats itself heavily and TOP_K is 3.
        if any(cosine_similarity(vec, chosen) > MMR_MAX_SIMILARITY for chosen in picked):
            continue
        if is_primer(chunk):
            primers += 1
        out.append(chunk)
        picked.append(vec)
        if len(out) >= top_k:
            break
    return out


def format_context(chunks: list[dict]) -> str:
    """Turn ranked chunks into a labeled block, shaped for gemma2:2b specifically.

    The instructions do three jobs. They make the retrieved text authoritative — the model
    otherwise answers DEVCON questions from its own vague pretraining even with the documents
    right there, which is the whole failure this block exists to prevent. They re-assert Kai's
    voice, because a bare "answer from the documents" makes the model recite chunk prose and drop
    the persona. And they keep the "ignore if unrelated" escape, the defensive half of the
    threshold gate, for chunks that squeaked past SIMILARITY_THRESHOLD.

    What they deliberately do NOT do is mention LENGTH, in either direction. This block lands in the
    strongest slot in the prompt (see the placement note below) and it fires only on RAG turns, so
    whatever it says about length overrides persona.txt on exactly the turns that carry the most
    facts. Both directions were tried on-device and both were wrong:

      "short, warm, spoken"   the original. Applied the STRONGEST pressure to compress on the turns
                              holding the MOST retrieved facts — TOP_K chunks in, two sentences out,
                              with the model picking arbitrarily which facts survived. That was the
                              "Kai's answers are vague" complaint.
      "long enough to cover
       what was asked"        the overcorrection, and worse. persona.txt was simultaneously asking
                              for two or three sentences and getting them — a plain greeting came
                              back at 9 words — while every RAG turn ran 104-130 words and hit
                              OLLAMA_NUM_PREDICT mid-sentence. The split by turn type was exact, and
                              it is the tell: if fixing the persona moves chat replies but not
                              DEVCON replies, the instruction fighting it is this line.

    So it says nothing about length now. Length is persona.txt's job and only persona.txt's, where
    one rule covers every turn and there is nothing positioned to outrank it.

    Three things about the LAYOUT are gemma2-specific rather than cosmetic:

    * The instructions sit AFTER the facts, not before. Gemma2 has no system role at all — its
      chat template has only `user` and `model` turns, so Ollama folds our system message into
      the first user turn and the persona stops being a privileged channel. What is left to lean
      on is position, and the last thing before the question is the strongest slot in the prompt.
      With RAG_CONTEXT_PLACEMENT="user" this block is immediately followed by the question, so
      trailing instructions land directly against it.
    * Short imperative lines, and no markdown bullets or headings. A 2B model mirrors the shape
      of its prompt far more readily than a 4B one, and Kai's output is spoken — a leaked "* "
      or "## " is a TTS artifact. ai/tts.py strips them defensively; not emitting them is better.
    * The chunks are Q&A pairs now, which is the single most useful shape this model could be
      handed — but it introduces a failure it did not have before, where a small model answers
      the DOCUMENT's question instead of the person's. Hence the explicit last line.

    Kept terse on purpose: OLLAMA_NUM_CTX is 2048, TOP_K chunks already dominate it, and every
    instruction token here competes with the persona for a 2B model's limited attention."""
    if not chunks:
        return ""
    lines = ["FACTS from Kai's own documents:"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[{i}] ({chunk_label(chunk)}) {chunk['text']}")
    lines.append(
        "Answer from them when they cover the question — names, numbers and dates exactly as "
        "written, never guessed or padded. Say you're not sure when the answer isn't here. If "
        "they only touch the question, answer it and use them to steer back to DEVCON; if they "
        "have nothing to do with it, leave them alone. Never read this block out loud and never "
        "mention documents or facts. Answer in Kai's own voice: warm and spoken. Answer the "
        "person's question below, not any question written above.")
    return "\n".join(lines)


def points_backwards(text: str) -> bool:
    """True if this turn leans on the previous one to be understood ("who runs it?"). Cheap
    pronoun test — see ANAPHORA_WORDS for why the pronoun is the signal rather than the length."""
    return any(tok.text in ANAPHORA_WORDS for tok in normalize_tokens(text))


def _build_query(query_text: str, previous_user_text: str | None) -> tuple[str, bool]:
    """The embedded query, and whether this turn is provably about DEVCON.

    Four rewrites, all retrieval-only. canonicalize_devcon() folds whatever Whisper made of the
    brand onto the documents' spelling. expand_tagalog_question_words() appends an English anchor
    for EMBED_MODEL, which cannot read Tagalog at all (A10). match_entities() does the same for
    program and chapter names, and appends the canonical form — a mangled "campus dev con" is not
    repaired by the brand matcher alone. Anaphora expansion prepends the previous turn when this
    one leans on it.

    The flag is the gate for the whole failsafe chain below, so it is set only by evidence in the
    text: the brand, or a name that exists nowhere but these documents."""
    query = canonicalize_devcon(query_text)
    if query != query_text:
        print(f"[rag] query canonicalized for retrieval: {query_text!r} -> {query!r}")
    brand = mentions_devcon(query)

    expanded = expand_tagalog_question_words(query)
    if expanded != query:
        print(f"[rag] Tagalog question word expanded for retrieval: {query!r} -> {expanded!r}")
        query = expanded

    entities = match_entities(query, _INDEX_ENTITIES)
    if entities:
        missing = [e for e in entities if e.casefold() not in query.casefold()]
        if missing:
            query = f"{query} {' '.join(missing)}"
            print(f"[rag] gazetteer hit, query expanded: {missing} -> {query!r}")
        brand = True

    if previous_user_text and points_backwards(query_text):
        query = f"{canonicalize_devcon(previous_user_text)} {query}"
        print(f"[rag] follow-up expanded with previous turn: {query!r}")
        brand = brand or mentions_devcon(query)
    return query, brand


def retrieve_context(query_text: str, previous_user_text: str | None = None) -> str:
    """The single entry point voice_assistant.py calls. Fail-open end to end: any failure
    anywhere returns "" — a broken/missing index behaves exactly as if RAG didn't exist. The
    caller's `query_text` is what reaches the LLM and the UI, untouched; everything here is
    retrieval-side only.

    Layered, cheapest first, and it stops at the first layer that finds something:

      1. dense retrieval over the rewritten query (see _build_query), with one slot open to a
         strong lexical hit dense missed (lexical_rescue)
      2. lexical overlap alone — when dense returned nothing at all
      3. sticky topic — a pronoun-free follow-up ("how many chapters?") retried with the brand
      4. a lowered threshold, brand-flagged turns only
      5. the pinned primer
      6. NO_CONTEXT_NOTICE

    Layers 4-6 fire ONLY when the turn is provably about DEVCON. That gate is the whole design:
    an unrelated question still falls through to "" exactly as before, while the question this
    robot exists to answer cannot come back empty — and "" is the dangerous state, the one where
    gemma2:2b answers about DEVCON from pretraining."""
    global _sticky_turns
    try:
        query, brand = _build_query(query_text, previous_user_text)
        sticky = _sticky_turns > 0
        _sticky_turns = STICKY_TURNS if brand else max(0, _sticky_turns - 1)

        query_vec = embed_query(query)
        chunks = rank_chunks(query_vec) if query_vec is not None else []
        if chunks:
            return format_context(lexical_rescue(query, chunks))

        chunks = lexical_rank(query)
        if chunks:
            print(f"[rag] dense retrieval empty; lexical fallback matched "
                  f"{[c.get('source') for c in chunks]}")
            return format_context(chunks)

        # Sticky retry re-embeds with the brand attached rather than lowering the bar — the turn
        # is not flagged, so it has not earned a weaker threshold, only a better query.
        if not brand and sticky and _INDEX_CHUNKS:
            retry_vec = embed_query(f"{DEVCON_CANONICAL} {query}")
            chunks = rank_chunks(retry_vec) if retry_vec is not None else []
            if chunks:
                # Deliberately does NOT renew the counter. A retry succeeding is not evidence the
                # speaker is still on the subject — it is evidence that forcing the subject in
                # retrieves something, which it always will. Renewing on it made the topic
                # permanent: every subsequent turn kept the flag alive by its own retry.
                print(f"[rag] sticky topic: retried as {DEVCON_CANONICAL} {query!r}")
                return format_context(chunks)

        if not brand:
            return ""

        if query_vec is not None:
            chunks = rank_chunks(query_vec, threshold=FALLBACK_THRESHOLD, top_k=FALLBACK_TOP_K)
            if chunks:
                print(f"[rag] failsafe: nothing over {SIMILARITY_THRESHOLD}, took "
                      f"{chunks[0].get('source')} at >={FALLBACK_THRESHOLD}")
                return format_context(chunks)

        chunks = primer_chunks()
        if chunks:
            print("[rag] failsafe: falling back to the pinned primer")
            return format_context(chunks)

        print("[rag] failsafe: no primer indexed — returning the don't-guess notice")
        return NO_CONTEXT_NOTICE
    except Exception as exc:
        _note_error("retrieve_context", exc)
        return ""
