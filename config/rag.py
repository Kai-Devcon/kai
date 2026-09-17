"""Retrieval-augmented generation knobs. Consumed by ai/rag.py and ai/index_documents.py.
(DOCUMENTS_DIR / INDEX_PATH are __file__-derived paths and stay in ai/rag.py.)"""

# ~90MB, 384-dim, CPU-only via onnxruntime (already installed) — no torch dependency, so
# installing fastembed can't disturb the Jetson's custom-built CUDA-enabled torch.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# bge-small-en-v1.5 was trained with task prefixes; using the wrong one (or none) measurably
# degrades retrieval quality. Indexed chunks and live queries must use the matching prefix —
# swap these together with EMBED_MODEL if you change models.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX    = "search_query: "

# Small on purpose: OLLAMA_NUM_CTX is trimmed to 1024 in config/voice.py to keep gemma3:4b
# fully GPU-resident, so even TOP_K retrieved chunks must leave room for persona+history+query.
CHUNK_SIZE_CHARS    = 800
CHUNK_OVERLAP_CHARS = 150

# ── Fuzzy "DEVCON" spotting (ai/query_alias.py) ────────────────────────────────
# Every chunk in documents/ spells the brand "DEVCON"; Whisper frequently does not ("defcon",
# "dev com", "Devon", "de con"). bge-small embeds those as different words, so the query drifts
# off the very chunks that answer it. The query is rewritten to DEVCON_CANONICAL before embedding.
DEVCON_CANONICAL = "DEVCON"

# Matched against, not matched literally — like WAKE_PHRASE_NAMES, a second spelling widens the
# neighborhood cheaply. "devkhan" exists only to catch "dev khan" / "dev kan", which sit at 0.61
# against "devcon" and would otherwise need a threshold low enough to sweep in real words.
DEVCON_SPELLINGS = ("devcon", "devkhan")

# Measured with difflib over ~60 renderings and ~50 distractors: every plausible mishearing
# ("devcom", "debcon", "defcon", "devkon", "devcan", "devgon", "decon", "devon") lands at 0.833+,
# while the nearest real words ("recon", "devotion", "beckon", "second", "beacon", "device") top
# out at 0.727. 0.80 sits in that gap. Below ~0.72 there is no gap left to sit in.
DEVCON_MATCH_RATIO = 0.80

# The one real word inside the gap: "deacon" scores 0.833. Rare enough in a DEVCON robot's turns
# that it is cheaper to blocklist than to lower the threshold around.
DEVCON_BLOCKLIST = frozenset({"deacon"})

# Second, independent matcher OR'd with the ratio (ai/query_alias.py _skeleton). difflib compares
# characters in order, so it misses renderings that sound right but share few letters — a vowel
# shift moves "devcon" -> "davcan" -> "duvcun" and the ratio falls off while the word still sounds
# identical. Folding away vowels and collapsing consonants onto sound-alike classes reduces all of
# them to one key. Two mechanisms behind one gate: a rendering only has to be caught by either.
# Still checked after DEVCON_BLOCKLIST and the length window, which are what keep real words out.
DEVCON_SKELETON_CLASSES = {
    "b": "F", "f": "F", "p": "F", "v": "F",
    "c": "K", "g": "K", "j": "K", "k": "K", "q": "K", "x": "K",
    "d": "T", "t": "T",
    "l": "R", "r": "R",
    "m": "N", "n": "N",
    "s": "S", "z": "S",
}
# Dropped before classing: vowels carry the variation, h/w are near-silent ("dev khan" -> devkan).
DEVCON_SKELETON_DROP = frozenset("aeiouyhw")

# ── Entity gazetteer (built at index time, ai/index_documents.py) ──────────────
# The brand is not the only name Whisper mangles — "Geeks on a Beach" and "Campus DEVCON" are
# just as mishearable, and worse, a mishearing of a program name carries no DEVCON token at all,
# so none of the machinery above fires. The gazetteer is the document set's own vocabulary of
# distinctive names, harvested from its headings, and it does two jobs: it flags the turn as
# on-topic, and it puts the canonical spelling into the embedded query.
#
# Hand-listing these would drift the moment documents/ changes, so they are derived. The filter
# is document frequency: a heading is kept only if one of its words is RARE across the corpus.
# That is what separates "Geeks on a Beach" (beach: 1 chunk) from the style guide's "Theme" and
# "When in doubt", which are headings too and would otherwise flag every turn containing "theme".
GAZETTEER_MAX_DF_RATIO = 0.02   # rarest word must appear in <=2% of chunks
GAZETTEER_MIN_WORD_LEN = 4      # and be a real word, not "a"/"of"/"ph"
GAZETTEER_MAX_TOKENS   = 5      # headings longer than this are sentences, not names

# A ratio alone gets LOOSER as documents/ grows, which is backwards: the filter that decides what
# counts as a rare name should not relax just because there is more corpus. Measured — adding
# DEVCON_2025_Year-End_Chapter_Stats_Submissions.md took the corpus 332 -> 505 chunks, which moved
# the ratio ceiling 6 -> 10 and admitted five junk entries that the same filter had correctly
# rejected the day before: "Theme" (df 9), "Cagayan De Oro" (10), "Initial Insights from the
# Numbers" (6), and two form-field labels. "Theme" is the exact example the comment above warns
# about, and with it in the gazetteer "what's the theme of your talk?" was flagged as a DEVCON
# question and had "Theme" appended to its embedded query.
#
# So the ceiling is min(ratio, this). 5 is where the ratio sat when the filter was last known
# good, and it is a count rather than a proportion because "a name appears in a handful of
# chunks" is a statement about names, not about corpus size.
GAZETTEER_MAX_DF_ABS = 5

# The same argument in the other direction, and the compaction is what exposed it. A pure ratio
# also gets STRICTER as the corpus shrinks: consolidating documents/ into one 50-entry Q&A file
# took int(50 * 0.02) = 1, so a name had to appear in exactly ONE chunk of the entire corpus to
# be admitted, and every program mentioned in both its own entry and a "how do I join" entry was
# rejected. That is the same bug as GAZETTEER_MAX_DF_ABS with the sign flipped, so it gets the
# same kind of fix: the ceiling is clamped from BELOW as well as above. 3 keeps the small-corpus
# ceiling in the same "appears in a handful of chunks" band the absolute cap describes, and it
# changes nothing on a large corpus — at 505 chunks the ceiling was already 5.
GAZETTEER_MIN_DF_ABS = 3

# A one-word heading is a topic label, not a name — "Theme", "Voice", "Colour". Names that matter
# here are multi-word ("Movers and Shakers Award") or carry the brand ("DEVCON Kids"), and the
# brand always bypasses. Counted over content words (GAZETTEER_MIN_WORD_LEN and up), so
# "Cagayan De Oro" counts as one — "de" and "oro" are too short to identify anything.
GAZETTEER_MIN_CONTENT_WORDS = 2
# Higher than DEVCON_MATCH_RATIO: these are long multi-word strings, where difflib is far more
# discriminative and a loose threshold starts merging one program name into another.
GAZETTEER_MATCH_RATIO  = 0.86

# ── Lexical fallback (ai/rag.py lexical_rank) ─────────────────────────────────
# Dense retrieval is a single point of failure, and it fails in a specific way: rare tokens — a
# chapter city, a person's surname, a year — are exactly what bge-small smooths away, and exactly
# what identifies the one chunk that answers the question. An IDF-weighted overlap scan fails in
# the opposite direction (it is blind to paraphrase), so running it only when dense comes back
# short covers each one's blind spot with the other. Brute-force over the same cached chunks, no
# new dependency and no new index.
LEXICAL_MIN_SCORE = 0.34   # fraction of the query's IDF mass the chunk must contain

# Which of the query's words are allowed to carry weight at all. IDF weighting alone was measured
# insufficient: a word in 12 of 14 chunks still has a positive IDF, so a query made only of
# moderately common words ("is this a page about the brand") scored a perfect 1.0 against the
# first chunk that happened to contain them — and this layer runs precisely when dense retrieval
# already declined, so a spurious match here is context the turn should not have had. A hard
# ceiling keeps the layer doing the one thing it is good at: rare literal tokens.
LEXICAL_MAX_DF_RATIO = 0.05

# ── Failsafe chain (ai/rag.py retrieve_context) ───────────────────────────────
# Everything above widens the ENTRANCE to retrieval. These three guarantee an EXIT: on a turn
# that is provably about DEVCON, returning "" is the one outcome that must not happen, because
# "" is precisely the state in which the model answers from pretraining and confuses DEVCON
# Philippines with DEFCON the hacker conference.
#
# 1. Second-chance threshold. Only for brand-flagged turns — an ordinary question still fails
#    open at SIMILARITY_THRESHOLD. 0.32 is under the ~0.40 distractor band, so it admits weak
#    matches rather than noise, and top_k=1 keeps a weak match from crowding the prompt.
FALLBACK_THRESHOLD = 0.32
FALLBACK_TOP_K     = 1

# 2. Pinned primer. The handful of broad "what even is DEVCON" facts, injected verbatim when
#    retrieval found nothing at all. PRIMER_MAX_CHUNKS is how much of it that injection may use.
#
#    Addressed by SECTION now, not by filename. documents/ used to hold a hand-written
#    devcon_primer.txt alongside six long source documents; it is now one Q&A knowledge base
#    (devcon_faq_rag.md) whose front matter tags every entry with a `section:`. That file no
#    longer exists, so a filename-keyed primer silently resolved to [] and took the whole
#    last-resort layer offline — retrieval fell straight through to NO_CONTEXT_NOTICE.
#    PRIMER_SOURCE is kept as an OR'd alias so a reverted/older index still finds its primer.
PRIMER_SECTION    = "About DEVCON"
PRIMER_SOURCE     = "devcon_primer.txt"
# 2, not 4. The old primer was seven terse one-line facts; an "About DEVCON" entry is a full Q&A
# pair of ~400 chars, so four of them is ~400 tokens — a fifth of OLLAMA_NUM_CTX spent on a
# last-resort guess. Two entries still carry the founding, scale and mission lines.
PRIMER_MAX_CHUNKS = 2

#    It is indexed like any other document, so it also competes normally — and measured after the
#    reindex, it competes far too well. One dense, on-brand fact per line is the ideal shape for
#    this embedder, so primer lines took ALL THREE slots on "what is DEVCON?", "when did DEVCON
#    start?" and "who founded DEVCON?", and two of three on "what is DEVCON CREST?" — crowding out
#    the documents that actually answer the specific question.
#
#    Barring it from normal ranking overcorrects, though: measured, "who founded DEVCON?" then
#    returns a kai_facts line about who assembled the robot, and "when did DEVCON start?" returns
#    anniversary boilerplate with no date in it. The primer really is the best answer to a broad
#    question. So it gets exactly one slot — the accurate summary line stays available, and
#    TOP_K-1 slots are always left for whatever is specific to the question actually asked.
#    Applies to ranking only; the last-resort injection above still uses PRIMER_MAX_CHUNKS.
#
#    Raised 1 -> 2 on measurement. One slot was losing real answers to rounding: on "who founded
#    DEVCON?" the founder line scored 0.770 and the (wrong) 2009 line 0.771, so the single slot
#    went to the wrong sentence by a thousandth of a point and the right one was then discarded
#    for being a primer line too. The primer's seven lines all sit within ~0.02 of each other on
#    any broad DEVCON question, which makes the winner essentially noise. Two slots fixed that
#    question and cost nothing elsewhere; three changed nothing further, so the crowding-out this
#    cap exists to prevent starts above 2, not at it. Measured 79% -> 83% answer-in-context.
PRIMER_MAX_IN_RANKING = 2

# ── Lexical rescue (ai/rag.py retrieve_context) ───────────────────────────────
# LEXICAL_MIN_SCORE above describes a layer that only ran when dense retrieval came back EMPTY —
# which is almost never, because dense always returns something. The result was that the layer
# built for rare literal tokens never fired on the queries it is best at: "what is DEVCON CREST?"
# missed although "crest" appears in exactly 1 chunk of 505, and the mission line was missed
# although it contains the word "mission" verbatim.
#
# So lexical now also runs ALONGSIDE dense, and may claim this many of the TOP_K slots for a
# strong lexical hit that dense did not find, displacing the weakest dense chunk.
#
# 1, not 2, from measurement: a second rescue slot starts evicting dense results that were right,
# because two thirds of the context is then chosen by literal overlap. Set to 0 to disable and
# restore fallback-only behaviour exactly.
LEXICAL_RESCUE_SLOTS = 1

# The rescue needs its OWN, much stricter thresholds — it cannot reuse the two above. Measured:
# reusing them (df <= 5% of corpus, 0.34 of the IDF mass) fired the rescue on 7 of 8 off-topic
# queries and gained nothing at all (83% either way), because at that bar words like "anniversary",
# "theme" and "joke" qualify as rare and drag in style-guide chunks. The fallback's bar is loose
# on purpose — it only ever runs when dense found NOTHING, where any signal beats none. An
# always-on layer is a different job and needs a different number.
#
# Tightened to an absolute document count and most of the query's rare mass: 88%, and the rescue
# stops firing on off-topic turns entirely. The measurement was flat for every df ceiling from 1
# to 8 and every score from 0.5 to 1.0 — this is a plateau, not a knife-edge, so 3 and 0.5 are
# chosen as the middle of a stable range rather than a fitted optimum.
LEXICAL_RESCUE_MAX_DF    = 3     # the token must name something: <=3 chunks in the whole corpus
LEXICAL_RESCUE_MIN_SCORE = 0.5   # and the chunk must carry half the query's rare mass

# 3. The empty case, made safe. When even the primer is missing, say so in the prompt instead of
#    saying nothing — an explicit "you don't have this, don't invent it" beats the silence that
#    invites a confident hallucination.
NO_CONTEXT_NOTICE = (
    "The person is asking about DEVCON Philippines, and Kai's documents have nothing on this "
    "specific point. Say you're not sure and offer to help with something else — do NOT guess, "
    "and do NOT confuse DEVCON Philippines (a Filipino developer community) with any other "
    "conference of a similar name. Answer in Kai's own voice: short, warm, conversational."
)

# Topic stickiness. ANAPHORA_WORDS only expands turns carrying a pronoun, but plenty of
# follow-ups carry none — "how many chapters?", "when did it start?", "who founded it" — and on
# their own they retrieve nothing. After a brand-flagged turn, the brand is prepended to the next
# STICKY_TURNS queries, but ONLY as a retry after normal retrieval has already come back empty.
# That ordering is the safety: a genuine topic change retrieves its own answer and never reaches
# the retry, so the last subject cannot be dragged into it.
STICKY_TURNS = 2

# ── Follow-up query expansion (ai/rag.py retrieve_context) ─────────────────────
# Retrieval sees one utterance at a time, but a conversation does not repeat its subject: after
# "Tell me about DEVCON Kids", the next turn is "Who runs it?" — which on its own retrieved a
# chunk about an AI Code Camp. Prepending the previous user turn to the embedded query fixed 4
# of 5 measured follow-ups ("how many chapters does it have?" 0.662 -> 0.832, "when did it
# start?" 0.603 -> 0.847).
#
# Only for turns that actually point backwards. Expanding every turn would drag the last topic
# into a genuine subject change ("what's the weather?" right after a DEVCON question), and the
# pronoun IS the signal — every follow-up that needed the context had one, and none of the
# topic-switch controls did. Retrieval only: the LLM still receives the turn verbatim.
# Tagalog is listed alongside English because persona.txt tells Kai to answer in whichever
# language the person used, so the follow-up arrives in that language too ("sino sila?").
ANAPHORA_WORDS = frozenset({
    "it", "its", "it's", "they", "them", "their", "theirs", "he", "him", "his", "she", "her",
    "hers", "that", "this", "those", "these", "there",
    "sila", "siya", "nila", "niya", "kanila", "kanya", "ito", "iyan", "yan", "iyon", "yun",
})

# ── Tagalog question words, translated for retrieval only (A10) ────────────────────────────────
# EMBED_MODEL is English-only (bge-small-en-v1.5). A Tagalog query still carries "DEVCON" and any
# English loanwords, but the QUESTION WORD carries most of a short query's meaning ("sino" vs
# "kailan" is the difference between asking for a person and a date) and the embedder cannot read
# it at all. MEASURED 2026-09-17: "Sino ang nagtatag ng DEVCON?" (who founded DEVCON?) retrieved
# the wrong chunk entirely (a "where can I find DEVCON online" entry) while the identical English
# phrasing retrieved the correct one (Winston Damarillo) — same for "Ilang taon na ang DEVCON?"
# against "when did DEVCON start?" (2009). See docs/tickets/A10.
#
# Appended to the query before embedding, never substituted — same "retrieval-only, query
# untouched for the LLM" contract canonicalize_devcon() and match_entities() already keep. A short
# list, not a translator: this is meant to give the embedder a few extra English anchor words, not
# to translate the query.
#
# The question words alone were NOT enough: "Sino ang nagtatag ng DEVCON?" + "who" still retrieved
# the wrong chunk (measured 2026-09-17) — a bare "who" is too generic to out-rank the wrong entry
# ("Where can I find DEVCON online?"). The CONTENT word carries the actual meaning ("nagtatag" =
# founded) and is what the target document's own Q&A phrasing ("DEVCON's founder and president is
# Winston Damarillo") shares. A handful of the highest-traffic content words from documents/'s own
# FAQ phrasing are included below for the same reason — add more here, not a general dictionary,
# if a specific question is still missing its answer.
TAGALOG_QUESTION_WORDS = {
    "sino": "who", "ano": "what", "anong": "what", "kailan": "when", "saan": "where",
    "paano": "how", "bakit": "why", "ilan": "how many", "ilang": "how many",
    "magkano": "how much", "gaano": "how",
    "nagtatag": "founded", "itinatag": "founded", "tagapagtatag": "founder",
    "namumuno": "leads leader", "pinuno": "leader", "pangulo": "president",
}

# Per-source score bonus, added to cosine similarity before ranking (ai/rag.py rank_chunks).
# Not a general relevance dial — it exists because documents/ is ~70% one large DEVCON brand and
# content style guide, and its chunks crowd out Kai's own five-line fact sheet on exactly the
# questions a visitor asks a robot. Measured: "when is your birthday?" put "Kai's birthday is in
# June 2026." at rank 20 (0.532) behind 19 style-guide chunks, and Kai answered "July 9, 2026" —
# the version date out of the guide's filename. +0.08 lifts a first-person fact over that noise
# without letting it win a question it has nothing to say about (an unrelated kai_facts line sits
# at ~0.36, still far under SIMILARITY_THRESHOLD even boosted). Keep it small and keep it rare.
#
# Keyed by section since the compaction — same reason as PRIMER_SECTION. Kai's own facts used to
# be kai_facts.txt; they are now the "Kai (robot)" section of devcon_faq_rag.md, and with the
# whole corpus in one file a filename key can only boost everything or nothing.
SECTION_BOOST = {"Kai (robot)": 0.08}
# Retained and still applied (OR'd with SECTION_BOOST, larger wins) so an older index, or a
# hand-written file dropped back into documents/, keeps working without a config edit.
SOURCE_BOOST = {"kai_facts.txt": 0.08}

# ── Result diversity (ai/rag.py rank_chunks) ──────────────────────────────────
# Ranking by score alone will happily return the same fact three times. Measured on the 505-chunk
# index: 1347 chunk pairs sit above cosine 0.97 (1347 of them inside Nationwide_Chapters_Showcase
# alone — 87% of every possible pair in that file), and 4 of the 16 eval queries came back with
# two of their three chunks above 0.90 similar to each other. At TOP_K=3 inside a 1024-token
# context, a redundant slot is a third of everything Kai gets to see.
#
# A candidate is skipped when it is this close to a chunk already selected. Deliberately high,
# and set from the measurement rather than rounded: the restatements this exists to drop sit
# above 0.97, while two genuinely different paragraphs on one topic land around 0.84. 0.95 is in
# that gap, near the restatement end — the job is to drop repetition, not to force variety onto a
# question whose answer honestly lives in two adjacent paragraphs. bge-small packs everything
# into a narrow cone, so these numbers are much higher than raw cosine intuition suggests; do not
# lower this toward 0.8 without re-running scripts/rag_eval.py.
# Set to 1.0 to disable — that restores pure score ordering exactly.
MMR_MAX_SIMILARITY = 0.95

TOP_K = 3
# Tuning knob — verify empirically, don't trust blindly. Re-run BOTH harnesses after any change:
#   python3 -m scripts.rag_eval       (what the cutoff costs on the on-topic/off-topic split)
#   python3 -m scripts.rag_accuracy   (whether the answer still reaches the model)
#
# History: 0.5 -> 0.45 when per-fact chunks put "when is your birthday" at ~0.49. It then stayed
# at 0.45 through the P2 finding, which measured on-topic (0.572-0.843) and off-topic
# (0.541-0.686) score ranges OVERLAPPING and concluded — correctly, at the time — that no global
# cutoff separates them, so raising this only trades real answers for silence.
#
# 0.55 now, because consolidating documents/ into one Q&A knowledge base changed that
# distribution rather than that reasoning. Six sprawling source documents produced chunks that
# were fragments of arbitrary prose, and an off-topic question could land 0.686 against a
# paragraph that merely shared its register. Fifty self-contained question/answer entries score
# far more cleanly: measured after the reindex, on-topic 0.552-0.770 and off-topic 0.520-0.637 —
# still overlapping, but by 0.085 rather than 0.114, and the whole off-topic band moved down. In
# the sweep 0.55 keeps 8/8 on-topic while rejecting 4/8 off-topic, where 0.45 rejects 0/8, and
# answer-in-context accuracy is unchanged at 30/31.
#
# P2 is therefore REDUCED, not closed: half the off-topic queries still carry a documents block,
# and the survivors ("Kumusta ka na?" at 0.637) sit above every cutoff that leaves the on-topic
# set intact. Closing it still needs a non-score signal — see docs/plan/wip/resolution-plan.md —
# so do not chase the remaining half by raising this number further.
SIMILARITY_THRESHOLD = 0.55
