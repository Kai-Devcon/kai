import unittest

from ai.query_alias import (
    _skeleton, canonicalize_devcon, expand_tagalog_question_words, looks_like_devcon,
    match_entities, mentions_devcon,
)
from config.rag import DEVCON_CANONICAL


class TestLooksLikeDevcon(unittest.TestCase):
    def test_exact_spelling(self):
        self.assertTrue(looks_like_devcon("devcon"))

    def test_whisper_renderings(self):
        # Every one of these is a real faster-whisper output for the spoken word "DEVCON".
        for token in ("devcom", "debcon", "defcon", "devkon", "devcan", "devgon", "devcorn",
                      "devon", "decon", "devkhan", "devcons"):
            with self.subTest(token=token):
                self.assertTrue(looks_like_devcon(token))

    def test_real_words_rejected(self):
        for token in ("recon", "beacon", "second", "device", "devices", "devote", "devotion",
                      "beckon", "reckon", "develop", "lexicon", "silicon", "welcome"):
            with self.subTest(token=token):
                self.assertFalse(looks_like_devcon(token))

    def test_blocklisted_word_rejected_despite_high_ratio(self):
        # "deacon" scores 0.833, above DEVCON_MATCH_RATIO — only the blocklist stops it.
        self.assertFalse(looks_like_devcon("deacon"))

    def test_fragments_rejected_by_length_window(self):
        # Structural, not ratio: these must stay dead however the threshold is later tuned.
        for token in ("dev", "con", "de", "kon", ""):
            with self.subTest(token=token):
                self.assertFalse(looks_like_devcon(token))


class TestSkeletonMatcher(unittest.TestCase):
    """The second matcher, OR'd with the difflib ratio. Its job is the vowel shift, which costs
    difflib real score and costs the spoken word nothing."""

    def test_vowel_shifts_reduce_to_the_same_key(self):
        for token in ("davcan", "duvcun", "devkin", "devcun", "tefgon", "depcon"):
            with self.subTest(token=token):
                self.assertEqual(_skeleton(token), _skeleton("devcon"))
                self.assertTrue(looks_like_devcon(token))

    def test_adds_no_false_positives_to_the_known_rejects(self):
        # Same list the ratio must reject, re-asserted: a second matcher is only free if it does
        # not quietly widen what the first one refuses.
        for token in ("deafen", "deepen", "divine", "tavern", "defiant", "vacation", "beacon",
                      "recon", "second", "develop", "devices", "deacon"):
            with self.subTest(token=token):
                self.assertFalse(looks_like_devcon(token))

    def test_length_window_still_applies(self):
        # "dvcn" reduces to the right key but is a fragment — structure wins over the matchers.
        self.assertFalse(looks_like_devcon("dvcn"))


class TestMentionsDevcon(unittest.TestCase):
    def test_true_for_every_rendering_form(self):
        for text in ("what is DEVCON?", "tell me about defcon", "ano ang dev con",
                     "davcan po ba", "dev khan"):
            with self.subTest(text=text):
                self.assertTrue(mentions_devcon(text))

    def test_false_for_unrelated_turns(self):
        for text in ("what time is it?", "", "the deacon has a device", "tell me a joke"):
            with self.subTest(text=text):
                self.assertFalse(mentions_devcon(text))


class TestMatchEntities(unittest.TestCase):
    ENTITIES = ["DEVCON Kids", "Geeks on a Beach", "Campus DEVCON", "Movers and Shakers Award"]

    def test_exact_mention(self):
        self.assertEqual(match_entities("tell me about DEVCON Kids", self.ENTITIES),
                         ["DEVCON Kids"])

    def test_mishearing_resolves_to_the_canonical_spelling(self):
        # The point of the layer: the canonical string is what gets embedded, not what was said.
        self.assertEqual(match_entities("what is geeks on the beach", self.ENTITIES),
                         ["Geeks on a Beach"])

    def test_unrelated_text_matches_nothing(self):
        self.assertEqual(match_entities("what's the weather at the beach today?", self.ENTITIES),
                         [])

    def test_empty_gazetteer_is_inert(self):
        # An index built before this layer existed has no entities — it must degrade, not break.
        self.assertEqual(match_entities("tell me about DEVCON Kids", []), [])

    def test_multiple_hits_are_deduplicated(self):
        hits = match_entities("is Campus DEVCON the same as DEVCON Kids or campus devcon?",
                              self.ENTITIES)
        self.assertEqual(sorted(hits), ["Campus DEVCON", "DEVCON Kids"])


class TestCanonicalizeDevcon(unittest.TestCase):
    def test_empty_and_no_tokens(self):
        self.assertEqual(canonicalize_devcon(""), "")
        self.assertEqual(canonicalize_devcon("...!?"), "...!?")

    def test_unrelated_text_returned_unchanged(self):
        text = "what time is it?"
        self.assertIs(canonicalize_devcon(text), text)

    def test_rewrites_single_token(self):
        self.assertEqual(canonicalize_devcon("what is defcon?"), f"what is {DEVCON_CANONICAL}?")

    def test_rewrites_split_rendering(self):
        for spoken in ("dev con", "de con", "def con", "dev khan", "dev com"):
            with self.subTest(spoken=spoken):
                self.assertEqual(canonicalize_devcon(f"tell me about {spoken} please"),
                                 f"tell me about {DEVCON_CANONICAL} please")

    def test_does_not_swallow_the_following_word(self):
        # A token that matches on its own must never be joined with its neighbour — otherwise
        # "devcon po" merges to "devconpo" (0.857) and the Tagalog particle disappears.
        self.assertEqual(canonicalize_devcon("devcon po ba yan"), f"{DEVCON_CANONICAL} po ba yan")
        self.assertEqual(canonicalize_devcon("devcon ph events"),
                         f"{DEVCON_CANONICAL} ph events")

    def test_does_not_swallow_the_preceding_word(self):
        # "isdevcon" and "uydevcon" both score 0.857 against "devcon": without the fragment-length
        # guard these came out as "what DEVCON Philippines?" and "DEVCON ba yan".
        self.assertEqual(canonicalize_devcon("what is devcon Philippines?"),
                         f"what is {DEVCON_CANONICAL} Philippines?")
        self.assertEqual(canonicalize_devcon("uy devcon ba yan"),
                         f"uy {DEVCON_CANONICAL} ba yan")

    def test_rewrites_every_occurrence(self):
        self.assertEqual(canonicalize_devcon("is defcon the same as dev com?"),
                         f"is {DEVCON_CANONICAL} the same as {DEVCON_CANONICAL}?")

    def test_preserves_surrounding_text_exactly(self):
        # The rest of the string is sliced out of the original, so casing, digits, punctuation and
        # accents survive — rebuilding from normalized tokens would flatten all four.
        self.assertEqual(canonicalize_devcon("Ano ang DevCom 2026 hackathon, ha?"),
                         f"Ano ang {DEVCON_CANONICAL} 2026 hackathon, ha?")

    def test_case_insensitive(self):
        self.assertEqual(canonicalize_devcon("DEFCON rocks"), f"{DEVCON_CANONICAL} rocks")

    def test_canonical_input_is_idempotent(self):
        text = f"what does {DEVCON_CANONICAL} do?"
        self.assertEqual(canonicalize_devcon(text), text)

    def test_real_words_left_alone(self):
        text = "the deacon has a device and a second beacon"
        self.assertEqual(canonicalize_devcon(text), text)


class TestExpandTagalogQuestionWords(unittest.TestCase):
    """A10: EMBED_MODEL is English-only, so a Tagalog query drifts away from the English chunk
    that answers it — MEASURED 2026-09-17, "Sino ang nagtatag ng DEVCON?" retrieved the wrong
    chunk entirely while the English phrasing retrieved correctly. This appends an English anchor
    word so the embedder has something to work with, without touching what the LLM sees."""

    def test_empty_returns_empty(self):
        self.assertEqual(expand_tagalog_question_words(""), "")

    def test_no_question_word_returned_unchanged(self):
        text = "DEVCON po ba yan"
        self.assertIs(expand_tagalog_question_words(text), text)

    def test_appends_english_anchor(self):
        self.assertEqual(expand_tagalog_question_words("Sino ang nagtatag ng DEVCON?"),
                         "Sino ang nagtatag ng DEVCON? who founded")

    def test_multi_word_translation(self):
        self.assertEqual(expand_tagalog_question_words("Ilang taon na ang DEVCON?"),
                         "Ilang taon na ang DEVCON? how many")

    def test_case_insensitive(self):
        self.assertEqual(expand_tagalog_question_words("SINO ang founder?"),
                         "SINO ang founder? who")

    def test_each_english_word_appended_once(self):
        # "ilan" and "ilang" both mean "how many" — a query using both must not repeat the anchor.
        self.assertEqual(expand_tagalog_question_words("ilan o ilang chapters?"),
                         "ilan o ilang chapters? how many")

    def test_multiple_distinct_question_words_all_appended(self):
        self.assertEqual(expand_tagalog_question_words("sino at kailan nagtatag?"),
                         "sino at kailan nagtatag? who when founded")

    def test_original_text_preserved_exactly(self):
        # Additive only — casing, punctuation and the Tagalog itself must survive untouched, since
        # this is retrieval-only and never reaches the LLM or the transcript on /params.
        text = "Sino, talaga, ang nagtatag ng DEVCON?!"
        self.assertTrue(expand_tagalog_question_words(text).startswith(text))

    def test_english_words_do_not_false_trigger(self):
        # "who", "when" etc. are English already; nothing here should fire on an English question.
        text = "who founded DEVCON?"
        self.assertIs(expand_tagalog_question_words(text), text)


if __name__ == '__main__':
    unittest.main()
