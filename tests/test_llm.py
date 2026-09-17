"""The Ollama client: prompt assembly, the persona file, and the GPU-placement probe.

Moved out of tests/test_voice_assistant.py with ai/llm.py. The tests that drive a request through
VoiceAssistant (TestCallOllama, TestEnsureLlmWarm) stayed there — they are about the assistant's
use of this module, not about the module.
"""

import unittest
from unittest.mock import MagicMock, patch

import requests

import ai.llm as llm_mod
from ai.llm import (
    _DEFAULT_PERSONA, _log_llm_timings, build_chat_messages, load_persona, log_model_placement,
)
from config.voice import MAX_HISTORY_TURNS, OLLAMA_MODEL, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT


class TestBuildChatMessages(unittest.TestCase):
    def test_system_prompt_first(self):
        msgs = build_chat_messages("sys", [], "hello")
        self.assertEqual(msgs[0], {"role": "system", "content": "sys"})

    def test_appends_user_turn_last(self):
        msgs = build_chat_messages("sys", [], "hello")
        self.assertEqual(msgs[-1], {"role": "user", "content": "hello"})

    def test_includes_history_in_order(self):
        history = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        msgs = build_chat_messages("sys", history, "c")
        self.assertEqual(msgs, [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ])

    def test_truncates_to_max_history_turns(self):
        history = []
        for i in range(MAX_HISTORY_TURNS + 5):
            history.append({"role": "user", "content": f"u{i}"})
            history.append({"role": "assistant", "content": f"a{i}"})
        msgs = build_chat_messages("sys", history, "new")
        # system + capped history + new user turn
        self.assertEqual(len(msgs), 1 + MAX_HISTORY_TURNS * 2 + 1)

    def test_no_budget_means_no_trimming_beyond_max_history_turns(self):
        # Default behaviour, and every existing caller that doesn't pass budget_tokens, must be
        # unchanged by A7's addition.
        history = [{"role": "user", "content": "a" * 400},
                   {"role": "assistant", "content": "b" * 400}]
        msgs = build_chat_messages("sys", history, "c")
        self.assertEqual(len(msgs), 4)


class TestBuildChatMessagesBudget(unittest.TestCase):
    """A7: deterministic history trimming, since a standalone robot has nobody tailing the log the
    OLLAMA_CTX_WARN_FRACTION warning prints to — the warning alone was not enough."""

    def _pair(self, n_chars):
        return [{"role": "user", "content": "u" * n_chars},
                {"role": "assistant", "content": "a" * n_chars}]

    def test_keeps_everything_under_budget(self):
        history = self._pair(40)
        msgs = build_chat_messages("sys", history, "question", budget_tokens=1000)
        self.assertEqual(len(msgs), 4)

    def test_drops_the_oldest_pair_first(self):
        # Two pairs, a tight budget that only leaves room for the newest one plus system + question.
        history = self._pair(400) + self._pair(40)   # oldest first, as build_chat_messages expects
        msgs = build_chat_messages("sys", history, "q", budget_tokens=30)
        contents = [m["content"] for m in msgs]
        self.assertNotIn("u" * 400, contents, "the OLDEST pair must go first")
        self.assertIn("u" * 40, contents, "the newest pair must be kept as long as anything is")

    def test_never_drops_the_system_prompt_or_the_new_turn(self):
        history = self._pair(1000) * 3
        msgs = build_chat_messages("sys", history, "the question", budget_tokens=1)
        self.assertEqual(msgs[0], {"role": "system", "content": "sys"})
        self.assertEqual(msgs[-1], {"role": "user", "content": "the question"})

    def test_logs_when_it_actually_trims(self):
        history = self._pair(400) + self._pair(400)
        with patch("builtins.print") as mock_print:
            build_chat_messages("sys", history, "q", budget_tokens=30)
        self.assertTrue(any("trimmed" in c.args[0] for c in mock_print.call_args_list))

    def test_silent_when_nothing_needs_trimming(self):
        history = self._pair(10)
        with patch("builtins.print") as mock_print:
            build_chat_messages("sys", history, "q", budget_tokens=1000)
        mock_print.assert_not_called()


class TestLoadPersona(unittest.TestCase):
    def test_reads_custom_content(self):
        mock_path = MagicMock()
        mock_path.read_text.return_value = "Custom persona text.\n"
        with patch("ai.llm.PERSONA_PATH", mock_path):
            self.assertEqual(load_persona(), "Custom persona text.")

    def test_missing_file_falls_back_to_default(self):
        mock_path = MagicMock()
        mock_path.read_text.side_effect = OSError("no such file")
        with patch("ai.llm.PERSONA_PATH", mock_path):
            self.assertEqual(load_persona(), _DEFAULT_PERSONA)

    def test_empty_file_falls_back_to_default(self):
        mock_path = MagicMock()
        mock_path.read_text.return_value = "   \n"
        with patch("ai.llm.PERSONA_PATH", mock_path):
            self.assertEqual(load_persona(), _DEFAULT_PERSONA)


class TestLogModelPlacement(unittest.TestCase):
    def _resp(self, payload):
        r = MagicMock()
        r.json.return_value = payload
        return r

    def test_reports_a_full_gpu_offload(self):
        payload = {"models": [{"name": f"{OLLAMA_MODEL}", "size": 2000, "size_vram": 2000}]}
        with patch("ai.llm.requests.get", return_value=self._resp(payload)):
            out = log_model_placement()
        self.assertEqual(out["gpu_pct"], 100.0)

    def test_reports_a_partial_offload(self):
        """The ~2x slowdown this whole probe exists to make visible."""
        payload = {"models": [{"name": f"{OLLAMA_MODEL}", "size": 2000, "size_vram": 900}]}
        with patch("ai.llm.requests.get", return_value=self._resp(payload)):
            out = log_model_placement()
        self.assertEqual(out["gpu_pct"], 45.0)

    def test_never_raises_when_ollama_is_unreachable(self):
        with patch("ai.llm.requests.get",
                   side_effect=requests.exceptions.ConnectionError()):
            self.assertEqual(log_model_placement(), {})

    def test_handles_model_not_loaded(self):
        with patch("ai.llm.requests.get", return_value=self._resp({"models": []})):
            self.assertEqual(log_model_placement(), {})


class TestLogLlmTimingsContextBudget(unittest.TestCase):
    """A3: prompt_eval_count is measured on every turn and was dropped before reaching anything
    that could act on it. Ns are in nanoseconds; a nonzero prompt/gen duration is what keeps
    _log_llm_timings from taking its "nothing measured" early return."""

    def setUp(self):
        llm_mod._ctx_was_over = False   # edge-triggered module state — every test starts "under"

    def _data(self, prompt_n, gen_n=10):
        return {"prompt_eval_count": prompt_n, "prompt_eval_duration": 1,
                "eval_count": gen_n, "eval_duration": 1}

    def test_no_warning_when_comfortably_under_budget(self):
        with patch("builtins.print") as mock_print:
            _log_llm_timings(self._data(10))
        self.assertFalse(any("OLLAMA_NUM_CTX" in c.args[0] for c in mock_print.call_args_list))

    def test_warns_once_on_crossing_the_threshold(self):
        over = int(0.9 * OLLAMA_NUM_CTX)   # comfortably past OLLAMA_CTX_WARN_FRACTION (0.85)
        with patch("builtins.print") as mock_print:
            _log_llm_timings(self._data(over))
        lines = [c.args[0] for c in mock_print.call_args_list]
        self.assertTrue(any("OLLAMA_NUM_CTX" in l and "WARNING" in l for l in lines))

    def test_does_not_re_warn_on_a_second_turn_that_stays_over(self):
        # The NO_FACE precedent: a long conversation that stays over budget must not re-warn on
        # every single turn, or the log becomes one nobody reads.
        over = int(0.9 * OLLAMA_NUM_CTX)
        _log_llm_timings(self._data(over))
        with patch("builtins.print") as mock_print:
            _log_llm_timings(self._data(over))
        lines = [c.args[0] for c in mock_print.call_args_list]
        self.assertFalse(any("OLLAMA_NUM_CTX" in l and "WARNING" in l for l in lines))

    def test_warns_again_after_dropping_back_under_and_crossing_again(self):
        over = int(0.9 * OLLAMA_NUM_CTX)
        under = 10
        _log_llm_timings(self._data(over))          # crosses -> warns
        _log_llm_timings(self._data(under))          # drops back under -> clears the edge
        with patch("builtins.print") as mock_print:
            _log_llm_timings(self._data(over))       # crosses again -> should warn again
        lines = [c.args[0] for c in mock_print.call_args_list]
        self.assertTrue(any("OLLAMA_NUM_CTX" in l and "WARNING" in l for l in lines))

    def test_warns_when_the_reply_hit_num_predict_exactly(self):
        with patch("builtins.print") as mock_print:
            _log_llm_timings(self._data(10, gen_n=OLLAMA_NUM_PREDICT))
        lines = [c.args[0] for c in mock_print.call_args_list]
        self.assertTrue(any("OLLAMA_NUM_PREDICT" in l and "cut mid-word" in l for l in lines))

    def test_no_num_predict_warning_for_a_short_reply(self):
        with patch("builtins.print") as mock_print:
            _log_llm_timings(self._data(10, gen_n=5))
        lines = [c.args[0] for c in mock_print.call_args_list]
        self.assertFalse(any("cut mid-word" in l for l in lines))

    def test_a_response_with_no_timing_fields_warns_nothing(self):
        # The mocked case _log_llm_timings already guards: no prompt_eval_duration and no
        # eval_duration means the early return fires before either new warning could run.
        with patch("builtins.print") as mock_print:
            _log_llm_timings({})
        mock_print.assert_not_called()


if __name__ == "__main__":
    unittest.main()
