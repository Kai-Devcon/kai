import threading
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import requests

from ai import voice_assistant
from ai.tts import clean_for_speech
from config.voice import OLLAMA_MODEL
from ai.voice_assistant import (
    VoiceAssistant,
    MicChoice,
    WHISPER_BEAM_SIZE, WHISPER_INITIAL_PROMPT,
    NO_SPEECH_RESPONSE,
    STATUS_DONE,
    STATUS_ERROR,
    STATUS_IDLE,
    STATUS_RECORDING,
)


def make_assistant() -> VoiceAssistant:
    return VoiceAssistant()


def make_segment(text: str):
    seg = MagicMock()
    seg.text = text
    return seg


def block_real_audio_hardware(case):
    """Make it impossible for a test in `case` to reach a real capture device.

    Belt to the braces of patching resolve_capture_device by name. These tests already patch the
    resolution entry point on ai.voice_assistant, and that is what they mean to assert — but that
    patch protects the hardware only INCIDENTALLY, and only while the name it targets is still the
    one being called.

    MEASURED when it was not. Renaming resolve_input_device -> resolve_capture_device made six
    patches silently no-op, so the real ai.mic_device path ran: PortAudio enumerated the dev box's
    actual sound cards and the suite sat through 3 s liveness-probe timeouts on each. The run took
    **1040 seconds instead of 16**, and what it reported depended on which microphones the machine
    happened to have. Failing slowly and nondeterministically is much worse than failing.

    So this patches the layer where the hardware actually is, in the module that actually owns it. A
    future rename then fails immediately and for the right reason, instead of quietly going to the
    sound card.
    """
    for target, kw in (("ai.mic_device.sd.query_devices", {"return_value": []}),
                       ("ai.mic_device.sd.rec", {"side_effect": AssertionError(
                           "a test tried to open a real capture device — a patch target is stale")}),
                       ("ai.mic_device.subprocess.run", {"side_effect": FileNotFoundError(
                           "no amixer/pactl in a test")})):
        p = patch(target, **kw)
        p.start()
        case.addCleanup(p.stop)


class TestEnsureInputResolved(unittest.TestCase):
    """The assistant's own use of ai/mic_device — ordering and the pulse hand-back.

    Patched on ai.voice_assistant, not ai.mic_device: ensure_input_resolved() calls these by bare
    name, so it resolves them through this module's globals (they are re-exported there).
    """

    def setUp(self):
        block_real_audio_hardware(self)

    def test_ensure_input_resolved_delegates_the_whole_sequence(self):
        # This used to inline route/suspend/resolve/resume and assert their order. That sequence now
        # lives in mic_device.resolve_capture_device() — it had to, because capture grew a second
        # route needing the OPPOSITE pulse state, and four inlined copies would have been four
        # places to teach. So the assistant's contract here is delegation, and the ordering is
        # asserted where the ordering lives (tests/test_mic_device.py). See S16.
        va = make_assistant()
        with patch("ai.voice_assistant.resolve_capture_device",
                   return_value=MicChoice(3, 48000, 2, 0, "int16", True, "i2s", "APE")) as resolve:
            va.ensure_input_resolved()
        resolve.assert_called_once()
        self.assertEqual(va._capture_device, 3)
        self.assertTrue(va._capture_is_i2s)

    def test_ensure_input_resolved_applies_any_environment_the_choice_carries(self):
        # The pulse route names its source with PULSE_SOURCE, and start_recording() opens its stream
        # later — so the setting has to outlive the resolve, not just the probe.
        va = make_assistant()
        key = "KAI_TEST_PULSE_SOURCE"
        os.environ.pop(key, None)
        self.addCleanup(os.environ.pop, key, None)
        with patch("ai.voice_assistant.resolve_capture_device",
                   return_value=MicChoice(4, 16000, 1, 0, "int16", False, "pulse", "",
                                          {key: "some.source"})):
            va.ensure_input_resolved()
        self.assertEqual(os.environ.get(key), "some.source")


class TestStateMachine(unittest.TestCase):
    def setUp(self):
        block_real_audio_hardware(self)
        # ensure_input_resolved()/start_recording() shell out to `amixer` and `pactl`; stub them so
        # no test reconfigures real audio on the Jetson (or spawns a missing binary on a dev box).
        for name in ("apply_i2s_route", "free_i2s_device", "resume_pulse_source"):
            p = patch(f"ai.voice_assistant.{name}")
            setattr(self, f"mock_{name}", p.start())
            self.addCleanup(p.stop)
        self.mock_apply_i2s_route.return_value = False

    def test_starts_idle(self):
        va = make_assistant()
        self.assertEqual(va.get_status()["voice_status"], STATUS_IDLE)

    def test_start_recording_opens_stream(self):
        va = make_assistant()
        with patch("ai.voice_assistant.resolve_capture_device",
                   return_value=MicChoice(None, 16000, 1, 0, "int16")), \
             patch("ai.voice_assistant.sd.InputStream") as mock_stream_cls:
            mock_stream = MagicMock()
            mock_stream_cls.return_value = mock_stream
            result = va.start_recording()
        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(va.get_status()["voice_status"], STATUS_RECORDING)
        mock_stream.start.assert_called_once()

    def test_start_recording_opens_i2s_stereo_and_frees_pulse(self):
        va = make_assistant()
        with patch("ai.voice_assistant.resolve_capture_device",
                   return_value=MicChoice(3, 48000, 2, 0, "int16", True)), \
             patch("ai.voice_assistant.sd.InputStream") as mock_stream_cls:
            mock_stream_cls.return_value = MagicMock()
            va.start_recording()
        _, kwargs = mock_stream_cls.call_args
        self.assertEqual(kwargs["channels"], 2)
        self.assertEqual(kwargs["samplerate"], 48000)
        self.assertEqual(kwargs["device"], 3)
        # I2S capture must (re)free the card from pulse before opening the stream
        self.mock_free_i2s_device.assert_called()

    def test_start_recording_does_not_re_free_the_card_for_a_non_i2s_mic(self):
        # Handing pulse back on a non-I2S choice is resolve_capture_device()'s job now, and it is
        # asserted there. What matters HERE is the inverse of the I2S case above: a mic that is not
        # the raw device must NOT have the card yanked off pulse underneath it, which is exactly
        # what re-asserting the suspend would do — and for the pulse route it would be fatal.
        va = make_assistant()
        self.mock_free_i2s_device.reset_mock()
        with patch("ai.voice_assistant.resolve_capture_device",
                   return_value=MicChoice(None, 16000, 1, 0, "int16", False)), \
             patch("ai.voice_assistant.sd.InputStream") as mock_stream_cls:
            mock_stream_cls.return_value = MagicMock()
            va.start_recording()
        self.mock_free_i2s_device.assert_not_called()

    def test_start_recording_rejected_while_recording(self):
        va = make_assistant()
        with patch("ai.voice_assistant.resolve_capture_device",
                   return_value=MicChoice(None, 16000, 1, 0, "int16")), \
             patch("ai.voice_assistant.sd.InputStream"):
            va.start_recording()
            result = va.start_recording()
        self.assertIn("error", result)

    def test_stop_recording_rejected_while_idle(self):
        va = make_assistant()
        result = va.stop_recording()
        self.assertIn("error", result)

    def test_stop_recording_transitions_and_spawns_worker(self):
        va = make_assistant()
        with patch("ai.voice_assistant.resolve_capture_device",
                   return_value=MicChoice(None, 16000, 1, 0, "int16")), \
             patch("ai.voice_assistant.sd.InputStream"):
            va.start_recording()
        with patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            result = va.stop_recording()
        self.assertEqual(result, {"status": "ok"})
        mock_thread.start.assert_called_once()

    def test_start_recording_mic_failure_sets_error(self):
        va = make_assistant()
        with patch("ai.voice_assistant.resolve_capture_device",
                   return_value=MicChoice(None, 16000, 1, 0, "int16")), \
             patch("ai.voice_assistant.sd.InputStream", side_effect=OSError("no device")):
            result = va.start_recording()
        self.assertIn("error", result)
        self.assertEqual(va.get_status()["voice_status"], STATUS_ERROR)

    def test_device_resolution_only_probed_once(self):
        va = make_assistant()
        with patch("ai.voice_assistant.resolve_capture_device",
                   return_value=MicChoice(2, 44100, 2, 0, "int16")) as mock_resolve:
            va.ensure_input_resolved()
            va.ensure_input_resolved()
        mock_resolve.assert_called_once()
        self.assertEqual(va._capture_device, 2)
        self.assertEqual(va._capture_rate, 44100)
        self.assertEqual(va._capture_channels, 2)
        self.assertEqual(va._capture_channel, 0)


class TestOnAudioChunk(unittest.TestCase):
    def test_mono_chunk_stored_as_is(self):
        va = make_assistant()  # defaults: 1 channel, take channel 0
        indata = np.arange(10, dtype="int16").reshape(10, 1)
        va._on_audio_chunk(indata, 10, None, None)
        stored = va._audio_chunks[0]
        self.assertEqual(stored.shape, (10, 1))
        np.testing.assert_array_equal(stored[:, 0], np.arange(10))

    def test_stereo_chunk_keeps_only_taken_channel(self):
        va = make_assistant()
        va._capture_channels = 2
        va._capture_channel  = 0
        indata = np.zeros((10, 2), dtype="int16")
        indata[:, 0] = np.arange(10)     # left = real audio
        indata[:, 1] = 999               # right = digital silence stand-in
        va._on_audio_chunk(indata, 10, None, None)
        stored = va._audio_chunks[0]
        self.assertEqual(stored.shape, (10, 1))   # stays mono downstream
        np.testing.assert_array_equal(stored[:, 0], np.arange(10))

    def test_chunk_is_a_copy(self):
        # PortAudio reuses indata's buffer after the callback returns — the stored chunk must not
        # alias it.
        va = make_assistant()
        indata = np.ones((5, 1), dtype="int16")
        va._on_audio_chunk(indata, 5, None, None)
        indata[:] = 0
        self.assertTrue((va._audio_chunks[0] == 1).all())


class TestProcess(unittest.TestCase):
    def setUp(self):
        # _process -> _speak; with a real voice model present (on the Jetson) _speak would spawn a
        # worker that shells out to Piper/paplay. Disable TTS so these tests exercise the synchronous
        # text-timed pantomime fallback deterministically and never touch audio.
        p = patch("ai.voice_assistant.tts.enabled", return_value=False)
        p.start()
        self.addCleanup(p.stop)

    def test_empty_transcript_short_circuits_before_ollama(self):
        va = make_assistant()
        va._whisper_model = MagicMock()
        va._whisper_model.transcribe.return_value = ([], None)
        with patch.object(va, "_call_ollama") as mock_call:
            va._process(np.zeros((10, 1), dtype="int16"))
        mock_call.assert_not_called()
        status = va.get_status()
        self.assertEqual(status["voice_status"], STATUS_DONE)
        self.assertEqual(status["voice_response"], NO_SPEECH_RESPONSE)

    def test_transcript_concatenation(self):
        va = make_assistant()
        va._whisper_model = MagicMock()
        va._whisper_model.transcribe.return_value = (
            [make_segment("hello "), make_segment("world")], None
        )
        with patch.object(va, "_call_ollama", return_value="hi there"):
            va._process(np.ones((10, 1), dtype="int16"))
        status = va.get_status()
        self.assertEqual(status["voice_transcript"], "hello world")
        self.assertEqual(status["voice_response"], "hi there")
        self.assertEqual(status["voice_status"], STATUS_DONE)

    def test_history_updated_after_success(self):
        va = make_assistant()
        va._whisper_model = MagicMock()
        va._whisper_model.transcribe.return_value = ([make_segment("hi")], None)
        with patch.object(va, "_call_ollama", return_value="hello back"):
            va._process(np.ones((10, 1), dtype="int16"))
        self.assertEqual(va._history[-2], {"role": "user", "content": "hi"})
        self.assertEqual(va._history[-1], {"role": "assistant", "content": "hello back"})

    def test_exception_sets_error_status(self):
        va = make_assistant()
        va._whisper_model = MagicMock()
        va._whisper_model.transcribe.side_effect = RuntimeError("boom")
        va._process(np.ones((10, 1), dtype="int16"))
        status = va.get_status()
        self.assertEqual(status["voice_status"], STATUS_ERROR)
        self.assertIn("boom", status["voice_error"])

    def test_resamples_when_capture_rate_differs_from_whisper_rate(self):
        va = make_assistant()
        va._capture_rate = 44100
        va._whisper_model = MagicMock()
        va._whisper_model.transcribe.return_value = ([make_segment("hi")], None)
        audio = np.ones((44100, 1), dtype="int16")  # 1s at the mic's native rate
        va._transcribe(audio)
        call_args = va._whisper_model.transcribe.call_args
        samples = call_args[0][0]
        # resampled from 44100 to 16000 samples/sec — length should track the target rate
        self.assertAlmostEqual(len(samples), 16000, delta=100)
        self.assertTrue(call_args[1]["vad_filter"])


class TestIdentityCapture(unittest.TestCase):
    """Pinning who Kai is talking to. Extraction itself is tests/test_identity.py; this covers the
    lifetime, the epoch guard and the prompt placement. See docs/tickets/S12."""

    def test_name_is_none_until_offered(self):
        va = make_assistant()
        self.assertIsNone(va.person_name)
        va.note_identity("How many chapters does DEVCON have?")
        self.assertIsNone(va.person_name)

    def test_name_is_pinned(self):
        va = make_assistant()
        va.note_identity("My name is Jhondel")
        self.assertEqual(va.person_name, "Jhondel")

    def test_first_name_wins(self):
        """A later utterance cannot overwrite it. Re-extracting every turn would make the system
        prompt vary between turns, which is exactly what the system placement exists to avoid."""
        va = make_assistant()
        va.note_identity("My name is Jhondel")
        va.note_identity("My name is Somebody")
        self.assertEqual(va.person_name, "Jhondel")

    def test_reset_history_forgets_the_name(self):
        """The contract reset_history() already holds for the rolling history and the sticky RAG
        topic: the next person inherits nothing."""
        va = make_assistant()
        va.note_identity("My name is Jhondel")
        with patch("ai.voice_assistant.rag.reset_topic"):
            va.reset_history()
        self.assertIsNone(va.person_name)

    def test_stale_epoch_cannot_pin_a_name(self):
        """A session that ended while STT was still running must not hand its name to the next
        person — the same failure the history append is epoch-guarded against."""
        va = make_assistant()
        stale = va.epoch
        va.bump_epoch()
        va.note_identity("My name is Jhondel", epoch=stale)
        self.assertIsNone(va.person_name)

    def test_current_epoch_pins_normally(self):
        va = make_assistant()
        va.note_identity("My name is Jhondel", epoch=va.epoch)
        self.assertEqual(va.person_name, "Jhondel")

    def test_disabled_captures_nothing(self):
        with patch("ai.voice_assistant.IDENTITY_CAPTURE", False):
            va = make_assistant()
            va.note_identity("My name is Jhondel")
            self.assertIsNone(va.person_name)

    def test_one_breath_turn_also_pins_the_name(self):
        """say(use_llm=True) is the one-breath hands-free path — the whisper wake tier already holds
        the transcript, so "Hey Kai, my name is Jhondel" said without a pause runs here and never
        touches _process(). Found live: the mic path captured and this one did not, so the same
        sentence pinned a name or not depending purely on whether the speaker drew breath."""
        va = make_assistant()
        done = threading.Event()
        with patch.object(va, "_call_ollama", return_value="Hi there"), \
             patch.object(va, "_speak"):
            va.say("my name is Jhondel", use_llm=True, on_done=lambda *_: done.set())
            done.wait(5)
        self.assertEqual(va.person_name, "Jhondel")

    def test_verbatim_say_does_not_pin_a_name(self):
        """use_llm=False is the dashboard reading a line out loud, not somebody introducing
        themselves. Kai saying "my name is Kai" must not make Kai believe it is talking to Kai."""
        va = make_assistant()
        done = threading.Event()
        with patch.object(va, "_speak"):
            va.say("My name is Jhondel", use_llm=False, on_done=lambda *_: done.set())
            done.wait(5)
        self.assertIsNone(va.person_name)

    def test_name_goes_in_the_system_prompt_not_the_user_turn(self):
        """The OPPOSITE placement to the RAG context, and for the opposite reason: this string is
        constant for the rest of the session, so it costs one KV-prefix invalidation when learned
        rather than one per turn. See IDENTITY_PROMPT in config/voice.py."""
        va = make_assistant()
        va.note_identity("My name is Jhondel")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "reply"}}
        with patch("ai.voice_assistant.load_persona", return_value="PERSONA_TEXT"), \
             patch("ai.voice_assistant.rag.retrieve_context", return_value="CONTEXT_TEXT"), \
             patch("ai.llm.requests.post", return_value=mock_resp) as mock_post:
            va._call_ollama("hello")
        messages = mock_post.call_args[1]["json"]["messages"]
        system_msg, user_msg = messages[0], messages[-1]
        self.assertIn("PERSONA_TEXT", system_msg["content"])
        self.assertIn("Jhondel", system_msg["content"])
        self.assertNotIn("Jhondel", user_msg["content"])

    def test_prompt_is_unchanged_when_no_name_is_known(self):
        """With capture on but nobody having introduced themselves, the prompt must be identical to
        today's — the feature costs nothing until it fires."""
        va = make_assistant()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "reply"}}
        with patch("ai.voice_assistant.load_persona", return_value="PERSONA_TEXT"), \
             patch("ai.voice_assistant.rag.retrieve_context", return_value=""), \
             patch("ai.llm.requests.post", return_value=mock_resp) as mock_post:
            va._call_ollama("hello")
        system_msg = mock_post.call_args[1]["json"]["messages"][0]
        self.assertEqual(system_msg["content"], "PERSONA_TEXT")

    def test_system_prompt_is_stable_across_turns_once_learned(self):
        """The KV-prefix claim in config/voice.py, asserted rather than assumed: the system message
        must be byte-identical on consecutive turns even as the retrieved context changes, or the
        'one invalidation, then free' reasoning is wrong and the placement should change."""
        va = make_assistant()
        va.note_identity("My name is Jhondel")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "reply"}}
        seen = []
        with patch("ai.voice_assistant.load_persona", return_value="PERSONA_TEXT"), \
             patch("ai.voice_assistant.rag.retrieve_context", side_effect=["CTX_A", "CTX_B"]), \
             patch("ai.llm.requests.post", return_value=mock_resp) as mock_post:
            va._call_ollama("first")
            seen.append(mock_post.call_args[1]["json"]["messages"][0]["content"])
            va._call_ollama("second")
            seen.append(mock_post.call_args[1]["json"]["messages"][0]["content"])
        self.assertEqual(seen[0], seen[1])


class TestCallOllama(unittest.TestCase):
    def test_posts_expected_payload(self):
        va = make_assistant()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "reply"}}
        with patch("ai.voice_assistant.rag.retrieve_context", return_value=""), \
             patch("ai.llm.requests.post", return_value=mock_resp) as mock_post:
            result = va._call_ollama("hello")
        self.assertEqual(result, "reply")
        _, kwargs = mock_post.call_args
        # Assert against the config value, not a hardcoded name — this drifted once already when the
        # model was switched from gemma3:4b to fit the camera in 8 GB.
        self.assertEqual(kwargs["json"]["model"], OLLAMA_MODEL)
        self.assertEqual(kwargs["json"]["stream"], False)
        self.assertIn("num_ctx", kwargs["json"]["options"])

    def test_context_goes_in_the_user_turn_leaving_the_prefix_stable(self):
        """The RAG context must NOT sit in the system message. It changes every turn, and Ollama
        reuses its KV cache only for the longest common PREFIX — so context at the front
        re-evaluates the persona and all of history on every turn. In the user turn it leaves that
        prefix byte-identical. See RAG_CONTEXT_PLACEMENT in config/voice.py."""
        va = make_assistant()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "reply"}}
        with patch("ai.voice_assistant.load_persona", return_value="PERSONA_TEXT"), \
             patch("ai.voice_assistant.rag.retrieve_context", return_value="CONTEXT_TEXT"), \
             patch("ai.llm.requests.post", return_value=mock_resp) as mock_post:
            va._call_ollama("hello")
        messages = mock_post.call_args[1]["json"]["messages"]
        system_msg, user_msg = messages[0], messages[-1]
        self.assertEqual(system_msg["role"], "system")
        self.assertEqual(system_msg["content"], "PERSONA_TEXT")   # the persona ALONE — no context
        self.assertNotIn("CONTEXT_TEXT", system_msg["content"])
        self.assertEqual(user_msg["role"], "user")
        self.assertIn("CONTEXT_TEXT", user_msg["content"])
        self.assertIn("hello", user_msg["content"])

    def test_system_placement_restores_the_old_shape(self):
        """The documented revert (RAG_CONTEXT_PLACEMENT = "system") must put the context back in the
        system prompt exactly as before, so a quality regression can be undone with one config line."""
        va = make_assistant()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "reply"}}
        with patch("ai.voice_assistant.RAG_CONTEXT_PLACEMENT", "system"), \
             patch("ai.voice_assistant.load_persona", return_value="PERSONA_TEXT"), \
             patch("ai.voice_assistant.rag.retrieve_context", return_value="CONTEXT_TEXT"), \
             patch("ai.llm.requests.post", return_value=mock_resp) as mock_post:
            va._call_ollama("hello")
        messages = mock_post.call_args[1]["json"]["messages"]
        self.assertIn("PERSONA_TEXT", messages[0]["content"])
        self.assertIn("CONTEXT_TEXT", messages[0]["content"])
        self.assertEqual(messages[-1]["content"], "hello")

    def test_history_stores_the_raw_transcript_not_the_injected_context(self):
        """The prefix-stability win depends on this: if the context leaked into stored history, the
        prefix would differ next turn and the KV cache would be invalidated anyway."""
        va = make_assistant()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "reply"}}
        with patch("ai.voice_assistant.rag.retrieve_context", return_value="CONTEXT_TEXT"), \
             patch("ai.llm.requests.post", return_value=mock_resp), \
             patch.object(va, "_transcribe", return_value="hello"), \
             patch.object(va, "_speak"):
            va._process(np.zeros(16000, dtype=np.int16), rate=16000)
        self.assertEqual(va._history[0], {"role": "user", "content": "hello"})

    def test_records_stage_timings(self):
        va = make_assistant()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "reply"},
            "prompt_eval_count": 120, "prompt_eval_duration": 200_000_000,   # 200 ms
            "eval_count": 40, "eval_duration": 2_000_000_000,                # 2 s -> 20 tok/s
        }
        with patch("ai.voice_assistant.rag.retrieve_context", return_value=""), \
             patch("ai.llm.requests.post", return_value=mock_resp):
            va._call_ollama("hello")
        t = va.stage_timings()
        self.assertEqual(t["llm_prompt_ms"], 200)
        self.assertEqual(t["llm_gen_ms"], 2000)
        self.assertEqual(t["llm_tok_s"], 20.0)
        self.assertGreaterEqual(t["llm_ms"], 0)

    def test_missing_timing_fields_are_tolerated(self):
        """A response with no timing fields at all (an old Ollama, or a stub) must not raise —
        this sits on the hot path and may never cost a reply."""
        va = make_assistant()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "reply"}}
        with patch("ai.voice_assistant.rag.retrieve_context", return_value=""), \
             patch("ai.llm.requests.post", return_value=mock_resp):
            self.assertEqual(va._call_ollama("hello"), "reply")
        self.assertEqual(va.stage_timings()["llm_tok_s"], 0.0)

    def test_uses_only_persona_when_no_context(self):
        va = make_assistant()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "reply"}}
        with patch("ai.voice_assistant.load_persona", return_value="PERSONA_TEXT"), \
             patch("ai.voice_assistant.rag.retrieve_context", return_value=""), \
             patch("ai.llm.requests.post", return_value=mock_resp) as mock_post:
            va._call_ollama("hello")
        messages = mock_post.call_args[1]["json"]["messages"]
        self.assertEqual(messages[0]["content"], "PERSONA_TEXT")
        self.assertEqual(messages[-1]["content"], "hello")   # no context, so the turn is untouched

    def test_keep_alive_and_timeout_present(self):
        va = make_assistant()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "reply"}}
        with patch("ai.voice_assistant.rag.retrieve_context", return_value=""), \
             patch("ai.llm.requests.post", return_value=mock_resp) as mock_post:
            va._call_ollama("hello")
        _, kwargs = mock_post.call_args
        self.assertIn("timeout", kwargs)
        # keep_alive must be a JSON number (e.g. -1) — Ollama rejects the string "-1" as an
        # invalid Go duration with a 400, which silently broke every real request before.
        self.assertIsInstance(kwargs["json"]["keep_alive"], (int, float))

    def test_connection_error_raises_runtime_error(self):
        va = make_assistant()
        with patch("ai.voice_assistant.rag.retrieve_context", return_value=""), \
             patch("ai.llm.requests.post", side_effect=requests.exceptions.ConnectionError()):
            with self.assertRaises(RuntimeError):
                va._call_ollama("hello")


class TestEnsureLlmWarm(unittest.TestCase):
    def test_swallows_connection_error(self):
        va = make_assistant()
        with patch("ai.llm.requests.post", side_effect=requests.exceptions.ConnectionError()):
            va.ensure_llm_warm()  # must not raise

    def test_posts_a_trivial_request(self):
        va = make_assistant()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "hi"}}
        # log_model_placement patched out: it is diagnostics-only, and unpatched it would reach for a
        # real Ollama on localhost from the test suite.
        with patch("ai.voice_assistant.log_model_placement"), \
             patch("ai.llm.requests.post", return_value=mock_resp) as mock_post:
            va.ensure_llm_warm()
        mock_post.assert_called_once()

    def test_logs_model_placement_after_a_successful_warm(self):
        """The placement probe is the whole reason warming happens at startup — Ollama pins the
        CPU/GPU choice at load time, and a partial offload is otherwise invisible."""
        va = make_assistant()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "hi"}}
        with patch("ai.voice_assistant.log_model_placement") as mock_ps, \
             patch("ai.llm.requests.post", return_value=mock_resp):
            va.ensure_llm_warm()
        mock_ps.assert_called_once()

    def test_skips_placement_probe_when_warm_up_failed(self):
        va = make_assistant()
        with patch("ai.voice_assistant.log_model_placement") as mock_ps, \
             patch("ai.llm.requests.post",
                   side_effect=requests.exceptions.ConnectionError()):
            va.ensure_llm_warm()
        mock_ps.assert_not_called()


class TestSpeakingIntegration(unittest.TestCase):
    def setUp(self):
        # Same as TestProcess: keep the jaw pantomime synchronous and audio-free by disabling TTS.
        p = patch("ai.voice_assistant.tts.enabled", return_value=False)
        p.start()
        self.addCleanup(p.stop)

    def test_reply_opens_speaking_window(self):
        va = make_assistant()
        self.assertIsNone(va.speaking_openness())
        with patch("ai.voice_assistant.rag.retrieve_context", return_value=""):
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"message": {"content": "hello there friend"}}
            with patch("ai.llm.requests.post", return_value=mock_resp), \
                 patch.object(va, "_transcribe", return_value="hi"):
                va._process(np.zeros((10, 1), dtype="int16"))
        self.assertEqual(va.get_status()["voice_status"], STATUS_DONE)
        self.assertTrue(va.get_status()["voice_speaking"])
        self.assertIsNotNone(va.speaking_openness())

    def test_no_speech_still_animates(self):
        va = make_assistant()
        with patch.object(va, "_transcribe", return_value="   "):
            va._process(np.zeros((10, 1), dtype="int16"))
        self.assertEqual(va.get_status()["voice_response"], NO_SPEECH_RESPONSE)
        self.assertTrue(va.get_status()["voice_speaking"])


class _InlineThread:
    """threading.Thread stand-in that runs the target synchronously on .start(), so _speak's worker
    body is exercised deterministically in-process — no real thread, no shelling out to Piper/paplay
    (the tts.* calls are themselves mocked)."""
    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)


class TestSpeak(unittest.TestCase):
    def test_disabled_uses_text_timed_pantomime_without_synth(self):
        va = make_assistant()
        with patch("ai.voice_assistant.tts.enabled", return_value=False), \
             patch("ai.voice_assistant.tts.synthesize") as mock_synth:
            va._speak("Hello there. Nice to meet you.")
        mock_synth.assert_not_called()
        self.assertIsNotNone(va._speak_start)
        self.assertEqual(len(va._speak_segments), 2)   # one window per sentence

    def test_enabled_scales_window_to_audio_duration_and_plays(self):
        va = make_assistant()
        with patch("ai.voice_assistant.tts.enabled", return_value=True), \
             patch("ai.voice_assistant.tts.stop"), \
             patch("ai.voice_assistant.tts.synthesize", return_value="/tmp/kai_tts.wav") as mock_synth, \
             patch("ai.voice_assistant.tts.wav_duration", return_value=4.0), \
             patch("ai.voice_assistant.tts.wav_speech_span", return_value=(0.0, 4.0)), \
             patch("ai.voice_assistant.tts.play") as mock_play, \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va._speak("One two three. Four five six.")
        mock_synth.assert_called_once()
        mock_play.assert_called_once()
        self.assertIsNotNone(va._speak_start)
        self.assertAlmostEqual(va._speak_segments[-1][1], 4.0, places=6)  # window == real audio length

    def test_enabled_synth_failure_falls_back_to_pantomime(self):
        va = make_assistant()
        with patch("ai.voice_assistant.tts.enabled", return_value=True), \
             patch("ai.voice_assistant.tts.stop"), \
             patch("ai.voice_assistant.tts.synthesize", return_value=None), \
             patch("ai.voice_assistant.tts.wav_duration") as mock_dur, \
             patch("ai.voice_assistant.tts.play") as mock_play, \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va._speak("Hello there.")
        mock_play.assert_not_called()   # nothing to play
        mock_dur.assert_not_called()
        self.assertIsNotNone(va._speak_start)          # pantomime window still set so the jaw moves
        self.assertEqual(len(va._speak_segments), 1)

    def test_claiming_the_speaker_retires_the_previous_lines_jaw(self):
        # The filler-cut-by-reply bug: _begin_speech kills the outgoing audio, and the jaw is a
        # schedule that nothing else clears — so a filler cut mid-word went on miming the rest of
        # its sentence in silence, all through the Piper run before the reply had a window.
        # Observed from INSIDE play(), because _InlineThread runs the worker synchronously — by the
        # time speak_wav() returns, playback has "finished" and _end_speech has tidied up. Mid-play
        # is the state that matters here anyway: it is when the reply actually arrives.
        va = make_assistant()
        seen = {}

        def _reply_arrives_mid_filler(_path):
            seen["jaw_before"] = va._speak_start
            seen["end_before"] = va.audio_ends_at()
            with patch("ai.voice_assistant.tts.stop"):
                va._begin_speech()                     # the reply takes the speaker
            seen["jaw_after"] = va._speak_start
            seen["segs_after"] = va._speak_segments
            seen["end_after"] = va.audio_ends_at()

        with patch("ai.voice_assistant.tts.stop"), \
             patch("ai.voice_assistant.tts.wav_duration", return_value=9.0), \
             patch("ai.voice_assistant.tts.play", side_effect=_reply_arrives_mid_filler), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va.speak_wav("/tmp/filler.wav", "Hmm, let me think about that one.")

        self.assertIsNotNone(seen["jaw_before"])       # the filler was miming
        self.assertIsNotNone(seen["end_before"])
        self.assertIsNone(seen["jaw_after"], "the cut filler must stop miming")
        self.assertEqual(seen["segs_after"], ())
        self.assertIsNone(seen["end_after"],
                          "a stale end time would make the session cut the new line instantly")

    def test_stop_speech_cuts_the_audio_and_the_jaw_together(self):
        va = make_assistant()
        seen = {}

        def _interrupted(_path):
            with patch("ai.voice_assistant.tts.stop") as mock_stop:
                va.stop_speech()
            seen["stopped"] = mock_stop.call_count
            seen["jaw"] = va._speak_start
            seen["end"] = va.audio_ends_at()

        with patch("ai.voice_assistant.tts.stop"), \
             patch("ai.voice_assistant.tts.wav_duration", return_value=6.0), \
             patch("ai.voice_assistant.tts.play", side_effect=_interrupted), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va.speak_wav("/tmp/line.wav", "A sentence that is going to be interrupted.")

        self.assertEqual(seen["stopped"], 1)           # the audio was killed…
        self.assertIsNone(seen["jaw"])                 # …and the mouth stopped with it
        self.assertIsNone(seen["end"])

    def test_audio_ends_at_is_the_wav_end_and_gate_is_that_plus_the_tail(self):
        from config.wake import TTS_TAIL_MUTE_S
        va = make_assistant()
        seen = {}

        def _capture(_path):
            seen["ends_at"] = va.audio_ends_at()
            seen["gate"] = va._gate_until

        with patch("ai.voice_assistant.tts.enabled", return_value=True), \
             patch("ai.voice_assistant.tts.stop"), \
             patch("ai.voice_assistant.tts.synthesize", return_value="/tmp/kai_tts.wav"), \
             patch("ai.voice_assistant.tts.wav_duration", return_value=5.0), \
             patch("ai.voice_assistant.tts.play", side_effect=_capture), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va._speak("One two three.")
        self.assertIsNotNone(seen["ends_at"])
        # The two answer different questions: when the sound stops, and when the mic may reopen.
        self.assertAlmostEqual(seen["gate"] - seen["ends_at"], TTS_TAIL_MUTE_S, places=6)

    def test_the_jaw_stays_shut_until_the_audio_actually_starts(self):
        # Reported on the robot 2026-08-12: after "Hey Kai" the mouth moved and the "Yes?" arrived
        # afterwards. tts.play() only spawns paplay — the first sample is ~0.2 s behind it — but the
        # jaw window opened at the call, so on a line this short the mouth was most of the way
        # through its mime before there was any sound at all.
        from config.voice import TTS_PLAYBACK_LEAD_S

        self.assertGreater(TTS_PLAYBACK_LEAD_S, 0.0, "the rest of this test would prove nothing")
        va = make_assistant()
        # The span is pinned to the whole file so this test is about the LEAD alone — and so it
        # cannot read a real /tmp/kai_ack WAV when the suite happens to run on the robot.
        with patch("ai.voice_assistant.tts.stop"), \
             patch("ai.voice_assistant.tts.wav_duration", return_value=0.8), \
             patch("ai.voice_assistant.tts.wav_speech_span", return_value=(0.0, 0.8)), \
             patch("ai.voice_assistant.tts.play"), \
             patch("ai.voice_assistant.time.monotonic", return_value=100.0), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va.speak_wav("/tmp/kai_ack/kai_canned_ack.wav", "Yes?")
        self.assertAlmostEqual(va._speak_start, 100.0 + TTS_PLAYBACK_LEAD_S)
        # And the servo is told to keep the mouth closed through that window rather than being
        # handed a negative time it might clamp to "open".
        self.assertIsNone(va.speaking_openness(now=100.0))
        self.assertIsNone(va.speaking_openness(now=100.0 + TTS_PLAYBACK_LEAD_S - 0.01))
        self.assertIsNotNone(va.speaking_openness(now=100.0 + TTS_PLAYBACK_LEAD_S + 0.01))

    def test_the_jaw_is_timed_to_the_speech_not_the_file(self):
        # The other half of the same mismatch: Piper pads silence onto both ends, so the ack file is
        # 0.82 s for a word that lasts 0.54 s. Miming across the whole file left the mouth open a
        # quarter-second after the sound had stopped. The span here is the one measured off the
        # robot's own /tmp/kai_ack/kai_canned_ack.wav on 2026-08-12.
        from config.voice import TTS_PLAYBACK_LEAD_S

        va = make_assistant()
        seen = {}
        # Read from INSIDE play(): _InlineThread runs the worker synchronously, so by the time
        # speak_wav() returns _end_speech has already retired the audio end (see
        # test_claiming_the_speaker_retires_the_previous_lines_jaw).
        with patch("ai.voice_assistant.tts.stop"), \
             patch("ai.voice_assistant.tts.wav_duration", return_value=0.82), \
             patch("ai.voice_assistant.tts.wav_speech_span", return_value=(0.04, 0.58)), \
             patch("ai.voice_assistant.tts.play",
                   side_effect=lambda _p: seen.update(ends_at=va.audio_ends_at())), \
             patch("ai.voice_assistant.time.monotonic", return_value=100.0), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va.speak_wav("/tmp/kai_ack/kai_canned_ack.wav", "Yes?")
        audio_starts = 100.0 + TTS_PLAYBACK_LEAD_S
        self.assertAlmostEqual(va._speak_start, audio_starts + 0.04)      # skips the leading pad
        self.assertAlmostEqual(va._speak_segments[-1][1], 0.58 - 0.04)    # and stops at the word
        # The mic gate is NOT trimmed: the speaker is busy for the whole file either way.
        self.assertAlmostEqual(seen["ends_at"], audio_starts + 0.82)

    def test_an_unreadable_scan_falls_back_to_the_whole_file(self):
        # A scan that cannot read the WAV must cost the trim and nothing else — never a jaw that
        # fails to move, which is the one outcome worse than a mistimed one.
        va = make_assistant()
        with patch("ai.voice_assistant.tts.stop"), \
             patch("ai.voice_assistant.tts.wav_duration", return_value=4.0), \
             patch("ai.voice_assistant.tts.wav_speech_span", return_value=(0.0, 0.0)), \
             patch("ai.voice_assistant.tts.play"), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va.speak_wav("/tmp/line.wav", "One two three.")
        self.assertIsNotNone(va._speak_start)
        self.assertAlmostEqual(va._speak_segments[-1][1], 4.0)

    def test_a_pantomime_publishes_no_audio_end(self):
        # No sound is playing, so there is no end time — the session must fall back to its cap.
        va = make_assistant()
        with patch("ai.voice_assistant.tts.enabled", return_value=False):
            va._speak("Hello there.")
        self.assertIsNotNone(va._speak_start)   # the jaw still moves
        self.assertIsNone(va.audio_ends_at())

    def test_synthesizes_emoji_stripped_text(self):
        # The UI keeps emoji; the spoken text must not (real clean_for_speech runs — not mocked).
        va = make_assistant()
        with patch("ai.voice_assistant.tts.enabled", return_value=True), \
             patch("ai.voice_assistant.tts.stop"), \
             patch("ai.voice_assistant.tts.synthesize", return_value="/tmp/kai_tts.wav") as mock_synth, \
             patch("ai.voice_assistant.tts.wav_duration", return_value=2.0), \
             patch("ai.voice_assistant.tts.play"), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va._speak("Hello there! 😀🎉")
        # Positional arg only: _speak also passes the per-reply tempo jitter (ai/delivery), which is
        # not what this test is about. Two words, so no delivery shaping applies to the text itself.
        mock_synth.assert_called_once()
        self.assertEqual(mock_synth.call_args.args[0], "Hello there!")

    def test_spoken_text_is_delivery_shaped_but_the_ui_text_is_not(self):
        # The shaping (ai/delivery) exists because the voice model cannot be improved on this board —
        # see docs/plan/completed/expressive-voice-plan.md. It must reach Piper and stop there: the
        # dashboard shows what the LLM actually said, exactly as with the emoji stripping above.
        va = make_assistant()
        reply = "The DEVCON program runs all year long but the internships open in March."
        with patch("ai.voice_assistant.tts.enabled", return_value=True), \
             patch("ai.voice_assistant.tts.stop"), \
             patch("ai.voice_assistant.tts.synthesize", return_value="/tmp/kai_tts.wav") as mock_synth, \
             patch("ai.voice_assistant.tts.wav_duration", return_value=2.0), \
             patch("ai.voice_assistant.tts.play"), \
             patch("ai.voice_assistant.delivery.shape", return_value="SHAPED") as mock_shape, \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va._speak(reply)
        mock_shape.assert_called_once_with(reply)
        self.assertEqual(mock_synth.call_args.args[0], "SHAPED")

    def test_shaping_off_passes_no_length_scale_override(self):
        # Shaping off must be byte-identical to the behaviour before ai/delivery existed: no scale
        # argument at all, so ai/tts._run_piper goes on reading the live dashboard rate itself.
        va = make_assistant()
        with patch("ai.voice_assistant.tts.enabled", return_value=True), \
             patch("ai.voice_assistant.tts.stop"), \
             patch("ai.voice_assistant.tts.synthesize", return_value="/tmp/kai_tts.wav") as mock_synth, \
             patch("ai.voice_assistant.tts.wav_duration", return_value=2.0), \
             patch("ai.voice_assistant.tts.play"), \
             patch("ai.voice_assistant.delivery.enabled", return_value=False), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va._speak("The DEVCON program runs all year long and takes volunteers.")
        self.assertIsNone(mock_synth.call_args.kwargs["length_scale"])


class TestCleanForSpeech(unittest.TestCase):
    def test_strips_emoji_keeps_words(self):
        self.assertEqual(clean_for_speech("Hello there! 😀"), "Hello there!")

    def test_strips_symbols_dingbats_and_flags(self):
        self.assertEqual(clean_for_speech("Nice ✨🎉 job 🇵🇭"), "Nice job")

    def test_collapses_whitespace_left_behind(self):
        self.assertEqual(clean_for_speech("Yes 👍 sir"), "Yes sir")

    def test_keeps_normal_punctuation_and_ellipsis(self):
        # … (U+2026) and — (U+2014) live in General Punctuation and must be preserved.
        self.assertEqual(clean_for_speech("Wait… really? Yes—now!"), "Wait… really? Yes—now!")

    def test_emoji_only_becomes_empty(self):
        self.assertEqual(clean_for_speech("😀🎉👍"), "")

    def test_empty_and_none_safe(self):
        self.assertEqual(clean_for_speech(""), "")
        self.assertEqual(clean_for_speech(None), "")


class TestTranscribeAsync(unittest.TestCase):
    """STT only — the wake-phrase scan must not be able to touch turn state."""

    def _run(self, **transcribe_kw):
        va = make_assistant()
        seen = []
        with patch.object(va, "_transcribe", **transcribe_kw), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va.transcribe_async(np.ones(1000, dtype="int16"), 16000,
                                on_done=lambda tok, text, err: seen.append((tok, text, err)),
                                token=7, log_language=False)
        return va, seen

    def test_hands_back_the_transcript(self):
        _, seen = self._run(return_value="hey kai what time is it")
        self.assertEqual(seen, [(7, "hey kai what time is it", "")])

    def test_never_touches_turn_state(self):
        # Writing _status here would post a chat bubble for a sentence nobody addressed to Kai.
        va, _ = self._run(return_value="some overheard sentence")
        status = va.get_status()
        self.assertEqual(status["voice_status"], STATUS_IDLE)
        self.assertEqual(status["voice_transcript"], "")
        self.assertEqual(status["voice_response"], "")

    def test_errors_come_back_as_an_error_not_an_exception(self):
        _, seen = self._run(side_effect=RuntimeError("ctranslate2 exploded"))
        self.assertEqual(seen[0][0], 7)
        self.assertEqual(seen[0][1], "")
        self.assertIn("exploded", seen[0][2])

    def test_forwards_rate_and_suppresses_the_language_log(self):
        va = make_assistant()
        with patch.object(va, "_transcribe", return_value="") as mock_tr, \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va.transcribe_async(np.ones(10, dtype="int16"), 48000, on_done=lambda *a: None)
        self.assertEqual(mock_tr.call_args.kwargs["rate"], 48000)
        self.assertTrue(mock_tr.call_args.kwargs["log_language"])

    def test_broken_callback_does_not_kill_the_thread(self):
        va = make_assistant()
        with patch.object(va, "_transcribe", return_value="hi"), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread), \
             patch("builtins.print") as mock_print:
            va.transcribe_async(np.ones(10, dtype="int16"), 16000,
                                on_done=lambda *a: (_ for _ in ()).throw(ValueError("bad")))
        mock_print.assert_called_once()

    def test_language_log_is_suppressed_when_asked(self):
        va = make_assistant()
        va._whisper_model = MagicMock()
        va._whisper_model.transcribe.return_value = ([make_segment("hi")], MagicMock(language="en",
                                                                                    language_probability=0.9))
        with patch("builtins.print") as mock_print:
            va._transcribe(np.ones((10, 1), dtype="int16"), rate=16000, log_language=False)
        mock_print.assert_not_called()


class TestSayEpochAndCallback(unittest.TestCase):
    """say() is the one-breath turn path: the transcript already exists, so re-transcribing the same
    audio through process_utterance would double the latency of the fastest interaction."""

    def setUp(self):
        p = patch("ai.voice_assistant.tts.enabled", return_value=False)
        p.start()
        self.addCleanup(p.stop)

    def test_reports_done(self):
        va = make_assistant()
        seen = []
        with patch.object(va, "_call_ollama", return_value="a reply"), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va.say("what time is it", epoch=va.epoch, on_done=lambda ep, o: seen.append(o))
        self.assertEqual(seen, ["done"])
        self.assertEqual(va.get_status()["voice_response"], "a reply")

    def test_reports_error(self):
        va = make_assistant()
        seen = []
        with patch.object(va, "_call_ollama", side_effect=RuntimeError("ollama down")), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va.say("hello", epoch=va.epoch, on_done=lambda ep, o: seen.append(o))
        self.assertEqual(seen, ["error"])

    def test_rejects_a_stale_epoch_up_front(self):
        va = make_assistant()
        stale = va.epoch
        va.bump_epoch()
        result = va.say("hello", epoch=stale)
        self.assertEqual(result, {"error": "stale"})

    def test_stale_after_ollama_writes_nothing(self):
        va = make_assistant()
        seen = []

        def ollama(text):
            va.bump_epoch()
            return "a reply"

        with patch.object(va, "_call_ollama", side_effect=ollama), \
             patch.object(va, "_speak") as mock_speak, \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va.say("hello", epoch=va.epoch, on_done=lambda ep, o: seen.append(o))
        self.assertEqual(seen, ["stale"])
        self.assertEqual(va._history, [])
        mock_speak.assert_not_called()
        self.assertEqual(va.get_status()["voice_response"], "")

    def test_busy_is_reported_so_the_caller_can_fall_back(self):
        # Ignoring this return leaves the session in BUSY for SESSION_BUSY_MAX_S — 120 s of a robot
        # that looks hung.
        va = make_assistant()
        va._status = "thinking"
        self.assertIn("error", va.say("hello", epoch=va.epoch))

    def test_unversioned_say_behaves_exactly_as_before(self):
        va = make_assistant()
        va.bump_epoch()
        with patch.object(va, "_call_ollama", return_value="reply"), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            self.assertEqual(va.say("hello"), {"status": "ok"})
        self.assertEqual(va.get_status()["voice_response"], "reply")

    def test_broken_callback_does_not_lose_the_reply(self):
        va = make_assistant()
        with patch.object(va, "_call_ollama", return_value="reply"), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread), \
             patch("builtins.print"):
            va.say("hello", epoch=va.epoch,
                   on_done=lambda ep, o: (_ for _ in ()).throw(ValueError("bad")))
        self.assertEqual(va.get_status()["voice_response"], "reply")


def _info(language, probs=None, prob=0.9):
    """A stand-in for faster-whisper's TranscriptionInfo."""
    info = MagicMock()
    info.language = language
    info.language_probability = prob
    info.all_language_probs = probs
    return info


class TestLanguageRestriction(unittest.TestCase):
    """Whisper picks from all 99 languages otherwise, and on short audio it is confidently wrong —
    measured on the robot: en 0.34, cy (Welsh) 0.22, nn (Norwegian Nynorsk) 0.21."""

    def _va(self, results):
        """results: list of (text, info) returned by successive transcribe() calls."""
        va = make_assistant()
        va._capture_rate = 16000
        va._whisper_model = MagicMock()
        calls = []

        beams = []
        prompts = []

        def transcribe(samples, language=None, vad_filter=True, beam_size=None,
                       initial_prompt=None):
            calls.append(language)
            beams.append(beam_size)
            prompts.append(initial_prompt)
            text, info = results[min(len(calls) - 1, len(results) - 1)]
            return ([make_segment(text)], info)

        va._whisper_model.transcribe.side_effect = transcribe
        va._recorded_beam_sizes = beams   # so the beam-width wiring can be asserted on
        va._recorded_prompts = prompts    # ditto for the decoder-bias wiring
        return va, calls

    def test_allowed_language_uses_the_first_pass_only(self):
        va, calls = self._va([("magandang umaga", _info("tl", [("tl", 0.8), ("en", 0.1)]))])
        with patch("builtins.print"):
            text = va._transcribe(np.ones((16000, 1), dtype="int16"), rate=16000)
        self.assertEqual(text, "magandang umaga")
        self.assertEqual(calls, [None], "an allowed detection must not cost a second pass")

    def test_english_also_uses_one_pass(self):
        va, calls = self._va([("what time is it", _info("en", [("en", 0.9)]))])
        with patch("builtins.print"):
            va._transcribe(np.ones((16000, 1), dtype="int16"), rate=16000)
        self.assertEqual(calls, [None])

    def test_configured_beam_size_reaches_both_passes(self):
        """Beam width is a turn-latency knob (greedy measured ~10% faster per turn), so it must not
        silently fall back to faster-whisper's default of 5 — least of all on the second pass."""
        va, calls = self._va([
            ("Rwy'n hoffi coffi", _info("cy", [("cy", 0.5), ("tl", 0.3), ("en", 0.1)])),
            ("gusto ko ng kape", _info("tl", [("tl", 0.9)])),
        ])
        with patch("builtins.print"):
            va._transcribe(np.ones((16000, 1), dtype="int16"), rate=16000)
        self.assertEqual(calls, [None, "tl"], "expected the two-pass path for this fixture")
        self.assertEqual(va._recorded_beam_sizes, [WHISPER_BEAM_SIZE, WHISPER_BEAM_SIZE])

    def test_decoder_bias_reaches_both_passes(self):
        """WHISPER_INITIAL_PROMPT is how "DEVCON" gets spelled correctly in the first place — a
        second pass that drops it hands the fuzzy matcher a mishearing to repair for nothing."""
        va, calls = self._va([
            ("Rwy'n hoffi coffi", _info("cy", [("cy", 0.5), ("tl", 0.3), ("en", 0.1)])),
            ("gusto ko ng kape", _info("tl", [("tl", 0.9)])),
        ])
        with patch("builtins.print"):
            va._transcribe(np.ones((16000, 1), dtype="int16"), rate=16000)
        self.assertEqual(va._recorded_prompts, [WHISPER_INITIAL_PROMPT, WHISPER_INITIAL_PROMPT])

    def test_the_scan_path_is_never_biased(self):
        # The wake tier runs on a weak model over overheard room noise, where priming it with
        # DEVCON vocabulary shows up as invented DEVCON talk. It only needs "hey kai".
        va, calls = self._va([("hey kai", _info("en", [("en", 0.9)]))])
        va._scan_model = va._whisper_model
        with patch("builtins.print"):
            va._transcribe(np.ones((16000, 1), dtype="int16"), rate=16000, scan=True)
        self.assertEqual(va._recorded_prompts, [None])

    def test_disallowed_language_is_retranscribed_as_the_best_allowed(self):
        va, calls = self._va([
            ("Rwy'n hoffi coffi", _info("cy", [("cy", 0.5), ("tl", 0.3), ("en", 0.1)])),
            ("gusto ko ng kape", _info("tl", [("tl", 0.9)])),
        ])
        with patch("builtins.print"):
            text = va._transcribe(np.ones((16000, 1), dtype="int16"), rate=16000)
        self.assertEqual(calls, [None, "tl"], "must redo the pass forced to the best allowed language")
        self.assertEqual(text, "gusto ko ng kape", "the Welsh transcript must not be kept")

    def test_norwegian_falls_back_to_english_when_that_scores_higher(self):
        va, calls = self._va([
            ("Han er ikke en annen", _info("nn", [("nn", 0.6), ("en", 0.2), ("tl", 0.01)])),
            ("I don't know what it is", _info("en", [("en", 0.9)])),
        ])
        with patch("builtins.print"):
            text = va._transcribe(np.ones((16000, 1), dtype="int16"), rate=16000)
        self.assertEqual(calls, [None, "en"])
        self.assertEqual(text, "I don't know what it is")

    def test_an_explicitly_forced_language_is_never_second_guessed(self):
        va, calls = self._va([("hello", _info("cy", [("cy", 0.9)]))])
        with patch("ai.voice_assistant.WHISPER_LANGUAGE", "en"), patch("builtins.print"):
            va._transcribe(np.ones((16000, 1), dtype="int16"), rate=16000)
        self.assertEqual(calls, ["en"])

    def test_the_scan_path_is_pinned_and_not_second_guessed(self):
        # The scan only has to spot two English words, and it is latency-critical.
        va, calls = self._va([("hey guys", _info("cy", [("cy", 0.9)]))])
        va._scan_model = va._whisper_model
        with patch("ai.voice_assistant.WAKE_WHISPER_SCAN_LANGUAGE", "en"), patch("builtins.print"):
            va._transcribe(np.ones((16000, 1), dtype="int16"), rate=16000, scan=True)
        self.assertEqual(calls, ["en"])

    def test_empty_allow_list_restores_full_auto_detect(self):
        va, calls = self._va([("Rwy'n hoffi coffi", _info("cy", [("cy", 0.9)]))])
        with patch("ai.voice_assistant.WHISPER_LANGUAGES", ()), patch("builtins.print"):
            text = va._transcribe(np.ones((16000, 1), dtype="int16"), rate=16000)
        self.assertEqual(calls, [None])
        self.assertEqual(text, "Rwy'n hoffi coffi")

    def test_missing_probs_still_produces_a_usable_transcript(self):
        va, calls = self._va([
            ("garble", _info("cy", None)),
            ("what time is it", _info("en", None)),
        ])
        with patch("builtins.print"):
            text = va._transcribe(np.ones((16000, 1), dtype="int16"), rate=16000)
        self.assertEqual(calls, [None, "en"])
        self.assertEqual(text, "what time is it")


class TestTurnId(unittest.TestCase):
    """The dashboard appends a chat bubble when voice_turn_id changes. It must move exactly once per
    completed turn and never otherwise — inferring the same thing from voice_status reaching "done"
    caused three separate duplicate-bubble races."""

    def setUp(self):
        p = patch("ai.voice_assistant.tts.enabled", return_value=False)
        p.start()
        self.addCleanup(p.stop)

    def test_starts_at_zero(self):
        self.assertEqual(make_assistant().get_status()["voice_turn_id"], 0)

    def test_increments_once_per_successful_turn(self):
        va = make_assistant()
        with patch.object(va, "_transcribe", return_value="hi"), \
             patch.object(va, "_call_ollama", return_value="hello"):
            va._process(np.ones((10, 1), dtype="int16"))
        self.assertEqual(va.get_status()["voice_turn_id"], 1)
        with patch.object(va, "_transcribe", return_value="again"), \
             patch.object(va, "_call_ollama", return_value="sure"):
            va._process(np.ones((10, 1), dtype="int16"))
        self.assertEqual(va.get_status()["voice_turn_id"], 2)

    def test_increments_for_the_no_speech_reply(self):
        # That reply is shown to the user, so it is a turn as far as the chat log is concerned.
        va = make_assistant()
        with patch.object(va, "_transcribe", return_value="   "):
            va._process(np.ones((10, 1), dtype="int16"))
        self.assertEqual(va.get_status()["voice_turn_id"], 1)

    def test_increments_for_a_say_turn(self):
        va = make_assistant()
        with patch.object(va, "_call_ollama", return_value="reply"), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va.say("hello")
        self.assertEqual(va.get_status()["voice_turn_id"], 1)

    def test_does_not_increment_on_error(self):
        va = make_assistant()
        with patch.object(va, "_transcribe", side_effect=RuntimeError("whisper died")):
            va._process(np.ones((10, 1), dtype="int16"))
        self.assertEqual(va.get_status()["voice_turn_id"], 0)

    def test_does_not_increment_for_a_stale_turn(self):
        va = make_assistant()
        epoch = va.epoch

        def ollama(text):
            va.bump_epoch()
            return "a reply"

        with patch.object(va, "_transcribe", return_value="hi"), \
             patch.object(va, "_call_ollama", side_effect=ollama):
            va._process(np.ones((10, 1), dtype="int16"), epoch=epoch)
        self.assertEqual(va.get_status()["voice_turn_id"], 0)

    def test_is_unaffected_by_status_churn(self):
        # The whole point: clearing/reprojecting status must never look like a new turn.
        va = make_assistant()
        with patch.object(va, "_transcribe", return_value="hi"), \
             patch.object(va, "_call_ollama", return_value="hello"):
            va._process(np.ones((10, 1), dtype="int16"))
        before = va.get_status()["voice_turn_id"]
        for _ in range(5):
            va.clear_turn_status()
            va._status = STATUS_DONE
            va.clear_turn_status()
        self.assertEqual(va.get_status()["voice_turn_id"], before)


class TestClearTurnStatus(unittest.TestCase):
    """The dashboard posts a chat bubble on the transition INTO "done". A stale "done" surfacing when
    a hands-free session ends therefore duplicates the last question and answer."""

    def test_retires_done_to_idle(self):
        va = make_assistant()
        va._status = STATUS_DONE
        va.clear_turn_status()
        self.assertEqual(va.get_status()["voice_status"], STATUS_IDLE)

    def test_retires_error_too(self):
        va = make_assistant()
        va._status = STATUS_ERROR
        va.clear_turn_status()
        self.assertEqual(va.get_status()["voice_status"], STATUS_IDLE)

    def test_keeps_the_transcript_and_response_for_display(self):
        va = make_assistant()
        va._status, va._transcript, va._response = STATUS_DONE, "what time is it", "It's 3pm"
        va.clear_turn_status()
        status = va.get_status()
        self.assertEqual(status["voice_transcript"], "what time is it")
        self.assertEqual(status["voice_response"], "It's 3pm")

    def test_does_not_disturb_an_in_flight_turn(self):
        va = make_assistant()
        for live in ("recording", "transcribing", "thinking"):
            va._status = live
            va.clear_turn_status()
            self.assertEqual(va.get_status()["voice_status"], live)


class _FakeMic:
    """Stand-in for the shared always-open stream owned by ai/session.py."""

    def __init__(self, audio=None, rate=16000, arm_ok=True):
        self.audio = np.zeros((0, 1), dtype="int16") if audio is None else audio
        self.rate = rate
        self.arm_ok = arm_ok
        self.armed = 0
        self.harvested = 0
        self.last_preroll = None

    def arm_utterance(self, preroll=False):
        self.last_preroll = preroll
        if self.arm_ok:
            self.armed += 1
        return self.arm_ok

    def harvest_utterance(self):
        self.harvested += 1
        return self.audio, self.rate


class TestSharedMicCapture(unittest.TestCase):
    """With a session owning the stream, VoiceAssistant must never open one of its own — the raw I2S
    hw device admits a single opener, and a second one fails as intermittent silent capture."""

    def test_start_recording_arms_the_shared_mic_and_opens_no_stream(self):
        va = make_assistant()
        mic = _FakeMic()
        va.attach_mic(mic)
        with patch("ai.voice_assistant.sd.InputStream") as mock_stream_cls, \
             patch("ai.voice_assistant.tts.stop"):
            result = va.start_recording()
        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(mic.armed, 1)
        self.assertTrue(mic.last_preroll)
        mock_stream_cls.assert_not_called()
        self.assertEqual(va.get_status()["voice_status"], STATUS_RECORDING)

    def test_start_recording_skips_device_resolution_entirely(self):
        va = make_assistant()
        va.attach_mic(_FakeMic())
        with patch("ai.voice_assistant.resolve_capture_device") as mock_resolve, \
             patch("ai.voice_assistant.tts.stop"):
            va.start_recording()
        mock_resolve.assert_not_called()

    def test_arm_failure_reports_the_same_error_shape_as_before(self):
        va = make_assistant()
        va.attach_mic(_FakeMic(arm_ok=False))
        with patch("ai.voice_assistant.tts.stop"):
            result = va.start_recording()
        self.assertIn("error", result)
        self.assertEqual(va.get_status()["voice_status"], STATUS_ERROR)

    def test_stop_recording_harvests_and_passes_the_rate_through(self):
        va = make_assistant()
        mic = _FakeMic(audio=np.ones(1000, dtype="int16"), rate=16000)
        va.attach_mic(mic)
        with patch("ai.voice_assistant.tts.stop"):
            va.start_recording()
        with patch("ai.voice_assistant.threading.Thread") as mock_thread_cls:
            mock_thread_cls.return_value = MagicMock()
            va.stop_recording()
        self.assertEqual(mic.harvested, 1)
        self.assertEqual(mock_thread_cls.call_args.kwargs["kwargs"]["rate"], 16000)

    def test_stop_recording_never_touches_the_shared_stream(self):
        va = make_assistant()
        va.attach_mic(_FakeMic())
        sentinel = MagicMock()
        va._stream = sentinel   # a session's stream must survive a push-to-talk stop
        with patch("ai.voice_assistant.tts.stop"):
            va.start_recording()
        with patch("ai.voice_assistant.threading.Thread", return_value=MagicMock()):
            va.stop_recording()
        sentinel.close.assert_not_called()
        sentinel.stop.assert_not_called()


class TestLegacyCaptureIsBounded(unittest.TestCase):
    def test_buffer_stops_growing_past_the_hard_cap(self):
        # Nothing auto-stops a push-to-talk recording, so a stuck button used to grow this forever.
        from config.wake import CAPTURE_HARD_CAP_S

        va = make_assistant()
        va._capture_rate = 16000
        chunk = np.zeros((16000, 1), dtype="int16")   # 1 s each
        for _ in range(int(CAPTURE_HARD_CAP_S) + 30):
            va._on_audio_chunk(chunk, len(chunk), None, None)
        self.assertLessEqual(va._audio_samples, int(CAPTURE_HARD_CAP_S * 16000) + len(chunk))

    def test_keeps_the_most_recent_audio(self):
        va = make_assistant()
        va._capture_rate = 100   # tiny cap so the trim is easy to observe
        for value in range(60):
            va._on_audio_chunk(np.full((100, 1), value, dtype="int16"), 100, None, None)
        newest = va._audio_chunks[-1]
        self.assertEqual(int(newest[0, 0]), 59)


class TestEpoch(unittest.TestCase):
    def test_starts_at_zero_and_increments(self):
        va = make_assistant()
        self.assertEqual(va.epoch, 0)
        self.assertEqual(va.bump_epoch(), 1)
        self.assertEqual(va.epoch, 1)

    def test_none_is_always_current(self):
        va = make_assistant()
        va.bump_epoch()
        self.assertTrue(va._epoch_ok(None), "legacy unversioned callers must keep working")

    def test_stale_epoch_is_detected(self):
        va = make_assistant()
        stale = va.epoch
        va.bump_epoch()
        self.assertFalse(va._epoch_ok(stale))

    def test_reset_history_invalidates_in_flight_turns(self):
        # Otherwise a reply already inside _call_ollama appends itself after the clear, handing the
        # next person one turn of the last person's conversation.
        va = make_assistant()
        va._history = [{"role": "user", "content": "secret"}]
        epoch = va.epoch
        va.reset_history()
        self.assertEqual(va._history, [])
        self.assertFalse(va._epoch_ok(epoch))


class TestProcessEpochGuards(unittest.TestCase):
    def setUp(self):
        p = patch("ai.voice_assistant.tts.enabled", return_value=False)
        p.start()
        self.addCleanup(p.stop)

    def _assistant(self, transcript="hi"):
        va = make_assistant()
        va._whisper_model = MagicMock()
        va._whisper_model.transcribe.return_value = ([make_segment(transcript)], None)
        return va

    def test_stale_after_transcription_writes_nothing(self):
        va = self._assistant()
        epoch = va.epoch

        def transcribe(audio, rate=None):
            va.bump_epoch()   # a session reset lands while Whisper is running
            return "hello"

        with patch.object(va, "_transcribe", side_effect=transcribe), \
             patch.object(va, "_call_ollama") as mock_ollama:
            va._process(np.ones((10, 1), dtype="int16"), epoch=epoch)
        mock_ollama.assert_not_called()
        self.assertEqual(va.get_status()["voice_transcript"], "")

    def test_stale_after_ollama_does_not_append_history(self):
        va = self._assistant()
        epoch = va.epoch

        def ollama(text):
            va.bump_epoch()   # ~50 s cold load is plenty of time for the session to end
            return "a reply"

        with patch.object(va, "_transcribe", return_value="hello"), \
             patch.object(va, "_call_ollama", side_effect=ollama), \
             patch.object(va, "_speak") as mock_speak:
            va._process(np.ones((10, 1), dtype="int16"), epoch=epoch)
        self.assertEqual(va._history, [])
        mock_speak.assert_not_called()
        self.assertEqual(va.get_status()["voice_response"], "")

    def test_current_epoch_completes_normally(self):
        va = self._assistant()
        with patch.object(va, "_transcribe", return_value="hello"), \
             patch.object(va, "_call_ollama", return_value="a reply"):
            va._process(np.ones((10, 1), dtype="int16"), epoch=va.epoch)
        self.assertEqual(va.get_status()["voice_response"], "a reply")
        self.assertEqual(len(va._history), 2)

    def test_rate_is_forwarded_to_transcribe(self):
        va = self._assistant()
        with patch.object(va, "_transcribe", return_value="") as mock_tr:
            va._process(np.ones((10, 1), dtype="int16"), rate=48000)
        self.assertEqual(mock_tr.call_args.kwargs["rate"], 48000)


class TestOnDoneCallback(unittest.TestCase):
    def setUp(self):
        p = patch("ai.voice_assistant.tts.enabled", return_value=False)
        p.start()
        self.addCleanup(p.stop)

    def _run_turn(self, **transcribe_kw):
        # NB: not named _outcome — unittest.TestCase already owns that attribute.
        va = make_assistant()
        seen = []
        with patch.object(va, "_transcribe", **transcribe_kw), \
             patch.object(va, "_call_ollama", return_value="r"):
            va._process(np.ones((10, 1), dtype="int16"), epoch=va.epoch,
                        on_done=lambda ep, outcome: seen.append(outcome))
        return va, seen

    def test_reports_done_on_a_real_reply(self):
        _, seen = self._run_turn(return_value="hello")
        self.assertEqual(seen, ["done"])

    def test_reports_empty_on_no_transcript(self):
        _, seen = self._run_turn(return_value="   ")
        self.assertEqual(seen, ["empty"])

    def test_reports_error_on_failure(self):
        _, seen = self._run_turn(side_effect=RuntimeError("whisper died"))
        self.assertEqual(seen, ["error"])

    def test_reports_stale_when_the_session_moved_on(self):
        va = make_assistant()
        seen = []
        epoch = va.epoch
        with patch.object(va, "_transcribe", side_effect=lambda a, rate=None: va.bump_epoch() and "hi"):
            va._process(np.ones((10, 1), dtype="int16"), epoch=epoch,
                        on_done=lambda ep, outcome: seen.append(outcome))
        self.assertEqual(seen, ["stale"])

    def test_empty_transcript_is_not_spoken_twice(self):
        # The session speaks its own cached "didn't catch that", so _process must not also say it.
        va = make_assistant()
        with patch.object(va, "_transcribe", return_value=""), \
             patch.object(va, "_speak") as mock_speak:
            va._process(np.ones((10, 1), dtype="int16"), on_done=lambda ep, o: None)
        mock_speak.assert_not_called()

    def test_empty_transcript_is_still_spoken_without_a_session(self):
        va = make_assistant()
        with patch.object(va, "_transcribe", return_value=""), \
             patch.object(va, "_speak") as mock_speak:
            va._process(np.ones((10, 1), dtype="int16"))
        mock_speak.assert_called_once()

    def test_broken_callback_does_not_kill_the_turn_thread(self):
        va = make_assistant()
        with patch.object(va, "_transcribe", return_value="hello"), \
             patch.object(va, "_call_ollama", return_value="r"), \
             patch("builtins.print"):
            va._process(np.ones((10, 1), dtype="int16"),
                        on_done=lambda ep, o: (_ for _ in ()).throw(ValueError("bad")))
        self.assertEqual(va.get_status()["voice_response"], "r")


class TestProcessUtterance(unittest.TestCase):
    def test_spawns_a_worker_with_the_epoch_and_rate(self):
        va = make_assistant()
        with patch("ai.voice_assistant.threading.Thread") as mock_thread_cls:
            mock_thread_cls.return_value = MagicMock()
            result = va.process_utterance(np.ones(100, dtype="int16"), rate=16000, epoch=7)
        self.assertEqual(result, {"status": "ok"})
        kwargs = mock_thread_cls.call_args.kwargs["kwargs"]
        self.assertEqual(kwargs["rate"], 16000)
        self.assertEqual(kwargs["epoch"], 7)

    def test_rejected_while_thinking(self):
        va = make_assistant()
        va._status = "thinking"
        result = va.process_utterance(np.ones(100, dtype="int16"), rate=16000)
        self.assertIn("error", result)

    def test_clears_the_previous_turns_text(self):
        va = make_assistant()
        va._transcript, va._response, va._error = "old", "old", "old"
        with patch("ai.voice_assistant.threading.Thread", return_value=MagicMock()):
            va.process_utterance(np.ones(100, dtype="int16"), rate=16000)
        status = va.get_status()
        self.assertEqual(status["voice_transcript"], "")
        self.assertEqual(status["voice_response"], "")
        self.assertEqual(status["voice_error"], "")


class TestMicMuted(unittest.TestCase):
    """The self-hearing gate. With the mic always open, this is the only thing between Kai and
    answering his own reply."""

    def setUp(self):
        for name, kw in (("is_playing", {"return_value": False}),
                         ("quiet_since", {"return_value": float("inf")})):
            p = patch(f"ai.voice_assistant.tts.{name}", **kw)
            setattr(self, f"mock_{name}", p.start())
            self.addCleanup(p.stop)

    def test_open_when_nothing_is_happening(self):
        self.assertFalse(make_assistant().mic_muted(now=100.0))

    def test_muted_while_playback_runs(self):
        self.mock_is_playing.return_value = True
        self.assertTrue(make_assistant().mic_muted(now=100.0))

    def test_muted_during_the_settle_tail(self):
        from config.wake import TTS_TAIL_MUTE_S

        self.mock_quiet_since.return_value = TTS_TAIL_MUTE_S / 2
        self.assertTrue(make_assistant().mic_muted(now=100.0),
                        "paplay exits before the amp goes quiet — the tail covers that gap")

    def test_open_once_the_tail_expires(self):
        from config.wake import TTS_TAIL_MUTE_S

        self.mock_quiet_since.return_value = TTS_TAIL_MUTE_S + 0.01
        self.assertFalse(make_assistant().mic_muted(now=100.0))

    def test_muted_during_synthesis_before_any_audio_exists(self):
        # The window voice_speaking cannot see: the jaw envelope is only set after synth returns.
        va = make_assistant()
        va._tts_active = True
        self.assertTrue(va.mic_muted(now=100.0))

    def test_muted_until_the_computed_audio_end(self):
        va = make_assistant()
        va._gate_until = 150.0
        self.assertTrue(va.mic_muted(now=149.9))
        self.assertFalse(va.mic_muted(now=150.1))

    def test_speak_gates_before_the_worker_runs(self):
        va = make_assistant()
        with patch("ai.voice_assistant.tts.enabled", return_value=True), \
             patch("ai.voice_assistant.threading.Thread") as mock_thread_cls:
            mock_thread_cls.return_value = MagicMock()   # worker never runs
            va._speak("Hello there.")
        self.assertTrue(va._tts_active, "the gate must close before Piper is even started")

    def test_speak_sets_the_gate_from_the_wav_duration(self):
        from config.voice import TTS_PLAYBACK_LEAD_S
        from config.wake import TTS_TAIL_MUTE_S

        va = make_assistant()
        with patch("ai.voice_assistant.tts.enabled", return_value=True), \
             patch("ai.voice_assistant.tts.stop"), \
             patch("ai.voice_assistant.tts.synthesize", return_value="/tmp/kai_tts.wav"), \
             patch("ai.voice_assistant.tts.wav_duration", return_value=4.0), \
             patch("ai.voice_assistant.tts.play"), \
             patch("ai.voice_assistant.time.monotonic", return_value=100.0), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va._speak("One two three.")
        # The lead is in here because the 4 s of audio does not begin at the play() call — the gate
        # used to open TTS_PLAYBACK_LEAD_S early, into audio that was still coming out.
        self.assertAlmostEqual(va._gate_until,
                               100.0 + TTS_PLAYBACK_LEAD_S + 4.0 + TTS_TAIL_MUTE_S)

    def test_gate_is_released_even_when_the_worker_bails_out(self):
        va = make_assistant()
        with patch("ai.voice_assistant.tts.enabled", return_value=True), \
             patch("ai.voice_assistant.tts.stop"), \
             patch("ai.voice_assistant.tts.synthesize", side_effect=RuntimeError("piper died")), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            with self.assertRaises(RuntimeError):
                va._speak("Hello.")
        self.assertFalse(va._tts_active, "a stuck gate would deafen Kai permanently")


class TestSpeakStaleEpoch(unittest.TestCase):
    def test_abandoned_reply_is_never_played(self):
        # The bug this closes: synthesis takes 0.5-1.5 s, and tts.stop() alone only killed playback,
        # so a worker cancelled mid-Piper still spoke into whatever session came next.
        va = make_assistant()
        epoch = va.epoch

        def synth(text, length_scale=None):
            va.bump_epoch()
            return "/tmp/kai_tts.wav"

        with patch("ai.voice_assistant.tts.enabled", return_value=True), \
             patch("ai.voice_assistant.tts.stop"), \
             patch("ai.voice_assistant.tts.synthesize", side_effect=synth), \
             patch("ai.voice_assistant.tts.play") as mock_play, \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va._speak("A reply nobody is waiting for.", epoch=epoch)
        mock_play.assert_not_called()
        self.assertIsNone(va._speak_start, "and the jaw must not animate either")

    def test_current_epoch_still_plays(self):
        va = make_assistant()
        with patch("ai.voice_assistant.tts.enabled", return_value=True), \
             patch("ai.voice_assistant.tts.stop"), \
             patch("ai.voice_assistant.tts.synthesize", return_value="/tmp/kai_tts.wav"), \
             patch("ai.voice_assistant.tts.wav_duration", return_value=2.0), \
             patch("ai.voice_assistant.tts.play") as mock_play, \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va._speak("A wanted reply.", epoch=va.epoch)
        mock_play.assert_called_once()

    def test_unversioned_speak_always_plays(self):
        va = make_assistant()
        va.bump_epoch()
        with patch("ai.voice_assistant.tts.enabled", return_value=True), \
             patch("ai.voice_assistant.tts.stop"), \
             patch("ai.voice_assistant.tts.synthesize", return_value="/tmp/kai_tts.wav"), \
             patch("ai.voice_assistant.tts.wav_duration", return_value=2.0), \
             patch("ai.voice_assistant.tts.play") as mock_play, \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va._speak("Verbatim say() path.")
        mock_play.assert_called_once()


class TestSpeechOwnership(unittest.TestCase):
    """One speaker, one _tts_active boolean, and workers that finish out of order.

    Heard on the robot 2026-08-09 as "all the fillers are being spoken at once". A 7.5 s filler
    opener was still playing when the 2.8 s reply started; the opener's worker then cleared
    _tts_active out from under the reply. speech_in_flight() went False mid-answer, which dropped
    SPEAKING straight to COOLDOWN and told the filler loop nothing was playing, so it started more
    lines on top of the answer.
    """

    def test_a_late_worker_does_not_report_silence_for_a_newer_line(self):
        va = make_assistant()
        with patch("ai.voice_assistant.tts.stop"):
            first = va._begin_speech()
            second = va._begin_speech()
            va._end_speech(first)                  # the older line finishes last
            self.assertTrue(va.speech_in_flight(),
                            "an older line clearing the flag makes a live answer look finished")
            va._end_speech(second)
            self.assertFalse(va.speech_in_flight())

    def test_starting_a_line_cuts_whatever_is_still_playing(self):
        # tts.play() starts a SECOND paplay rather than replacing the first, so without this a
        # filler that outlives its turn keeps talking straight over the answer it was covering.
        va = make_assistant()
        with patch("ai.voice_assistant.tts.stop") as mock_stop:
            va._begin_speech()
        mock_stop.assert_called_once()

    def test_the_reply_cuts_a_filler_that_is_still_going(self):
        # The end-to-end shape, through the two public speech paths rather than the helpers.
        va = make_assistant()
        with patch("ai.voice_assistant.tts.wav_duration", return_value=7.5), \
             patch("ai.voice_assistant.tts.play"), \
             patch("ai.voice_assistant.tts.stop") as mock_stop, \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va.speak_wav("/tmp/kai_canned_filler_op_tl_0.wav", "opener", epoch=va.epoch)
            mock_stop.reset_mock()
            va.speak_wav("/tmp/kai_canned_ack.wav", "Yes?", epoch=va.epoch)
        mock_stop.assert_called_once()


class TestSpeakWav(unittest.TestCase):
    """The cached wake acknowledgement."""

    def test_plays_the_cached_file_and_syncs_the_jaw(self):
        va = make_assistant()
        # The span has to be faked alongside the duration: this names the ack WAV's REAL path, and
        # on the robot that file exists — so an unpatched wav_speech_span would scan it off disk and
        # this would assert against whatever Piper last wrote there.
        with patch("ai.voice_assistant.tts.wav_duration", return_value=0.6), \
             patch("ai.voice_assistant.tts.wav_speech_span", return_value=(0.0, 0.6)), \
             patch("ai.voice_assistant.tts.play") as mock_play, \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va.speak_wav("/tmp/kai_ack/kai_canned_ack.wav", "Yes?", epoch=va.epoch)
        mock_play.assert_called_once_with("/tmp/kai_ack/kai_canned_ack.wav")
        self.assertIsNotNone(va._speak_start)
        self.assertAlmostEqual(va._speak_segments[-1][1], 0.6, places=6)

    def test_does_not_synthesize_anything(self):
        va = make_assistant()
        with patch("ai.voice_assistant.tts.synthesize") as mock_synth, \
             patch("ai.voice_assistant.tts.wav_duration", return_value=0.6), \
             patch("ai.voice_assistant.tts.play"), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va.speak_wav("/tmp/x.wav", "Yes?")
        mock_synth.assert_not_called()

    def test_leaves_status_and_response_untouched(self):
        # Routing the ack through say() would post a "Kai: Yes?" chat bubble on every single wake.
        va = make_assistant()
        with patch("ai.voice_assistant.tts.wav_duration", return_value=0.6), \
             patch("ai.voice_assistant.tts.play"), \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va.speak_wav("/tmp/x.wav", "Yes?")
        status = va.get_status()
        self.assertEqual(status["voice_status"], STATUS_IDLE)
        self.assertEqual(status["voice_response"], "")

    def test_stale_epoch_is_silent(self):
        va = make_assistant()
        epoch = va.epoch
        va.bump_epoch()
        with patch("ai.voice_assistant.tts.wav_duration", return_value=0.6), \
             patch("ai.voice_assistant.tts.play") as mock_play, \
             patch("ai.voice_assistant.threading.Thread", _InlineThread):
            va.speak_wav("/tmp/x.wav", "Yes?", epoch=epoch)
        mock_play.assert_not_called()


if __name__ == '__main__':
    unittest.main()
