import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai import tts


def _fake_proc(returncode=0, stderr=b"", alive=False):
    """A stand-in for a Popen: communicate() returns immediately, poll() reports alive/dead."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.returncode = returncode
    proc.communicate.return_value = (b"", stderr)
    proc.poll.return_value = None if alive else returncode
    return proc


class _ResetProcState(unittest.TestCase):
    """tts holds module-level subprocess handles; every test starts from a clean slate."""

    def setUp(self):
        tts._synth_proc = None
        tts._current_proc = None
        tts._last_end = 0.0
        # Pretend the output card profile has already been asserted. These tests patch Popen but not
        # subprocess.run, so a False here would let play()'s lazy assert shell out to a real pactl.
        # TestOutputCardProfile clears it deliberately, with subprocess.run patched.
        tts._profile_applied = True

    tearDown = setUp


class TestCleanForSpeech(unittest.TestCase):
    def test_strips_emoji(self):
        self.assertEqual(tts.clean_for_speech("hello 😀 there"), "hello there")

    def test_keeps_smart_punctuation(self):
        # General Punctuation is deliberately excluded from the strip ranges
        self.assertEqual(tts.clean_for_speech("it's — really"), "it's — really")

    def test_emoji_only_returns_empty(self):
        self.assertEqual(tts.clean_for_speech("😀🚀"), "")

    def test_none_returns_empty(self):
        self.assertEqual(tts.clean_for_speech(None), "")

    def test_strips_markdown_emphasis(self):
        # Piper voices these as words: "**world**" measured 1.90s longer than "world".
        self.assertEqual(tts.clean_for_speech("hello **bold** and *italic*"),
                         "hello bold and italic")

    def test_strips_backticks(self):
        self.assertEqual(tts.clean_for_speech("say `hello` now"), "say hello now")

    def test_strips_wrapped_aside(self):
        # The shape gemma actually produced on a live turn.
        self.assertEqual(tts.clean_for_speech("* I don't have the weather. *"),
                         "I don't have the weather.")

    def test_strips_leading_list_and_heading_markers(self):
        self.assertEqual(tts.clean_for_speech("- one\n- two"), "one two")
        self.assertEqual(tts.clean_for_speech("## Heading"), "Heading")

    def test_keeps_hyphenated_words_and_mid_sentence_dashes(self):
        # Only line-leading markers go — a hyphen inside a word is speakable text.
        self.assertEqual(tts.clean_for_speech("push-to-talk still works"),
                         "push-to-talk still works")
        self.assertEqual(tts.clean_for_speech("wait - what?"), "wait - what?")

    def test_keeps_underscores(self):
        # Measured silent through Piper, so there is nothing to buy by removing them.
        self.assertEqual(tts.clean_for_speech("file_name here"), "file_name here")

    def test_markdown_only_returns_empty(self):
        self.assertEqual(tts.clean_for_speech("**"), "")


class TestClampForSpeech(unittest.TestCase):
    """Kai can't hear while talking, so an over-long reply is an over-long deaf spell."""

    def test_short_text_untouched(self):
        self.assertEqual(tts.clamp_for_speech("Hello there.", 400), "Hello there.")

    def test_zero_max_disables_clamping(self):
        long = "x" * 1000
        self.assertEqual(tts.clamp_for_speech(long, 0), long)

    def test_prefers_a_sentence_boundary(self):
        text = "First sentence here. Second sentence here. Third one here."
        out = tts.clamp_for_speech(text, 45)
        self.assertEqual(out, "First sentence here. Second sentence here.")

    def test_falls_back_to_a_word_boundary(self):
        # One long sentence: no usable break, so don't cut mid-word.
        text = "alpha bravo charlie delta echo foxtrot golf hotel india"
        out = tts.clamp_for_speech(text, 20)
        self.assertTrue(out.endswith("…"))
        self.assertLessEqual(len(out), 21)
        self.assertNotIn("brav…", out)

    def test_ignores_a_boundary_that_would_lose_most_of_the_reply(self):
        text = "Hi. " + "a long continuation without any sentence break at all here"
        out = tts.clamp_for_speech(text, 40)
        self.assertNotEqual(out, "Hi.", "a 4-char sentence must not swallow a 40-char budget")

    def test_never_exceeds_the_budget_by_more_than_the_ellipsis(self):
        for n in (5, 10, 50, 200):
            self.assertLessEqual(len(tts.clamp_for_speech("word " * 200, n)), n + 1)

    def test_none_is_empty(self):
        self.assertEqual(tts.clamp_for_speech(None, 400), "")


class TestLiveSettings(_ResetProcState):
    """TTS reads its dashboard knobs at the point of use, so a change applies to the very next thing
    Kai says — no restart, no engine reload."""

    # Every knob _run_piper pulls. Kept as one dict so a new prosody parameter fails loudly here
    # (KeyError) rather than silently never being passed.
    LIVE = {"tts_volume": 1.75, "tts_length_scale": 1.0, "tts_enabled": True,
            "tts_sentence_silence": 0.35, "tts_noise_scale": 0.667, "tts_noise_w": 0.8}

    @staticmethod
    def _argv_after(popen, flag):
        argv = popen.call_args[0][0]
        return argv[argv.index(flag) + 1]

    def test_enabled_follows_the_setting(self):
        with patch("ai.tts.voice_model_path", return_value=Path(__file__)):
            with patch("ai.tts.settings.get", return_value=False):
                self.assertFalse(tts.enabled())
            with patch("ai.tts.settings.get", return_value=True):
                self.assertTrue(tts.enabled())

    def test_piper_gets_the_live_rate(self):
        proc = _fake_proc()
        live = {**self.LIVE, "tts_length_scale": 0.8}
        with patch("ai.tts.voice_model_path", return_value=Path(__file__)), \
             patch("ai.tts.settings.get", side_effect=lambda name: live[name]), \
             patch("subprocess.Popen", return_value=proc) as popen, \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.stat", return_value=MagicMock(st_size=1024)):
            tts._run_piper("hi", Path("/tmp/x.wav"))
        self.assertEqual(self._argv_after(popen, "--length-scale"), "0.8")

    def test_piper_gets_the_live_prosody_parameters(self):
        """The three parameters that were never passed before 2026-08-10. --sentence-silence is the
        one that matters: Piper's default is 0, which is why Kai used to run four sentences together
        without a breath. See config/voice.py for the measurements."""
        proc = _fake_proc()
        live = {**self.LIVE, "tts_sentence_silence": 0.4, "tts_noise_scale": 0.7,
                "tts_noise_w": 1.1}
        with patch("ai.tts.voice_model_path", return_value=Path(__file__)), \
             patch("ai.tts.settings.get", side_effect=lambda name: live[name]), \
             patch("subprocess.Popen", return_value=proc) as popen, \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.stat", return_value=MagicMock(st_size=1024)):
            tts._run_piper("hi", Path("/tmp/x.wav"))
        self.assertEqual(self._argv_after(popen, "--sentence-silence"), "0.4")
        self.assertEqual(self._argv_after(popen, "--noise-scale"), "0.7")
        self.assertEqual(self._argv_after(popen, "--noise-w-scale"), "1.1")

    def test_the_flag_spellings_are_the_ones_this_piper_accepts(self):
        """Pinned deliberately. These were read off `python3 -m piper --help` on the robot against
        piper-tts 1.4.2; Piper rejects an unknown flag by exiting non-zero, which _run_piper reports
        as a failed synthesis — i.e. a wrong spelling here makes EVERY reply silent, and the log says
        nothing about flags. If a piper upgrade renames one, this test is where it should break."""
        proc = _fake_proc()
        with patch("ai.tts.voice_model_path", return_value=Path(__file__)), \
             patch("ai.tts.settings.get", side_effect=lambda name: self.LIVE[name]), \
             patch("subprocess.Popen", return_value=proc) as popen, \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.stat", return_value=MagicMock(st_size=1024)):
            tts._run_piper("hi", Path("/tmp/x.wav"))
        argv = popen.call_args[0][0]
        for flag in ("--length-scale", "--sentence-silence", "--noise-scale", "--noise-w-scale"):
            self.assertIn(flag, argv)

    def test_normalisation_is_skipped_only_when_sox_will_do_it_instead(self):
        """Piper normalises every sentence to full scale, which is why no two sentences Kai says ever
        differed in peak level. Handing that to sox's single `gain -n -1` keeps the relative
        differences — but only if sox is actually in the chain, since raw Piper output is ~10 dB
        quieter. TTS_POST_PROCESS off must therefore leave Piper normalising."""
        for normalize, post, expected in ((False, True, True), (False, False, False),
                                          (True, True, False)):
            with self.subTest(normalize=normalize, post_process=post):
                proc = _fake_proc()
                with patch("ai.tts.TTS_PIPER_NORMALIZE", normalize), \
                     patch("ai.tts.TTS_POST_PROCESS", post), \
                     patch("ai.tts.voice_model_path", return_value=Path(__file__)), \
                     patch("ai.tts.settings.get", side_effect=lambda name: self.LIVE[name]), \
                     patch("subprocess.Popen", return_value=proc) as popen, \
                     patch("pathlib.Path.is_file", return_value=True), \
                     patch("pathlib.Path.stat", return_value=MagicMock(st_size=1024)):
                    tts._run_piper("hi", Path("/tmp/x.wav"))
                argv = popen.call_args[0][0]
                self.assertEqual("--no-normalize" in argv, expected)
                # Whatever happens, the output file must stay the last argument pair.
                self.assertEqual(argv[-2], "-f")

    def test_a_length_scale_override_leaves_the_other_parameters_live(self):
        """synthesize_to_duration drives --length-scale directly to fit a cached line to a target
        length. That override must not quietly reset the prosody parameters to Piper's defaults, or a
        fitted filler line would come out with no sentence pause while every reply had one."""
        proc = _fake_proc()
        with patch("ai.tts.voice_model_path", return_value=Path(__file__)), \
             patch("ai.tts.settings.get", side_effect=lambda name: self.LIVE[name]), \
             patch("subprocess.Popen", return_value=proc) as popen, \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.stat", return_value=MagicMock(st_size=1024)):
            tts._run_piper("hi", Path("/tmp/x.wav"), length_scale=2.5)
        self.assertEqual(self._argv_after(popen, "--length-scale"), "2.5")
        self.assertEqual(self._argv_after(popen, "--sentence-silence"), "0.35")

    def test_synthesis_does_not_apply_volume(self):
        # Piper's --volume is normalised straight back out by TTS_POST_EFFECTS (`gain -n -1`), and
        # values above ~1.2 clip the raw audio. Volume belongs at playback — see play().
        proc = _fake_proc()
        live = self.LIVE
        with patch("ai.tts.voice_model_path", return_value=Path(__file__)), \
             patch("ai.tts.settings.get", side_effect=lambda name: live[name]), \
             patch("subprocess.Popen", return_value=proc) as popen, \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.stat", return_value=MagicMock(st_size=1024)):
            tts._run_piper("hi", Path("/tmp/x.wav"))
        self.assertNotIn("--volume", popen.call_args[0][0])

    @staticmethod
    def _paplay_argv(popen):
        """The paplay invocations only.

        Deliberately not popen.call_args (the LAST call): the suite runs daemon threads that also
        shell out (session ack prewarm, TTS workers), so under load one of those can land inside this
        patch and make the last call somebody else's.
        """
        return [c.args[0] for c in popen.call_args_list
                if c.args and c.args[0] and c.args[0][0] == "paplay"]

    def test_playback_applies_the_live_volume_to_the_sink_input(self):
        proc = _fake_proc()
        proc.returncode = 0
        with patch("ai.tts.settings.get", return_value=1.5), \
             patch("subprocess.Popen", return_value=proc) as popen:
            tts.play(Path("/tmp/x.wav"))
        calls = self._paplay_argv(popen)
        self.assertTrue(calls, "play() must invoke paplay")
        # PA_VOLUME_NORM is 65536, so 1.5 -> 98304.
        self.assertIn("--volume=98304", calls[0])

    def test_playback_volume_is_never_negative(self):
        proc = _fake_proc()
        proc.returncode = 0
        with patch("ai.tts.settings.get", return_value=-3.0), \
             patch("subprocess.Popen", return_value=proc) as popen:
            tts.play(Path("/tmp/x.wav"))
        calls = self._paplay_argv(popen)
        self.assertTrue(calls, "play() must invoke paplay")
        self.assertIn("--volume=0", calls[0])


class TestSoxChain(unittest.TestCase):
    """The effect ORDER is the whole design (see ai/tts._sox_chain), and it is not visible from the
    config lists on their own — so it is pinned here."""

    LOUD = ["compand", "0.3,1", "6:-70,-60,-20", "-5", "-90", "0.2", "gain", "-n", "-1"]

    def test_highpass_runs_before_the_compand(self):
        # After the compressor has spent its headroom on sub-90 Hz energy, removing it is too late.
        with patch("ai.tts.TTS_POST_HIGHPASS", ["highpass", "90"]), \
             patch("ai.tts.TTS_POST_EFFECTS", self.LOUD), \
             patch("ai.tts.TTS_POST_ROOM", []):
            chain = tts._sox_chain()
        self.assertLess(chain.index("highpass"), chain.index("compand"))

    def test_room_runs_after_the_compand_and_is_renormalised(self):
        # Reverb adds energy after the loudness chain's own `gain -n -1`, so the peak must be pulled
        # back down or the output sits above -1 dB.
        with patch("ai.tts.TTS_POST_HIGHPASS", ["highpass", "90"]), \
             patch("ai.tts.TTS_POST_EFFECTS", self.LOUD), \
             patch("ai.tts.TTS_POST_ROOM", ["reverb", "18", "50", "28", "100", "0", "-4"]):
            chain = tts._sox_chain()
        self.assertGreater(chain.index("reverb"), chain.index("compand"))
        self.assertEqual(chain[-3:], ["gain", "-n", "-1"])
        self.assertGreater(len(chain) - 3, chain.index("reverb"))

    def test_both_empty_is_byte_identical_to_the_old_chain(self):
        """The revert path. Emptying both constants must reproduce the pre-2026-08-11 command line
        exactly, not merely something equivalent."""
        with patch("ai.tts.TTS_POST_HIGHPASS", []), \
             patch("ai.tts.TTS_POST_EFFECTS", self.LOUD), \
             patch("ai.tts.TTS_POST_ROOM", []):
            self.assertEqual(tts._sox_chain(), self.LOUD)

    def test_no_room_means_no_second_normalise(self):
        with patch("ai.tts.TTS_POST_HIGHPASS", ["highpass", "90"]), \
             patch("ai.tts.TTS_POST_EFFECTS", self.LOUD), \
             patch("ai.tts.TTS_POST_ROOM", []):
            self.assertEqual(tts._sox_chain().count("gain"), 1)


class TestOutputCardProfile(_ResetProcState):
    """PulseAudio flips this dongle to its digital (S/PDIF) profile on its own, which deletes
    TTS_SINK and makes every reply inaudible without raising anything. apply_output_profile() is the
    counter-measure, so it has to be both effective and incapable of costing us a reply itself."""

    @staticmethod
    def _pactl_argv(run):
        return [c.args[0] for c in run.call_args_list
                if c.args and c.args[0] and c.args[0][0] == "pactl"]

    def test_asserts_the_configured_card_and_profile(self):
        with patch("ai.tts.subprocess.run") as run:
            self.assertTrue(tts.apply_output_profile())
        argv = self._pactl_argv(run)
        self.assertEqual(argv, [["pactl", "set-card-profile", tts.TTS_CARD, tts.TTS_CARD_PROFILE]])
        # pactl needs XDG_RUNTIME_DIR to find the Pulse socket under the @reboot cron autostart.
        self.assertEqual(run.call_args.kwargs["env"]["XDG_RUNTIME_DIR"], tts.TTS_XDG_RUNTIME)
        self.assertIsNotNone(run.call_args.kwargs["timeout"], "must not be able to wedge the worker")

    def test_profile_keeps_the_input_half(self):
        # This one card carries the dongle's mic too: the bare "output:analog-stereo" profile fixes
        # playback by dropping that capture source. Guard against someone "simplifying" the string.
        self.assertIn("+input:", tts.TTS_CARD_PROFILE)
        self.assertTrue(tts.TTS_CARD_PROFILE.startswith("output:analog-stereo"))

    def test_disabled_toggle_skips_pactl(self):
        with patch("ai.tts.TTS_ASSERT_CARD_PROFILE", False), \
             patch("ai.tts.subprocess.run") as run:
            self.assertFalse(tts.apply_output_profile())
        run.assert_not_called()

    def test_missing_pactl_returns_false_without_raising(self):
        # Runs on a daemon speak worker with no error handling above it — must never escape.
        with patch("ai.tts.subprocess.run", side_effect=FileNotFoundError("no pactl")), \
             patch("builtins.print") as mock_print:
            self.assertFalse(tts.apply_output_profile())
        mock_print.assert_called_once()

    def test_pactl_failure_returns_false_without_raising(self):
        for exc in (subprocess.CalledProcessError(1, "pactl"),
                    subprocess.TimeoutExpired("pactl", 5.0)):
            with self.subTest(exc=type(exc).__name__):
                with patch("ai.tts.subprocess.run", side_effect=exc), \
                     patch("builtins.print"):
                    self.assertFalse(tts.apply_output_profile())

    def test_play_asserts_the_profile_once_per_process_not_per_reply(self):
        tts._profile_applied = False
        proc = _fake_proc(returncode=0)
        with patch("ai.tts.settings.get", return_value=1.0), \
             patch("ai.tts.subprocess.run") as run, \
             patch("subprocess.Popen", return_value=proc):
            tts.play(Path("/tmp/x.wav"))
            tts.play(Path("/tmp/x.wav"))
        self.assertEqual(len(self._pactl_argv(run)), 1,
                         "the profile assert is a per-process cost, not a per-reply one")

    def test_a_failed_playback_reasserts_the_profile_before_retrying(self):
        # The whole point of the retry: an identical retry fails identically when the cause is the
        # card having flipped and taken TTS_SINK with it.
        proc = _fake_proc(returncode=1)
        # `stderr` is a Popen *instance* attribute, so spec=Popen doesn't expose it — assign it.
        proc.stderr = MagicMock()
        proc.stderr.read.return_value = b"Failure: No such entity"
        with patch("ai.tts.settings.get", return_value=1.0), \
             patch("ai.tts.subprocess.run") as run, \
             patch("ai.tts.time.sleep"), \
             patch("subprocess.Popen", return_value=proc), \
             patch("builtins.print"):
            tts.play(Path("/tmp/x.wav"))
        self.assertEqual(len(self._pactl_argv(run)), 1,
                         "retry must re-assert the profile, not just wait and hope")

    def test_successful_playback_does_not_reassert(self):
        proc = _fake_proc(returncode=0)
        with patch("ai.tts.settings.get", return_value=1.0), \
             patch("ai.tts.subprocess.run") as run, \
             patch("subprocess.Popen", return_value=proc):
            tts.play(Path("/tmp/x.wav"))
        self.assertEqual(self._pactl_argv(run), [], "nothing to fix on the happy path")


class TestRunPiper(_ResetProcState):
    def test_publishes_handle_so_stop_can_cancel_it(self):
        seen = []
        proc = _fake_proc()
        proc.communicate.side_effect = lambda **kw: (seen.append(tts._synth_proc), (b"", b""))[1]
        with patch("ai.tts.voice_model_path", return_value=Path(__file__)), \
             patch("subprocess.Popen", return_value=proc), \
             patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.stat", return_value=MagicMock(st_size=1024)):
            self.assertTrue(tts._run_piper("hi", Path("/tmp/x.wav")))
        self.assertIs(seen[0], proc, "synth handle must be visible to stop() while Piper runs")
        self.assertIsNone(tts._synth_proc, "handle must be cleared once the synth finishes")

    def test_communicate_failure_is_contained(self):
        # This runs on a daemon speak worker with no error handling above it, so an escaping
        # exception would kill the jaw animation silently.
        proc = _fake_proc()
        proc.communicate.side_effect = OSError("broken pipe")
        with patch("ai.tts.voice_model_path", return_value=Path(__file__)), \
             patch("subprocess.Popen", return_value=proc), \
             patch("builtins.print") as mock_print:
            self.assertFalse(tts._run_piper("hi", Path("/tmp/x.wav")))
        mock_print.assert_called_once()
        self.assertIsNone(tts._synth_proc, "handle must be cleared even on failure")

    def test_negative_returncode_is_silent_cancellation(self):
        # stop() terminates the synth -> negative returncode. That is a newer turn taking over,
        # not a failure, so it must not be logged as one.
        proc = _fake_proc(returncode=-15)
        with patch("ai.tts.voice_model_path", return_value=Path(__file__)), \
             patch("subprocess.Popen", return_value=proc), \
             patch("builtins.print") as mock_print:
            self.assertFalse(tts._run_piper("hi", Path("/tmp/x.wav")))
        mock_print.assert_not_called()

    def test_nonzero_returncode_logs_and_fails(self):
        proc = _fake_proc(returncode=1, stderr=b"piper: bad model")
        with patch("ai.tts.voice_model_path", return_value=Path(__file__)), \
             patch("subprocess.Popen", return_value=proc), \
             patch("builtins.print") as mock_print:
            self.assertFalse(tts._run_piper("hi", Path("/tmp/x.wav")))
        mock_print.assert_called_once()

    def test_missing_voice_model_fails_before_spawning(self):
        with patch("ai.tts.voice_model_path", return_value=Path("/nope/missing.onnx")), \
             patch("subprocess.Popen") as mock_popen, \
             patch("builtins.print"):
            self.assertFalse(tts._run_piper("hi", Path("/tmp/x.wav")))
        mock_popen.assert_not_called()

    def test_hung_piper_is_killed_and_reaped_within_the_call(self):
        # A5: nothing watches the background warm/rewarm threads' calls to _run_piper, so a wedged
        # Piper must be bounded HERE rather than relying on a caller-side deadline — otherwise the
        # calling thread (and its Popen child) is leaked for the life of the process.
        proc = _fake_proc(alive=True)
        proc.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="piper", timeout=tts.TTS_PIPER_TIMEOUT_S),
            (b"", b""),   # the reap call after kill()
        ]
        with patch("ai.tts.voice_model_path", return_value=Path(__file__)), \
             patch("subprocess.Popen", return_value=proc), \
             patch("builtins.print") as mock_print:
            self.assertFalse(tts._run_piper("hi", Path("/tmp/x.wav")))
        proc.kill.assert_called_once()
        mock_print.assert_called_once()
        self.assertIsNone(tts._synth_proc, "handle must be cleared after a timeout")


class TestSynthesizeTo(_ResetProcState):
    def test_returns_none_for_unspeakable_text(self):
        with patch("ai.tts._run_piper") as mock_run:
            self.assertIsNone(tts.synthesize_to("😀", Path("/tmp/kai_ack/a.wav")))
        mock_run.assert_not_called()

    def test_writes_to_its_own_path_not_the_shared_output(self):
        dest = Path("/tmp/kai_ack/ack.wav")
        with patch("ai.tts._run_piper", return_value=True) as mock_run, \
             patch("ai.tts._post_process", return_value=dest), \
             patch("pathlib.Path.mkdir"):
            self.assertEqual(tts.synthesize_to("Yes?", dest), dest)
        raw_used = mock_run.call_args[0][1]
        self.assertEqual(raw_used, Path("/tmp/kai_ack/ack_raw.wav"))
        # A cached line sharing _OUTPUT_WAV would be overwritten by the next reply.
        self.assertNotEqual(dest, tts._OUTPUT_WAV)
        self.assertNotEqual(raw_used, tts._RAW_WAV)

    def test_falls_back_to_raw_file_moved_onto_dest(self):
        dest = Path("/tmp/kai_ack/ack.wav")
        raw = Path("/tmp/kai_ack/ack_raw.wav")
        with patch("ai.tts._run_piper", return_value=True), \
             patch("ai.tts._post_process", return_value=raw), \
             patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.replace") as mock_replace:
            self.assertEqual(tts.synthesize_to("Yes?", dest), dest)
        mock_replace.assert_called_once_with(dest)

    def test_synth_failure_returns_none(self):
        with patch("ai.tts._run_piper", return_value=False), patch("pathlib.Path.mkdir"):
            self.assertIsNone(tts.synthesize_to("Yes?", Path("/tmp/kai_ack/ack.wav")))


class TestPrewarmCanned(_ResetProcState):
    def test_returns_empty_when_tts_disabled(self):
        with patch("ai.tts.enabled", return_value=False), \
             patch("ai.tts.synthesize_to") as mock_synth:
            self.assertEqual(tts.prewarm_canned({"ack": "Yes?"}, "/tmp/kai_ack"), {})
        mock_synth.assert_not_called()

    def test_maps_keys_to_distinct_paths(self):
        with patch("ai.tts.enabled", return_value=True), \
             patch("ai.tts.synthesize_to", side_effect=lambda t, d: d):
            out = tts.prewarm_canned({"ack": "Yes?", "err": "Oops."}, "/tmp/kai_ack")
        self.assertEqual(set(out), {"ack", "err"})
        self.assertEqual(len(set(out.values())), 2, "each canned line needs its own file")

    def test_one_failure_omits_only_that_key(self):
        def synth(text, dest):
            return None if "err" in str(dest) else dest

        with patch("ai.tts.enabled", return_value=True), \
             patch("ai.tts.synthesize_to", side_effect=synth), \
             patch("builtins.print"):
            out = tts.prewarm_canned({"ack": "Yes?", "err": "Oops."}, "/tmp/kai_ack")
        self.assertEqual(set(out), {"ack"})


class TestWavSpeechSpan(unittest.TestCase):
    """The jaw is timed to the SOUND in a WAV, not the file — Piper pads silence onto both ends,
    0.29 s of the 0.82 s ack WAV on the robot. See ai/tts.wav_speech_span."""

    def _write(self, path, blocks, rate=22050, channels=1):
        """Write a WAV from (seconds, amplitude) blocks, so a test can say '0.2 s of silence, then
        0.5 s of tone' and read the span straight back out."""
        frames = bytearray()
        for secs, amp in blocks:
            for i in range(int(rate * secs)):
                # Alternate sign every sample: a DC block would have an RMS of `amp` too, but a
                # square wave is closer to what the scan actually meets.
                s = amp if i % 2 else -amp
                frames += struct.pack("<h", s) * channels
        with wave.open(str(path), "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(bytes(frames))
        return path

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.tmp = Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def test_trims_the_silence_piper_pads_onto_both_ends(self):
        wav = self._write(self.tmp / "ack.wav",
                          [(0.20, 0), (0.50, 8000), (0.30, 0)])
        start, end = tts.wav_speech_span(wav)
        self.assertAlmostEqual(start, 0.20, places=1)
        self.assertAlmostEqual(end, 0.70, places=1)
        self.assertAlmostEqual(tts.wav_duration(wav), 1.00, places=2)   # the file is longer

    def test_internal_pauses_stay_inside_the_span(self):
        # A gap between two sentences is not the end of the speech. Only the outer edges are trimmed
        # — the per-sentence pauses are the envelope's job, not this one's.
        wav = self._write(self.tmp / "two.wav",
                          [(0.10, 0), (0.30, 8000), (0.40, 0), (0.30, 8000), (0.10, 0)])
        start, end = tts.wav_speech_span(wav)
        self.assertAlmostEqual(start, 0.10, places=1)
        self.assertAlmostEqual(end, 1.10, places=1)

    def test_stereo_is_scanned_on_whole_frames(self):
        # The post-processing chain writes 2ch (TTS_POST_CHANNELS); a scan stepping by samples
        # rather than frames would read half-frames and mistime every reply that goes through it.
        wav = self._write(self.tmp / "stereo.wav",
                          [(0.20, 0), (0.50, 8000), (0.30, 0)], channels=2)
        start, end = tts.wav_speech_span(wav)
        self.assertAlmostEqual(start, 0.20, places=1)
        self.assertAlmostEqual(end, 0.70, places=1)

    def test_a_silent_wav_reports_the_whole_file(self):
        # Falling back to the file is what keeps a failed scan from freezing the jaw shut.
        wav = self._write(self.tmp / "quiet.wav", [(0.50, 0)])
        self.assertEqual(tts.wav_speech_span(wav), (0.0, 0.5))

    def test_an_unreadable_wav_reports_an_empty_span(self):
        # The caller reads it as "no trim" and uses the file length. Silent by design: wav_duration
        # has already warned about this same file.
        self.assertEqual(tts.wav_speech_span(self.tmp / "nope.wav"), (0.0, 0.0))


class TestIsPlaying(_ResetProcState):
    def test_false_when_nothing_started(self):
        self.assertFalse(tts.is_playing())

    def test_true_while_process_alive(self):
        tts._current_proc = _fake_proc(alive=True)
        self.assertTrue(tts.is_playing())

    def test_false_once_process_exits(self):
        tts._current_proc = _fake_proc(alive=False)
        self.assertFalse(tts.is_playing())


class TestQuietSince(_ResetProcState):
    def test_infinite_before_anything_plays(self):
        self.assertEqual(tts.quiet_since(now=100.0), float("inf"))

    def test_zero_while_playing(self):
        tts._current_proc = _fake_proc(alive=True)
        tts._last_end = 50.0
        self.assertEqual(tts.quiet_since(now=100.0), 0.0)

    def test_elapsed_since_last_end(self):
        tts._last_end = 90.0
        self.assertAlmostEqual(tts.quiet_since(now=100.5), 10.5)

    def test_never_negative_on_clock_skew(self):
        tts._last_end = 200.0
        self.assertEqual(tts.quiet_since(now=100.0), 0.0)


class TestPlay(_ResetProcState):
    def test_stamps_last_end_so_the_quiet_tail_can_start(self):
        proc = _fake_proc()
        with patch("subprocess.Popen", return_value=proc), \
             patch("ai.tts.time.monotonic", return_value=123.0):
            tts.play(Path("/tmp/x.wav"))
        proc.wait.assert_called_once()
        self.assertEqual(tts._last_end, 123.0)
        self.assertIsNone(tts._current_proc)

    def test_spawn_failure_only_logs(self):
        with patch("subprocess.Popen", side_effect=OSError("no paplay")), \
             patch("builtins.print") as mock_print:
            tts.play(Path("/tmp/x.wav"))
        mock_print.assert_called_once()
        self.assertEqual(tts._last_end, 0.0)

    def test_does_not_clear_a_newer_playback_handle(self):
        # stop() + a new play() can land while this one is still in proc.wait().
        old, new = _fake_proc(), _fake_proc(alive=True)

        def wait():
            tts._current_proc = new

        old.wait.side_effect = wait
        with patch("subprocess.Popen", return_value=old):
            tts.play(Path("/tmp/x.wav"))
        self.assertIs(tts._current_proc, new)


class TestStop(_ResetProcState):
    def test_noop_when_nothing_running(self):
        tts.stop()   # must not raise
        self.assertIsNone(tts._current_proc)

    def test_terminates_playback_and_clears_handle(self):
        proc = _fake_proc(alive=True)
        tts._current_proc = proc
        tts.stop()
        proc.terminate.assert_called_once()
        self.assertIsNone(tts._current_proc)

    def test_terminates_an_in_flight_synth(self):
        # The bug this guards: a worker cancelled mid-Piper used to finish synthesizing and then
        # play its reply into whatever session came next.
        synth = _fake_proc(alive=True)
        tts._synth_proc = synth
        tts.stop()
        synth.terminate.assert_called_once()
        self.assertIsNone(tts._synth_proc)

    def test_terminates_both_stages_at_once(self):
        synth, play = _fake_proc(alive=True), _fake_proc(alive=True)
        tts._synth_proc, tts._current_proc = synth, play
        tts.stop()
        synth.terminate.assert_called_once()
        play.terminate.assert_called_once()

    def test_starts_the_quiet_tail_from_the_cut(self):
        tts._current_proc = _fake_proc(alive=True)
        with patch("ai.tts.time.monotonic", return_value=77.0):
            tts.stop()
        self.assertEqual(tts._last_end, 77.0)

    def test_cancelling_a_synth_alone_does_not_stamp_last_end(self):
        # No audio was in the air, so there is no quiet tail to wait out.
        tts._synth_proc = _fake_proc(alive=True)
        tts.stop()
        self.assertEqual(tts._last_end, 0.0)

    def test_skips_terminate_on_already_dead_process(self):
        proc = _fake_proc(alive=False)
        tts._current_proc = proc
        tts.stop()
        proc.terminate.assert_not_called()

    def test_survives_terminate_raising(self):
        proc = _fake_proc(alive=True)
        proc.terminate.side_effect = OSError("gone")
        tts._current_proc = proc
        tts.stop()   # must not propagate


if __name__ == "__main__":
    unittest.main()
