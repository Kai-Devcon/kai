"""Microphone discovery: device ranking, capture rates, liveness probing, ALSA/pulse plumbing.

These moved out of tests/test_voice_assistant.py with the code they cover. They exercise the layer
BELOW the assistant — which mic to open and how — and none of them needs a VoiceAssistant, an LLM
or a Whisper model.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from ai import mic_device
from ai.mic_device import (
    _candidate_input_devices,
    _capture_rates_for,
    _classify_device,
    _is_speaker_card,
    _probe_is_live,
    _profile_for,
    apply_i2s_route,
    free_i2s_device,
    refresh_devices,
    resolve_input_device,
    resume_pulse_source,
    resume_pulse_sources,
)
from config.voice import I2S_PROBE_SILENT_RETRIES, USB_PROBE_SILENT_RETRIES


def reset_module_state():
    """Clear every cache ai/mic_device keeps between resolves.

    Three of them now, and all three are deliberately process-lifetime: the skip log (one line per
    device name, not per resolve), the speaker's ALSA card index (a subprocess we refuse to run once
    per candidate), and the list of pulse sources awaiting a resume. Left alone, they leak state
    across tests in whatever order the runner happens to pick."""
    mic_device._speaker_card_logged.clear()
    mic_device._speaker_card = None
    mic_device._suspended_sources.clear()


class TestCandidateInputDevices(unittest.TestCase):
    def setUp(self):
        reset_module_state()

    def test_default_device_listed_first(self):
        devices = [
            {"name": "card0 (hw:0,0)", "max_input_channels": 2},
            {"name": "card1 (hw:1,0)", "max_input_channels": 2},
        ]
        with patch("ai.mic_device.sd.default") as mock_default:
            mock_default.device = [1, 1]
            candidates = _candidate_input_devices(devices)
        self.assertEqual(candidates[0], 1)

    def test_dedupes_duplicate_subdevices_of_same_card(self):
        devices = [
            {"name": "APE (hw:1,0)", "max_input_channels": 16},
            {"name": "APE (hw:1,1)", "max_input_channels": 16},
            {"name": "APE (hw:1,2)", "max_input_channels": 16},
            {"name": "USB Mic (hw:0,0)", "max_input_channels": 2},
        ]
        with patch("ai.mic_device.sd.default") as mock_default:
            mock_default.device = [-1, -1]
            candidates = _candidate_input_devices(devices)
        # only one representative for card 1, plus card 0 — not all 3 hw:1,* duplicates
        card_1_hits = [i for i in candidates if i in (0, 1, 2)]
        self.assertEqual(len(card_1_hits), 1)
        self.assertIn(3, candidates)

    def test_skips_output_only_devices(self):
        devices = [
            {"name": "HDMI out (hw:0,3)", "max_input_channels": 0},
            {"name": "Mic (hw:1,0)", "max_input_channels": 2},
        ]
        with patch("ai.mic_device.sd.default") as mock_default:
            mock_default.device = [-1, -1]
            candidates = _candidate_input_devices(devices)
        self.assertNotIn(0, candidates)
        self.assertIn(1, candidates)


class TestSpeakerCardIsNeverCaptured(unittest.TestCase):
    """The 2026-08-11 segfault: capturing the card the speaker plays out of.

    The C-Media dongle is both the only output sink and an input device. Raw capture there, plus the
    `pactl set-card-profile` tts.play() runs before the first reply, took the process down at the
    startup greeting — and the relaunch greeted the room a second time. See SPEAKER_CARD_NAME_HINTS
    in config/voice.py.
    """

    def setUp(self):
        reset_module_state()

    def test_an_input_on_the_speakers_card_is_not_a_candidate(self):
        devices = [
            {"name": "USB Audio Device: - (hw:0,0)", "max_input_channels": 1},
            {"name": "NVIDIA Jetson Orin Nano APE: - (hw:2,1)", "max_input_channels": 16},
        ]
        with patch("ai.mic_device.sd.default") as mock_default:
            mock_default.device = [-1, -1]
            candidates = _candidate_input_devices(devices)
        self.assertEqual(candidates, [1])

    def test_the_system_default_is_dropped_too_when_it_points_at_that_card(self):
        # The default seed bypasses the classification loop, so it needs the check of its own.
        devices = [
            {"name": "USB Audio Device: - (hw:0,0)", "max_input_channels": 1},
            {"name": "APE (hw:2,1)", "max_input_channels": 16},
        ]
        with patch("ai.mic_device.sd.default") as mock_default:
            mock_default.device = [0, 0]
            candidates = _candidate_input_devices(devices)
        self.assertNotIn(0, candidates)

    def test_a_pulse_default_entry_is_kept(self):
        # "default"/"pulse" do not match the hints, and that asymmetry is the point: going through
        # pulse is the SAFE way to touch that card, because pulse coordinates access to it.
        devices = [
            {"name": "default", "max_input_channels": 32},
            {"name": "USB Audio Device: - (hw:0,0)", "max_input_channels": 1},
        ]
        with patch("ai.mic_device.sd.default") as mock_default:
            mock_default.device = [0, 0]
            candidates = _candidate_input_devices(devices)
        self.assertEqual(candidates, [0])

    def test_a_separate_usb_mic_is_unaffected(self):
        # The guard names the speaker's card, not "anything USB" — a real USB mic stays the fallback.
        devices = [{"name": "USB PnP Sound Device: - (hw:1,0)", "max_input_channels": 1}]
        with patch("ai.mic_device.sd.default") as mock_default:
            mock_default.device = [-1, -1]
            candidates = _candidate_input_devices(devices)
        self.assertEqual(candidates, [0])

    def test_emptying_the_hints_restores_the_old_behaviour(self):
        devices = [{"name": "USB Audio Device: - (hw:0,0)", "max_input_channels": 1}]
        with patch("ai.mic_device.sd.default") as mock_default, \
             patch("ai.mic_device.SPEAKER_CARD_NAME_HINTS", ()):
            mock_default.device = [-1, -1]
            candidates = _candidate_input_devices(devices)
        self.assertEqual(candidates, [0])

    def test_resolution_prefers_no_mic_over_the_speakers_card(self):
        """The trade, asserted so nobody has to rediscover it.

        With the I2S mic reading silent and only the speaker's card left, resolution falls through to
        the pulse-mediated default (device=None) instead of handing back the dongle. That run may be
        deaf; the alternative was a SIGSEGV at the greeting and a relaunch that greeted again.
        """
        devices = [
            {"name": "APE tegra-dlink-0 (hw:APE,0)", "max_input_channels": 16,
             "default_samplerate": 48000.0},
            {"name": "USB Audio Device: - (hw:0,0)", "max_input_channels": 1,
             "default_samplerate": 44100.0},
        ]
        probed = []

        def probe(device, rate, channels, take_channel, retries=0):
            probed.append(device)
            return False            # the I2S mic reads silent, as it did on the robot

        with patch("ai.mic_device.sd.query_devices", return_value=devices), \
             patch("ai.mic_device.sd.default") as mock_default, \
             patch("ai.mic_device._probe_is_live", side_effect=probe):
            mock_default.device = [-1, -1]
            choice = resolve_input_device()
        self.assertIsNone(choice.device)
        self.assertNotIn(1, probed)   # never even opened for a probe


class TestPulseSuspend(unittest.TestCase):
    def setUp(self):
        reset_module_state()

    def test_free_i2s_device_suspends_source(self):
        from ai.mic_device import I2S_PULSE_SOURCE
        with patch("ai.mic_device.I2S_SUSPEND_PULSE", True), \
             patch("ai.mic_device.PULSE_SUSPEND_ALL_SOURCES", False), \
             patch("ai.mic_device.subprocess.run") as mock_run:
            free_i2s_device()
        args, _ = mock_run.call_args
        self.assertEqual(args[0], ["pactl", "suspend-source", I2S_PULSE_SOURCE, "1"])

    def test_every_capture_source_is_released_not_just_i2s(self):
        # A source pulse holds makes that device's liveness probe time out, which reads as "not live" —
        # enough to skip the real mic and fall back to a 44.1 kHz pulse device that cannot be resampled
        # to 16 kHz. Reachable once pulseaudio started at boot and held the USB card.
        from ai.mic_device import I2S_PULSE_SOURCE
        listing = (f"0\t{I2S_PULSE_SOURCE}\tmodule-alsa-card.c\ts16le 2ch 44100Hz\tSUSPENDED\n"
                   "1\talsa_input.usb-C-Media_Audio-00.mono-fallback\tmodule-alsa-card.c\t"
                   "s16le 1ch 44100Hz\tIDLE\n"
                   "2\talsa_output.usb-C-Media_Audio-00.analog-stereo.monitor\tmodule-alsa-card.c\t"
                   "s16le 2ch 44100Hz\tIDLE\n")

        def run(cmd, **kw):
            out = MagicMock()
            out.stdout = listing if cmd[:3] == ["pactl", "list", "short"] else ""
            return out

        with patch("ai.mic_device.I2S_SUSPEND_PULSE", True), \
             patch("ai.mic_device.PULSE_SUSPEND_ALL_SOURCES", True), \
             patch("ai.mic_device.subprocess.run", side_effect=run) as mock_run:
            free_i2s_device()

        suspended = [c.args[0][2] for c in mock_run.call_args_list
                     if c.args[0][:2] == ["pactl", "suspend-source"]]
        self.assertIn(I2S_PULSE_SOURCE, suspended)
        self.assertIn("alsa_input.usb-C-Media_Audio-00.mono-fallback", suspended)
        self.assertEqual(len(suspended), 2, "the I2S source must not be suspended twice")
        self.assertFalse([s for s in suspended if s.endswith(".monitor")],
                         "monitors are output taps and hold no capture hardware")

    def test_missing_pactl_while_enumerating_does_not_raise(self):
        with patch("ai.mic_device.I2S_SUSPEND_PULSE", True), \
             patch("ai.mic_device.PULSE_SUSPEND_ALL_SOURCES", True), \
             patch("ai.mic_device.subprocess.run", side_effect=FileNotFoundError("no pactl")):
            free_i2s_device()   # must not raise

    def test_resume_pulse_source_unsuspends(self):
        from ai.mic_device import I2S_PULSE_SOURCE
        with patch("ai.mic_device.I2S_SUSPEND_PULSE", True), \
             patch("ai.mic_device.subprocess.run") as mock_run:
            resume_pulse_source()
        args, _ = mock_run.call_args
        self.assertEqual(args[0], ["pactl", "suspend-source", I2S_PULSE_SOURCE, "0"])

    def test_disabled_toggle_skips_pactl(self):
        with patch("ai.mic_device.I2S_SUSPEND_PULSE", False), \
             patch("ai.mic_device.subprocess.run") as mock_run:
            free_i2s_device()
            resume_pulse_source()
        mock_run.assert_not_called()

    def test_missing_pactl_does_not_raise(self):
        with patch("ai.mic_device.I2S_SUSPEND_PULSE", True), \
             patch("ai.mic_device.subprocess.run", side_effect=FileNotFoundError("no pactl")):
            free_i2s_device()      # must not raise
            resume_pulse_source()  # must not raise


class TestApplyI2SRoute(unittest.TestCase):
    def test_applies_every_control_when_amixer_succeeds(self):
        from ai.mic_device import I2S_ROUTE_CONTROLS
        with patch("ai.mic_device.I2S_APPLY_ROUTE_ON_STARTUP", True), \
             patch("ai.mic_device.subprocess.run") as mock_run:
            ok = apply_i2s_route()
        self.assertTrue(ok)
        self.assertEqual(mock_run.call_count, len(I2S_ROUTE_CONTROLS))
        # each invocation is a non-shell amixer cset on the configured card
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0][0], "amixer")
        self.assertTrue(kwargs.get("check"))

    def test_disabled_toggle_skips_amixer(self):
        with patch("ai.mic_device.I2S_APPLY_ROUTE_ON_STARTUP", False), \
             patch("ai.mic_device.subprocess.run") as mock_run:
            ok = apply_i2s_route()
        self.assertFalse(ok)
        mock_run.assert_not_called()

    def test_missing_amixer_returns_false_without_raising(self):
        with patch("ai.mic_device.I2S_APPLY_ROUTE_ON_STARTUP", True), \
             patch("ai.mic_device.subprocess.run", side_effect=FileNotFoundError("no amixer")):
            self.assertFalse(apply_i2s_route())   # must not raise

    def test_failed_control_stops_early(self):
        import subprocess as _sp
        with patch("ai.mic_device.I2S_APPLY_ROUTE_ON_STARTUP", True), \
             patch("ai.mic_device.subprocess.run",
                   side_effect=_sp.CalledProcessError(1, "amixer")) as mock_run:
            ok = apply_i2s_route()
        self.assertFalse(ok)
        self.assertEqual(mock_run.call_count, 1)   # bails after the first failure, no 9x spam

class TestClassifyDevice(unittest.TestCase):
    def test_i2s_matches_ape_and_tegra_dlink(self):
        self.assertEqual(_classify_device("APE (hw:APE,0)"), "i2s")
        self.assertEqual(_classify_device("tegra-dlink-0 (hw:1,0)"), "i2s")

    def test_usb_match(self):
        self.assertEqual(_classify_device("Some USB Audio (hw:0,0)"), "usb")

    def test_case_insensitive(self):
        self.assertEqual(_classify_device("my usb mic"), "usb")
        self.assertEqual(_classify_device("Tegra-DLink capture"), "i2s")

    def test_other_when_no_hint_matches(self):
        self.assertEqual(_classify_device("Generic onboard analog"), "other")

    def test_i2s_wins_over_usb_when_both_present(self):
        # Contrived name containing both hints — I2S is checked first.
        self.assertEqual(_classify_device("APE USB bridge"), "i2s")


class TestResolveInputDevice(unittest.TestCase):
    def setUp(self):
        reset_module_state()

    def test_prefers_live_i2s_over_usb(self):
        devices = [
            {"name": "USB Mic (hw:0,0)", "max_input_channels": 2, "default_samplerate": 44100.0},
            # Real Jetson APE reports a misleading default_samplerate (44100) but the hw device is
            # locked to its 48 kHz route rate — resolution must ignore the advertised rate.
            {"name": "NVIDIA Jetson Orin Nano APE: - (hw:1,0)", "max_input_channels": 16, "default_samplerate": 44100.0},
        ]
        # Both live; the I2S device must be probed first and win.
        with patch("ai.mic_device.sd.query_devices", return_value=devices), \
             patch("ai.mic_device.sd.default") as mock_default, \
             patch("ai.mic_device._probe_is_live", return_value=True):
            mock_default.device = [-1, -1]
            choice = resolve_input_device()
        self.assertEqual(choice.device, 1)         # the APE/I2S device
        self.assertEqual(choice.rate, 48000)       # pinned to the I2S clock rate (pulse suspended)
        self.assertEqual(choice.channels, 2)       # captured stereo
        self.assertEqual(choice.take_channel, 0)   # left slot
        self.assertTrue(choice.is_i2s)

    def test_falls_back_to_usb_when_i2s_silent(self):
        devices = [
            {"name": "APE tegra-dlink-0 (hw:APE,0)", "max_input_channels": 16, "default_samplerate": 48000.0},
            {"name": "USB Mic (hw:0,0)", "max_input_channels": 2, "default_samplerate": 44100.0},
        ]
        # I2S (probed first) reads silent, USB is live.
        with patch("ai.mic_device.sd.query_devices", return_value=devices), \
             patch("ai.mic_device.sd.default") as mock_default, \
             patch("ai.mic_device._probe_is_live", side_effect=[False, True]):
            mock_default.device = [-1, -1]
            choice = resolve_input_device()
        self.assertEqual(choice.device, 1)         # the USB device
        # NOT 44100, which is what this device advertises. 44100 does not divide into SAMPLE_RATE,
        # so MicStream cannot build a decimator for it and the session dies on open — the whole
        # point of _capture_rates_for. Only divisible rates are ever offered.
        self.assertEqual(choice.rate % 16000, 0)
        self.assertEqual(choice.channels, 1)       # mono
        self.assertEqual(choice.take_channel, 0)
        self.assertFalse(choice.is_i2s)

    def test_usb_that_rejects_16k_is_opened_at_48k_not_its_advertised_44100(self):
        """The 2026-08-09 robot failure, end to end.

        The C-Media dongle advertises default_samplerate=44100 and its hw params are
        `S16_LE mono, RATE: [44100 48000]` — so 16 kHz cannot be opened at all and 44100 cannot be
        resampled. The old code took the advertised rate and handed back 44100, and MicStream.open()
        died on `decimation needs an integer ratio, got 44100 -> 16000`, which took hands-free AND
        push-to-talk down. 48000 was available the entire time.

        SPEAKER_CARD_NAME_HINTS is emptied here on purpose. On the real robot this exact device is now
        skipped outright, because it is also the speaker (2026-08-11 — see
        TestSpeakerCardIsNeverCaptured). What is under test in THIS case is the rate arithmetic, which
        every non-I2S device still depends on, so the guard is switched off rather than the device
        renamed — a renamed device would stop being the dongle whose hw params are quoted above.
        """
        devices = [
            {"name": "USB Audio Device: - (hw:0,0)", "max_input_channels": 1,
             "default_samplerate": 44100.0},
        ]

        def probe(device, rate, channels, take_channel, retries=0):
            return rate in (44100, 48000)      # exactly what this dongle supports

        with patch("ai.mic_device.sd.query_devices", return_value=devices), \
             patch("ai.mic_device.SPEAKER_CARD_NAME_HINTS", ()), \
             patch("ai.mic_device.sd.default") as mock_default, \
             patch("ai.mic_device._probe_is_live", side_effect=probe):
            mock_default.device = [-1, -1]
            choice = resolve_input_device()
        self.assertEqual(choice.device, 0)
        self.assertEqual(choice.rate, 48000)       # the one rate that both opens and resamples
        self.assertFalse(choice.is_i2s)

    def test_device_that_opens_at_no_usable_rate_is_skipped_not_returned(self):
        """A 44.1-kHz-only device must be passed over, not handed back.

        Returning it is strictly worse than falling through: it looks like success and then fails
        at Decimator construction, where the only recovery is the session refusing to start.
        """
        devices = [
            {"name": "Fussy Mic (hw:1,0)", "max_input_channels": 1, "default_samplerate": 44100.0},
        ]
        with patch("ai.mic_device.sd.query_devices", return_value=devices), \
             patch("ai.mic_device.sd.default") as mock_default, \
             patch("ai.mic_device._probe_is_live", return_value=False):
            mock_default.device = [-1, -1]
            choice = resolve_input_device()
        self.assertIsNone(choice.device)
        self.assertEqual(choice.rate, 16000)


class TestCaptureRatesFor(unittest.TestCase):
    def test_i2s_is_pinned_to_the_route_rate_and_ignores_the_advertised_one(self):
        # The real APE device advertises 44100 while the route runs at 48 kHz. Trusting the
        # advertised rate here would garble speech even when it happened to be divisible.
        self.assertEqual(_capture_rates_for("i2s", 44100), (48000,))

    def test_every_offered_rate_divides_into_the_pipeline_rate(self):
        for kind in ("usb", "other"):
            for advertised in (0, 8000, 44100, 48000, 96000):
                for rate in _capture_rates_for(kind, advertised):
                    self.assertEqual(rate % 16000, 0,
                                     f"{rate} from kind={kind} advertised={advertised}")

    def test_indivisible_advertised_rate_is_dropped_entirely(self):
        self.assertNotIn(44100, _capture_rates_for("usb", 44100))

    def test_divisible_advertised_rate_leads_so_the_native_rate_is_tried_first(self):
        # Opening a device at its own rate avoids a driver-side resample, so prefer it — but only
        # because it passed the divisibility filter, never on the strength of being advertised.
        self.assertEqual(_capture_rates_for("usb", 48000)[0], 48000)
        self.assertEqual(_capture_rates_for("other", 32000)[0], 32000)

    def test_no_duplicate_rates_so_no_device_is_probed_twice_at_one_rate(self):
        for advertised in (16000, 32000, 44100, 48000):
            rates = _capture_rates_for("usb", advertised)
            self.assertEqual(len(rates), len(set(rates)))

    def test_default_is_last_resort(self):
        # No I2S/USB present: an 'other' device that's live is chosen, captured mono.
        devices = [
            {"name": "Generic onboard (hw:1,0)", "max_input_channels": 2, "default_samplerate": 48000.0},
        ]
        with patch("ai.mic_device.sd.query_devices", return_value=devices), \
             patch("ai.mic_device.sd.default") as mock_default, \
             patch("ai.mic_device._probe_is_live", return_value=True):
            mock_default.device = [-1, -1]
            choice = resolve_input_device()
        self.assertEqual(choice.device, 0)
        self.assertEqual(choice.channels, 1)

    def test_falls_back_when_nothing_is_live(self):
        devices = [
            {"name": "Silent onboard (hw:1,0)", "max_input_channels": 2, "default_samplerate": 48000.0},
        ]
        with patch("ai.mic_device.sd.query_devices", return_value=devices), \
             patch("ai.mic_device.sd.default") as mock_default, \
             patch("ai.mic_device._probe_is_live", return_value=False):
            mock_default.device = [-1, -1]
            choice = resolve_input_device()
        self.assertIsNone(choice.device)
        self.assertEqual(choice.rate, 16000)
        self.assertEqual(choice.channels, 1)

    def test_falls_back_when_query_raises(self):
        with patch("ai.mic_device.sd.query_devices", side_effect=OSError("no audio subsystem")):
            choice = resolve_input_device()
        self.assertIsNone(choice.device)
        self.assertEqual(choice.rate, 16000)


class TestProbeIsLive(unittest.TestCase):
    def test_returns_true_above_threshold(self):
        with patch("ai.mic_device.sd.rec", return_value=np.full((100, 1), 100, dtype="int16")), \
             patch("ai.mic_device.sd.wait"):
            self.assertTrue(_probe_is_live(0, 16000, 1, 0))

    def test_returns_false_on_silence(self):
        with patch("ai.mic_device.sd.rec", return_value=np.zeros((100, 1), dtype="int16")), \
             patch("ai.mic_device.sd.wait"):
            self.assertFalse(_probe_is_live(0, 16000, 1, 0))

    def test_returns_false_on_exception(self):
        with patch("ai.mic_device.sd.rec", side_effect=OSError("busy")):
            self.assertFalse(_probe_is_live(0, 16000, 1, 0))

    def test_stereo_measures_only_the_taken_channel(self):
        # INMP441 shape: left (col 0) loud, right (col 1) digital silence -> live on channel 0.
        rec = np.zeros((100, 2), dtype="int16")
        rec[:, 0] = 100
        with patch("ai.mic_device.sd.rec", return_value=rec), \
             patch("ai.mic_device.sd.wait"):
            self.assertTrue(_probe_is_live(0, 48000, 2, 0))

    def test_a_silent_first_read_is_retried_and_the_device_can_come_back(self):
        """The INMP441 warm-up: silent on the first capture after the route is applied, live after.

        Before the retry, that single early read condemned the preferred mic for the whole life of
        the process and Kai ran the entire session on the fallback USB mic.
        """
        silent = np.zeros((100, 2), dtype="int16")
        live = np.zeros((100, 2), dtype="int16")
        live[:, 0] = 100
        with patch("ai.mic_device.sd.rec", side_effect=[silent, silent, live]), \
             patch("ai.mic_device.sd.wait"), \
             patch("ai.mic_device.time.sleep"):
            self.assertTrue(_probe_is_live(5, 48000, 2, 0, retries=3))

    def test_retries_are_bounded_and_a_dead_device_still_reads_dead(self):
        with patch("ai.mic_device.sd.rec", return_value=np.zeros((100, 2), dtype="int16")) as rec, \
             patch("ai.mic_device.sd.wait"), \
             patch("ai.mic_device.time.sleep"):
            self.assertFalse(_probe_is_live(5, 48000, 2, 0, retries=3))
        self.assertEqual(rec.call_count, 4)      # the first read plus exactly three retries

    def test_a_device_that_refuses_to_open_is_not_retried(self):
        """An open failure is a definite answer. Retrying it burns LIVE_PROBE_TIMEOUT_S multiples on
        the session start path — which is the hang the timeout exists to prevent."""
        with patch("ai.mic_device.sd.rec", side_effect=OSError("busy")) as rec, \
             patch("ai.mic_device.time.sleep"):
            self.assertFalse(_probe_is_live(5, 48000, 2, 0, retries=3))
        self.assertEqual(rec.call_count, 1)

    def test_retries_default_to_off_so_other_devices_are_read_once(self):
        with patch("ai.mic_device.sd.rec", return_value=np.zeros((100, 1), dtype="int16")) as rec, \
             patch("ai.mic_device.sd.wait"):
            self.assertFalse(_probe_is_live(0, 48000, 1, 0))
        self.assertEqual(rec.call_count, 1)

    def test_each_device_kind_is_retried_by_its_own_budget(self):
        devices = [
            {"name": "APE tegra-dlink-0 (hw:APE,0)", "max_input_channels": 16,
             "default_samplerate": 44100.0},
            {"name": "USB Mic (hw:0,0)", "max_input_channels": 1, "default_samplerate": 44100.0},
        ]
        seen = []

        def probe(device, rate, channels, take_channel, retries=0):
            seen.append((device, retries))
            return device == 1 and rate == 48000

        with patch("ai.mic_device.sd.query_devices", return_value=devices), \
             patch("ai.mic_device.sd.default") as mock_default, \
             patch("ai.mic_device._probe_is_live", side_effect=probe):
            mock_default.device = [-1, -1]
            resolve_input_device()
        # Both kinds get retries now that a USB mic can be plugged into a running robot and needs a
        # moment to settle — but not the same budget. The I2S mic's is sized for a documented boot
        # race; the USB mic's for enumeration settling, and it is paid on every dead candidate at
        # startup, so it stays strictly smaller. See _profile_for and the two constants.
        self.assertTrue(all(r == I2S_PROBE_SILENT_RETRIES for d, r in seen if d == 0), seen)
        self.assertTrue(all(r == USB_PROBE_SILENT_RETRIES for d, r in seen if d == 1), seen)
        self.assertGreater(I2S_PROBE_SILENT_RETRIES, USB_PROBE_SILENT_RETRIES)

    def test_stereo_silent_on_taken_channel_reads_dead(self):
        # Signal only in the untaken channel -> the taken (left) channel is silent -> not live.
        rec = np.zeros((100, 2), dtype="int16")
        rec[:, 1] = 100
        with patch("ai.mic_device.sd.rec", return_value=rec), \
             patch("ai.mic_device.sd.wait"):
            self.assertFalse(_probe_is_live(0, 48000, 2, 0))


class TestProbeExplainsItself(unittest.TestCase):
    """A rejected probe must say WHY, and the two reasons must be distinguishable.

    Both used to return a bare False. "The device refused to open" and "the mic is silent" then
    looked identical from the log — just `i2s=False` with no reason — and on 2026-08-07 that turned a
    startup race into a hardware investigation. They need different fixes (check what is holding the
    card vs. check the wiring), so the log has to say which one happened.
    """

    def test_open_failure_reports_the_exception(self):
        with patch("ai.mic_device.sd.rec", side_effect=OSError("Device unavailable")), \
             patch("builtins.print") as out:
            self.assertFalse(_probe_is_live(5, 48000, 2, 0))
        logged = " ".join(str(c) for c in out.call_args_list)
        self.assertIn("rejected the probe", logged)
        self.assertIn("Device unavailable", logged, "the real reason must survive to the log")
        self.assertIn("OSError", logged)

    def test_silence_is_reported_as_silence_not_as_an_error(self):
        with patch("ai.mic_device.sd.rec", return_value=np.zeros((100, 1), dtype="int16")), \
             patch("ai.mic_device.sd.wait"), patch("builtins.print") as out:
            self.assertFalse(_probe_is_live(5, 48000, 1, 0))
        logged = " ".join(str(c) for c in out.call_args_list)
        self.assertIn("read as silent", logged)
        self.assertNotIn("rejected the probe", logged, "silence is not an open failure")

    def test_a_live_device_stays_quiet(self):
        # One line per REJECTED candidate; the success path must not add noise to every startup.
        with patch("ai.mic_device.sd.rec",
                   return_value=np.full((100, 1), 100, dtype="int16")), \
             patch("ai.mic_device.sd.wait"), patch("builtins.print") as out:
            self.assertTrue(_probe_is_live(5, 48000, 1, 0))
        self.assertEqual(out.call_args_list, [])


PACTL_CARDS = (
    "Card #0\n"
    "\tName: alsa_card.usb-C-Media_Electronics_Inc._USB_Audio_Device-00\n"
    "\tDriver: module-alsa-card.c\n"
    "\tProperties:\n"
    '\t\talsa.card = "2"\n'
    '\t\talsa.card_name = "USB Audio Device"\n'
    "Card #1\n"
    "\tName: alsa_card.platform-sound\n"
    "\tProperties:\n"
    '\t\talsa.card = "0"\n'
)


class TestSpeakerCardResolution(unittest.TestCase):
    """Tier 1: which card the speaker is on, asked of pactl rather than guessed from a name.

    This is the change that makes a USB mic usable at all. The name-substring rule blocks anything
    called "USB Audio Device", which is the most common name a cheap USB mic enumerates under — so
    the guard protecting the speaker was also silently discarding real microphones before they were
    ever probed.
    """

    def setUp(self):
        reset_module_state()

    def _pactl(self, stdout=PACTL_CARDS):
        return patch("ai.mic_device.subprocess.run", return_value=MagicMock(stdout=stdout))

    def test_the_speakers_own_card_is_still_blocked(self):
        # The whole point of the guard, and the 2026-08-11 segfault it prevents.
        with self._pactl():
            self.assertTrue(_is_speaker_card("USB Audio Device: - (hw:2,0)"))

    def test_a_usb_mic_sharing_the_speakers_name_on_another_card_is_allowed(self):
        # Same name, different card. Under the old name-substring rule this device was dropped
        # before it was ever probed, with no log line saying a microphone had been rejected.
        with self._pactl():
            self.assertFalse(_is_speaker_card("USB Audio Device: - (hw:3,0)"))

    def test_the_i2s_mic_is_never_the_speakers_card(self):
        with self._pactl():
            self.assertFalse(_is_speaker_card("APE tegra-dlink-0 (hw:APE,0)"))

    def test_the_card_index_is_resolved_once_and_cached(self):
        # One subprocess per process, not one per candidate device per resolve.
        with patch("ai.mic_device.subprocess.run",
                   return_value=MagicMock(stdout=PACTL_CARDS)) as run:
            for _ in range(5):
                _is_speaker_card("USB Audio Device: - (hw:2,0)")
        self.assertEqual(run.call_count, 1)

    def test_a_name_belonging_to_another_pulse_card_is_not_confused_for_the_speaker(self):
        # alsa.card = "0" belongs to the platform card, not TTS_CARD. Reading the wrong block would
        # block the onboard card and let the speaker through — the exact inversion of the guard.
        with self._pactl():
            self.assertFalse(_is_speaker_card("tegra-snd (hw:0,0)"))

    def test_falls_back_to_the_name_hints_when_pactl_is_missing(self):
        # A dev box, or pulse not running. Must still BLOCK: the failure this prevents is a segfault
        # mid-greeting, and the failure it causes is one skipped candidate.
        with patch("ai.mic_device.subprocess.run", side_effect=FileNotFoundError("no pactl")):
            self.assertTrue(_is_speaker_card("USB Audio Device: - (hw:0,0)"))
            self.assertFalse(_is_speaker_card("USB PnP Sound Device (hw:1,0)"))

    def test_falls_back_to_the_name_hints_when_pactl_does_not_know_tts_card(self):
        with self._pactl(stdout="Card #0\n\tName: alsa_card.something-else\n"):
            self.assertTrue(_is_speaker_card("USB Audio Device: - (hw:0,0)"))

    def test_the_card_name_property_is_not_mistaken_for_the_card_index(self):
        # `alsa.card_name` sits next to `alsa.card` in the same Properties block and a prefix match
        # would take the product name as the index — which matches no hw:<N> and silently disables
        # the whole tier. Fixture puts it first so the wrong parse would win.
        reordered = PACTL_CARDS.replace(
            '\t\talsa.card = "2"\n\t\talsa.card_name = "USB Audio Device"\n',
            '\t\talsa.card_name = "USB Audio Device"\n\t\talsa.card = "2"\n')
        self.assertNotEqual(reordered, PACTL_CARDS, "the fixture reorder did not apply")
        with self._pactl(stdout=reordered):
            self.assertTrue(_is_speaker_card("USB Audio Device: - (hw:2,0)"))
            self.assertFalse(_is_speaker_card("USB Audio Device: - (hw:3,0)"))

    def test_the_tier_one_toggle_forces_the_name_rule(self):
        with patch("ai.mic_device.SPEAKER_CARD_FROM_TTS_CARD", False), \
             patch("ai.mic_device.subprocess.run",
                   return_value=MagicMock(stdout=PACTL_CARDS)) as run:
            self.assertTrue(_is_speaker_card("USB Audio Device: - (hw:9,0)"))
            run.assert_not_called()


class TestMicPreference(unittest.TestCase):
    """MIC_PREFERENCE reorders the probe. It must never exclude a kind."""

    DEVICES = [
        {"name": "APE tegra-dlink-0 (hw:APE,0)", "max_input_channels": 16},
        {"name": "USB PnP Sound Device (hw:1,0)", "max_input_channels": 1},
        {"name": "some other card (hw:2,0)", "max_input_channels": 2},
    ]

    def setUp(self):
        reset_module_state()

    def _order(self, pref):
        with patch("ai.mic_device._preference", return_value=pref), \
             patch("ai.mic_device.sd.default") as default, \
             patch("ai.mic_device.subprocess.run", side_effect=FileNotFoundError("no pactl")):
            default.device = [-1, -1]
            return _candidate_input_devices(self.DEVICES)

    def test_auto_keeps_the_historical_order(self):
        self.assertEqual(self._order("auto"), [0, 1, 2])

    def test_preferring_usb_probes_it_first(self):
        self.assertEqual(self._order("usb"), [1, 0, 2])

    def test_preferring_i2s_probes_it_first(self):
        self.assertEqual(self._order("i2s"), [0, 1, 2])

    def test_every_kind_is_still_probed_whichever_is_preferred(self):
        # A preference that could filter a device out could leave Kai deaf when only the other mic
        # is plugged in. It reorders; it never excludes.
        for pref in ("auto", "i2s", "usb"):
            self.assertCountEqual(self._order(pref), [0, 1, 2], pref)

    def test_an_unrecognised_preference_reads_as_auto(self):
        # Operator-writable, so a typo must not change which mic Kai picks in a way nobody can
        # explain afterwards.
        self.assertEqual(self._order("nonsense"), [0, 1, 2])

    def test_preference_falls_back_to_config_when_settings_is_unreadable(self):
        from ai.mic_device import _preference
        with patch("ai.mic_device.settings.get", side_effect=RuntimeError("boom")):
            self.assertEqual(_preference(), "auto")


class TestCaptureProfiles(unittest.TestCase):
    def test_the_i2s_mic_is_stereo_on_its_left_slot(self):
        channels, take, retries = _profile_for("i2s")
        self.assertEqual((channels, take), (2, 0))
        self.assertEqual(retries, I2S_PROBE_SILENT_RETRIES)

    def test_other_kinds_are_mono_on_channel_zero(self):
        for kind in ("usb", "other"):
            channels, take, retries = _profile_for(kind)
            self.assertEqual((channels, take), (1, 0), kind)
            self.assertEqual(retries, USB_PROBE_SILENT_RETRIES, kind)

    def test_a_usb_mic_does_not_inherit_the_i2s_take_channel(self):
        # Harmless only because I2S_TAKE_CHANNEL happens to be 0 today. Pin the independence, so
        # retuning it for a differently-wired board cannot silently read a USB mic's empty slot.
        with patch("ai.mic_device.I2S_TAKE_CHANNEL", 1):
            self.assertEqual(_profile_for("usb")[1], 0)
            self.assertEqual(_profile_for("i2s")[1], 1)


class TestPulseSuspendResumeSymmetry(unittest.TestCase):
    """Everything free_i2s_device() suspended must come back when we don't end up on the raw device.

    Invisible while USB was only ever a boot-time fallback onto a raw hw device — a suspended source
    does not stop a raw open. It stops being invisible when a run is *meant* to end on the USB mic.
    """

    SOURCES = "0\talsa_input.usb-mic\t\n1\talsa_output.x.monitor\t\n2\talsa_input.other\t\n"

    def setUp(self):
        reset_module_state()

    def _run(self):
        def run(cmd, **kw):
            if cmd[:3] == ["pactl", "list", "short"]:
                return MagicMock(stdout=self.SOURCES)
            return MagicMock(stdout="")
        return patch("ai.mic_device.subprocess.run", side_effect=run)

    def _suspends(self, mock, on):
        return [c.args[0][2] for c in mock.call_args_list
                if c.args[0][:2] == ["pactl", "suspend-source"] and c.args[0][3] == on]

    def test_every_suspended_source_is_resumed(self):
        with patch("ai.mic_device.I2S_SUSPEND_PULSE", True), \
             patch("ai.mic_device.PULSE_SUSPEND_ALL_SOURCES", True), \
             self._run() as mock:
            free_i2s_device()
            suspended = self._suspends(mock, "1")
            resume_pulse_sources()
            resumed = self._suspends(mock, "0")
        self.assertCountEqual(suspended, resumed)
        self.assertIn("alsa_input.usb-mic", resumed)

    def test_resuming_twice_does_not_resume_the_extra_sources_twice(self):
        # Not a strict no-op, and deliberately so: the tracked list is consumed, but the fallback
        # for callers that resume without ever having suspended (VoiceAssistant does this) still
        # hands back the I2S source. That one extra pactl call is idempotent — un-suspending a
        # source that is not suspended does nothing — whereas re-resuming the whole enumerated set
        # would be real work on every non-I2S resolve.
        from ai.mic_device import I2S_PULSE_SOURCE
        with patch("ai.mic_device.I2S_SUSPEND_PULSE", True), \
             patch("ai.mic_device.PULSE_SUSPEND_ALL_SOURCES", True), \
             self._run() as mock:
            free_i2s_device()
            resume_pulse_sources()
            mock.reset_mock()
            resume_pulse_sources()
        self.assertEqual(self._suspends(mock, "0"), [I2S_PULSE_SOURCE])

    def test_resuming_without_a_suspend_still_hands_back_the_i2s_source(self):
        # The historical behaviour, for callers that resume on a path that never suspended.
        from ai.mic_device import I2S_PULSE_SOURCE
        with patch("ai.mic_device.I2S_SUSPEND_PULSE", True), \
             patch("ai.mic_device.subprocess.run") as mock:
            resume_pulse_sources()
        self.assertEqual(mock.call_args.args[0],
                         ["pactl", "suspend-source", I2S_PULSE_SOURCE, "0"])

    def test_the_old_name_is_still_the_same_function(self):
        # Imported under the old name by ai/mic_stream.py, ai/voice_assistant.py and
        # scripts/wake_test.py.
        self.assertIs(resume_pulse_source, resume_pulse_sources)


class TestRefreshDevices(unittest.TestCase):
    def setUp(self):
        reset_module_state()

    def test_it_reinitialises_portaudio(self):
        with patch("ai.mic_device.sd._terminate") as term, \
             patch("ai.mic_device.sd._initialize") as init:
            self.assertTrue(refresh_devices())
        term.assert_called_once()
        init.assert_called_once()

    def test_it_drops_the_speaker_card_cache(self):
        # A re-plug renumbers ALSA cards, so an index resolved before the change may name a
        # different card after it.
        mic_device._speaker_card = "2"
        with patch("ai.mic_device.sd._terminate"), patch("ai.mic_device.sd._initialize"):
            refresh_devices()
        self.assertIsNone(mic_device._speaker_card)

    def test_a_portaudio_failure_is_reported_not_raised(self):
        # A stale device list means the mic may be invisible, not that startup should die.
        with patch("ai.mic_device.sd._terminate", side_effect=RuntimeError("busy")), \
             patch("ai.mic_device.sd._initialize"):
            self.assertFalse(refresh_devices())


class TestResolvedChoiceNamesItsKind(unittest.TestCase):
    def setUp(self):
        reset_module_state()

    def _resolve(self, devices):
        with patch("ai.mic_device.sd.query_devices", return_value=devices), \
             patch("ai.mic_device.sd.default") as default, \
             patch("ai.mic_device.subprocess.run", side_effect=FileNotFoundError("no pactl")), \
             patch("ai.mic_device._probe_is_live", return_value=True):
            default.device = [-1, -1]
            return resolve_input_device()

    def test_a_usb_choice_says_usb(self):
        # is_i2s=False cannot distinguish "the USB mic the operator chose" from "whatever the system
        # default turned out to be", and the dashboard has to be able to name the mic.
        mic = self._resolve([{"name": "USB PnP Sound Device (hw:1,0)", "max_input_channels": 1}])
        self.assertEqual(mic.kind, "usb")
        self.assertFalse(mic.is_i2s)

    def test_an_i2s_choice_says_i2s(self):
        mic = self._resolve([{"name": "APE tegra-dlink-0 (hw:APE,0)", "max_input_channels": 16}])
        self.assertEqual((mic.kind, mic.is_i2s), ("i2s", True))

    def test_the_give_up_fallback_says_other(self):
        with patch("ai.mic_device.sd.query_devices", side_effect=RuntimeError("no portaudio")):
            self.assertEqual(resolve_input_device().kind, "other")

if __name__ == "__main__":
    unittest.main()
