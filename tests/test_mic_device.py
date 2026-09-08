"""Microphone discovery: device ranking, capture rates, liveness probing, ALSA/pulse plumbing.

These moved out of tests/test_voice_assistant.py with the code they cover. They exercise the layer
BELOW the assistant — which mic to open and how — and none of them needs a VoiceAssistant, an LLM
or a Whisper model.
"""

import os
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
    resolve_capture_device,
    resolve_input_device,
    resolve_pulse_device,
    resume_pulse_source,
    resume_pulse_sources,
)
from config.voice import (
    ANALOG_PROBE_SILENT_RETRIES, I2S_PROBE_SILENT_RETRIES, SAMPLE_RATE, USB_PROBE_SILENT_RETRIES,
)


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


class TestPulseStaysOffTheCardWeOpenRaw(unittest.TestCase):
    """The card we are about to capture raw must NOT be handed back to pulse before we open it.

    This build runs no module-suspend-on-idle, so pulse holds every card it owns open permanently
    and re-grabs a resumed one within milliseconds. Resuming between resolve and open is what made a
    healthy BY-PM700 probe live at 48 kHz and then fail EVERY open with "Device unavailable"
    [PaErrorCode -9985], forever, on 2026-08-27. `is_i2s` was standing in for "raw device" and is
    the wrong question: a USB mic on hw:3,0 is exactly as exclusive with is_i2s False.
    """

    SHORT = '0\talsa_input.usb-mic\t\n1\talsa_output.x.monitor\t\n2\talsa_input.other\t\n'
    # `pactl list sources` - one block per source, the one property we need out of each.
    LONG = (
        'Source #0\n\tName: alsa_input.usb-mic\n\tProperties:\n\t\talsa.card = "3"\n'
        'Source #2\n\tName: alsa_input.other\n\tProperties:\n\t\talsa.card = "0"\n'
    )

    def setUp(self):
        reset_module_state()

    def _run(self, long_out=None):
        long_out = self.LONG if long_out is None else long_out

        def run(cmd, **kw):
            if cmd[:3] == ["pactl", "list", "short"]:
                return MagicMock(stdout=self.SHORT)
            if cmd[:3] == ["pactl", "list", "sources"]:
                return MagicMock(stdout=long_out)
            return MagicMock(stdout="")
        return patch("ai.mic_device.subprocess.run", side_effect=run)

    def _resumed(self, mock):
        return [c.args[0][2] for c in mock.call_args_list
                if c.args[0][:2] == ["pactl", "suspend-source"] and c.args[0][3] == "0"]

    def test_the_chosen_cards_source_is_not_resumed(self):
        with (
            patch("ai.mic_device.I2S_SUSPEND_PULSE", True),
            patch("ai.mic_device.PULSE_SUSPEND_ALL_SOURCES", True),
            self._run() as mock,
        ):
            free_i2s_device()
            mock.reset_mock()
            resume_pulse_sources(keep_card="3")
            resumed = self._resumed(mock)
        self.assertNotIn("alsa_input.usb-mic", resumed)   # hw:3 — the mic we are about to open
        self.assertIn("alsa_input.other", resumed)        # hw:0 — nothing to do with our capture

    def test_the_kept_source_is_resumed_by_a_later_unqualified_resume(self):
        # Left in the tracked list rather than dropped, so the card is handed back the moment we
        # stop needing it exclusively — otherwise the source stays muted for the life of the process.
        with (
            patch("ai.mic_device.I2S_SUSPEND_PULSE", True),
            patch("ai.mic_device.PULSE_SUSPEND_ALL_SOURCES", True),
            self._run() as mock,
        ):
            free_i2s_device()
            resume_pulse_sources(keep_card="3")
            mock.reset_mock()
            resume_pulse_sources()
            resumed = self._resumed(mock)
        self.assertIn("alsa_input.usb-mic", resumed)

    def test_an_unattributable_source_is_left_suspended(self):
        # Cannot tell which card it is on and we are opening one raw: guessing wrong here costs a
        # muted source somewhere else, guessing wrong the other way costs the microphone.
        with (
            patch("ai.mic_device.I2S_SUSPEND_PULSE", True),
            patch("ai.mic_device.PULSE_SUSPEND_ALL_SOURCES", True),
            self._run(long_out="") as mock,
        ):
            free_i2s_device()
            mock.reset_mock()
            resume_pulse_sources(keep_card="3")
            resumed = self._resumed(mock)
        self.assertEqual(resumed, [])

    def test_no_card_resumes_everything(self):
        from ai.mic_device import I2S_PULSE_SOURCE
        # A pulse-mediated PCM ("default"/"pulse") is not exclusive, so there is nothing to protect.
        with (
            patch("ai.mic_device.I2S_SUSPEND_PULSE", True),
            patch("ai.mic_device.PULSE_SUSPEND_ALL_SOURCES", True),
            self._run() as mock,
        ):
            free_i2s_device()
            mock.reset_mock()
            resume_pulse_sources()
            resumed = self._resumed(mock)
        self.assertCountEqual(
            resumed, [I2S_PULSE_SOURCE, "alsa_input.usb-mic", "alsa_input.other"])


class TestAlsaCardOfDeviceName(unittest.TestCase):
    def test_raw_devices_yield_their_card_and_plugins_yield_nothing(self):
        from ai.mic_device import _alsa_card_of
        self.assertEqual(_alsa_card_of("BY-PM700: USB Audio (hw:3,0)"), "3")
        self.assertEqual(_alsa_card_of("NVIDIA Jetson Orin Nano APE: - (hw:2,1)"), "2")
        # Named plugin PCMs are not raw and not exclusive — pulse may keep them.
        self.assertEqual(_alsa_card_of("default"), "")
        self.assertEqual(_alsa_card_of("pulse"), "")
        self.assertEqual(_alsa_card_of(""), "")


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

class TestAnalogMicKind(unittest.TestCase):
    """A 3.5mm mic, which on this board can only arrive through a USB audio adapter.

    That is the whole difficulty: the adapter IS a USB sound card, so every signal that would
    distinguish it from a native USB mic is either shared with one or absent. The rule is therefore
    name-based and deliberately allowed to be wrong — it decides label and probe order, never
    whether a device can be opened.
    """

    def setUp(self):
        reset_module_state()

    def test_an_adapter_is_analog_and_not_usb_despite_the_usb_in_its_name(self):
        # The load-bearing assertion of the whole kind. ANALOG_MIC_NAME_HINTS must be matched before
        # USB_MIC_NAME_HINTS: check them the other way round and every analog adapter classifies as
        # 'usb', so the kind exists but is never reached.
        self.assertEqual(_classify_device("USB Audio CODEC (hw:2,0)"), "analog")
        self.assertEqual(_classify_device("GeneralPlus USB Audio Device (hw:3,0)"), "analog")

    def test_a_native_usb_mic_is_still_usb(self):
        self.assertEqual(_classify_device("USB PnP Sound Device (hw:1,0)"), "usb")

    def test_the_i2s_mic_still_wins_over_everything(self):
        # An APE device is never reclassified, whatever else its name happens to contain.
        self.assertEqual(_classify_device("APE tegra-dlink-0 (hw:APE,0)"), "i2s")

    def test_an_unrecognised_adapter_degrades_to_usb_rather_than_disappearing(self):
        # The property that makes a name-based rule acceptable: getting the name wrong costs a
        # label, not a microphone. Adapter names genuinely are not distinctive.
        with patch("ai.mic_device.ANALOG_MIC_NAME_HINTS", ()):
            self.assertEqual(_classify_device("USB Audio CODEC (hw:2,0)"), "usb")

    def test_analog_is_probed_mono_on_channel_zero(self):
        channels, take, retries = _profile_for("analog")
        self.assertEqual((channels, take), (1, 0))
        self.assertEqual(retries, ANALOG_PROBE_SILENT_RETRIES)

    def test_analog_does_not_inherit_the_i2s_take_channel(self):
        with patch("ai.mic_device.I2S_TAKE_CHANNEL", 1):
            self.assertEqual(_profile_for("analog")[1], 0)

    def test_a_resolved_adapter_reports_kind_analog(self):
        devices = [{"name": "USB Audio CODEC (hw:2,0)", "max_input_channels": 1}]
        with patch("ai.mic_device.sd.query_devices", return_value=devices), \
             patch("ai.mic_device.sd.default") as default, \
             patch("ai.mic_device.subprocess.run", side_effect=FileNotFoundError("no pactl")), \
             patch("ai.mic_device._probe_is_live", return_value=True):
            default.device = [-1, -1]
            mic = resolve_input_device()
        self.assertEqual(mic.kind, "analog")
        self.assertFalse(mic.is_i2s)

    def test_an_adapter_on_a_card_that_is_not_the_speakers_is_not_excluded(self):
        # An adapter with a headphone jack is a card with outputs, so it looks structurally like the
        # speaker dongle. It is only usable because the exclusion is keyed on TTS_CARD's ALSA index
        # rather than on a name -- pin that, because reverting it would silently lose the mic.
        with patch("ai.mic_device.subprocess.run",
                   return_value=MagicMock(stdout=PACTL_CARDS)):        # speaker is ALSA card 2
            self.assertFalse(_is_speaker_card("USB Audio CODEC (hw:5,0)"))
            self.assertTrue(_is_speaker_card("USB Audio CODEC (hw:2,0)"))


class TestPreferenceAcrossFourKinds(unittest.TestCase):
    DEVICES = [
        {"name": "APE tegra-dlink-0 (hw:APE,0)", "max_input_channels": 16},
        {"name": "USB PnP Sound Device (hw:1,0)", "max_input_channels": 1},
        {"name": "USB Audio CODEC (hw:2,0)", "max_input_channels": 1},
        {"name": "some other card (hw:3,0)", "max_input_channels": 2},
    ]

    def setUp(self):
        reset_module_state()

    def _order(self, pref):
        with patch("ai.mic_device._preference", return_value=pref), \
             patch("ai.mic_device.sd.default") as default, \
             patch("ai.mic_device.subprocess.run", side_effect=FileNotFoundError("no pactl")):
            default.device = [-1, -1]
            return _candidate_input_devices(self.DEVICES)

    def test_auto_puts_analog_after_usb(self):
        # Deliberately after, not before: analog is the newer kind, and ordering it earlier would
        # silently change which mic an already-working robot picks.
        self.assertEqual(self._order("auto"), [0, 1, 2, 3])

    def test_preferring_analog_probes_the_adapter_first(self):
        self.assertEqual(self._order("analog"), [2, 0, 1, 3])

    def test_preferring_a_kind_leaves_the_rest_in_the_standard_order(self):
        # "prefer X" is "X, then exactly what you would have got anyway" — the easiest rule to
        # predict, and the one the dashboard's wording promises.
        self.assertEqual(self._order("usb"), [1, 0, 2, 3])
        self.assertEqual(self._order("i2s"), [0, 1, 2, 3])

    def test_every_kind_is_still_probed_under_every_preference(self):
        for pref in ("auto", "i2s", "usb", "analog", "nonsense"):
            self.assertCountEqual(self._order(pref), [0, 1, 2, 3], pref)

    def test_the_preference_reader_accepts_analog(self):
        from ai.mic_device import _preference
        with patch("ai.mic_device.settings.get", return_value="analog"):
            self.assertEqual(_preference(), "analog")

    def test_the_preference_reader_still_rejects_nonsense(self):
        from ai.mic_device import _preference
        for bad in ("other", "3.5mm", "", None):
            with patch("ai.mic_device.settings.get", return_value=bad):
                self.assertEqual(_preference(), "auto", bad)

class PulseRouteCase(unittest.TestCase):
    """Shared rig for the pulse-mediated route. No pactl, no PortAudio, no real devices."""

    # The one-dongle build: a card carrying BOTH the speaker and a mic jack, plus a pulse-backed
    # entry. Card 2 is TTS_CARD's ALSA index per PACTL_CARDS.
    DEVICES = [
        {"name": "APE tegra-dlink-0 (hw:APE,0)", "max_input_channels": 16},   # 0  raw I2S
        {"name": "USB Audio Device: - (hw:2,0)", "max_input_channels": 1},    # 1  the MIC JACK, raw
        {"name": "pulse", "max_input_channels": 32},                          # 2  pulse-mediated
        {"name": "default", "max_input_channels": 32},                        # 3  pulse-mediated
    ]

    def setUp(self):
        reset_module_state()
        for p in (patch("ai.mic_device.PULSE_CAPTURE_ENABLED", True),
                  patch("ai.mic_device.PULSE_CAPTURE_SOURCE", "alsa_input.usb-dongle.mono-fallback"),
                  patch("ai.mic_device.PULSE_CAPTURE_ENV_VAR", "KAI_TEST_PULSE_SOURCE")):
            p.start()
            self.addCleanup(p.stop)
        os.environ.pop("KAI_TEST_PULSE_SOURCE", None)
        self.addCleanup(os.environ.pop, "KAI_TEST_PULSE_SOURCE", None)

    def rig(self, live=(), devices=None):
        """Patch PortAudio and pactl. `live` is the device indices whose probe succeeds."""
        return (
            patch("ai.mic_device.sd.query_devices", return_value=devices or self.DEVICES),
            patch("ai.mic_device.sd.default", MagicMock(device=[-1, -1])),
            patch("ai.mic_device.subprocess.run", return_value=MagicMock(stdout=PACTL_CARDS)),
            patch("ai.mic_device._probe_is_live",
                  side_effect=lambda idx, *a, **k: idx in live),
        )

    def run_rigged(self, fn, live=(), devices=None):
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in self.rig(live, devices):
                stack.enter_context(p)
            return fn()


class TestPulseRouteIsReachable(PulseRouteCase):
    """The point of the whole ticket: the mic jack on the speaker's own dongle becomes usable."""

    def test_the_raw_mic_jack_on_the_speakers_card_is_still_refused(self):
        # The guard this route goes AROUND, not through. Raw capture there is the 2026-08-11
        # segfault and stays blocked forever — asserted first, because everything else in this class
        # would be a regression if this ever stopped holding.
        cands = self.run_rigged(lambda: _candidate_input_devices(self.DEVICES), live=())
        self.assertNotIn(1, cands, "raw hw:2,0 is the speaker's card and must never be a candidate")

    def test_the_pulse_route_can_capture_that_same_card(self):
        mic = self.run_rigged(resolve_pulse_device, live=(2,))
        self.assertIsNotNone(mic)
        self.assertEqual(mic.kind, "pulse")
        self.assertEqual(mic.device, 2)
        self.assertFalse(mic.is_i2s)

    def test_it_asks_for_the_pipeline_rate_so_no_decimator_is_needed(self):
        # Pulse resamples for us, which deletes the 44.1 kHz integer-ratio problem for this route
        # rather than solving it: a card that can only do 44.1 kHz is unusable raw, fine through here.
        mic = self.run_rigged(resolve_pulse_device, live=(2,))
        self.assertEqual(mic.rate, SAMPLE_RATE)

    def test_the_choice_carries_the_environment_that_selects_the_source(self):
        # Not a global default-source change: this steers OUR stream and nothing else's.
        mic = self.run_rigged(resolve_pulse_device, live=(2,))
        self.assertEqual(mic.env,
                         {"KAI_TEST_PULSE_SOURCE": "alsa_input.usb-dongle.mono-fallback"})

    def test_the_probe_environment_does_not_leak_out_of_the_resolve(self):
        # MicStream re-applies it from MicChoice.env before opening, so it must not be left behind
        # here — a stray PULSE_SOURCE would silently steer every later recording in the process.
        self.run_rigged(resolve_pulse_device, live=(2,))
        self.assertIsNone(os.environ.get("KAI_TEST_PULSE_SOURCE"))

    def test_it_restores_a_pre_existing_value_rather_than_deleting_it(self):
        os.environ["KAI_TEST_PULSE_SOURCE"] = "someone.elses.choice"
        self.run_rigged(resolve_pulse_device, live=(2,))
        self.assertEqual(os.environ["KAI_TEST_PULSE_SOURCE"], "someone.elses.choice")

    def test_device_names_are_matched_exactly_not_as_substrings(self):
        # "default" inside a longer name means something else entirely, and opening the wrong device
        # here is how a route that is supposed to be safe would turn out to be the raw one.
        devices = [{"name": "Default Audio Thing (hw:9,0)", "max_input_channels": 2}]
        self.assertIsNone(self.run_rigged(resolve_pulse_device, live=(0,), devices=devices))

    def test_config_order_wins_over_enumeration_order(self):
        # PULSE_CAPTURE_DEVICE_NAMES is ("pulse", "default"): "pulse" is the explicit one, so it is
        # tried first even though "default" enumerates earlier here.
        devices = [{"name": "default", "max_input_channels": 32},
                   {"name": "pulse", "max_input_channels": 32}]
        mic = self.run_rigged(resolve_pulse_device, live=(0, 1), devices=devices)
        self.assertEqual(mic.device, 1)


class TestPulseRouteOwnsThePulseEntries(PulseRouteCase):
    """With the route on, the raw phase must not also claim the pulse-backed devices.

    Found by test_falling_through_to_the_pulse_route_resumes_first before this existed: the raw
    phase picked "pulse" as kind="other" and pre-empted the route. Two things wrong with that, and
    the second is the one that bites — the raw phase probes with every source suspended, so it reads
    them silent and wastes the probe; and if it ever DID read one live it would hand back a choice
    with no env, recording from pulse's default source instead of PULSE_CAPTURE_SOURCE. The same
    card by luck rather than on purpose, and unnamed on the dashboard.
    """

    def test_the_raw_phase_skips_pulse_backed_entries(self):
        cands = self.run_rigged(lambda: _candidate_input_devices(self.DEVICES))
        self.assertNotIn(2, cands)      # "pulse"
        self.assertNotIn(3, cands)      # "default"
        self.assertIn(0, cands)         # the I2S mic is untouched

    def test_it_skips_them_even_when_one_is_the_system_default(self):
        # The default-device seed bypasses the classification loop, so it needs the check of its own
        # — the same shape of bug the speaker-card guard already had to fix once.
        from contextlib import ExitStack
        with ExitStack() as st:
            st.enter_context(patch("ai.mic_device.sd.query_devices", return_value=self.DEVICES))
            st.enter_context(patch("ai.mic_device.sd.default", MagicMock(device=[3, 3])))
            st.enter_context(patch("ai.mic_device.subprocess.run",
                                   return_value=MagicMock(stdout=PACTL_CARDS)))
            cands = _candidate_input_devices(self.DEVICES)
        self.assertNotIn(3, cands)

    def test_with_the_flag_off_they_are_the_last_resort_fallback_again(self):
        # Exactly the pre-S16 behaviour: pulse-mediated capture stays reachable as the system
        # default, just unnamed and unselectable.
        with patch("ai.mic_device.PULSE_CAPTURE_ENABLED", False):
            cands = self.run_rigged(lambda: _candidate_input_devices(self.DEVICES))
        self.assertIn(2, cands)
        self.assertIn(3, cands)


class TestPulseRouteDegradesQuietly(PulseRouteCase):
    def test_disabled_is_a_no_op_and_touches_nothing(self):
        with patch("ai.mic_device.PULSE_CAPTURE_ENABLED", False), \
             patch("ai.mic_device.sd.query_devices") as q:
            self.assertIsNone(resolve_pulse_device())
        q.assert_not_called()

    def test_an_unset_source_is_refused_rather_than_guessed(self):
        # Recording from "whatever pulse's default source happens to be" is a worse failure than
        # not recording: the default moves as devices come and go, so it would work until it didn't.
        with patch("ai.mic_device.PULSE_CAPTURE_SOURCE", ""):
            self.assertIsNone(self.run_rigged(resolve_pulse_device, live=(2, 3)))

    def test_no_pulse_backed_device_present_returns_none(self):
        devices = [{"name": "USB PnP Sound Device (hw:1,0)", "max_input_channels": 1}]
        self.assertIsNone(self.run_rigged(resolve_pulse_device, live=(0,), devices=devices))

    def test_a_silent_pulse_source_falls_through(self):
        self.assertIsNone(self.run_rigged(resolve_pulse_device, live=()))


class TestTwoPhaseResolve(PulseRouteCase):
    """The load-bearing change: each route is probed in the pulse state IT needs.

    A raw hw open of the INMP441 needs pulse off the card; a pulse-mediated capture needs the source
    pulse holds to be un-suspended. Opposite states, so one prelude cannot serve both — which is
    exactly why the pulse fallback that existed on paper never worked.
    """

    def _trace(self, live=(), pref="auto", devices=None):
        """Record the order of route/suspend/resume/probe so the phase ordering is observable."""
        from contextlib import ExitStack
        calls = []
        with ExitStack() as st:
            st.enter_context(patch("ai.mic_device.sd.query_devices",
                                   return_value=devices or self.DEVICES))
            st.enter_context(patch("ai.mic_device.sd.default", MagicMock(device=[-1, -1])))
            st.enter_context(patch("ai.mic_device.subprocess.run",
                                   return_value=MagicMock(stdout=PACTL_CARDS)))
            st.enter_context(patch("ai.mic_device.apply_i2s_route",
                                   side_effect=lambda: calls.append("route")))
            st.enter_context(patch("ai.mic_device.free_i2s_device",
                                   side_effect=lambda: calls.append("suspend")))
            st.enter_context(patch("ai.mic_device.resume_pulse_sources",
                                   side_effect=lambda keep_card="":
                                       calls.append(f"resume:{keep_card}" if keep_card
                                                     else "resume")))
            st.enter_context(patch("ai.mic_device._preference", return_value=pref))

            def probe(idx, *a, **k):
                calls.append(f"probe:{idx}")
                return idx in live

            st.enter_context(patch("ai.mic_device._probe_is_live", side_effect=probe))
            mic = resolve_capture_device()
        return mic, calls

    def test_preferring_pulse_probes_it_before_anything_is_suspended(self):
        # The whole fix in one assertion. Suspending first is what made this route read as silent.
        mic, calls = self._trace(live=(2,), pref="pulse")
        self.assertEqual(mic.kind, "pulse")
        self.assertEqual(calls[:2], ["route", "probe:2"])
        self.assertNotIn("suspend", calls)

    def test_the_raw_phase_probes_with_sources_suspended(self):
        mic, calls = self._trace(live=(0,), pref="i2s")
        self.assertEqual(mic.kind, "i2s")
        self.assertLess(calls.index("suspend"), calls.index("probe:0"))

    def test_a_chosen_i2s_mic_keeps_its_own_card_suspended(self):
        # The raw I2S device needs the card off pulse to stay off it. resume_pulse_sources() now
        # runs unconditionally after the raw phase (see resolve_capture_device / keep_card on
        # MicChoice) — the card is still kept suspended, it is just done by naming it rather than
        # by skipping the call outright.
        mic, calls = self._trace(live=(0,), pref="auto")
        self.assertEqual(mic.kind, "i2s")
        self.assertIn("resume:APE", calls)

    def test_a_chosen_raw_usb_mic_also_keeps_its_own_card_suspended(self):
        # The regression this design has to keep fixed (see 11ae9bc "Stop handing pulse the USB mic
        # back before opening it"): a USB mic is just as raw and just as exclusive as the I2S device,
        # with is_i2s False — so the card it resolved to, not is_i2s, is what must stay suspended.
        devices = [{"name": "USB PnP Sound Device (hw:1,0)", "max_input_channels": 1}]
        mic, calls = self._trace(live=(0,), pref="auto", devices=devices)
        self.assertEqual(mic.kind, "usb")
        self.assertIn("resume:1", calls)

    def test_falling_through_to_the_pulse_route_resumes_first(self):
        # Ordering is the point: resuming AFTER the raw phase rather than before it is what lets
        # both phases run in one resolve.
        mic, calls = self._trace(live=(2,), pref="auto")
        self.assertEqual(mic.kind, "pulse")
        self.assertIn("resume", calls)
        self.assertLess(calls.index("resume"), calls.index("probe:2"))

    def test_a_real_raw_mic_beats_the_pulse_route(self):
        # The pulse route is only preferred over MicChoice(None, ...) — "whatever the system default
        # turns out to be", which on this build is the same card reached by luck instead of on
        # purpose. A named raw mic wins.
        devices = [{"name": "USB PnP Sound Device (hw:1,0)", "max_input_channels": 1},
                   {"name": "pulse", "max_input_channels": 32}]
        mic, _ = self._trace(live=(0, 1), pref="auto", devices=devices)
        self.assertEqual(mic.kind, "usb")

    def test_preferring_pulse_still_falls_back_to_a_raw_mic(self):
        # No preference may leave Kai deaf — the rule every kind obeys.
        mic, calls = self._trace(live=(0,), pref="pulse")
        self.assertEqual(mic.kind, "i2s")
        self.assertIn("suspend", calls)

    def test_with_the_flag_off_the_sequence_is_exactly_what_it_always_was(self):
        # A robot that works today must not change behaviour because this landed.
        with patch("ai.mic_device.PULSE_CAPTURE_ENABLED", False):
            mic, calls = self._trace(live=(0,), pref="auto")
        self.assertEqual(calls[:2], ["route", "suspend"])
        self.assertEqual(mic.kind, "i2s")

    def test_nothing_live_anywhere_still_returns_the_give_up_choice(self):
        mic, _ = self._trace(live=(), pref="auto")
        self.assertIsNone(mic.device)
        self.assertEqual(mic.kind, "other")

    def test_pulse_is_a_selectable_preference_but_not_a_device_bucket(self):
        # "pulse" is a route, not a bucket of hardware — the card it reaches is deliberately absent
        # from every bucket. So it belongs in _PREFERENCES and not in _KINDS.
        self.assertIn("pulse", mic_device._PREFERENCES)
        self.assertNotIn("pulse", mic_device._KINDS)
        self.assertNotIn("other", mic_device._PREFERENCES)

if __name__ == "__main__":
    unittest.main()
