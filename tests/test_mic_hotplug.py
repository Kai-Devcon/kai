"""The hot-plug card watcher: what counts as a change, and when it is safe to say so.

No audio, no PortAudio, no /proc — CardWatcher reads a path and is driven by a caller-supplied
clock, so everything here is a temp file and a float.
"""

import os
import tempfile
import unittest

from ai.mic_hotplug import CardWatcher

ONE_CARD = " 0 [APE            ]: tegra-ape - NVIDIA Jetson\n"
TWO_CARDS = ONE_CARD + " 1 [Device         ]: USB-Audio - USB PnP Sound Device\n"


class WatcherTestCase(unittest.TestCase):
    """A watcher over a real file we can rewrite between polls."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = os.path.join(self.dir.name, "cards")
        self.write(ONE_CARD)
        # poll_s=0 so every call polls: this suite is about the settle logic, not the poll interval,
        # and a test that has to advance the clock past both is testing two things at once.
        self.w = CardWatcher(path=self.path, enabled=True, poll_s=0.0, settle_s=2.0)

    def write(self, text):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(text)


class TestSteadyState(WatcherTestCase):
    def test_the_first_poll_establishes_a_baseline_and_is_never_a_change(self):
        # Otherwise every process would "hot-plug" once at startup and re-resolve a mic it had just
        # resolved.
        self.assertFalse(self.w.poll(0.0))
        self.assertEqual(self.w.changes, 0)

    def test_an_unchanging_card_set_never_fires(self):
        self.w.poll(0.0)
        for t in range(1, 20):
            self.assertFalse(self.w.poll(float(t)))
        self.assertEqual(self.w.changes, 0)

    def test_reordered_lines_are_not_a_change(self):
        # The file's line order is not a promise; only the set of cards is.
        self.w.poll(0.0)
        self.write("".join(reversed(TWO_CARDS.splitlines(keepends=True))))
        self.w.poll(1.0)
        self.write(TWO_CARDS)
        self.assertFalse(self.w.poll(2.0))   # same set, so the pending change resolved to nothing
        self.assertEqual(self.w.changes, 0)


class TestSettling(WatcherTestCase):
    def test_a_settled_change_fires_exactly_once(self):
        self.w.poll(0.0)
        self.write(TWO_CARDS)
        self.assertFalse(self.w.poll(1.0))            # seen, not settled
        self.assertTrue(self.w.poll(1.0 + 2.0))       # settled
        self.assertFalse(self.w.poll(10.0))           # and never again for the same change
        self.assertFalse(self.w.poll(100.0))
        self.assertEqual(self.w.changes, 1)

    def test_a_change_is_not_reported_before_the_settle_window(self):
        # USB enumeration is not atomic: the card appears before its PCM devices are openable, so
        # firing early probes a device that is not ready and reads it as silent — which would skip
        # the very mic that was just plugged in.
        self.w.poll(0.0)
        self.write(TWO_CARDS)
        self.assertFalse(self.w.poll(1.0))
        self.assertFalse(self.w.poll(2.9))            # 1.9 s in, window is 2.0
        self.assertTrue(self.w.poll(3.0))

    def test_a_set_that_keeps_moving_restarts_the_window(self):
        self.w.poll(0.0)
        self.write(TWO_CARDS)
        self.assertFalse(self.w.poll(1.0))
        self.write(TWO_CARDS + " 2 [Third]: something else\n")
        self.assertFalse(self.w.poll(2.5))            # would have settled, but the set moved again
        self.assertFalse(self.w.poll(4.0))            # 1.5 s into the NEW window
        self.assertTrue(self.w.poll(4.5))

    def test_a_change_that_reverts_before_settling_never_happened(self):
        # A card that appears and disappears inside the window is a flap, not a mic.
        self.w.poll(0.0)
        self.write(TWO_CARDS)
        self.assertFalse(self.w.poll(1.0))
        self.write(ONE_CARD)
        self.assertFalse(self.w.poll(1.5))
        self.assertFalse(self.w.poll(30.0))
        self.assertEqual(self.w.changes, 0)

    def test_removal_fires_the_same_as_insertion(self):
        # Unplugging is the case where Kai must fall BACK to the other mic, so it cannot be the
        # quiet one.
        self.write(TWO_CARDS)
        self.w.poll(0.0)
        self.write(ONE_CARD)
        self.w.poll(1.0)
        self.assertTrue(self.w.poll(3.0))


class TestPolling(WatcherTestCase):
    def test_the_file_is_not_read_more_often_than_the_poll_interval(self):
        w = CardWatcher(path=self.path, enabled=True, poll_s=3.0, settle_s=0.0)
        w.poll(0.0)
        self.write(TWO_CARDS)
        self.assertFalse(w.poll(1.0))    # inside the interval — not even read
        self.assertFalse(w.poll(2.9))
        self.assertFalse(w.poll(3.0))    # first read of the new set: seen, not yet confirmed
        self.assertTrue(w.poll(6.0))     # second read agrees -> report

    def test_a_change_always_needs_two_reads_even_with_no_settle_window(self):
        # The timer alone cannot rule out a torn read of a file the kernel is rewriting; a second
        # agreeing read can. So settle_s=0 still costs two polls, and that is the contract, not an
        # off-by-one — see the module docstring.
        w = CardWatcher(path=self.path, enabled=True, poll_s=0.0, settle_s=0.0)
        w.poll(0.0)
        self.write(TWO_CARDS)
        self.assertFalse(w.poll(1.0))
        self.assertTrue(w.poll(1.0))


class TestDegradation(WatcherTestCase):
    def test_a_missing_path_disables_the_watcher_instead_of_raising(self):
        # The dev box and every non-Linux machine. Callers must not need a platform check, and a
        # missing file must not be re-read every few seconds for the life of the process.
        w = CardWatcher(path=os.path.join(self.dir.name, "nope"), enabled=True, poll_s=0.0)
        self.assertFalse(w.poll(0.0))
        self.assertFalse(w.active)
        self.assertFalse(w.poll(1.0))

    def test_disabled_in_config_never_reads_anything(self):
        w = CardWatcher(path=self.path, enabled=False, poll_s=0.0, settle_s=0.0)
        self.assertFalse(w.active)
        self.write(TWO_CARDS)
        self.assertFalse(w.poll(0.0))
        self.assertFalse(w.poll(100.0))


if __name__ == "__main__":
    unittest.main()
