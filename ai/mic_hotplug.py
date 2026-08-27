"""Noticing that the set of sound cards changed, so a mic can be swapped without a restart.

Kai has two microphones with different failure modes — the onboard INMP441 and a USB mic — and until
now choosing between them was a startup-only decision. Plugging a USB mic into a running robot did
nothing at all, and not for want of a retry path: ai/session.py's `reresolve_mic` already re-runs the
whole discovery sequence on demand, but it re-ran it against a device list PortAudio had snapshotted
at Pa_Initialize and never refreshed. The new card was invisible to it. Two things were missing, and
this module is the first: something that notices, and (in ai/mic_device.refresh_devices) something
that makes the list current.

WHY /proc/asound/cards AND NOT PortAudio. Polling sd.query_devices() and diffing it is the obvious
implementation and it cannot work — for the reason above, that list does not change. Forcing a
PortAudio re-init on each poll WOULD refresh it, and would also invalidate the live InputStream
several times a minute, which is the one thing the whole always-open-stream design exists to avoid.
The kernel's own card list has neither problem: one small read, no audio state touched, no PortAudio
involved. That also makes this module trivially testable — it reads a path, so a test hands it a
temp file and writes to it.

WHY IT SETTLES BEFORE REPORTING. USB enumeration is not atomic. The card appears in this file before
its PCM devices are openable, so re-resolving the instant the file changes probes a device that is
not ready and reads it as silent — which, because a silent device is skipped, is exactly the outcome
"plug the mic in" was supposed to produce and would instead have prevented. A change is therefore
reported only once the card set has held steady for the settle window, and a set that keeps churning
(a hub re-enumerating) reports nothing until it stops.

That makes the guarantee two-part, and both halves matter: a change must be seen on two SEPARATE
polls *and* have held for settle_s. The second read is not redundant with the timer — it is what
rules out a torn read of a file the kernel is rewriting, which no amount of waiting on a single
sample can do. So a change costs at least two poll intervals even when settle_s is 0.

Deliberately does no I/O beyond that read, spawns no thread and owns no state but its own: it is
polled from the session tick, which already runs at a known rate and already knows whether Kai is
mid-conversation. Deciding *when* it is safe to act on a change is the session's business, not this
module's — see ConversationSession._tick.

Non-Linux and dev boxes have no such file. That is not an error and not a platform check the caller
should have to make: the watcher disables itself on the first miss and answers False forever after.
"""

from __future__ import annotations

from config.voice import (
    MIC_HOTPLUG_CARDS_PATH, MIC_HOTPLUG_ENABLED, MIC_HOTPLUG_POLL_S, MIC_HOTPLUG_SETTLE_S,
)


class CardWatcher:
    """Reports each settled change to the set of ALSA cards, exactly once.

    Driven by poll(now) from the caller's own clock — no thread, no time.monotonic() of its own — so
    the session's fake-clock tests can step it through an enumeration without sleeping.
    """

    def __init__(self, path: str = MIC_HOTPLUG_CARDS_PATH, enabled: bool = MIC_HOTPLUG_ENABLED,
                 poll_s: float = MIC_HOTPLUG_POLL_S, settle_s: float = MIC_HOTPLUG_SETTLE_S) -> None:
        self.path     = path
        self.enabled  = bool(enabled)
        self.poll_s   = float(poll_s)
        self.settle_s = float(settle_s)
        self.changes  = 0            # settled changes reported, for /params
        self._known: str | None = None    # the fingerprint we last reported (or started with)
        self._pending: str | None = None  # a different fingerprint, not yet steady long enough
        self._pending_since = 0.0
        self._next_poll = 0.0
        self._disabled_reason = "" if self.enabled else "disabled in config"

    @property
    def active(self) -> bool:
        return self.enabled and not self._disabled_reason

    def _fingerprint(self) -> str | None:
        """The card list as an order-insensitive string, or None if it can't be read.

        Sorted because the file's line order is not a promise, and hashing the whole text would make
        an unrelated reformat read as a hot-plug. Any read failure disables the watcher rather than
        being retried: the file either exists on this machine or it does not, and re-reading a
        missing path every 3 s for the life of the process is pure noise."""
        try:
            with open(self.path, encoding="utf-8", errors="replace") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
        except OSError as exc:
            self._disabled_reason = f"{type(exc).__name__}: {exc}"
            print(f"[mic] hot-plug watching is off — cannot read {self.path} "
                  f"({self._disabled_reason}); mics are chosen at startup and by the dashboard "
                  f"button only", flush=True)
            return None
        return "\n".join(sorted(lines))

    def poll(self, now: float) -> bool:
        """True exactly once per settled change in the card set. Cheap to call every tick.

        "Settled" means seen on two separate polls AND unchanged for settle_s — see the module
        docstring for why both, and why that costs at least two poll intervals."""
        if not self.active or now < self._next_poll:
            return False
        self._next_poll = now + self.poll_s

        current = self._fingerprint()
        if current is None:
            return False
        if self._known is None:          # first read establishes the baseline, never a "change"
            self._known = current
            return False
        if current == self._known:
            self._pending = None         # a change that reverted before settling never happened
            return False

        if current != self._pending:     # still moving — restart the settle window
            self._pending = current
            self._pending_since = now
            return False
        if now - self._pending_since < self.settle_s:
            return False

        self._known, self._pending = current, None
        self.changes += 1
        print(f"[mic] the set of sound cards changed and has settled — re-resolving the microphone "
              f"(change {self.changes})", flush=True)
        return True
