# S15 — A 3.5mm mic has no place in the input model, only a USB card it borrows

| | |
|---|---|
| **Tier** | 3 |
| **Severity** | Low (enhancement, not a defect) |
| **Effort** | Low |
| **Confidence** | High |
| **Lens** | Software |

> **Status: FIXED** — branch `feat/analog-mic-kind`. `analog` is a fourth mic kind: classified
> before the generic `usb` hint, selectable in `MIC_PREFERENCE` and on the dashboard, and reported
> as `sess_mic_kind`. Acceptance criteria checked off in place below.
>
> **Not verified on hardware.** Everything here is covered by unit tests on a dev box; nobody has
> yet plugged a 3.5mm mic into a USB adapter on the robot. See *Verification* for what is still owed.

## Location

- `config/voice.py` — `I2S_MIC_NAME_HINTS`, `USB_MIC_NAME_HINTS`, `MIC_PREFERENCE`,
  `SPEAKER_CARD_NAME_HINTS`, `FALLBACK_CAPTURE_RATES`
- `ai/mic_device.py` — `_classify_device()`, `_profile_for()`, `_candidate_input_devices()`,
  `MicChoice.kind`
- `settings.py` — the `mic_preference` spec's `choices`
- `web/frontend/dashboard.html` — `MIC_NAMES`, `MicRecoveryCard`'s Prefer selector

## Problem

Kai's input model has exactly two named kinds, `i2s` and `usb` (plus `other` for the system
default). A 3.5mm analog microphone is a real third option — it is what most people already own,
and unlike the INMP441 it needs no soldering — but it has nowhere to be named.

It also has no independent existence on this board. The Jetson's own analog input "enumerates as a
normal input device but isn't wired to anything, so it only captures digital silence"
(`config/voice.py`), so a 3.5mm mic reaches Kai only through a USB audio adapter. Electrically that
adapter *is* a USB sound card, which means today the mic works — and is then labelled `usb`,
indistinguishable from a native USB microphone.

That indistinguishability is the whole defect, and it is small but real:

1. **The dashboard cannot say which mic Kai is on** when both a USB mic and a 3.5mm adapter are
   plugged in. `sess_mic_kind` reads `usb` for both. The Microphone card's whole purpose is
   answering "did that do what I meant", and with two `usb` devices it cannot.
2. **`MIC_PREFERENCE` cannot express the choice.** `usb` picks whichever of the two probes live
   first, which is ALSA enumeration order — not a preference.
3. **The one hazard is silent.** A 3.5mm adapter with a headphone jack is a card with outputs, so
   it looks structurally like the speaker dongle. The speaker-card exclusion is now keyed on
   `TTS_CARD`'s ALSA index rather than a name (2026-08-27), so a second adapter is *not* wrongly
   excluded — but nothing documents that a 3.5mm mic is only safe *because* of that change, and a
   build that reverts to name matching would silently lose the mic.

## Non-goals

- **The Jetson's onboard analog jack.** Not wired on this board; supporting it is a hardware
  question, not a software one.
- **The speaker dongle's own mic jack.** That card is excluded from capture because raw capture on
  it segfaulted the process at the startup greeting (2026-08-11). Capturing it would require a
  pulse-mediated path that never touches raw ALSA, and that is a separate ticket with a much worse
  failure mode. This ticket assumes a *separate* USB adapter.
- **Device pinning by exact name.** Considered; a preference that reorders can never leave Kai deaf,
  whereas a pin that filters can. Not worth the surface area yet.

## Proposal

Add `analog` as a fourth kind. It changes **labelling and probe order only** — never whether a
device can be opened.

1. `ANALOG_MIC_NAME_HINTS` in `config/voice.py`, matched **before** `USB_MIC_NAME_HINTS`, because a
   USB→3.5mm adapter matches `"usb"` too and the more specific rule has to win.
2. `_profile_for("analog")` returns the same mono / channel-0 treatment as `usb`, and the same
   silent-read retry budget — an adapter is a USB device and settles the same way.
3. `_candidate_input_devices()` orders buckets as *preferred kind first, then the standard order*.
4. `MIC_PREFERENCE`, the `mic_preference` setting's `choices`, and the dashboard's Prefer selector
   all gain `analog`.

**The classification is a heuristic and must degrade to cosmetic.** Adapter names are not
distinctive — `"USB PnP Sound Device"` is used by both adapters and standalone USB mics — so an
unmatched adapter classifies as `usb` and *still works exactly as it does today*. Getting it wrong
costs a label and a position in the probe order, never a working microphone. That is the property
that makes a name-based rule acceptable here.

**Rejected: infer it structurally** from the card also having output channels. An adapter with a
headphone jack does, but so does every USB mic with a monitor output (Yeti, AT2020USB+), and a
mic-in-only adapter does not. It is not a discriminator, and a wrong structural guess would be
harder to explain than a wrong name guess because there would be no list to edit.

## Acceptance Criteria

- [x] A device matching `ANALOG_MIC_NAME_HINTS` classifies as `analog`, not `usb`, even though its
      name also contains `"usb"`
- [x] `MIC_PREFERENCE = "analog"` probes analog devices first and still probes every other kind
      afterwards, so it cannot leave Kai deaf
- [x] `_profile_for("analog")` is mono on channel 0 and does not inherit `I2S_TAKE_CHANNEL`
- [x] `MicChoice.kind` and `sess_mic_kind` report `analog`
- [x] `/audio/reresolve` reports `kind: "analog"`, and the dashboard names it
- [x] `settings.py` accepts `analog` for `mic_preference` and rejects anything else
- [x] An adapter whose name is not in the hints still resolves and opens exactly as before
- [x] An analog adapter on a card that is not `TTS_CARD` is not excluded as the speaker's card
- [x] Hot-plugging the adapter is picked up by the existing `CardWatcher` with no new machinery
- [ ] **Verified on the robot** with a real 3.5mm mic in a USB adapter — see Verification

## Verification

Unit-tested on a dev box; the hardware half is outstanding because the repo is the only thing
available right now.

Still owed, on the Jetson:

1. `python -c "import sounddevice as sd; print(sd.query_devices())"` with the adapter plugged in —
   record its **actual** name and add it to `ANALOG_MIC_NAME_HINTS` if the defaults do not match it.
   The defaults are plausible, not measured.
2. `arecord -D hw:<N>,0 --dump-hw-params` to confirm a usable rate exists. **A device offering only
   44.1 kHz is skipped by design** — `FALLBACK_CAPTURE_RATES` only offers rates that divide into
   16 kHz, because an integer-ratio decimator cannot resample 44100 and returning it took the whole
   session down once (2026-08-09). Several cheap adapters are 44.1-only, and that is the single most
   likely way this fails in the field.
3. Confirm `sess_mic_kind` reads `analog`, and that `MIC_PREFERENCE` moves between all three real
   mics with the dashboard button.
4. Confirm TTS is still audible afterwards — the adapter has outputs, so it is worth re-checking
   that `TTS_SINK` still resolves to the C-Media dongle and not to the new card.

## Notes

- **Buying guidance belongs in the docs, not here, but the two gotchas are:** a 3.5mm *electret*
  mic needs plug-in power, which not every adapter supplies; and a 4-pole TRRS headset plug does not
  work in a separate mic jack — a 3-pole TRS mic does. Neither is visible from software, and both
  present as "the mic reads as silent".
- Depends on the speaker-card exclusion being keyed on `TTS_CARD`'s ALSA index rather than a name
  (2026-08-27). Reverting that would exclude any adapter reporting a generic
  `"USB Audio Device"` name, which is most of them.
