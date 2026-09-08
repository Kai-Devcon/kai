# S16 — The mic jack on the speaker's dongle is unreachable, and the safe route to it is unwired

| | |
|---|---|
| **Tier** | 2 |
| **Severity** | Medium (enhancement; the refusal it removes is protecting against a crash) |
| **Effort** | Medium |
| **Confidence** | Medium — the mechanism is sound, one runtime behaviour is unmeasured |
| **Lens** | Software |

> **Status: FIXED (code) / OPEN (hardware)** — branch `feat/analog-mic-kind`. `resolve_capture_device()`
> resolves in two phases, each with the pulse state that route needs; the `pulse` kind is selectable
> and off by default. Every code-side criterion below is checked and unit-tested. **The last one is
> not, and cannot be here** — it needs the robot. Do not close this ticket on the strength of a green
> test run.

## Location

- `ai/mic_device.py` — `_is_speaker_card()`, `resolve_input_device()`, `free_i2s_device()`,
  `resume_pulse_sources()`, `MicChoice`
- `ai/mic_stream.py` — `open()`'s route → suspend → resolve → resume sequence
- `config/voice.py` — `SPEAKER_CARD_*`, `PULSE_SUSPEND_ALL_SOURCES`, `TTS_CARD`, `TTS_CARD_PROFILE`
- `ai/voice_assistant.py` — `ensure_input_resolved()` runs the same sequence for the legacy path
- `scripts/wake_test.py`, `scripts/mic_survey.py` — third and fourth copies of the sequence

## Problem

A single USB dongle with two 3.5mm jacks — speaker out and mic in — is the ordinary way to give Kai
both. It is also the hardware this build already has: `config/voice.py` records the C-Media dongle as
"both the only output sink (`TTS_SINK`) and an input device (`"USB Audio Device: - (hw:0,0)"`, one
mono input)". **Kai cannot use that mic input at all.**

One dongle is **one ALSA card**, and the card — not the jack — is the unit of configuration:

```
card N  ──┬── playback  hw:N,0    pulse sink:   alsa_output.usb-<product>-00.analog-stereo
          └── capture   hw:N,0    pulse source: alsa_input.usb-<product>-00.mono-fallback
```

`pactl set-card-profile`, which `tts.play()` asserts before the first reply, acts on the whole card.
On 2026-08-11 that re-opened the card's ALSA devices underneath a live **raw** PortAudio capture
stream and the process took SIGSEGV at the startup greeting; the relaunch greeted the room a second
time. So `_is_speaker_card()` now drops every input device on that card, keyed on `TTS_CARD`'s ALSA
index. Correct, and it must stay for raw capture.

**But the exclusion is broader than the hazard.** The hazard is *raw ALSA capture* on that card, not
capture on that card. `config/voice.py` already states the way out: "Pulse coordinates access to the
card, so it is safe where a raw open is not." Nothing in the codebase acts on that sentence.

Three things stand between here and a working mic on that jack:

1. **The exclusion is card-wide, not route-wide.** `_is_speaker_card()` cannot distinguish "open
   `hw:N,0` raw" from "open this card's source through pulse", so it refuses both.
2. **The one safe route is probed with pulse suspended.** `MicStream.open()` calls
   `free_i2s_device()` — which under `PULSE_SUSPEND_ALL_SOURCES` suspends *every* capture source —
   *before* `resolve_input_device()`, and `resume_pulse_sources()` only runs afterwards. A
   pulse-mediated candidate is therefore probed while the source it needs is suspended. The two
   routes need **opposite pulse states**, and the current single-phase sequence can only provide
   one. This is the actual reason the fallback that exists on paper does not work.
3. **It cannot be asked for.** The pulse-mediated route is reachable today only as the last-resort
   system default, labelled `other`, after every named kind has failed. `MIC_PREFERENCE` cannot
   select it.

## Proposal

Add a **pulse-mediated capture route** as its own kind, `pulse`, and make the pulse state a function
of which route is being probed rather than a fixed prelude.

1. **Two-phase resolve.** Phase order follows `MIC_PREFERENCE`:
   - *pulse phase* — pulse UP, probe pulse-mediated candidates.
   - *raw phase* — pulse suspended (today's `free_i2s_device()`), probe I2S / USB / analog.

   Each phase leaves pulse in the state the *other* phase needs before handing over, so neither can
   poison the other's probe. This is the load-bearing change; the rest is plumbing.

2. **Target the source explicitly, do not move the system default.** `PULSE_SOURCE` in the process
   environment selects which source a libpulse client records from, so the route opens PortAudio's
   pulse-backed device with `PULSE_SOURCE=<PULSE_CAPTURE_SOURCE>` set. That is the mirror of what
   playback already does with `paplay --device=TTS_SINK`, and it mutates no global state.

3. **Narrow the exclusion to raw capture.** `_is_speaker_card()` keeps refusing raw `hw:N` devices on
   `TTS_CARD`'s index — unchanged, permanently. A pulse-mediated candidate for the same card is
   allowed, because it is a different mechanism with a different failure mode.

4. **No decimator.** Pulse resamples, so the route asks for 16 kHz directly and
   `MicStream.open()`'s existing `if mic.rate == SAMPLE_RATE: self._decim = None` applies. This
   deletes the 44.1 kHz integer-ratio problem for this route rather than solving it.

5. **Off by default.** `PULSE_CAPTURE_ENABLED = False`. A robot that works today must not change
   behaviour because this landed.

## Non-goals

- **Raw capture on the speaker's card.** Stays blocked forever. This ticket does not weaken that
  guard; it routes around it.
- **Echo cancellation / barge-in.** Sharing one card puts the mic and speaker on a common ground as
  well as a common chassis, so bleed gets *worse*, not better. The existing mute gate is still the
  only defence and barge-in stays off.
- **Re-deriving `TTS_SINK` / `TTS_CARD` automatically.** Still hand-configured. If the operator moves
  the speaker to a different dongle they must update both, and `scripts/mic_survey.py` reports
  whether they resolve.

## The unmeasured part

**Does `pactl set-card-profile` disturb a live pulse-mediated capture?** Going through pulse means
there is no raw ALSA stream for the profile change to yank, which is the entire reason this should be
safe — but nobody has measured it on this hardware.

If it does disturb the stream, the existing mic watchdog (`MIC_STALL_S`, `MIC_REOPEN_BACKOFF_S`)
reopens it, so the failure mode is a reopen at the first reply rather than a crash. That is the
argument for building it behind a flag rather than waiting: the downside is bounded and already
handled, and the flag makes it revertible without a code change.

`scripts/mic_survey.py --probe --suspend-pulse` measures the *other* open question — whether a
pulse-mediated capture delivers audio while sources are suspended — which is what justifies the
two-phase split above.

## Acceptance Criteria

- [x] A pulse-mediated candidate on `TTS_CARD`'s ALSA card is probed and can be selected, while a
      **raw** `hw:N,0` device on that same card is still refused
- [x] The pulse phase probes with pulse sources UP; the raw phase probes with them suspended; neither
      leaves the other's state behind
- [x] `MIC_PREFERENCE = "pulse"` selects the route; every other kind is still probed after it, so it
      cannot leave Kai deaf
- [x] `PULSE_CAPTURE_ENABLED = False` reproduces today's behaviour exactly, probe order included
- [x] The route opens at 16 kHz with no decimator, so a 44.1 kHz-only card is usable through it
- [x] `MicChoice.kind` / `sess_mic_kind` / `/audio/reresolve` report `pulse`, and the dashboard names
      it as the mic jack on the speaker's dongle
- [x] `resume_pulse_sources()` is still symmetric with `free_i2s_device()` across both phases
- [x] The legacy per-turn path (`ensure_input_resolved`) is not left behind by the two-phase change
- [ ] **Verified on the robot:** the mic works, and TTS is still audible after several replies —
      specifically that the first reply's `set-card-profile` does not kill capture

## Notes

- Blocked-on-hardware, not blocked-on-code: everything above is unit-testable, and the last two
  criteria are not.
- Related: S15 (the `analog` kind, for a 3.5mm mic on a *separate* adapter — the configuration that
  needs none of this).
- The suspend → resolve → resume sequence is duplicated in four places
  (`ai/mic_stream.py`, `ai/voice_assistant.py`, `scripts/wake_test.py`, `scripts/mic_survey.py`).
  Two phases make that duplication materially worse. Consider hoisting it into one function in
  `ai/mic_device.py` as part of this work rather than after it.
