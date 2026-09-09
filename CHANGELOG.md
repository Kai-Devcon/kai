# Changelog

Notable changes to Kai, newest first.

Entries dated **2026-08-07 and earlier** were reconstructed from the running log that used to live
in `README.md`'s TL;DR section — they are reproduced verbatim, so their wording, tense and any
claims they make are as originally written. Anything they say about the code reflects the state on
that date, not today's.

The style is deliberate: each entry says what changed, and — where it was hard-won — *why*, with the
measurement that justified it. That is the same convention the code comments follow, and it is the
reason this file is worth keeping by hand rather than generating from commit subjects.

Conventions:

- **Newest first.** Multiple entries on one date stay in the order they were originally written.
- One heading per change, not per commit. A change that took six commits gets one entry.
- Behaviour, measurements and reverts belong here. Refactors with no observable effect do not,
  unless they change how the thing is operated or debugged.

---

## 2026-09-08 — A stale default device index could kill the session-start thread outright

Found by a test guard, not by the robot. `_candidate_input_devices()` bounds-checked the system
default index only where it read that device's NAME — the append that made it a *candidate* checked
merely that it was non-negative. An index past the end of the device list therefore became a
candidate, and `resolve_input_device()` then does `devices[idx]` and raises `IndexError`.

**Nothing caught it.** The `try/except` in `resolve_input_device()` wraps `sd.query_devices()` alone,
not the candidate loop; `MicStream.open()` has no guard around resolving; and `face_track`'s
`_start_session()` has none around `_session.start()`. So the exception would kill the
`kai-session-start` thread, skipping all 14 `SESSION_START_ATTEMPTS` retries — a permanently deaf
robot, from one stale integer, with a bare traceback as the only clue.

Latent for as long as it existed, because `sd.default.device` and `sd.query_devices()` are the same
PortAudio state and normally agree. What made it reachable is `refresh_devices()`, added days
earlier for hot-plug: it tears that state down and rebuilds it, which is exactly when the two can
disagree. A hot-plug feature turned a dormant bug into a live one.

**How it surfaced is worth recording.** Renaming `resolve_input_device` to
`resolve_capture_device` made six patch targets in `tests/test_voice_assistant.py` silently no-op,
so those tests ran the real `ai/mic_device` path: PortAudio enumerated the dev box's actual sound
cards and the suite sat through a 3 s liveness-probe timeout on each. **The run took 1040 seconds
instead of 16**, and what it reported depended on which microphones the machine had. Those tests
never meant to touch hardware — they were protected only incidentally, by patching a name.

So `block_real_audio_hardware()` now patches the layer where the hardware actually is, in the module
that owns it, for the two classes that resolve devices. A stale patch target now fails in
milliseconds and for the right reason. Writing that guard is what raised the `IndexError` that
exposed the bug above.

---

## 2026-09-08 — The mic jack on the speaker's own dongle, reached through PulseAudio

A single USB dongle with two 3.5mm jacks — speaker out, mic in — is the ordinary way to give Kai
both, and Kai could not use the mic half at all. Now it can, as a fourth kind, `pulse`.
`PULSE_CAPTURE_ENABLED`, off by default. See
`docs/tickets/S16-mic-on-the-speakers-own-dongle.md`.

**Why it was refused.** One dongle is one ALSA card, and the card — not the jack — is the unit of
configuration. `pactl set-card-profile`, which `tts.play()` asserts before the first reply, acts on
the whole card; on 2026-08-11 that re-opened the card's ALSA devices underneath a live **raw**
capture stream and the process took SIGSEGV at the startup greeting, then greeted the room again on
relaunch. So raw capture there is refused. That guard is unchanged and stays: this route goes
*around* it, it does not relax it.

**The load-bearing change is that the two routes need opposite pulse states.** A raw hw open of the
INMP441 needs the card taken off pulse or it is locked to 44100 with noise injected. A
pulse-mediated capture needs the source pulse holds to be *un*-suspended or it delivers silence.
`MicStream.open()` ran `free_i2s_device()` once at the top and then resolved — so the pulse route was
always probed in the one state guaranteed to fail it. That is why the fallback that existed on paper
never worked, and no amount of naming or preference would have fixed it. Resolving is now two
phases, each run in the state it needs, ordered by `MIC_PREFERENCE`, and both always run so no
preference can leave Kai deaf.

That sequence also stopped being copy-pasted. `resolve_capture_device()` in `ai/mic_device.py` owns
it; `ai/mic_stream.py`, `ai/voice_assistant.py` and `scripts/wake_test.py` all call it instead of
carrying their own copy. Four copies of a one-phase prelude was tolerable; four copies of a
two-phase one was not, which is what forced the hoist.

Two smaller things fell out of it:

- **The route owns the pulse-backed device entries outright** when enabled. A test caught the raw
  phase claiming `"pulse"` as `kind="other"` and pre-empting the route — which would have recorded
  from pulse's *default* source rather than `PULSE_CAPTURE_SOURCE`: the same card by luck instead of
  on purpose, and unnamed on the dashboard. With the flag off they go back to being the last-resort
  fallback, exactly as before.
- **The source is selected with `PULSE_SOURCE` in our own environment**, not `pactl
  set-default-source`, which would change what every other program on the box records from. It
  travels on `MicChoice.env` so the *stream* is created with it too, not just the probe — a probe-only
  setting would have recorded from the right source once and the wrong one forever after.

**Also: no decimator on this route.** Pulse resamples, so it asks for 16 kHz directly. That deletes
the 44.1 kHz integer-ratio problem for this route rather than solving it — a dongle that can only do
44.1 kHz is unusable raw and perfectly usable through here.

**Not verified on hardware, and one thing is genuinely unmeasured:** whether
`set-card-profile` disturbs a live pulse-mediated capture at all. Going through pulse means there is
no raw stream to yank, which is the whole reason it should be safe, but nobody has measured it on
this hardware. If it does, the mic watchdog reopens the stream, so the cost is a reopen at the first
reply rather than a crash — a bounded downside, which is the argument for shipping it behind a flag
instead of waiting. `scripts/mic_survey.py --probe --suspend-pulse` measures the other half.

**The physical trade is not fixable in software:** one card puts the mic and speaker on a common
ground as well as a common chassis, so playback bleed into the mic gets worse, not better. Two
dongles remain the lower-risk arrangement and need none of this.

---

## 2026-09-08 — A 3.5mm mic is a third named input, not an unlabelled USB card

`analog` joins `i2s` and `usb` as a mic kind: selectable in `MIC_PREFERENCE` and on the dashboard,
reported as `sess_mic_kind`, and picked up by the existing hot-plug watcher with no new machinery.
See `docs/tickets/S15-analog-mic-is-not-a-selectable-input.md`.

The awkward part is that there is nothing to detect. The Jetson's own analog input is not wired to
anything, so a 3.5mm mic reaches Kai only through a USB→3.5mm adapter — and that adapter *is* a USB
sound card. Every signal that might distinguish it from a native USB mic is either shared with one
or absent, so the rule is a name match (`ANALOG_MIC_NAME_HINTS`) and nothing more, checked **before**
`USB_MIC_NAME_HINTS` because an adapter's name contains "usb" too.

**That rule is allowed to be wrong, and the design depends on it.** Classification decides a label
and a position in the probe order — never whether a device can be opened. An adapter whose name is
not in the list classifies as `usb` and works exactly as it did before; the cost is that the
dashboard cannot tell two USB inputs apart and `MIC_PREFERENCE` cannot choose between them. Nothing
about a wrong guess costs a working microphone, which is what makes a name-based rule acceptable for
hardware this un-namable. The defaults are plausible, not measured — `docs/hardware.md` has the
one-liner that prints what yours actually reports.

Rejected: inferring `analog` structurally from the card also having output channels. An adapter with
a headphone jack does, but so does every USB mic with a monitor output, and a mic-in-only adapter
does not. Not a discriminator — and a wrong structural guess is harder to explain than a wrong name
guess, because there would be no list to correct.

`MIC_PREFERENCE` is now "preferred kind first, then the standard order" rather than a hand-written
table per value, so adding a kind no longer means enumerating permutations. `analog` sits after
`usb` in `auto` on purpose: it is the newer kind, and ordering it earlier would silently change
which mic an already-working robot picks.

**Not verified on hardware.** Unit-tested only — nobody has plugged a 3.5mm mic into an adapter on
the robot yet. The three ways this fails that software cannot see are documented in
`docs/hardware.md`: an adapter offering only 44.1 kHz cannot be used at all (an integer-ratio
decimator cannot resample it — the 2026-08-09 incident), an electret mic needs plug-in power the
adapter may not supply, and a 4-pole TRRS plug puts the mic on the wrong contact. All three present
identically in the log as `read as silent`.

---

## 2026-08-27 — A USB mic is now a microphone Kai has, not one he settles for

Kai could always fall back to a USB mic when the INMP441 read silent, but only as a fallback, only
at startup, and — on the most common hardware — not at all. Three separate problems, and the first
one is the reason this was worth doing rather than tuning:

- **The guard against capturing the speaker's card was also throwing away real microphones.**
  Capturing raw on the C-Media dongle while `tts.play()` reconfigures it segfaulted the process at
  the startup greeting (2026-08-11), so input devices matching `"usb audio device"` are dropped. The
  match is on the PortAudio *name* — and `"USB Audio Device"` is the most common name a cheap USB
  mic enumerates under. A perfectly good mic was being discarded before it was ever probed, with no
  log line saying a microphone had been rejected. The card is now identified by resolving
  `TTS_CARD` to its ALSA card index through `pactl list cards` and matching that against the
  `hw:<N>` in the device name. That is exact: two devices that share a product name are still on
  different cards. Where `pactl` cannot answer — no pulse, a dev box — the name rule still applies,
  because the failure it prevents is a segfault and the failure it causes is one skipped candidate.
- **A mic plugged into a running Kai did nothing, and not for want of a retry path.**
  `/audio/reresolve` already re-ran the whole discovery sequence on demand — against a device list
  PortAudio snapshots at `Pa_Initialize` and never refreshes. The new card was invisible to it, so
  the button could do everything right and still find nothing. `refresh_devices()` forces the
  re-scan, from the only two places where no stream is open: inside `MicStream.reopen()`, and before
  `start()` on the never-started path. `ai/mic_hotplug.py` watches `/proc/asound/cards` — the
  kernel's list, not PortAudio's — and the session re-resolves at its next quiet moment. Never
  mid-turn: a swap that cut someone off mid-sentence would be worse than waiting one turn. A change
  must be seen on two separate polls and hold for 2 s, because USB enumeration is not atomic and the
  card appears before its PCM devices are openable — probing early reads the new mic as silent,
  which skips it, which is precisely the outcome plugging it in was meant to produce.
- **Startup giving up was the one case hot-plug could not reach.** If all 14 start attempts fail
  there is no tick loop, so nothing polls the watcher — and that is exactly the state a robot boots
  into with no mic attached, i.e. the most likely moment for someone to plug one in.
  `watch_for_a_mic()` keeps that already-finished startup thread watching instead.

Also: `MIC_PREFERENCE` (`auto` / `i2s` / `usb`, live on the Microphone card) decides which kind is
probed first. It **reorders and never excludes**, so preferring one mic cannot leave Kai deaf when
only the other is plugged in. It is the one dashboard knob that does not apply instantly — a
preference is only read while a mic is being resolved — which is why it sits next to the button that
resolves one rather than in the Settings tab, where the footer promises everything is immediate.
A USB mic now also gets one silent-read retry of its own (`USB_PROBE_SILENT_RETRIES`) and its own
channel treatment, instead of inheriting `I2S_TAKE_CHANNEL`, which was harmless only because it
happens to be 0. And `resume_pulse_sources()` now hands back every source `free_i2s_device()`
suspended rather than just the I2S one — invisible while USB was only ever a raw-device fallback,
and wrong the moment a run is *meant* to end on the USB mic.

`sess_mic_kind`, `sess_mic_hotswaps` and `sess_mic_hotplug_watching` are on `/params`, and
`/audio/reresolve` now returns `kind`, because "not the I2S mic" stopped being an honest label once
the alternative had a name worth saying.

---

## 2026-08-27 — Pulse was taking the USB mic back in the gap between finding it and opening it

The BY-PM700 above was found correctly and then failed every single open. The log said so plainly
and still read as a hardware fault:

```
[mic] selected usb mic: device 25 (BY-PM700: USB Audio (hw:3,0)) at 48000 Hz
[mic] resolved device=25 rate=48000 ch=1 kind=usb i2s=False — opening stream…
[mic] ERROR: could not open microphone: Error opening InputStream: Device unavailable [-9985]
```

Found live at 48 kHz by a probe that opened the device for real, then unavailable a few milliseconds
later, forever, on a fourteen-attempt retry loop that never converged.

The gap between those two lines is `resume_pulse_sources()`. It ran on every non-I2S resolve, and
the entry above had just widened it to hand back *every* source rather than only the I2S one — which
is correct for the case it was written for and wrong here. This build loads no
`module-suspend-on-idle`, so pulseaudio holds every card it owns open permanently: handing card 3
back means pulse re-grabs `hw:3,0` within milliseconds, and a raw ALSA device admits exactly one
opener. Kai was giving the microphone away and then failing to open it, once per attempt.

The mistake underneath is that `is_i2s` was being used to mean "we are about to open a raw device".
It never meant that. A USB mic resolves to `hw:<n>,0` — just as raw, just as exclusive — with
`is_i2s` False, and that was invisible for exactly as long as USB was a fallback nobody expected a
run to end on. `MicChoice` now carries the ALSA `card` of the resolved device (empty for the
pulse-mediated PCMs, which are not exclusive), and `resume_pulse_sources(keep_card=...)` hands back
every card *except* the one about to be opened raw. Sources pulse cannot attribute to a card are
left suspended, because guessing wrong in that direction costs a muted source somewhere else and
guessing wrong in the other one costs the microphone. The kept source is retained in the tracked
list, so the next unqualified resume still hands it back.

**And the error that made this expensive to read.** Interleaved with the above, on the reopen path:

```
[mic] resolved device=25 … opening stream…
[mic] reopening (attempt 1)
[mic] ERROR: could not open microphone: Error opening InputStream:
      Illegal combination of I/O devices [PaErrorCode -9993]
```

`paBadIODeviceCombination` is PortAudio's error for an input and an output device on different host
APIs. It cannot honestly occur on an input-only stream, and it did not: the stall watchdog's
`reopen()` called `refresh_devices()` — `Pa_Terminate` then `Pa_Initialize` — on the audio worker
thread while the dashboard's re-resolve button was inside `Pa_OpenStream` on the HTTP thread. The
host API being rebuilt underneath an open surfaces as that error, and it sends you looking for a
duplex device problem on a stream that has no output side. `ai/mic_device.pa_lock` now serialises
the two: `MicStream.open()` holds it across resolve *and* open, `reopen()` holds it across
close → refresh → open, and `refresh_devices()` takes it around the terminate/init pair.

Verified on the robot: `sess_mic_live: true`, `sess_mic_kind: "usb"`, one watchdog reopen at
startup and none since.

---

## 2026-08-12 — The Bisaya filler was Cebu Bisaya, in a room full of Bukidnon and Iligan

Read for dialect rather than for code. The `ceb` bank in `config/filler.py` is understandable
anywhere Cebuano is spoken, but it was not written for the chapters that will hear it: DEVCON lists
**Bukidnon and Iligan** as active and Cebu as a separate chapter entirely
(`documents/devcon_faq_rag.md`). One line named a Cebu City street; four carried grammar errors that
mark the text as written by a non-speaker rather than as a regional accent. All four openers and one
stall changed. No code changed — this is content only, so the 47 tests in `tests/test_filler.py`
that police the bank (ASCII-only, sentence shape, stall length) still pass unaltered.

- **`"tiangge sa Colon"` was the only outright geographic error.** Colon Street is Cebu City. It
  means nothing in Malaybalay or Valencia and reads as a foreign reference in Iligan — worse than no
  local reference at all. Now `"tiangge sa merkado"`, deliberately generic, because Bukidnon and
  Iligan share no market between them. A demo pinned to one city should name that city's (Cogon on
  the CDO side, Palao for Iligan); the comment in the file says so.
- **Four grammar errors and one Tagalog particle.** `"Wa nako kadungog"` -> `"Wa ko kadungog"`
  (`nako` is genitive and collides with the `ka-` verb); `"mao paspas kaayo ko"` -> `"maong ..."`
  (bare `mao` cannot carry "that's why"); `"sa akong nawong"` -> `"sa akong ulo"` (`nawong` is the
  face, and the Jetson is in her head); `"ayaw ko'g i-off"` -> `"ayaw ko i-off"`; `"Wala koy
  kinahanglan nga internet"` -> `"Dili ko manginahanglan ug internet"` (the first is a literal
  translation of the English). The stall `"Balik-balika nga."` -> `"Balika daw."` — `nga` as a
  softener is Tagalog; in Cebuano it is a linker.
- **What was deliberately left alone.** The `gina-` progressive (`"gina-process"`, `"Ginaproseso"`)
  is a genuine Mindanao marker where Cebu City leans on `gi-` — it is the most locally-correct thing
  in the bank and a tidy-up toward textbook Cebuano would remove it. `"huwat"` stays misspelled for
  `"hulat"` because it is a choice made for the **phonemizer**, not the reader: every line here is
  spoken by an American English voice under English phonetics, and `"hulat"` invites the /hjuː/
  reading. Both are now written down in the file so the next editor does not "fix" them.

**`scripts/filler_check.py` cannot score this bank, and the run proves it rather than assuming it.**
All 14 `ceb` lines came back at overlap **0.00** — including the nine stalls this change never
touched. That is the checker's documented limit (Whisper transcribing Bisaya through a
mostly-English model), not a defect in the lines: the failure it exists to catch is espeak spelling
a word out letter by letter, and nothing did. `"Ay, huwat sa."` was heard as `"I, you, what's up?"`
and `"Hapit na."` as `"Happy New Year!"` — wrong words, but words. The objective half of the check
did pass: measured on the robot, the four openers run **7.06 / 7.66 / 7.71 / 8.69 s** against
`FILLER_MAX_LINE_S` 10.0, and every stall **0.88-1.61 s** against `FILLER_MAX_STALL_S` 1.8, so
nothing new will be silently dropped by the length cap at prewarm. WAVs kept at
`/tmp/kai_filler_ceb` for the by-ear pass, which a native Bukidnon or Iligan speaker still owes this
change. Note that the script's own `MAX_STALL_S` (1.2) is stricter than the constant that actually
governs shipping (1.8), so it flags four stalls that the robot will happily cache.

Suite unchanged by this entry — no code was touched, and `tests/test_filler.py` is 47/47. The two
platform-split pre-existing failures it ran into are the ones the entry below already names and
confirms independently: on the Jetson,
`TestAmbientAdaptation::test_without_adaptation_the_same_room_never_closes_the_utterance`
(measured here as deterministic — three full runs of three, and again in isolation, with uptime at
216 808 s, so it is not the clock flake), and on Windows, `TestImportPvporcupine`. Restart-only: the bank is pre-synthesised at startup, so the old WAVs play until
`face_track.py` is restarted.

## 2026-08-12 — The jaw moved before the "Yes?" came out

Reported on the robot: say "Hey Kai", and the mouth moves first with the acknowledgement arriving
afterwards. Two independent errors, both at the start of the file and both pushing the same way, so
they added. Neither was visible on a long reply — on an 0.82 s word they are the whole illusion.
Suite: 1268 passed, 2685 subtests (was 1260), plus the one pre-existing failure this box always has
(`test_audio.py::TestImportPvporcupine` shells out to `cat /proc/cpuinfo`, so it can only pass on
Linux — verified against a stashed tree). On the Jetson: 1268 passed and a different pre-existing
failure, `TestAmbientAdaptation::test_without_adaptation_the_same_room_never_closes_the_utterance`,
likewise confirmed to fail without this change.

- **Playback does not start when it is asked to.** `ai/voice_assistant._speak` and `speak_wav` both
  opened the jaw window at `time.monotonic()` immediately before calling `tts.play()` — but that
  call only spawns `paplay`, and the first sample is not audible until the process has connected to
  PulseAudio and the stream buffer has filled. Measured with `pactl list sink-inputs` sampled
  through a 5 s playback using the same flags `ai/tts.py` passes: Buffer Latency 57–120 ms plus Sink
  Latency 63–98 ms, i.e. ~190–210 ms of pipeline before anything is heard, plus paplay's own spawn
  (~30 ms warm, ~210 ms on a process's first playback, which also pays `apply_output_profile()`).
  The schedule now starts at `now + TTS_PLAYBACK_LEAD_S` (0.22), and `speaking_openness_at()`
  already returned `None` ahead of `start`, so the jaw simply stays shut through that window.
- **The file is longer than the speech.** Piper pads silence onto both ends of every WAV it writes.
  The ack the robot is actually playing is 0.824 s, of which 0.040 s is leading pad and 0.244 s is
  trailing pad with `TTS_POST_ROOM`'s reverb decaying into it — so barely two thirds of that file is
  the word "Yes?", and `_speak_segments_for_duration` was miming across all of it, holding the mouth
  open a quarter-second after the sound stopped. New `tts.wav_speech_span()` finds the audible span
  and the jaw is timed to that instead. Threshold swept 50–3000 over the robot's own WAVs rather
  than picked: there is a plateau at 300–1000 (the ack's span lands within 20 ms across it),
  `SPEAK_SILENCE_PEAK` is 600. The scan costs 1.3 ms on the ack, 1.6 ms on an 8 s line. It is a scan
  rather than a constant because the pad is not fixed — the canned lines are re-synthesized at every
  startup, and the restart that deployed this wrote an ack of the same 0.824 s with the speech at
  0.080–0.620 instead of 0.040–0.580.

Verified on the robot by sampling the `jaw` servo angle from `/params` against the `paplay` process
while firing a bare `POST /voice/wake`: the jaw holds at its closed 90° for the first 0.31 s of
playback, opens at +0.36, peaks at 170° at +0.56, and is shut again by +0.97 — against a predicted
0.30–0.84 plus the servo's own slew. It used to start opening at +0.00. Note for anyone repeating
this: `/params`' `voice_speaking` cannot see it, because `session._project_status` hard-overrides
that field to `True` for the whole of `STATE_ACK` — it reports session state, not the jaw envelope.

Both offsets are applied in one place, `_arm_jaw_for_wav`, so the reply, the ack, the canned
failures and every filler line line up the same way. The mic gate moved with the first fix but not
the second, and that asymmetry is deliberate: it protects against Kai hearing itself, so it wants
the whole file plus the lead — it had been opening ~0.22 s early, into audio still coming out. The
fallbacks are intact and both fail toward today's behaviour: a WAV that cannot be scanned, or is
silent throughout, mimes the whole file rather than not moving the jaw at all. `SPEAK_TRIM_SILENCE`
and `TTS_PLAYBACK_LEAD_S = 0.0` restore the old timing exactly.

`FILLER_MAX_STALL_S` deliberately still measures the file: `config/filler.py` had been carrying a
note that measuring speech instead "belongs in its own pass", and this is that pass — but what that
cap bounds is how long the speaker is busy and Kai is deaf, which includes the pad.

## 2026-08-11 — A long filler line now sits out two exchanges before it can come back

The once-per-conversation rule (`_filler_used_openers`) only covers the FIRST lap through a
language's openers. After that the rung is empty permanently and the only thing left was a
single-entry back-to-back guard — so on the four-line `ceb` and `en` pools, which lap in four
exchanges, a conversation could settle into A B A B for the rest of the demo. Two lines alternating
announces itself as canned faster than either line simply recurring does.

`FILLER_OPENER_COOLDOWN_TURNS = 2` (`config/filler.py`): the session keeps the openers of the last
two exchanges instead of just the last one, so the tightest possible recurrence is A B C A. Not
larger, because a pool must hold more than the window for it to bind at all — and the length cap can
leave a language with as few as two warm openers (robot, 2026-08-09: `ceb 1op/10st, en 2op/10st`
after a restart). For that case `ai/filler._off_cooldown` relaxes the window one exchange at a time,
**oldest bar first**: a flat "bar all of them, else allow everything" fallback would hand back the
line that had just played, which is the one bar that matters most. A two-line pool therefore
alternates rather than repeating. Stalls are untouched — their no-repeat story is per-wait, not
per-exchange. Set the constant to 0 to switch the window off entirely. Restart-only.

Suite green: 1304 passed, 2710 subtests, the three new cases here being the only tests added.
Deployed and exercised on the robot — five
`/voice/wake` turns ran clean through `recording → transcribing → thinking → done` — but **which
opener each turn drew is unverified**: nothing about the filler is published on `/params`, so that
needs `/tmp/face-servo.log`, which needs a shell.

## 2026-08-11 — The startup greeting was spoken twice, a minute apart, with the jaw frozen partway through the first one

Not a bug in the greeting. `face_track` **segfaulted at the first playback** and
`scripts/autostart.sh` did its job — so the room heard the whole line again from the replacement
process. Two fixes: the crash, and the fact that a relaunch is audibly indistinguishable from a boot.
Suite green: 1258 passed, 2685 subtests (was 1244). Both verified on the robot.

- **Capture had landed on the same sound card as the speaker.** From `/tmp/face-servo.log`: across 22
  runs the I2S liveness probe timed out twice (`LIVE_PROBE_TIMEOUT_S = 3.0`), and both times selection
  fell through to `[mic] resolved device=0 rate=48000 ch=1 i2s=False` — input device 0 is
  `USB Audio Device: - (hw:0,0)`, the C-Media dongle, which is also the only output sink
  (`alsa_output.usb-C-Media…analog-stereo`). One of those two runs died `rc=139` (SIGSEGV) **32 s in,
  exactly as the greeting's first audio began** — which is when `tts.play()` runs its
  once-per-process `pactl set-card-profile` against that card, re-opening its ALSA devices underneath
  a live raw PortAudio capture stream. A milder form of the same collision is in the log as `could not
  set output card profile (… exit 1)` followed by `playback failed (Stream error: No such entity)`. So
  `SPEAKER_CARD_NAME_HINTS` (`config/voice.py`) drops such devices from `_candidate_input_devices()`
  entirely rather than ranking them last: they are not a worse mic, they are the one choice that can
  take the process down. **The trade is stated, not hidden** — when the I2S mic fails to probe, that
  run gets the pulse-mediated system default (`MicChoice(None, …)`; pulse coordinates access to the
  card, so it is safe where a raw open is not), and if nothing usable is left Kai is deaf for that run
  and logs why. `/audio/reresolve` retries it without a restart. Verified on the robot: device 0 is
  gone from the candidate list, the INMP441 still wins with `resolved device=5 … i2s=True`, and the
  named `default`/`pulse` entries are deliberately still there — they do not match the hints, which is
  the point.
- **`_greeted` could not see a relaunch, because it is a per-instance latch.** It correctly stops
  `reresolve_mic()` re-greeting inside one process, but the second greeting came from a *different*
  process 57 s later. The fact is now also written to `GREETING_STAMP_PATH`
  (`/tmp/kai_ack/kai_greeted.stamp`), and a process starting within `GREETING_REPEAT_SUPPRESS_S`
  (90.0) of the last greeting stays quiet and says so: `greeting suppressed — a Kai process greeted
  39s ago … this is what a crash-and-relaunch looks like from here`. **The stamp is written before the
  audio starts, not after** — the run this exists for died partway through its own greeting, and a
  stamp written on completion would never have been written by that run at all. 90 s is sized against
  the relaunch cost (5 s supervisor backoff + up to 15 s waiting for the capture device to be released
  + ~30 s of startup ≈ 50 s at the fastest), not against taste; it also covers double-tapping
  `/restart`. It lives under `ACK_WAV_DIR` because it wants that directory's lifetime: **/tmp is
  cleared on reboot, so a genuine cold boot always greets.** Verified live — two restarts 45 s apart
  gave one 9.4 s greeting and one suppression line.

Measured while looking, and unchanged by this entry: the greeting is **9.0–9.3 s of audio** (141
chars, three sentences, ~1.05 s of it the `tts_sentence_silence` pauses) and starts **~31 s after
launch**, because `_warm_all()` puts the four canned lines first — including up to four Piper passes
fitting the `thinking` line to `THINKING_SOUND_TARGET_S` — and only then synthesises the greeting
live, on the most contended 30 s the Jetson has. The comment on `GREETING_TEXT` still claims the
middle clause costs "roughly three extra seconds"; that estimate predates the current line.

Still true, and worth knowing: `tts.play()` retries a failed `paplay` **from the start of the WAV**,
guarded only on "has a newer utterance superseded me". It was innocent here — the log shows a real
crash, not a retry — but a `paplay` that fails *partway* through still replays the whole line.

## 2026-08-11 — One corrupted byte on the servo wire could slam the head into its stop

Suite green: 1254 passed, 2710 subtests (was 1249, 2710). Implements
[R4](docs/tickets/R4-firmware-servo-limits-mismatch.md). **Firmware — inert until the Arduino is
flashed, and not compiled here: no Arduino toolchain was available on either box.**

- **The limit that protects the SG90 existed only on the host.** `config/servo.py`'s `SERVO_MIN`/
  `SERVO_MAX` (10/170) are applied in `servo/servo.py`'s `send()` and `send_jaw()`, but the sketch
  constrained to the full `0..180` in all three of its `constrain()` calls. The host's clamp is
  applied and then *destroyed in transit* if the line corrupts, and the link is fire-and-forget by
  design — no checksum, no echo, no ack — so nothing downstream can tell a mangled line from a real
  one. `ANGLE_MIN`/`ANGLE_MAX` now live in the sketch, which is the copy that survives the wire.
- **`String::toInt()` returns 0 for anything it cannot parse, and 0 is the worst value on this
  wire.** Not an inert default — a hard slam to the end of travel. So the one input the old parser
  could not report was also the most damaging thing it could command, and the CH340 is documented as
  flapping under servo brownout, which makes corruption *correlated* with the condition that makes a
  stall worse: the stall current from a slam to 0 lands on the same rail whose sag caused the flap.
  `parseAngle()` replaces it — rejects an empty field, any non-digit, and anything over 3 digits.
- **A line is now applied whole or not at all.** Every field is parsed before any servo is written,
  so a good pan with a corrupt jaw moves nothing. The tilt field is validated and then thrown away:
  there is no tilt hardware (R10), but garbage in tilt means the *line* is corrupt, and the pan
  field sitting beside it has no better claim to being intact.
- **The "keep these in step with config/servo.py" comment is backed by a test.** R4 asked only for
  the comment. `tests/test_servo.py::TestFirmwareAngleLimits` reads the real `.ino` and fails if the
  constants drift, if a full-range `constrain(..., 0, 180)` reappears, or if `toInt()` comes back.
  It strips C++ comments before searching — the first version matched this sketch's own explanation
  of why `toInt()` is wrong and failed on the change that fixed it.

What is still true: the host-side clamps are untouched, so this is defence in depth rather than a
relocation, and `G:` gesture lines and the `J` fast channel dispatch exactly as before.

**Compiled and flashed the same day, after an earlier note here wrongly said there was no Arduino
toolchain.** That check had been run on the Windows dev box rather than the robot, and it searched
for `gcc-avr` — a Debian package name, never a binary; the binary is `avr-gcc`. The Jetson has had
`avr-gcc`, `avrdude 6.3`, `arduino-builder` and `arduino-core-avr` all along. The new firmware is
**6122 bytes against the old 6262** — `parseAngle()` costs less than the `String::toInt()`
instantiations it removes — with zero warnings under `-warnings all` and RAM unchanged at 262 bytes.

The board had to be probed, because it enumerates as a bare CH340 (`1a86:7523`) with no Arduino
VID/PID: **ATmega328P, signature `0x1e950f`, optiboot at 115200** (57600 and 19200 do not sync).
Flash verified twice — avrdude's own verify plus an independent readback diff, 0 of 6122 bytes
mismatched — and the previous firmware was read off the chip beforehand and kept at
`~/firmware-backups/servo_serial-PRE-R4-20260811-083436.hex`.

On the live link the new firmware boots to `READY`, accepts every legal form, survives empty fields,
letters for digits, signed values, run-together lines, raw binary noise and truncated numbers, and
still answers a reset with `READY` afterwards — so the parser does not wedge, which is the failure
mode a hand-rolled one would actually have. **What could not be checked remotely is that the rejected
lines produced no motion**: the link is fire-and-forget with no echo, so the board says nothing that
distinguishes "rejected" from "moved". That half needs someone watching the head.

## 2026-08-11 — The module that decides whether Kai has a camera had no tests

Suite green: 1301 passed, 2710 subtests (was 1270, 2710). Implements
[S8](docs/tickets/S8-camera-supervisor-untested.md). Tests and one extracted method; no behaviour
change.

- **`app/camera_supervisor.py` was the only substantial module in the repo nothing imported from a
  test.** Its own docstring had advertised testability since it was written — "it knows about
  `vision/` and `settings.py`; it does not know about Flask, MediaPipe or the servo, so it can be
  driven with a fake camera and a fake clock" — and nothing took it up.
- **The untested logic is where the recorded bugs are.** The depth-3 swap queue exists because a
  depth-1 one silently evicted a swap `CameraThread` had not applied yet. The `showing_live` gate on
  the stall check exists because a stalled CSI pipeline was reported as a live feed at 0 fps. The
  `cheap = not device_signature()` branch exists because a backoff was punishing failures that cost
  microseconds to discover. Nothing was stopping a refactor from undoing any of them.
- **`run()`'s while body became `_step() -> float`.** The ticket's own suggestion, and a pure move:
  the decision logic is byte-identical and only the two loop-carried locals moved onto the instance,
  so a single pass is meaningful on its own. `_step()` also consumes `_probe_now` and publishes
  `next_probe_at`, so a test drives exactly what the supervisor does rather than a subset. `run()`
  is now the wait and nothing else — the same shape as `ConversationSession.tick(now)`.
- **31 tests, and they were verified by mutation rather than by passing.** "All green" is also what
  an empty test file reports. Ten deliberate regressions were applied one at a time — each undoing a
  behaviour the module was written to have — and the suite re-run against every one: **10 of 10
  caught**, including all three of the recorded bugs above, the `and last` guard that stops a camera
  being judged dead before its first frame, and `--no-camera` losing to the stored setting. Source
  restored afterwards; the committed file differs from `main` only by the `_step()` extraction.

Two cases the ticket did not ask for and the tests cover anyway: `--no-camera` releasing a camera
already held must report the *locked* reason rather than "camera off (settings)", or the dashboard
blames the wrong thing; and `_report_failure` publishes the reason to the dashboard on every call
even though it only logs on a change — the rate limit is on the log, not on the state.

## 2026-08-11 — Every extra dashboard tab took the session lock 20 more times a second

Suite green: 1270 passed, 2710 subtests (was 1263, 2710). Implements
[S2](docs/tickets/S2-params-sse-snapshot-per-client.md).

- **The cost scaled with the number of open tabs, on the locks the robot needs most.** Each
  connected browser gets its own Flask generator thread calling `params_snapshot()` 20 times a
  second, and that builds a ~70-key dict from four sources. One of them, `session.get_status()`,
  holds the session `RLock` for most of its body and takes the assistant lock *inside* it — the same
  `RLock` the ~30 blocks/s audio worker needs for VAD and the 20 Hz session tick holds. Two or three
  tabs at a venue is a realistic load that had never been tested.
- **And it was pure overhead.** The publishers only write at 25 Hz (`WEB_PUBLISH_INTERVAL`), so most
  of that work produced JSON byte-identical to the previous tick. `DashboardState.cached_snapshot()`
  now sits in front, with `WEB_PUBLISH_INTERVAL` as the max age — the rate the data can actually
  change at, rather than a number chosen to feel safe.
- **Measured builds per second: 1 tab 20.0 → 20.0, 2 tabs 40.0 → 20.0, 3 tabs 60.0 → 20.0, 5 tabs
  100.0 → 20.0, 8 tabs 160.0 → 20.0.** Flat. **The single-tab case saves nothing** —
  `_PARAMS_POLL_S` (0.05) is longer than `WEB_PUBLISH_INTERVAL` (0.04), so one client's polls always
  find the cache expired. The ticket is about load scaling with tab count, and that is what goes
  away; nobody testing with one tab open will see a difference, which is worth knowing before
  someone concludes the change did nothing.
- **One builder, not a thundering herd.** The ticket sketched the build outside any lock and accepted
  duplicate builds under a race as harmless. It is harmless for correctness, but "a new client
  connecting does not trigger an extra build" is one of the ticket's own criteria, and a wave of
  tabs hitting a cold cache is exactly that case. A second `_build_lock` admits one builder; the
  losers wait, re-check, and find the fresh snapshot. `build()` still never runs under the lock that
  guards the cached value, so a slow `session.get_status()` cannot serialise readers.

What is unchanged: `params_snapshot()` itself is untouched and still directly testable, the SSE key
set and per-client cadence are identical, and `dashboard.html` needed no changes. What is unproven:
the criterion that matters most — three `/params` clients plus a `/video` client with
`[control] N Hz` holding and `sess_blocks_dropped` flat — measures contention on the robot and stays
open.

## 2026-08-11 — The main loop ran 200 times a second to decide it had nothing to do

Suite green: 1263 passed, 2710 subtests (was 1254, 2710). Implements
[R2](docs/tickets/R2-idle-spin-loop-gil-contention.md). Measured **185.0 Hz → 22.0 Hz on the idle
path, 8.4× fewer iterations** — on the dev box, not the Jetson; see the caveat at the end.

- **The loop polled for frames instead of waiting for them.** `CameraThread.latest()` is
  non-blocking and returns `None` until a fresh frame arrives, so `face_track.run()` slept
  `NO_FRAME_SLEEP` (5 ms) and went round again. With `--no-camera`, an unprobed camera, a stalled
  feed, or simply between frames at 30 fps, that is ~200 iterations a second — each doing real work
  before deciding there was nothing to do: a `settings.get()` under an `RLock`, a
  `speaking_openness()` call taking the assistant lock, a time comparison.
- **All of it holds the GIL, which `config/tracking.py` records as the resource this box actually
  runs out of.** The note on `INFERENCE_FPS` measured raising perception 15 → 22 fps collapsing the
  pure-Python control thread from ~14 Hz to 6–10 Hz, with CPU, memory and thermals all in hand. A
  permanent 200 Hz Python loop contends for exactly that, alongside the 15 Hz control thread, the
  20 Hz session tick and the ~30 blocks/s audio worker — and it is worst in the degraded states
  (`--no-camera`, stalled feed) where the servo and voice paths are the only things still working.
- **`CameraThread` now signals a stored frame; the loop blocks on it.** A `threading.Event` set and
  cleared *inside* `_lock` alongside `_frame`, which makes "event set" and "frame waiting" the same
  fact rather than two that can disagree — the failure mode being a set event with nothing behind
  it, which restores the full-speed spin silently and with nothing else looking different.
  `close()` sets it too, so shutdown does not pay out a timeout on a loop that is already stopping.
- **This is not a latency trade.** The old path slept a fixed 5 ms and could not notice a frame that
  arrived 0.1 ms into it; the loop now wakes on the store itself, so a frame is picked up sooner
  than the poll managed on average.
- **The wait is bounded by `WEB_PUBLISH_INTERVAL` (0.04), not `JAW_SEND_INTERVAL` (0.05).** R2
  suggested the latter, and it would have quietly dropped `/params` and the `cam_retry_in_s`
  countdown from 25 Hz to 20 Hz — which the same ticket's third acceptance criterion forbids. The
  ticket's criteria disagreed with each other and the tighter obligation wins. Written as
  `NO_FRAME_WAIT = WEB_PUBLISH_INTERVAL` rather than a literal so the two cannot drift, with
  `tests/test_settings.py::TestIdleWaitBounds` pinning it against `JAW_SEND_INTERVAL` as well — the
  two constants live in config modules that deliberately import nothing, so a test is the only place
  that relationship can be stated.

What is still unproven: the measurement above is off-robot, and the *reason* for the change — GIL
headroom for the 15 Hz control thread — is a Jetson claim. `[control] N Hz` with `--no-camera`
should now read at or above its old value, and that number is worth capturing before and after on
the next deploy. R2's two on-hardware criteria stay open until it is.

## 2026-08-11 — Kai was telling people it runs on Micro:bit, Qwen and Google AI Suite

Suite green: 1237 passed, 2680 subtests (unchanged). Retrieval was working correctly the whole
time — the document it retrieved was wrong.

- **The `kai-stack` entry in `documents/devcon_faq_rag.md` listed three things this repo has never
  used.** Asked what powers it, Kai read back "Claude Code, Micro:bit, Qwen, Google AI Suite, and
  NVIDIA" — verbatim from the chunk, which wins that question comfortably on `SECTION_BOOST`'s
  +0.08 for `"Kai (robot)"`. The LLM is `gemma2:2b` (`config/voice.py`), the microcontroller is an
  Arduino on `/dev/ttyUSB0` (`config/servo.py`, `servo/servo.py`), and there is no cloud AI call
  anywhere in the tree — STT is faster-whisper, embeddings are `BAAI/bge-small-en-v1.5` through
  fastembed's local ONNX runtime, TTS is Piper. The entry also contradicted its own neighbour
  `kai-hardware`, two lines up, which correctly says "no cloud round trips". It now names the real
  stack, and stays speakable: "Gemma 2", not `gemma2:2b`.
- **The accuracy harness had been scoring the false fact as a pass.** `scripts/rag_accuracy.py`
  asked "what tools were used to build you?" and looked for the needle `"Micro:bit"` — which the
  document did contain, so the case went green every run. The harness only checks that a document
  reached the model, never that the document was true; that is a real limit of it, not a bug, and
  the mitigation is to point needles at facts a reader can diff against the code. Needle is now
  `"Arduino"`, and deliberately not `"Claude Code"` — the Jumpstart internship entry names Claude
  Code too, so a substring test on it would pass while retrieving the wrong entry.
- **Wording was measured, not just written.** The first draft opened "Kai was built with Claude
  Code" and cost a case: "who made you?" started retrieving the stack entry instead of `kai-origin`
  ("Cohort 4"), 29/31 -> 28/31. Rephrased to "its code was written with Claude Code" and the origin
  entry keeps the question. Final: **29/31 answer-in-context, unchanged from before the edit**, and
  `scripts/rag_eval` is identical — on-topic 0.552–0.770 8/8, off-topic 0.520–0.637 with 4/8
  rejected at `SIMILARITY_THRESHOLD` 0.55, overlap 0.085. The numbers recorded against that
  constant in `config/rag.py` still hold.

`documents/.rag_index.json` was rebuilt (`python3 -m ai.index_documents`, 51 chunks) — the edit does
nothing until it is, and `_warn_if_stale()` is what says so at startup if someone forgets. One claim
in the neighbouring `kai-hardware` entry is still unverified and was left alone: it says Kai uses
"TensorRT for near-zero-latency neural network execution", but TensorRT appears only in
`requirements.lock.txt` (it ships with JetPack) and in `docs/plan/wip/tensorrt-plan.md`, which is a
plan, not a build. Nothing in the tree runs it.

## 2026-08-11 — Kai never took a breath, and spoke in no room at all

Suite green: 1249 passed, 2710 subtests (was 1239, 2705). Every number below was measured on the
robot, most of them from the WAV the live `_speak` path produced, and every part reverts from
`config/voice.py` alone. One judgement is deliberately left open — see the last bullet.

- **Piper's `--sentence-silence` was never passed, and its default is 0.** `ai/tts._run_piper` built
  its command line from `-m`, `--length-scale` and `-f`, so a four-sentence reply came out as one
  continuous run. Measured longest interior silences at the three sentence boundaries of the same
  reply, raw pre-sox: **0.20, 0.17, 0.14 s** — not pauses at all, just the decay of a phrase ending,
  against 0.4–0.7 s for conversational speech. With `TTS_SENTENCE_SILENCE_S = 0.35` the live path
  measures **0.59, 0.56, 0.54 s**. Note what already existed: `ai/delivery.py`'s `DELIVERY_PAUSE` was
  measured and tuned to buy a 0.156 s breath *inside* long sentences, while the boundary a person
  leans on hardest got nothing.
- **The two VITS noise parameters were never passed either — and they do not do what was expected, so
  they ship at the voice's own values.** `noise_scale` and `noise_w` control how much a line varies;
  `voices/en_US-hfc_female-medium.onnx.json` carries the stock 0.667 / 0.8. Four repeats per config of
  one long line, because VITS draws fresh noise every run and one sample cannot tell an effect from a
  draw: today **9.58 st** mean p10–p90 intonation range, at `noise_scale 0.8 / noise_w 1.1` **9.43
  st**, and within-config spread is **±0.4–0.9 st**. The knobs move intonation by *less than the noise
  floor*, and the "livelier" setting measures marginally flatter. An 8-cell sweep (`noise_w` 0.8→1.2 ×
  `noise_scale` 0.667/0.8) stayed inside 8.8–10.0 st. Plumbed and dashboard-settable so an ear can
  overrule the numbers, defaulted to no change. **Add to the killed-hypothesis table in
  [docs/plan/completed/expressive-voice-plan.md](docs/plan/completed/expressive-voice-plan.md): the
  VITS noise parameters do not increase intonation range.**
- **The output chain was loudness-only; it now has EQ and a room.** `TTS_POST_EFFECTS` was
  `compand … gain -n -1` and nothing else, so every reply arrived bone-dry, dead-centre and flat — a
  synthetic cue independent of which model produced it. `TTS_POST_HIGHPASS` runs **before** the compand
  (energy a PAM8403 into a small driver cannot reproduce otherwise eats the headroom the compressor
  then reacts to) and `TTS_POST_ROOM` **after** it, followed by a second `gain -n -1` because reverb
  adds energy past the chain's own normalise. That order lives in `ai/tts._sox_chain()` and is pinned
  by tests, since it is invisible from the config lists; emptying both constants reproduces the old
  command line exactly, asserted rather than assumed. Confirmed live: the sub-90 Hz share of a real
  reply fell from **0.55% raw to 0.18%** through the chain.
- **The reverb costs nothing the other contracts care about**, which is what made it safe to ship.
  Duration is **unchanged** — 7.84 s for the test line and 1.27 s for a filler stall under every chain
  tried — because sox decays the tail into the silence Piper already pads onto every WAV instead of
  appending to the file. So `FILLER_MAX_STALL_S` (1.8) and `FILLER_MAX_LINE_S` need no re-deriving, and
  the jaw window (sized from `wav_duration`) does not change. Trailing silence actually **shrinks**,
  0.13 → 0.10 s, which is the safe direction. Whisper transcribed all four candidate chains
  identically, so the room costs nothing a decoder can detect.
- **Tagalog replies were going out with no breaths at all.** `DELIVERY_BREATH_CONJUNCTIONS` was
  English-only, so `ai/delivery.shape()` matched nothing in the language the room actually speaks.
  Added on a Tagalog speaker's approval: `pero`, `kasi`, `kaya`, `tapos`, `kung`, `habang`, `para`.
  **The semicolon buys more in Tagalog than in English** — same sentence, same voice: plain
  **0.07 s**, comma **0.15 s**, semicolon **0.36 s**, against 0.156 s for English. Verified in the live
  path by toggling `delivery_shaping` around one line: **off gives 0.05/0.05/0.06/0.05 s** (word
  boundaries only), **on gives a 0.27 s breath**. Whisper confirms the punctuation is never voiced.
  - **`at` and `o` are deliberately excluded** — both too common and too short, and a breath before
    every "at" is a stutter rather than a rhythm. `kasi` and `kaya` are likeliest to misfire, since
    they sit mid-clause more often than "because"/"so" do; drop those two first if it sounds choppy.
  - **No Tagalog openers, and the asymmetry is the point.** Breaths insert punctuation and add no
    words, so they cannot be mispronounced. An opener hands the `en_US` voice a new Tagalog word to
    say — and there is no Filipino alternative offline: Piper ships no `tl`/`fil`/`ceb` across **173
    voices / 54 language codes**, espeak-ng has no Tagalog phonemizer, and sherpa-onnx pre-converts
    only 8 MMS languages, Tagalog not among them. Swapping *only* the espeak phonemizer to
    `id`/`ms`/`es` does work and keeps Kai's pitch (200–206 Hz vs 206 Hz), but a native speaker judged
    the result no better. Considered and dropped; `config/filler.py`'s claim that a "Tagalog voice"
    exists was corrected in passing.
- **Sentences may now differ in loudness.** `TTS_PIPER_NORMALIZE = False` passes `--no-normalize`, so
  Piper stops normalising **every sentence** to full scale — per-sentence peaks measured **0.00 dB
  apart**, meaning no two sentences Kai ever spoke differed in peak level — and sox's single
  `gain -n -1` normalises the whole reply instead. Honest about the size: 3.42 dB of raw variation
  arrives as **0.79 dB**, because the compand's upward compression eats the rest. Recovering it means
  softening `compand`, which is what keeps Kai audible in a loud room; a venue beats 2.6 dB of
  sentence dynamics. Coupled in code so the flag is only passed when `TTS_POST_PROCESS` is on —
  without sox the raw output is ~10 dB quieter and nothing would put the level back.
- **Still open, and it can only be closed in a venue: the reverb.** Everything above was judged at a
  desk, and reverb trades against intelligibility in ambient noise — the same trade the wake path
  already lost once to a loud room (`config/wake.py`'s ambient adaptation). If Kai gets harder to
  understand at an event, `TTS_POST_ROOM = []` keeps the EQ and drops the space. A stronger variant
  (`30 50 45 100 0 -2`) was rendered alongside the shipped one for comparison.

The flag spellings are pinned by a test on purpose. Piper rejects an unknown flag by exiting non-zero,
`_run_piper` reports that as a failed synthesis, and the result is **every reply silent** with nothing
flag-shaped in the log — so `--sentence-silence`, `--noise-scale`, `--noise-w-scale` and
`--no-normalize` were read off `python3 -m piper --help` against the pinned `piper-tts==1.4.2` on this
robot rather than taken from any documentation. Re-check after a piper upgrade; that test is where it
should break.

## 2026-08-10 — Kai forgot your name six questions after you gave it

Suite green: 1239 passed, 2705 subtests (was 1190, 2700). Implements
[S12](docs/tickets/S12-no-identity-within-a-session.md); deployed and exercised on the robot.

Two things the live run turned up that the tests could not, both recorded in the ticket: Whisper
mishears names (`[identity] talking to 'Jandal'` was a correct extraction of a misheard "Jhondel",
which nothing in this design defends against), and the model uses the name more often than
`IDENTITY_PROMPT` asks it to.

- **A name offered in speech was an ordinary history turn, so the rolling cap evicted it.** The
  prompt is `system + capped history + user turn` (`ai/llm.build_chat_messages`) with no slot for a
  fact about the *speaker*, so "I'm Jhondel" landed in `_history` and was dropped by
  `MAX_HISTORY_TURNS = 6` exchanges later — unrecoverable, because nothing had extracted it. Kai now
  pins the first name it is offered on `VoiceAssistant._person_name` and appends `IDENTITY_PROMPT` to
  the system prompt, so it outlives the window entirely. Published as `sess_person` on `/params`.
- **Extraction is a regex, not a second LLM call.** `ai/identity.py` is pure stdlib, in the shape of
  `ai/wake_phrase.py` and for the same reason — the bugs here are all false accepts and casing, so it
  must be testable with plain strings. Asking Ollama to pull the name out would put another
  round-trip on the path [R5](docs/tickets/R5-serialised-first-audio-latency.md) exists to shorten.
- **The anchors are two-tier, because the risk is not symmetric.** A missed name costs nothing; a
  wrong name is said out loud to somebody standing in front of the robot. Strong anchors ("my name
  is X", "call me X", "ako si X" — `si` is a personal-name marker, introducing a name is its
  grammatical job) are taken as-is. The weak tier ("I'm X") is accepted only with a capitalised
  first letter *and* a miss on `IDENTITY_STOPWORDS`, which is what separates "I'm Jhondel" from
  "I'm fine", "I'm from Cebu" and "I'm a developer".
- **The name is session-scoped, on the seam that already existed.** `reset_history()` clears it
  alongside the rolling history and the sticky RAG topic, because all three answer the same question
  — what may the next person inherit? Nothing. It survives a "hey Kai" landing in `LISTEN_WAIT`,
  matching the history rule there, and `note_identity()` is epoch-guarded so a session that ended
  while STT was still running cannot hand its name to whoever is next.
- **Both conversational paths capture, which took a live run to find.** `note_identity()` was hooked
  only into `_process()`, the mic-turn path. The **one-breath** turn — "Hey Kai, my name is Jhondel"
  said without pausing — runs through `say()` instead, because the whisper wake tier already holds
  the transcript. On the robot, `sess_person` stayed `''` while Kai cheerfully replied "Hi Jhondel!":
  the model read the name out of the user turn, which looks identical from outside and pins nothing.
  So the same sentence captured or did not purely on whether the speaker drew breath. Now hooked in
  both, gated on `use_llm` so the verbatim `/voice/say` route cannot make Kai think it is talking to
  itself.
- **The prompt placement is the opposite call to the RAG context, deliberately — and the reasoning
  behind it is currently unmeasurable.** `RAG_CONTEXT_PLACEMENT = "user"` keeps per-turn-varying text
  out of the cached prefix; this string does not vary once learned, so the system position should
  cost one invalidation and nothing after. On the robot it cannot be confirmed: every `[llm] turn:`
  line is preceded by `MODEL RELOADED: ~200-360ms — placement was re-decided`, so **no KV prefix
  survives between turns at all** and there is nothing to invalidate. What was measured is that the
  injection costs nothing detectable — 258-304 ms prompt eval with a name pinned, inside the
  215-465 ms spread without one. Noted at the constant in `config/voice.py`, to be re-measured if the
  per-turn reload is ever fixed. That reload also makes `RAG_CONTEXT_PLACEMENT`'s optimisation inert.
- **`ai/persona.txt` now says to use the conversation it already has.** One line, no code: refer back
  to what was said earlier, notice a repeated question, pick up a dropped thread — and never claim to
  remember an earlier *visit*, which would be a lie the history cannot support. `load_persona()`
  re-reads the file on every call, so this is revertible on the live robot without a restart.

## 2026-08-10 — Long replies were being cut off mid-sentence, with the jaw still moving

Two independent bugs behind one symptom. Suite green: 1180 passed, 2675 subtests (was 1173).

- **`STATE_SPEAKING` guillotined every reply at 20 s.** `_enter_speaking` armed
  `SESSION_SPEAK_MAX_UNKNOWN_S` unconditionally, and that was the only deadline a *healthy* reply
  ever got — `SESSION_SPEAK_GRACE_S`, commented "allowed overrun past the WAV's own duration",
  reached only the canned branch. The clock also starts before Piper does, because `on_done` fires
  from the turn worker the moment the reply text exists. Meanwhile `TTS_MAX_SPOKEN_CHARS` (500)
  allows ~90 words — about 31 s at `SPEAK_SEC_PER_WORD`. So any answer past ~18 s was cut mid-word
  by the guard against a *wedged* paplay. `VoiceAssistant.audio_ends_at()` now publishes the WAV's
  measured end and `session._speaking_deadline()` prefers it, falling back to the 20 s cap only
  when no length is known — a pantomime, an unreadable header, or the synthesis window. The
  backstop is intact: a wedged paplay either publishes no end time or overruns the one it did.
- **Cut audio left the jaw miming on.** The jaw is a `(start, segments)` schedule that
  `face_track.py` reads every frame; `tts.stop()` only kills the subprocess, and the sole reset in
  the class was `start_recording()`'s, covering push-to-talk alone. So a filler cut mid-word by an
  arriving reply went on mouthing the rest of its sentence in silence — up to `FILLER_MAX_LINE_S`,
  plus the 0.5–1.5 s Piper run before the reply had a window of its own. `_begin_speech()` now
  retires the outgoing schedule at the one seam every speech path already passes through, and
  `stop_speech()` pairs the two for the four sites in `ai/session.py` that cut audio directly
  (ack timeout, speak timeout, push-to-talk interrupt, session end).

Clearing the stale end time in `_begin_speech` is load-bearing for the first fix: `_enter_speaking`
arms from `on_done`, which fires after `_speak()` has claimed the speaker but before its worker
knows a duration. A left-over end time from the previous line would cut the new one instantly.

## 2026-08-10 — Documentation restructure

Docs only — no code behaviour changed. The full test suite is green before and after
(1173 passed, 2675 subtests).

- **`README.md` is now an overview only.** It had grown to 1207 lines and carried the running
  update log, the full setup guide, the tuning reference, the R&D write-ups and the FAQ inline.
  Every section moved to `docs/` verbatim; the README keeps the orientation and links out.
- **This changelog was created**, from the dated entries that used to live in the README's TL;DR.
  They are reproduced verbatim and re-ordered newest-first.
- **`docs/plan/` gained `completed/` and `wip/`.** The six planning documents were filed by
  whether work remains against them; only `expressive-voice-plan.md` is finished (concluded at its
  own abort gate). See `docs/plan/README.md` for what is outstanding in each.
- **`docs/tickets/` added** — 24 implementation-ready tickets from a two-lens (robotics +
  software) codebase review, grouped into four tiers by severity x inverse effort. Nothing in them
  is implemented. See `docs/tickets/README.md`.
- Comment references to the moved documents were updated in `ai/delivery.py`, `config/voice.py`,
  `config/rag.py`, `scripts/rag_eval.py`, `scripts/tts_setup_models.sh`,
  `scripts/tts_setup_kokoro.sh` and `tests/test_voice_assistant.py`.

## 2026-08-07 — A DEVCON question can no longer come back empty
- Fuzzy matching widened the *entrance* to retrieval. Nothing guaranteed an *exit*: when nothing
  cleared `SIMILARITY_THRESHOLD`, `retrieve_context()` returned `""` — and `""` is the dangerous
  state, the one where the model answers about DEVCON out of pretraining. Eight layers now sit
  around it, four preventive and four as the floor. **Requires a reindex** (`python3 -m
  ai.index_documents`) — the index format gained a `title` per chunk and an `entities` list; an
  older index still loads, it just runs without those two layers.
- **Before the text exists.** `WHISPER_INITIAL_PROMPT` seeds the turn decoder with the vocabulary
  it is about to need, so the mishearing often never happens. This is the only layer that reaches
  multi-word names — *"geeks on the beach"* is ordinary English, so no matcher can safely flag it,
  but a primed decoder writes *"Geeks on a Beach"* in the first place. Not applied to the wake
  scan: a weak model primed with DEVCON vocabulary invents DEVCON talk out of room noise.
- **A second matcher, OR'd with the ratio.** `difflib` compares characters *in order*, so a vowel
  shift (`devcon` → `davcan` → `duvcun`) costs it real score and costs the spoken word nothing.
  `_skeleton()` drops vowels and folds consonants onto sound-alike classes, reducing all of them
  to one key. Checked against all 2111 words in `documents/`: **zero** new false positives.
- **An entity gazetteer, derived not hand-listed.** A mangled program name carries no DEVCON token
  at all, so nothing above fires on it. `build_gazetteer()` harvests names from the documents' own
  headings at index time — 14 from 82 headings — and both filters are load-bearing: rarity alone
  let *"When in doubt"* and *"Color palette (official, exact)"* through, since a style guide's
  section headings are rare words too. A name-shape test (`_is_name_shaped`) is what separates a
  label from an instruction. Derived, because a hand-written list goes stale *silently* on the
  next content drop — and silent staleness is the exact failure this layer exists to prevent.
- **Titles in every chunk's embedding.** Breadcrumbs only reach chunks under a heading; a
  mid-document paragraph, a `.pdf`, or a line of `kai_facts.txt` can carry no mention of DEVCON at
  all — and those are the chunks a *perfectly transcribed* "what does DEVCON do?" scores worst
  against. The filename is the one piece of provenance every chunk has. Embedded, not stored in
  `text`: retrieval needs it, the prompt does not. **This shifts every score slightly — re-check
  `SIMILARITY_THRESHOLD` after the reindex.**
- **Then the floor, in order: lexical → sticky → lowered threshold → primer → notice.**
  - `lexical_rank()` is IDF-weighted token overlap, and it fails in the opposite direction to a
    dense embedder — blind to paraphrase, exact on the rare literal token (a chapter city, a
    surname, a year) that bge-small smooths away. IDF weighting alone proved too loose (a word in
    12 of 14 chunks still has positive IDF, and a query of merely common words scored a perfect
    1.0), so `LEXICAL_MAX_DF_RATIO` hard-gates which words may carry weight at all.
  - `STICKY_TURNS` covers the follow-up that carries neither pronoun nor brand — *"how many
    chapters?"*, *"when did it start?"* — which `ANAPHORA_WORDS` cannot see. It only ever runs as
    a **retry after normal retrieval already came back empty**, so a genuine topic change
    retrieves its own answer and never reaches it. It deliberately does not renew on a successful
    retry: doing so made the topic permanent, since every later turn kept the flag alive by its
    own retry. Cleared with the conversation by `reset_history()`.
  - `FALLBACK_THRESHOLD = 0.32` (`top_k=1`), then the pinned `documents/devcon_primer.txt`, then
    `NO_CONTEXT_NOTICE` — an explicit *"you don't have this, don't guess, and don't confuse
    DEVCON Philippines with the other conference"*, which beats the silence that invites a
    confident hallucination.
- **The primer needed capping (`PRIMER_MAX_IN_RANKING = 1`), measured after the reindex.** It is
  indexed like any other document, and one dense on-brand fact per line turns out to be the ideal
  shape for `bge-small`: primer lines took **all three** `TOP_K` slots on *"what is DEVCON?"*,
  *"when did DEVCON start?"* and *"who founded DEVCON?"*, crowding out the documents that answer
  the specific question. Barring it from ranking overcorrected in the other direction — *"who
  founded DEVCON?"* then returned a `kai_facts` line about who assembled the *robot*, and *"when
  did DEVCON start?"* returned anniversary boilerplate with no date in it. One slot is the
  measured middle: the accurate summary stays, `TOP_K - 1` slots are always left for whatever is
  specific. The last-resort injection is uncapped — when it fires there is nothing left to crowd.
- **The gate is the whole design.** Those last three fire only when the turn is *provably* about
  DEVCON — the brand however it was spelled, or a gazetteer name. An unrelated question still
  falls through to `""` exactly as before. `tests/test_rag.py` asserts both halves against the
  same index and the same scores, with only the flag differing.
- **`SIMILARITY_THRESHOLD` was re-checked after the reindex, and the finding is not the one that
  was expected.** The reindex itself is a wash — 21 real queries moved by ±0.05, on-topic mean
  **+0.003**, off-topic mean **−0.017**, so separation is marginally *better* and 0.45 needs no
  adjustment for it. But 0.45 is not separating anything on this corpus, and was not before this
  change either. Measured top scores: on-topic DEVCON **0.68–0.81**, Kai facts **0.61–0.76**,
  and *off-topic* — "tell me a joke", "what time is it?", "what is the capital of Japan?" —
  **0.51–0.64**. `bge-small` compresses everything into a narrow high band and 0.45 sits under
  all of it, so **every turn retrieves something**.
  - No single threshold fixes it: off-topic tops out at 0.637 ("tell me a joke") while the
    lowest legitimate hit is 0.613 ("when is your birthday?", already carrying its `SOURCE_BOOST`).
    They overlap. Raising the bar to 0.66 gives perfect off-topic rejection and keeps every DEVCON
    question — and loses the birthday, which is the exact case `SOURCE_BOOST` was added for.
  - The 236-chunk omnibus style guide (**71% of the index**) was the obvious suspect and is not
    the cause: excluding it only drops the off-topic ceiling from 0.637 to 0.614, still above
    that 0.613 floor. It is worth splitting anyway — it is a *writing* guide, not knowledge Kai
    should answer visitors from.
  - **Consequence for the layers above:** while nothing ever returns empty, the floor never gets
    reached. Layers 3–7 (lexical, sticky, lowered threshold, primer injection, notice) are
    insurance that stays dormant on today's corpus and index; they earn their keep when the index
    is missing, broken, or mid-rebuild, and the moment the threshold is raised to something that
    actually rejects. Actively working today: the decoder bias, the skeleton matcher, the
    gazetteer's query expansion, the per-chunk titles, and the primer as a ranked document.

## 2026-08-07 — Bare "hey" wakes Kai on the Whisper tier
- The wake phrase on tier 3 is now **"hey"** — the name is optional. `"Hey Kai"` still works and still
  wins when the name is recognized; `"hey"` alone is a third accepted form in `ai/wake_phrase.py`.
- Why: the NAME slot is where this tier loses real wakes. The `tiny` scan model renders "Kai" as
  *guy*, *gai*, *chi*, *嘿哀* — `WAKE_PHRASE_NAMES` is a list we only ever extend **after** being
  ignored. The prefix is one common English word the model gets right. A false reject means the
  feature does nothing; a false accept costs an ack and a listening window that self-ends.
- **Deliberately narrow.** Bare-prefix matching requires `"hey"` at **token 0** — a mid-sentence
  "...and I was like hey, no" cannot fire — and only `"hey"`, never the rest of
  `WAKE_PHRASE_PREFIXES`. `"okay"`/`"ok"`/`"oy"`/`"ey"` open ordinary sentences and `"hi"` is how
  people greet each other in the room; all stay two-word-only.
- **Exact match, not a ratio** — the one place in this file that abandons fuzzy matching, and it was
  measured: `"they"` scores **0.857** against `"hey"`, the *identical* score to `"heyy"`, a genuine
  drawn-out wake. No threshold separates them at three characters, and with no second token there is
  nothing left to disconfirm a bad guess. Repeated letters are collapsed (`"heeeyyy"` → `"hey"`)
  instead. Do not reintroduce a ratio here.
- **Accepted cost, stated plainly:** greeting any person by name now wakes Kai — `"hey Chris"`,
  `"hey guys"`, `"hey everyone"`. That is indistinguishable from a wake at token 0 and no matcher
  tuning fixes it. The real fix is tier 2 (openWakeWord), which spots the phrase acoustically.
- **This only affects tier 3.** Porcupine (`wake/hey-kai.ppn`) and openWakeWord (`wake/hey_kai.onnx`)
  are trained blobs that still require the full "Hey Kai" and cannot be widened from config — so
  which phrase works depends on which tier won at startup. Consider pinning
  `WAKE_ENGINE_FORCE = "whisper"` while evaluating this.
- Rollback is one line: `WAKE_PHRASE_SOLO_PREFIXES = ()` in `config/wake.py` restores strict
  two-word matching. Tests pin both configurations.

## 2026-08-07 — Why the wake word barely worked — three bugs, none of them in the matcher
- Shortening the phrase made it *worse*, and chasing that turned up two more problems upstream. The
  matcher was fine; everything below it was tuned for a different job.
- **The scan path was silently eating short wake phrases.** `WAKE_WHISPER_MIN_UTTERANCE_S` was 0.35,
  copied from the turn path's `MIN_UTTERANCE_S`. *"Hey Kai"* is ~0.65 s of voiced audio and cleared
  it easily; *"hey"* alone is 0.25–0.35 s — sitting exactly **on** the threshold, so a crisp one was
  discarded before Whisper ever ran. Now 0.15. `sess_scan_skip_short` is the counter that shows it.
- **~3.2 s of deafness after every wake attempt.** One `SpeechGate` served both paths, so a wake scan
  inherited `VAD_HANGOVER_S = 1.5` — a value that exists so a speaker pausing mid-sentence isn't cut
  off, which cannot happen while saying one word. Charged to every wake: 1.5 s hangover + ~0.75 s
  transcribe + 1.0 s cooldown, and **nothing is captured during the last two**. So a missed wake
  followed by the natural response — saying it again straight away — put the retry inside the dead
  window. That is most of why this felt broken rather than merely slow. New `WAKE_SCAN_HANGOVER_S =
  0.45` via `SpeechGate.set_hangover()`, set at each capture's onset: ~1 s faster per wake, dead
  window down to ~1.8 s, and a shorter clip for Whisper to decode.
- **In a noisy room the tier was structurally deaf — Whisper never ran once.** `VAD_RMS_FLOOR_HOLD`
  (250) is the bar to *keep* an utterance open. Once ambient noise sits above it the hangover clock
  can never run out, so the scan utterance never closes, hits `WAKE_WHISPER_MAX_UTTERANCE_S` (6 s),
  is thrown away as `too_long`, and arms the 3 s long cooldown. A 6-on/3-off cycle with **zero**
  transcriptions, and `sess_wake_ok` still reporting `True`. Signature on `/params`:
  `sess_scan_skip_long` climbing while `sess_scan_checks` stays flat.
  - Both floors were one room's measurement — 40 s in a quiet room, p95 and p50 of *that* room. They
    now track ambient instead of being pinned to it: `ambient` is the **quietest frame in a sliding
    1.5 s window** (a minimum, not an average — speech is loud and intermittent, so an average would
    be dragged up by the very person trying to wake Kai), and the floors are lifted to clear it.
  - **Adaptation is a no-op in the room the constants came from**, by construction: that session's
    p50 was 124, and 124 × 5.2 = 650 (the open floor), 124 × 2.0 = 250 (the hold floor). The
    multipliers were chosen to reproduce the measured tuning, and a test asserts it.
  - Frozen while an utterance is open. Continuous speech contains no true silence, so a minimum taken
    mid-turn settles on the speaker's quietest syllable and lifts the hold floor out from under
    them — re-creating the exact bug the hold floor was added to fix.
  - The lift is capped at 4× the configured floor. Deafness is strictly worse than false onsets: a
    false onset costs one discarded Whisper run, while a floor above the speaker's own voice makes
    the feature do nothing and report no error. In a room that loud the answer is tier 2.
- New on `/params`: `sess_rms_ambient`, `sess_rms_floor_live`, `sess_rms_hold_live`.
  **`sess_rms_ambient` is the number to look at when the wake word works at your desk and not in the
  venue.** `WAKE_AMBIENT_ADAPT = False` pins the old behaviour.
- Not done, and worth doing next: `WAKE_WHISPER_SCAN_MODEL` is still `tiny`, chosen when the matcher
  needed two words including a name `tiny` mangles constantly. Form C needs one common English word,
  so `base` (+470 ms measured) is probably now the better trade — and the hangover fix hands back
  more than it costs. Test it on the robot before switching.

---

## 2026-08-06 — Documents first, and "DEVCON" by ear
- **Retrieved documents now outrank the model's own knowledge.** The context block used to be
  introduced as *"Reference information (use only if relevant…)"*, which reads as optional — the
  model answered DEVCON questions out of its vague pretraining with the real text sitting right
  there in the prompt. `rag.format_context()` now presents the chunks as Kai's own documents, tells
  the model to answer from them and nothing else (names, numbers, dates as written, no guessing, say
  "not sure" when they don't cover it) — **and in the same breath re-asserts her voice**, because a
  bare "answer from the documents" makes a small model recite chunk prose and drop the persona.
  `persona.txt` carries the precedence rule too, so it survives even if the context block is
  trimmed. The "ignore if unrelated" escape stays: it is the defensive half of
  `SIMILARITY_THRESHOLD`.
- **Fuzzy "DEVCON" matching on the query (`ai/query_alias.py`).** Every chunk in `documents/` spells
  the brand `DEVCON`; Whisper spells it `defcon`, `dev com`, `debcon`, `Devon`, `de con`, `dev khan`.
  `bge-small` embeds those as *different words*, so the single most on-topic question Kai can be
  asked drifted below threshold and retrieved **nothing**. The query is now folded onto the
  canonical spelling before embedding — retrieval only, never the transcript on `/params` and never
  the turn handed to the LLM, since a wrong guess must not put words in the speaker's mouth. Pure
  stdlib `difflib`, reusing `wake_phrase.py`'s offset-carrying tokenizer.
  - Threshold `DEVCON_MATCH_RATIO = 0.80` was measured, not guessed: plausible mishearings land at
    0.833+ while the nearest real words (`recon`, `devotion`, `beckon`, `second`, `beacon`,
    `device`) top out at 0.727. `deacon` is the one word inside that gap and is blocklisted.
  - Split renderings are joined ("dev con" → `DEVCON`) **only when both halves are too short to
    stand alone**. Without that guard `isdevcon` scores 0.857 and *"what is DEVCON Philippines?"*
    came out as *"what DEVCON Philippines?"* — the guard also protects the trailing side, so
    "devcon po" keeps its `po` and "devcon ph" stays intact.
- Knobs in `config/rag.py`; `tests/test_query_alias.py` covers the renderings, the distractors and
  both swallowing bugs.
- **Watch the context budget.** `OLLAMA_NUM_CTX` is 1024 and `TOP_K = 3`. Indexed chunks average
  230 chars (median 140), so a normal retrieval is ~175 tokens and fits fine — but three worst-case
  800-char chunks are ~600 tokens, and with the persona, the new header, three history pairs and
  `OLLAMA_NUM_PREDICT = 96` that overflows, and what gets dropped is the *front* of the prompt, i.e.
  the persona. If Kai ever sounds flat and generic on a long-document question, set `TOP_K = 2`
  before touching `OLLAMA_NUM_CTX` (raising the context window is what breaks GPU residency).

## 2026-07-28 — Wake-word fallback chain
- Hands-free no longer depends on one vendor. Kai tries **three** wake engines in order and keeps the
  first that initializes: **Porcupine → openWakeWord → Whisper phrase spotting**. A tier failing is a
  logged reason, never a dead robot, and `sess_wake_engine` on `/params` says which one is live.
- Why: Porcupine has three independent ways to be unavailable — a cloud account and key, a
  `.ppn` compiled per-platform, and a CPU allow-list that **does not include this board** (the Orin's
  Cortex-A78AE reports `0xd42`, absent from every published version 2.1→4.0). Depending on all three
  holding forever was not a plan.
- **Tier 3 needs no setup at all.** It reuses the already-resident faster-whisper, so Kai always has
  *some* hands-free path. It also fixes the limitation noted in the entry below: because the
  transcript already contains what followed the wake words, *"Hey Kai, what time is it?"* is answered
  **in one breath** — no ack, no second turn. The trade-offs are ~0.4-1.0 s before the ack and the
  fact that it transcribes nearby speech locally to look for the phrase; both are bounded and
  documented in `wake/README.md`.
- New: `ai/wake_phrase.py` (pure-stdlib fuzzy matcher), `WakeEngine`/`PorcupineEngine`/
  `OpenWakeWordEngine`/`WhisperWakeEngine` and the `WakeDetector` chain in `ai/audio.py`, two scan
  states in the session FSM, `transcribe_async()` and `say(epoch=, on_done=)`.
- **Fixed a latent bug that would have made openWakeWord silently never fire:** `MicStream` sized its
  wake frame assembler in `__init__`, *before* `wake.open()` knows which tier won and therefore what
  frame size it wants. Porcupine is 512 in both places so it worked by luck; openWakeWord wants 1280
  and would have been fed 512-sample frames forever, with scores pinned near zero and
  `sess_wake_ok` still reporting `True`. `sess_wake_frame` on `/params` is the proof it took effect.
- Matching Whisper transcripts turned out to be the subtle part. `"kai" in text` is worse than
  useless — an adversarial sweep found it firing on `"Okay Google"`, `"okay okay i get it"`,
  `"hi hi hi"`, `"hey Kyle"`, `"hey Kayla"` and `"sabihin mo kay Kai"` (talking *about* Kai). All
  measured, all fixed, all pinned as tests. `"hey Kaye"` still matches and that is deliberate — it is
  a near-homophone the acoustic tiers can't separate either.
- Also added `WHISPER_CPU_THREADS = 4`: ctranslate2 defaults to *every* core, and STT now runs per
  nearby utterance rather than per button press. Left uncapped it starves the servo control loop and
  shows up as jittery face tracking rather than as anything audio-shaped — watch `[control] N Hz`.
- Setup for tiers 2 and 3, plus the "hey kai" training recipe: **`wake/README.md`**.

## 2026-07-28 — Hands-free conversation — "Hey Kai"
- Kai no longer needs a button. Say **"Hey Kai"**, he answers *"Yes?"*, and then you just talk:
  silence ends your turn, and the conversation ends itself when you stop talking or walk away.
  The dashboard mic button and spacebar still work exactly as before, as a fallback.
- **The camera does not wake Kai** — deliberately. Entry is the wake word only, so it works in the
  dark, from another room, and with nobody in frame. Vision is used only to help decide when a
  conversation is *over*.
- New pieces: `config/wake.py` (all tunables), `ai/audio.py` (resampler, framing, pre-roll, wake
  and VAD wrappers), `ai/session.py` (the one open stream + the state machine),
  `vision/presence.py` (a three-valued presence sink face_track feeds at `INFERENCE_FPS`).
- **Enable it with `--wake`** (already added to `scripts/autostart.sh`). Without the flag, or
  without a Porcupine key/`.ppn`, hands-free is simply off and push-to-talk is untouched.
- One-time setup:
  ```bash
  pip3 install pvporcupine
  pip3 install webrtcvad || pip3 install webrtcvad-wheels   # no aarch64 wheel; the fallback is prebuilt
  mkdir -p ~/.config/kai && printf '%s' 'YOUR_KEY' > ~/.config/kai/porcupine.key
  chmod 600 ~/.config/kai/porcupine.key
  ```
  Generate a custom "Hey Kai" keyword at `console.picovoice.ai` into `wake/hey-kai.ppn`. **The
  `.ppn` is platform-specific** — pick the ARM/Linux (aarch64) target, or Porcupine raises
  `PorcupineInvalidArgumentError` on the Jetson. The access key is never stored in `config/`.
- **Tune it on-device with `scripts/wake_test.py`** — it prints rolling RMS, VAD decisions and wake
  hits and nothing else, so setting `VAD_RMS_FLOOR` takes minutes instead of an afternoon of
  reading `/tmp/face-servo.log`. It cannot run at the same time as `face_track.py`: the raw I2S hw
  device admits exactly **one** opener, which is the single fact that shaped this whole design —
  hence one always-open stream fanned out to Porcupine, the VAD and the utterance buffer, rather
  than a self-contained wake module opening its own mic.
- **The mic is now open all the time, so Kai can hear himself.** A gate drops audio blocks before
  any DSP whenever Kai's own audio could be in the air, sized from the WAV's duration plus
  `TTS_TAIL_MUTE_S` — *not* from `paplay` exiting, which happens once the file is in the
  PulseAudio sink buffer, several hundred ms before the amp is actually quiet. That gap is exactly
  how a robot ends up answering itself; raise the tail first if it ever does.
- Voice barge-in is **not** supported (no echo cancellation, mic and speaker share the chassis), so
  the wake word is ignored while Kai is thinking or speaking. Pressing the dashboard mic button
  *does* interrupt him — a button press is unambiguous intent.
- New endpoints: `POST /voice/wake` (fire the wake word by hand — invaluable for telling "the
  session machine is broken" apart from "Porcupine isn't hearing me") and `POST /session/end`.
  ~40 additive `sess_*` fields ride the existing `/params` SSE stream, and session state is
  projected onto the `voice_status`/`voice_speaking` fields the dashboard already reads — so the
  existing UI shows hands-free state with **zero** frontend changes.
- Also fixed along the way, all pre-existing: `ai/tts.py` could not cancel an in-flight Piper
  synth (only playback), so an abandoned reply still got spoken; `reset_history()` was dead code
  and racy against an in-flight Ollama call, which could re-append one turn *after* the clear; and
  a push-to-talk recording had no maximum length, so a stuck button grew the buffer until the
  process died. A turn `epoch` now versions every async result and is dropped on mismatch.
- **Supersedes** the "text-only for now" note in the 2026-07-06 voice-assistant entry below and
  the onboard-audio "Next Steps" section: Piper TTS, the I2S mic and the speaker all shipped, and
  the jaw is synced to real audio duration rather than a text-timed estimate. The live model is
  `gemma2:2b` (switched from `gemma3:4b` to fit the camera in 8 GB), so read `config/voice.py`
  rather than the model names in the older entries.

## 2026-07-06 — Voice assistant
- Kai can now hear and reply: push-to-talk button on the web dashboard records from the
  Jetson's mic, transcribes locally with `faster-whisper`, and sends the text to a local
  Ollama model (`gemma3:4b`) for a reply — shown as text on the dashboard (`voice_assistant.py`).
- **Text-only for now** — speaker/TTS output is a planned next phase.
  *(Superseded 2026-07-28: Piper TTS shipped, and push-to-talk is no longer the only way in —
  see the hands-free entry above. `gemma3:4b` below is now `gemma2:2b`; check `config/voice.py`.)*
- One-time setup: `ollama pull gemma3:4b` (Ollama must already be installed and running).
  `faster-whisper`, `sounddevice`, and `requests` are already part of the environment — no
  new `pip install` needed.
- New endpoints: `POST /voice/start`, `POST /voice/stop`. Live status/transcript/response
  ride the existing `/params` SSE stream (`voice_status`, `voice_transcript`, `voice_response`,
  `voice_error` fields).
- Sanity-check the mic before use: `python3 -c "import sounddevice as sd; print(sd.query_devices())"`.

## 2026-07-06 — RAG + editable persona
- Kai can now answer from your own documents and its personality is editable without touching code:
  - Drop `.txt`/`.md`/`.pdf` files into `documents/`, then run `python3 -m ai.index_documents`
    from the project root (not `python3 ai/index_documents.py` — the imports are absolute)
    whenever files are added or changed — rebuilds `documents/.rag_index.json` from scratch
    each run (`rag.py`, `index_documents.py`).
  - Edit `persona.txt` at the project root to change Kai's personality — takes effect on the
    very next voice turn, no restart needed (`voice_assistant.load_persona()`).
  - Retrieval is fail-open: no index, nothing relevant enough (`SIMILARITY_THRESHOLD = 0.5`),
    or any failure anywhere just means Kai answers without extra context — unrelated
    questions behave exactly as before this feature existed.
  - Web search was discussed and intentionally left out of this round.
- **Important hardware finding:** embeddings do **not** run through Ollama. Measured directly
  on this Jetson: `gemma3:4b` and *any* Ollama-served embedding model (tried `nomic-embed-text`
  at 595MB resident, then even `all-minilm` at 76MB) cannot both stay loaded — loading either
  one always evicts the other, even with `keep_alive: -1` on both, because there's only
  ~200-300MB genuinely free once gemma3:4b is resident. That would have turned every
  RAG-enabled voice turn into a double reload (~7-13s embedding reload + ~48-51s gemma3:4b
  reload ≈ 60+ seconds), regressing the exact "thinking takes a long time" problem fixed
  earlier this session. Instead, `rag.py` embeds with **fastembed** (`BAAI/bge-small-en-v1.5`,
  ~90MB, ONNX/CPU via the already-installed `onnxruntime` — zero torch dependency, so it can't
  disturb the Jetson's custom CUDA-enabled torch build) running in-process in `face_track.py`,
  entirely decoupled from Ollama's GPU memory management. Query embedding takes ~0.03s once
  warm; `OLLAMA_NUM_CTX` stayed at `1024` with no need to bump it — verified via `ollama ps`
  that `gemma3:4b` remains 100% GPU-resident even with retrieved context injected (chunks are
  kept small on purpose, `CHUNK_SIZE_CHARS = 800`, specifically to fit this budget).
- One-time setup: `pip3 install pypdf` (only needed for indexing `.pdf` files — `fastembed` was
  installed as part of this feature and downloads its embedding model from Hugging Face on
  first use, cached afterward like `faster-whisper`'s model).
- New pre-warm threads at startup (same pattern as the voice assistant's): `rag.load_index()`
  and `rag.ensure_model_loaded()`, alongside the existing three.

## 2026-06-17 — Post-development additions
- LOFI face parameter capture — yaw, pitch, roll, mouth, eyes, smile/kiss, distance; same algorithm as face-detection-movements; use `--lofi` flag for 19-digit output string
- Auto USB driver loading — `face_track.py` now loads `ch341.ko` automatically if `/dev/ttyUSB0` is missing; no manual `modprobe` step needed
- All files consolidated into the `face-servo` directory

## 2026-06-15 → 2026-06-17 — Initial R&D

- **2026-06-15** — First hardware test. Ran into issues early: missing wires and servo condition not checked beforehand. Session postponed to the following day.
- **2026-06-16** — Main R&D day. Got everything working — built the full face tracking pipeline, solved all hardware and software challenges, completed most of the development.
- **2026-06-17** — Post-development. Consolidated findings, wrote the README and documentation, created the TL;DR, implemented Y-axis tilt (code complete, untested — only 1 working servo available), added LOFI-compatible face parameter capture (yaw, pitch, roll, mouth, eyes, smile, distance — same algorithm as face-detection-movements), auto USB driver loading on startup (no more manual `modprobe` step), and cleaned up the project structure.

