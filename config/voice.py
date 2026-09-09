"""Voice assistant: audio capture, Whisper STT, Ollama LLM, jaw-speaking pantomime.
Consumed by ai/voice_assistant.py. (STATUS_* strings, NO_SPEECH_RESPONSE, PERSONA_PATH and
the default persona stay in voice_assistant.py — protocol/structural, not tuning.)"""

SAMPLE_RATE      = 16000   # Whisper's required input rate — captured audio is resampled to this
CHANNELS         = 1

# The Jetson's onboard analog input enumerates as a normal input device but isn't wired to
# anything, so it only captures digital silence (which makes Whisper hallucinate filler like
# "You"). resolve_input_device() probes candidates for real signal and skips silent ones.
LIVE_PROBE_DURATION_S    = 0.3
LIVE_PROBE_RMS_THRESHOLD = 5.0   # low bar — just rules out true digital silence
# Ceiling on one probe. sd.wait() has no timeout of its own, and a device that opens but never
# delivers frames (pulse holding the suspended APE card, a half-open I2S route, another opener on
# the single-opener raw device) blocks it forever. That block lands on the session start path, so
# without this a hung probe leaves hands-free permanently disabled — and silently, because
# MicStream.start() never returns to report the failure or trigger the push-to-talk fallback.
LIVE_PROBE_TIMEOUT_S     = 3.0

# How many EXTRA times to re-read the I2S mic before believing it is silent, and how long to wait
# between tries. Applies to the I2S device only — see resolve_input_device().
#
# The INMP441 can read as exact digital silence on the first capture taken shortly after the XBAR
# route is applied, while reading normally a second or two later. Measured on 2026-08-09: the boot
# probe returned rms=0.0, yet the same device on the same route, probed repeatedly from a shell,
# returned rms 124-435 every time. `arecord` on the raw device confirmed real audio throughout, so
# the mic and the wiring were never the problem — a single 0.3 s read simply landed too early.
#
# The cost of that one bad read is out of all proportion to it: a device that reads silent is
# skipped for the entire lifetime of the process, so Kai spends the whole session on the fallback
# USB mic and only a restart can reconsider. Retrying is bounded (retries x (probe + delay), once at
# startup) and only ever runs for a device that has ALREADY read silent, so a genuinely dead mic
# costs about two extra seconds at boot and nothing afterwards.
#
# Deliberately NOT applied to the USB/system-default devices: they have no such warm-up, so silence
# from them is a real answer and retrying it would only slow startup down.
I2S_PROBE_SILENT_RETRIES = 3
I2S_PROBE_RETRY_DELAY_S  = 0.4

# ── Mic preference ──────────────────────────────────────────────────────────────
# Kai's default mic is the onboard INMP441 I2S MEMS mic (enabled at the hardware level — see
# mictest/RESULTS.md); a USB mic is the fallback; anything else / the system default is the last
# resort. resolve_input_device() classifies each input device by these name substrings
# (case-insensitive) and probes them in that priority order, picking the first that captures real
# signal. Requires the ALSA I2S2->ADMAIF1 route to be active (i2s-mic-route.service); if the I2S
# mic is absent or reads silent, selection degrades gracefully to USB, then the system default.
I2S_MIC_NAME_HINTS = ("APE", "tegra-dlink", "i2s")  # INMP441 enumerates on card "APE" / PCM "tegra-dlink-0"
USB_MIC_NAME_HINTS = ("usb",)

# A 3.5mm analog mic, which on this board can only arrive through a USB audio adapter.
#
# The Jetson's own analog input is not wired to anything (see the top of this file), so there is no
# native analog capture path at all. A USB->3.5mm adapter therefore IS a USB sound card
# electrically, and that is the whole reason this list has to be matched BEFORE USB_MIC_NAME_HINTS:
# an adapter's name contains "usb" too, and the more specific rule has to win.
#
# WHAT THIS BUYS, precisely: a label and a position in the probe order. Nothing here decides whether
# a device can be opened. An adapter whose name is not in this list classifies as "usb" and works
# exactly as it does today — so a wrong guess costs the dashboard the ability to tell two USB inputs
# apart, and costs MIC_PREFERENCE the ability to choose between them. It never costs a microphone.
# That property is what makes a name-based rule acceptable for something this un-namable.
#
# THESE DEFAULTS ARE PLAUSIBLE, NOT MEASURED. Adapter names are genuinely not distinctive —
# "USB PnP Sound Device" is used by cheap adapters AND by standalone USB mics — so confirm yours and
# edit this list:
#
#   python3 -c "import sounddevice as sd; print(sd.query_devices())"
#
# Rejected alternative: infer "analog" structurally from the card also having output channels. An
# adapter with a headphone jack does — but so does every USB mic with a monitor output, and a
# mic-in-only adapter does not. It is not a discriminator, and a wrong structural guess is harder to
# explain than a wrong name guess because there is no list to correct.
ANALOG_MIC_NAME_HINTS = ("usb audio codec", "usb advanced audio", "generalplus", "line in", "line-in")

# How many EXTRA silent reads to forgive on an analog adapter. Same as USB and for the same reason:
# it is a USB device and settles like one. Named separately anyway, because the two are tuned by
# different facts — this one by how the adapter's ADC comes up, USB_PROBE_SILENT_RETRIES by how a
# native USB mic does — and collapsing them would make either one impossible to move alone.
ANALOG_PROBE_SILENT_RETRIES = 1

# ── Capturing the mic jack on the SPEAKER's own dongle, through PulseAudio ──────
# A single USB dongle with two 3.5mm jacks — speaker out, mic in — is the ordinary way to give Kai
# both, and it is the hardware this build already has (see SPEAKER_CARD_NAME_HINTS below: the
# C-Media dongle is both TTS_SINK and a one-channel input). Kai could not use that mic at all.
#
# WHY NOT, precisely. One dongle is ONE ALSA card, and the card is the unit of configuration:
#
#   card N --+-- playback hw:N,0   sink:   alsa_output.usb-<product>-00.analog-stereo
#            +-- capture  hw:N,0   source: alsa_input.usb-<product>-00.mono-fallback
#
# `pactl set-card-profile`, which tts.play() asserts before the first reply, acts on the whole card.
# On 2026-08-11 that re-opened the card's ALSA devices underneath a live RAW PortAudio capture stream
# and the process took SIGSEGV at the startup greeting. So raw capture on that card is refused, and
# stays refused — see _is_speaker_card().
#
# But the hazard is RAW capture on that card, not capture on that card. Pulse serialises access to
# the card, so a pulse-mediated stream has no raw ALSA device for a profile change to pull out from
# under it. That is the route this enables, and it is the same claim already made further down: "Pulse
# coordinates access to the card, so it is safe where a raw open is not."
#
# STILL UNMEASURED, and worth knowing before you turn this on: whether set-card-profile disturbs a
# live pulse-mediated capture at all. If it does, the mic watchdog (MIC_STALL_S) reopens the stream,
# so the cost is a reopen at the first reply rather than a crash. That bounded downside is why this
# ships behind a flag instead of waiting for hardware.
#
# OFF BY DEFAULT. A robot that works today must not change behaviour because this landed.
PULSE_CAPTURE_ENABLED = False

# The pulse source to record from. `pactl list short sources` — it is the alsa_input.* name on the
# same card as TTS_SINK. Set to "" to use whatever pulse's default source is, which is a worse idea
# than it sounds: the default moves when devices come and go.
PULSE_CAPTURE_SOURCE = "alsa_input.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.mono-fallback"

# How the source is targeted: PULSE_SOURCE in the process environment, which libpulse reads when a
# recording stream is created. Deliberately NOT `pactl set-default-source`, which would change what
# every other program on the box records from — this build already treats global audio state as
# something to assert narrowly and put back (see free_i2s_device / resume_pulse_sources). It is the
# mirror of what playback does with `paplay --device=TTS_SINK`.
PULSE_CAPTURE_ENV_VAR = "PULSE_SOURCE"

# Which PortAudio device entry is the pulse-backed one, tried in order. These are the names ALSA's
# pulse plugin exposes; "pulse" is the explicit one and "default" is usually routed to it on a box
# where pulse is running. Matched as case-insensitive EXACT names, not substrings, because "default"
# appearing inside a longer device name means something else entirely.
PULSE_CAPTURE_DEVICE_NAMES = ("pulse", "default")

# Capture rate for the pulse route. Pulse resamples for us, so ask for the rate the pipeline wants
# and skip the decimator entirely (MicStream.open() already does this when rate == SAMPLE_RATE).
# That deletes the 44.1 kHz integer-ratio problem for this route rather than solving it: a card that
# can only do 44.1 kHz is unusable raw but perfectly usable through pulse.
PULSE_CAPTURE_RATE = SAMPLE_RATE

# Which kind of mic to probe FIRST: "auto" (i2s, then usb, then analog, then everything else),
# "i2s", "usb", "analog", or "pulse" (the mic jack on the speaker's own dongle — needs
# PULSE_CAPTURE_ENABLED, see below).
#
# This REORDERS the probe, it never excludes a kind. Preferring "usb" still falls through to the
# INMP441 when no USB mic is live, and so on for every value. A preference that could leave Kai deaf would be a
# worse control than no control: the whole point of resolve_input_device() is that it keeps looking
# until something actually captures signal, and a filter would defeat that. Live-settable from the
# dashboard (settings.py), so switching mics does not need a restart — and an unrecognised value
# falls back to "auto" rather than raising, because this one is operator-writable.
MIC_PREFERENCE = "auto"

# How many EXTRA times to re-read a USB/other mic before believing it is silent. See
# I2S_PROBE_SILENT_RETRIES above for the mechanism.
#
# Zero was right while USB was only ever the boot-time fallback: those devices have no warm-up, so
# silence from them was a real answer and retrying only slowed startup. Hot-plug changes that. A USB
# mic re-probed seconds after it was plugged in has genuinely not settled — the card appears in
# /proc/asound/cards before its PCM devices are usable — and one mistimed read would send Kai back to
# the I2S mic for the rest of the run. Kept to 1 rather than I2S's 3: this is settling, not the
# documented boot race, and it is paid on every non-live candidate at startup.
USB_PROBE_SILENT_RETRIES = 1

# Input devices that must NEVER be captured, because they are the SPEAKER's card.
#
# On this build the C-Media dongle is both the only output sink (TTS_SINK below) and an input device
# ("USB Audio Device: - (hw:0,0)", one mono input). Capture there is RAW ALSA with pulse suspended
# (see free_i2s_device), while playback goes through pulse on the same card — and tts.play() runs
# `pactl set-card-profile` against that card immediately before the first paplay, i.e. re-opens its
# ALSA devices underneath a live PortAudio capture stream.
#
# Measured on the robot 2026-08-11. Across 22 runs in one log the I2S probe timed out twice
# (LIVE_PROBE_TIMEOUT_S), and both times selection fell to `device=0 rate=48000 ch=1 i2s=False` —
# the speaker. One of those two runs died with SIGSEGV at 32 s, exactly as the startup greeting's
# first audio began:
#
#   [session] greeting: Hi, I'm Kai. ...
#   autostart.sh: line 196: 50270 Segmentation fault (core dumped) python3 .../face_track.py
#   [autostart] face_track exited rc=139 after 32s — restarting in 5s
#
# and the relaunch greeted the room a second time. A milder form of the same collision is in the same
# log as `could not set output card profile (... exit 1)` followed by `playback failed (Stream error:
# No such entity)` — the sink disappearing out from under a reply.
#
# The trade, stated plainly: when the I2S mic fails to probe, that run now gets the pulse-mediated
# system default instead of the raw dongle (MicChoice(None, ...) — and MicStream calls
# resume_pulse_source() on any non-I2S choice, so that path is not left suspended). Pulse coordinates
# access to the card, so it is safe where a raw open is not. If nothing usable is left, Kai is deaf
# for that run and the log says so — strictly better than a segfault at the greeting, and
# /audio/reresolve retries the mic without a restart.
#
# HOW a device is recognised as the speaker's card, in two tiers.
#
# Tier 1 (preferred, SPEAKER_CARD_FROM_TTS_CARD below): resolve TTS_CARD — the pulse card name this
# build already plays through — to its ALSA card index via `pactl list cards`, and match that against
# the `hw:<card>` in the PortAudio device name. This is exact. It identifies the one card that is
# genuinely dangerous to capture raw, and nothing else.
#
# Tier 2 (fallback, the hints below): case-insensitive substrings of the PortAudio device name, the
# original rule. Used only when tier 1 cannot answer — no pactl, no pulse, a dev box, or a TTS_CARD
# that pulse does not know. The order matters and is deliberate: when in doubt this must still BLOCK,
# because the failure it prevents is a segfault mid-greeting and the failure it causes is one skipped
# candidate.
#
# Tier 1 exists because tier 2 is too coarse now that a USB mic is a supported input rather than a
# last resort. "USB Audio Device" is the most common name a cheap USB mic enumerates under, so the
# hint that blocks the C-Media dongle also blocks a perfectly good separate mic — before it is ever
# probed, with no log line saying a mic was rejected. Keyed on the card index instead, the two are
# never confused: they are different cards even when they share a name.
#
# Keep the hints specific for the same reason as before — they name the speaker's card, not "any USB
# audio thing" — and a build whose speaker is not this dongle needs them changed with it. Set to ()
# AND SPEAKER_CARD_FROM_TTS_CARD False to allow capturing from the speaker's card again, i.e. the
# pre-2026-08-11 behaviour and the crash it carried.
SPEAKER_CARD_FROM_TTS_CARD = True
SPEAKER_CARD_NAME_HINTS = ("usb audio device",)

# ── Microphone hot-plug ─────────────────────────────────────────────────────────
# Noticing that a USB mic was plugged in or pulled out, so switching mics does not need a restart.
#
# Consumed by ai/mic_hotplug.py (the watcher) and ai/session.py (which acts on it), but it lives here
# with the rest of the "which mic" config rather than in wake.py: everything that decides WHICH
# device Kai captures from should be readable in one place.
#
# WHY A FILE AND NOT PortAudio. The obvious implementation — poll sd.query_devices() and diff it —
# cannot work. PortAudio snapshots the ALSA device list at Pa_Initialize and never refreshes it, so a
# card plugged in after startup is invisible to query_devices() for the life of the process. (That is
# also why /audio/reresolve could not find a newly plugged mic before this: it re-ran the whole
# discovery path against a stale list. See refresh_devices() in ai/mic_device.py.) Re-initialising
# PortAudio on every poll WOULD refresh it, and would also tear down the live InputStream several
# times a minute. /proc/asound/cards is the kernel's own list, costs one small read, and touches no
# audio state whatsoever.
MIC_HOTPLUG_ENABLED    = True
MIC_HOTPLUG_CARDS_PATH = "/proc/asound/cards"
MIC_HOTPLUG_POLL_S     = 3.0
# Act only once the card set has held steady this long. USB enumeration is not atomic — the card
# appears in /proc/asound/cards before its PCM devices are openable — so re-resolving the instant the
# file changes probes a device that is not ready yet and reads it as silent.
MIC_HOTPLUG_SETTLE_S   = 2.0
# Floor between two automatic re-resolves. A re-resolve costs a stream teardown, a PortAudio re-init
# and up to several liveness probes, so a flapping cable (or a hub that re-enumerates under load)
# must not be able to spend the whole session doing that. The operator's dashboard button is not
# subject to this — a human pressing it means "now".
MIC_HOTPLUG_COOLDOWN_S = 20.0

# Rates to try for a NON-I2S mic, in order, before giving up on that device. The I2S mic is
# clock-locked to I2S_CAPTURE_RATE below and ignores this list.
#
# This is an arithmetic constraint, not a preference. MicStream resamples with an integer-ratio
# decimator (ai/audio.py Decimator), so a capture rate that does not divide SAMPLE_RATE cannot be
# used AT ALL — and the failure is not graceful. On 2026-08-09 the robot booted with the I2S mic
# reading silent, fell back to the USB C-Media mic, and took the whole voice session down with it:
#
#   [mic] resolved device=0 rate=44100 ch=1 i2s=False - opening stream...
#   [mic] ERROR: cannot resample 44100 Hz: decimation needs an integer ratio, got 44100 -> 16000
#
# MicStream.open() returned False, so ConversationSession.start() returned False, so there was no
# capture stream at all — hands-free off, push-to-talk deaf, sess_state stuck on "disabled". The mic
# hardware was fine the whole time; only the rate was wrong.
#
# resolve_input_device() used to hand back whatever ALSA advertised as `default_samplerate`, which
# for that dongle is 44100 — the one rate it cannot use. The device's real capability (via
# `arecord -D hw:0,0 --dump-hw-params`) is S16_LE mono at RATE: [44100 48000]: 16 kHz is impossible
# and 48 kHz is the only usable rate on the whole device. Nothing about the advertised rate said so.
#
# So we no longer trust the advertised rate — we offer only divisible rates and let the liveness
# probe, which opens the device for real, decide which one the hardware actually accepts. A device
# where none of these open is skipped rather than returned, because returning it is the bug above.
FALLBACK_CAPTURE_RATES = (16000, 48000, 32000)   # each must divide evenly into SAMPLE_RATE

# The INMP441 records 2-channel S32 with real audio only in the LEFT slot (its L/R pin is tied to
# GND); a raw hw: device won't down-mix, so capture stereo and keep this channel. USB/other mics
# capture mono (CHANNELS above).
I2S_CAPTURE_CHANNELS = 2
I2S_TAKE_CHANNEL     = 0   # left channel = the live one
# Capture the INMP441 at its true I2S clock rate. The route runs I2S2 at 48 kHz; if pulseaudio is
# left in charge it locks the card to 44100 AND adds noise, which pitch-shifts/garbles speech so
# Whisper hears nothing. We suspend pulse (see below) and open the raw hw device at this rate.
# _transcribe resamples 48000 -> SAMPLE_RATE (16000). Set None to use the advertised rate instead.
I2S_CAPTURE_RATE     = 48000

# pulseaudio grabs the APE card and forces 44100 + injects noise (measured ambient RMS ~9500 vs
# ~900 on the raw device), garbling the INMP441 so speech won't transcribe. Suspend its capture
# source so the app can open the raw hw device cleanly at 48 kHz. The source name comes from
# `pactl list sources short`. Best-effort; set False to disable (e.g. if pulse is removed).
I2S_SUSPEND_PULSE = True
I2S_PULSE_SOURCE  = "alsa_input.platform-sound.analog-stereo"

# Release EVERY pulseaudio capture source, not just the I2S one, before probing mics.
# Any source pulse holds makes the liveness probe of that device block until it times out, and a
# timed-out probe is treated as "not live" — so the real mic gets skipped and Kai falls back to a
# pulse-backed 44.1 kHz device, which then fails outright ("decimation needs an integer ratio, got
# 44100 -> 16000") and takes hands-free down with it. Only became reachable once pulseaudio began
# starting at boot (`loginctl enable-linger`, needed so replies are not silent before anyone logs in) —
# before that nothing held the USB card this early. Monitors are skipped: they are output taps and hold
# no capture hardware.
PULSE_SUSPEND_ALL_SOURCES = True

# ── I2S capture route (applied at app startup) ────────────────────────────────────
# The INMP441 overlay is pinmux-only, so the ALSA XBAR/I2S2 capture path must be set up at
# runtime (see mictest/RESULTS.md). Rather than depend on the external i2s-mic-route.service or a
# manual SSH session, the app applies this route itself once, via `amixer`, right before it probes
# for a mic (resolve_input_device -> apply_i2s_route). Best-effort: if `amixer` or the APE card is
# absent (e.g. on a dev box, or before the device-tree overlay loads) it's skipped and selection
# falls back to the USB mic. Set the toggle False to rely solely on the systemd service instead.
I2S_APPLY_ROUTE_ON_STARTUP = True
I2S_ROUTE_CARD = "APE"
# (control name, value) pairs, applied in order — the exact working sequence from RESULTS.md.
I2S_ROUTE_CONTROLS = (
    ("I2S2 codec master mode",        "cbs-cfs"),   # Jetson = I2S master
    ("I2S2 codec frame mode",         "i2s"),
    ("I2S2 Sample Rate",              "48000"),
    ("I2S2 Capture Audio Bit Format", "32"),
    ("I2S2 Client Bit Format",        "32"),
    ("I2S2 Client Channels",          "2"),
    ("I2S2 Capture Audio Channels",   "2"),
    ("I2S2 FSYNC Width",              "31"),
    ("ADMAIF1 Mux",                   "I2S2"),       # XBAR: I2S2 -> ADMAIF1 (capture)
)

# "base" over "small" is the single biggest turn-latency win available. Measured on this box with a
# 2.94 s utterance, int8/cpu/4-threads, vad_filter on, 3 runs each:
#   small + auto-detect  7.81 s (2.66x realtime)  <- was the default, ~50% of the whole turn budget
#   base  + auto-detect  2.38 s (0.81x realtime)  <- 5.4 s faster
# The cost is accuracy, and it lands hardest on Tagalog — if transcripts start coming back wrong,
# this line is the first thing to revert. REVERT: "small"
WHISPER_MODEL    = "base"
WHISPER_DEVICE   = "cpu"   # keep off CUDA — leaves iGPU memory for Ollama/MediaPipe
WHISPER_COMPUTE  = "int8"
WHISPER_LANGUAGE = None     # None = auto-detect within WHISPER_LANGUAGES; force one with "en"/"tl"
# Restrict auto-detect to just the languages Kai is actually spoken to in. Whisper always picks from
# all 99 otherwise, and on short or unclear audio it is confidently wrong: measured on this box, a
# 2 s clip scored en 0.34, **cy (Welsh) 0.22, nn (Norwegian Nynorsk) 0.21** — which is why replies
# came back in Spanish and Norwegian.
#
# Costs nothing in the normal case: `transcribe(language=None)` already returns probabilities for
# every language, so the first pass is reused whenever it lands on an allowed one. Only an utterance
# detected as something else pays a second pass, forced to the best allowed language — and that is
# precisely the case that was previously returned as garbage. (A pre-emptive detect_language() pass
# was measured at 88% of a full transcribe, so doing it every time would nearly double turn latency.)
#
# Empty tuple or None = allow all 99, i.e. the old behaviour.
WHISPER_LANGUAGES = ("en", "tl")
# ctranslate2 defaults to EVERY core, so one transcription takes ~40% of all six — competing with
# MediaPipe and the servo control loop, and showing up as jittery face tracking rather than as
# anything audio-shaped. That was tolerable when STT only ran on a button press; the whisper wake
# tier runs it per nearby utterance. Capped at 4 of 6 to leave the tracking loop its headroom.
# Watch `[control] N Hz` in the log after changing this. None = ctranslate2's default (all cores).
WHISPER_CPU_THREADS = 4
# Beam width for the turn transcribe. faster-whisper defaults to 5; greedy (1) measured 1.29 s vs
# 1.44 s on base+forced-en and 4.05 s vs 4.46 s on small — a consistent ~10% off every turn, with no
# transcript differences observed on the test utterances. Raise back to 5 if accuracy regresses.
WHISPER_BEAM_SIZE = 1
# Decoder bias for the turn transcribe. Whisper has never heard of DEVCON and renders it as
# whatever it does know — "defcon", "dev com", "Devon" — which ai/query_alias.py then has to
# repair by guesswork. Seeding the decoder with the vocabulary it is about to need is the cheaper
# fix, and it is the only one that reaches multi-word names: "geeks on the beach" is a perfectly
# ordinary English phrase, so no fuzzy matcher can safely flag it, but a primed decoder writes
# "Geeks on a Beach" in the first place.
#
# Kept to a bare comma-separated name list, and deliberately short. faster-whisper prepends this
# as previous-context tokens, so a long or sentence-shaped prompt gets *continued* rather than
# used as vocabulary — that is the same mechanism that makes Whisper emit filler on silence (see
# vad_filter in _transcribe), and it gets worse the more prose you give it. Not applied to the
# wake scan: that tier only needs "hey kai" and runs on a weaker model, where the bias would
# show up as invented DEVCON talk in overheard room noise. None/"" disables.
WHISPER_INITIAL_PROMPT = ("DEVCON Philippines, DEVCON PH, DevConnect Philippines, Campus DEVCON, "
                          "Geeks on a Beach, Jumpstart internships, Winston Damarillo, Kai.")

# ── Input level normalisation (ASR only) ──────────────────────────────────────────
# Whisper receives `audio / 32768.0` and nothing else — so how loud the speaker was is passed
# straight through to the decoder. That is fine for someone leaning over the robot and bad for
# someone across the room: level falls ~6 dB per doubling of distance, so a talker at 2 m arrives
# several times quieter than the close-mic audio every other constant here was tuned against, and
# Whisper degrades on quiet input for no reason other than the level.
#
# This is a LEVEL fix, not a noise fix. It changes no signal-to-noise ratio and cannot rescue audio
# the mic never really captured; it only stops the decoder being handed a needlessly small number.
# Applied in ai/voice_assistant._transcribe, i.e. to the ASR path ONLY — the VAD floors and the
# acoustic wake engines work in absolute int16 units on the un-normalised signal and must keep
# doing so (see MicStream._asr_signal in ai/session.py for where that split lives).
ASR_NORMALIZE = True
# Target RMS in float units (1.0 = full scale). 0.06 is about -24 dBFS — a normal speech level,
# comfortably clear of the decoder's quiet end and still ~24 dB below clipping.
ASR_NORMALIZE_TARGET_RMS = 0.06
# Ceiling on the lift, for the same reason WAKE_AMBIENT_MAX_LIFT exists: an unbounded gain applied
# to a near-silent buffer produces loud noise, and loud noise is exactly what Whisper hallucinates
# sentences out of. 8x is ~18 dB, i.e. it reaches roughly 8 m of extra distance and then stops.
ASR_NORMALIZE_MAX_GAIN = 8.0
# Never let the lift clip. Checked against the actual peak, so a quiet utterance containing one
# transient (a door, a table knock) is limited by that transient rather than squared off.
ASR_NORMALIZE_PEAK_CEILING = 0.95
# Below this input RMS the buffer is left alone. There is no speech in something this quiet, and
# amplifying it only manufactures the near-silence hallucinations TRANSCRIPT_MAX_NO_SPEECH_PROB
# below exists to catch. ~-66 dBFS.
ASR_NORMALIZE_MIN_RMS = 0.0005
# Whether the WAKE SCAN's audio is normalised too (the "tiny" model, tier 3). Separate from the
# turn path on purpose: the scan runs on every nearby utterance including room noise, and that
# model is documented here as confidently hallucinating whole sentences out of noise. Turn this
# half off first if false wakes increase — it leaves the turn path, where the win is, untouched.
ASR_NORMALIZE_SCAN = True

# ── Transcript sanity gate ────────────────────────────────────────────────────────
# WHISPER_LANGUAGES above restricts the detected-language LABEL. It never looks at the text, and that
# is a real hole: on unintelligible audio Whisper routinely labels a clip "en" — which IS allowed, so
# no re-transcribe fires — and then emits something that is not English or Tagalog at all. Observed
# on this robot: "hey kai" decoded as '嘿哀' and 'Hẹc gai!'. That garbage reached the LLM as if it were
# a question, and Kai answered it.
#
# So the OUTPUT gets checked too. English and Tagalog are both Latin script (Tagalog's only extras
# are ñ and Spanish loanword accents, all Latin), which makes this an unusually clean test: a
# transcript whose letters are mostly non-Latin cannot be either language, whatever the label says.
TRANSCRIPT_SCRIPT_GUARD = True
# Fraction of ALPHABETIC characters that must be Latin. Not 1.0 — one stray glyph in an otherwise
# good sentence is not worth throwing a whole turn away. Punctuation, digits, spaces and emoji are
# not alphabetic and are ignored, so they can neither trip the check nor mask a failure.
TRANSCRIPT_MIN_LATIN_RATIO = 0.80

# Confidence floor on the decode itself. faster-whisper applies its own log_prob_threshold to decide
# whether to RETRY at a higher temperature, but it still returns whatever it ended up with — nothing
# was rejecting a confidently-wrong decode of room noise.
#
# -1.0 is faster-whisper's own notion of "this decode went badly", so it is the natural bar. Move
# toward -0.5 to reject more aggressively if noise still gets through; make it more negative if real
# speech in a loud room starts being discarded. None disables this half.
TRANSCRIPT_MIN_AVG_LOGPROB = -1.0
# Whisper's own estimate that the clip is silence. vad_filter already drops non-speech stretches, so
# this only catches what survives it — a high value WITH text attached means it decoded words out of
# something it simultaneously believes is silence, which is the signature of hallucinated filler
# ("Thank you.", "You", "Thanks for watching!"). None disables.
TRANSCRIPT_MAX_NO_SPEECH_PROB = 0.80

OLLAMA_URL       = "http://localhost:11434/api/chat"
OLLAMA_MODEL     = "gemma2:2b"  # switched from gemma3:4b (~4.3GB) to fit the camera in 8GB. REVERT: "gemma3:4b"
OLLAMA_TIMEOUT_S = 90      # generous — measured ~54s cold-load-to-response on-device
# Kai is a single-purpose appliance, not a shared server — Ollama has no reason to evict
# gemma3:4b between turns. A short keep_alive (Ollama's default is 5m) meant any gap longer
# than that between push-to-talk uses paid the ~48s reload cost again. Keep it loaded for the
# life of the Ollama service instead.
OLLAMA_KEEP_ALIVE = -1     # must be a JSON number, not a string — Ollama treats "-1" (string)
                           # as an invalid Go duration and returns 400 Bad Request
# Our prompts (short system prompt + a few history turns + a short reply) need nowhere near
# Ollama's 4096-token default context. On this 8GB Jetson, requesting the full default context
# left gemma3:4b's KV cache too big to fit alongside the camera/MediaPipe process, so Ollama
# split the model 45%/55% CPU/GPU (much slower generation). Trimming num_ctx frees enough
# memory for a full GPU offload — measured ~2x faster token generation as a result.
# 2048 (was 1024): that reasoning was measured on gemma3:4b, which no longer runs here. On
# gemma2:2b the extra KV cache costs ~35MB (2232MB -> 2198MB available, measured on-device with
# the camera up) and buys the headroom RAG turns actually need — a 6-turn session peaked at 959
# tokens of the old 1024, i.e. 94% full, and Ollama silently drops history to fit (it preserves
# the system prompt, so the documents survive and the conversation is what rots). 4096 is NOT
# available: the llama runner terminates on load. Changing this forces one model reload, so do
# it while face_track.py is stopped rather than mid-conversation.
# The whole 8 GB shared-memory budget this number lives inside — what is resident, what is left,
# and why the reload needs the camera down — is written up in docs/memory-budget.md. Read it
# before changing this line or the model.
OLLAMA_NUM_CTX   = 2048

# GPU layers for Ollama. None = let Ollama auto-decide the GPU/CPU split (fast — gemma2:2b fits
# the GPU alongside the camera on a freshly-booted/defragmented Jetson). 0 = force CPU (reliable
# but too slow for conversation). A positive int forces that many layers on GPU.
# The `cudaMalloc: out of memory` 500s seen earlier were GPU *fragmentation* after hours of model
# thrashing — a reboot defragments and GPU inference is fast again. If OOM recurs on long uptimes,
# free GPU headroom (e.g. disable the desktop GUI: `systemctl set-default multi-user.target`).
OLLAMA_NUM_GPU   = None
# Hard cap on generated tokens. Generation was previously unbounded while TTS_MAX_SPOKEN_CHARS
# silently discarded the overflow — so a long reply cost generation time AND synthesis time for text
# nobody ever heard. Every token saved here is paid back twice, once in generation (~27 tok/s) and
# again in Piper synthesis (~0.55x realtime). None = uncapped.
#
# 160 (was 96, briefly 192): 96 was sized as ~2x the 36-52 tokens measured back when persona.txt
# demanded "1 to 3 short sentences" unconditionally. That line is gone — persona.txt now lets the
# question set the length, up to a hard four spoken sentences, which is ~125 tokens. 96 would cut
# that mid-sentence; 160 only truncates runaways. 192 was tried first against a looser persona that
# invited five sentences, and on-device that produced 134-word replies — correct, well grounded, and
# far too long to listen to. The persona is what fixed that, not this number.
#
# Deliberately set ABOVE what TTS_MAX_SPOKEN_CHARS (500) allows, so the CHARACTER clamp is the one
# that normally binds. That ordering matters for how a runaway sounds: tts.clamp_for_speech backs up
# to a sentence end, while num_predict stops the model wherever the 160th token fell — usually
# mid-word. Keep this the looser of the two if you retune either.
OLLAMA_NUM_PREDICT = 160

# Ceiling on the `/api/ps` placement probe (ai/voice_assistant.log_model_placement). Purely
# diagnostic, and it runs on the startup warm path, so it must not be able to hold anything up if
# the Ollama service is wedged — short on purpose, and every failure only logs.
OLLAMA_PS_TIMEOUT_S = 3.0

# Log Ollama's own per-request timings (prompt eval / generation / model load) after every reply.
# One line per turn, and the only way to tell a slow prompt from slow generation from a model
# reload — the three causes of "Kai feels slow", each with a different fix. Set False to quieten it.
OLLAMA_LOG_TIMINGS = True

# WHERE the retrieved RAG context is placed in the prompt.
#   "user"   — prepended to the user's turn (default). Keeps the system prompt and the whole rolling
#              history byte-identical between turns, so Ollama's KV cache can reuse them as a prefix
#              and only the new context + question is evaluated.
#   "system" — appended to the system prompt (the original behaviour). The retrieved text then sits
#              at the FRONT of the prompt and changes every turn, which invalidates the cached
#              prefix and re-evaluates the persona and all MAX_HISTORY_TURNS of history, every turn.
# "user" is the faster placement and is also the shape Gemma2's chat template actually expects (it
# alternates user/assistant strictly, so a mid-conversation system message is not representable).
# REVERT to "system" if retrieved facts stop being treated as authoritative — ai/rag.format_context's
# header was tuned with the block in the system position. Verify with the known DEVCON questions.
RAG_CONTEXT_PLACEMENT = "user"

# user+assistant pairs kept in the rolling context. 6 (was 3): at 3, Kai forgot the opening
# question by turn 4 and then answered about it confidently anyway ("what was the first thing I
# asked you?" -> the wrong program). This is the cap that was doing the forgetting, well before
# Ollama's truncation would. Raise it only alongside OLLAMA_NUM_CTX — the two are one budget.
MAX_HISTORY_TURNS = 6

# ── Who Kai is talking to ───────────────────────────────────────────────────────
# A name offered in speech ("I'm Jhondel", "ako si Jhondel") is pinned for the life of the session
# and injected into the system prompt, so it outlives the MAX_HISTORY_TURNS window directly above,
# which would otherwise evict it six exchanges later. Extraction is ai/identity.py — pure stdlib, no
# second LLM call. See docs/tickets/S12. False disables the whole path and restores the
# persona+history prompt byte for byte.
IDENTITY_CAPTURE = True

# Injected into the SYSTEM prompt, not the user turn — the opposite placement to
# RAG_CONTEXT_PLACEMENT above, and for the opposite reason. The retrieved context changes every
# turn, so parking it at the front of the prompt invalidates Ollama's cached prefix every turn. This
# string does NOT change once learned, so in principle it costs one prefix invalidation at the
# moment the name is captured and nothing afterwards.
#
# MEASURED 2026-08-10, and the "one invalidation" half is NOT confirmed — it is currently
# unmeasurable on this robot. Every `[llm] turn:` line in /tmp/face-servo.log is preceded by
# `MODEL RELOADED: ~200-360ms — placement was re-decided`, on every turn, so there is no surviving
# KV prefix between turns for anything to invalidate. What WAS measured is that the injection costs
# nothing detectable: turns with a name pinned evaluated their prompt in 258-304 ms
# (2654-3394 tok/s), inside the spread of turns without one (215-465 ms).
# Re-measure if the per-turn reload is ever fixed — that is the point at which the prefix reasoning
# starts to mean something. If prompt_eval_* then spikes every turn rather than once, the string is
# being rebuilt and this placement is wrong.
#
# The "not in every reply" clause is load-bearing. Without it the model opens more or less every
# sentence with the name, which reads worse than never using it at all.
IDENTITY_PROMPT = ("The person you are talking to is called {name}. Use their name naturally in "
                   "conversation, not in every reply.")

# Length bounds on the captured word. The floor rejects initialisms and stray particles that survive
# the stop-list ("I'm K", "ako si a"); the ceiling rejects a run-on where an anchor matched inside an
# unrelated clause.
IDENTITY_MIN_LEN = 2
IDENTITY_MAX_LEN = 20

# Require a capitalised first letter for the WEAK anchors ("I'm X", "ako'y X") only. Whisper
# capitalises proper nouns fairly reliably, and that is what separates "I'm Jhondel" from "I'm fine"
# once the stop-list has taken the common cases. Set False only if real names are being missed
# BECAUSE of casing — measure first, because the weak tier is only safe with the corroboration on.
# The strong anchors ("my name is X", "ako si X") never consult this.
IDENTITY_WEAK_ANCHORS_NEED_CAPITAL = True

# Words that are never a name, checked casefolded. This is the list that will need editing after a
# false accept is heard at an event — which is why it is here and not in ai/identity.py. Everything
# in it is reachable through a weak anchor: "I'm fine", "I'm from Cebu", "I'm a developer",
# "I'm just looking", "ako'y masaya". Tagalog function words are included for the same reason the
# English ones are.
IDENTITY_STOPWORDS = frozenset({
    # English answers to "how are you" and friends
    "fine", "good", "great", "ok", "okay", "well", "alright", "sorry", "sure", "here", "back",
    "done", "ready", "busy", "tired", "hungry", "happy", "sad", "curious", "confused", "lost",
    "new", "old", "young", "late", "early", "right", "wrong", "serious", "kidding", "joking",
    # articles, determiners, prepositions, and the rest of the connective tissue
    "a", "an", "the", "from", "in", "at", "on", "with", "for", "to", "of", "about", "into", "over",
    "and", "but", "or", "so", "not", "no", "yes", "very", "really", "just", "still", "also",
    "going", "trying", "looking", "asking", "talking", "wondering", "thinking", "saying", "doing",
    "there", "this", "that", "these", "those", "it", "its", "my", "your", "our", "their",
    # roles people offer instead of a name
    "student", "teacher", "developer", "engineer", "designer", "programmer", "founder", "intern",
    "volunteer", "member", "speaker", "organizer", "organiser", "guest", "visitor", "user",
    # Tagalog function words and common answers reachable through "ako'y" / "ako si"
    "po", "ang", "ng", "sa", "na", "ay", "din", "rin", "lang", "naman", "kasi", "pala", "ba",
    "hindi", "oo", "opo", "sige", "salamat", "maayos", "masaya", "malungkot", "pagod", "gutom",
    "taga", "galing", "dito", "diyan", "doon", "ito", "iyan", "iyon", "siya", "sila", "kami",
})

# ── Jaw "speaking" pantomime ────────────────────────────────────────────────────
# Kai has no audio (yet), so when a reply is produced we drive the jaw servo for a window
# sized to how long that text would take to say aloud. The mouth opens once per sentence:
# ramps open at the start, holds open while "spoken", closes at the end, with a short closed
# pause between sentences. The schedule is built in ai/speak_envelope.py and face_track.py reads
# speaking_openness() each frame.
SPEAK_SEC_PER_WORD   = 0.34   # ~175 wpm — sets how long each sentence stays open for N words
SPEAK_MIN_SENTENCE_S = 0.6    # floor so a one-word sentence still visibly opens the jaw
SPEAK_MAX_S          = 15.0   # overall ceiling so a runaway reply can't pin the jaw forever
SPEAK_GAP_S          = 0.20   # closed pause between sentences
SPEAK_AMP            = 1.00   # how far the mouth opens (1.0 = fully open) while a sentence is said
SPEAK_OPEN_S         = 0.22   # ramp-open time at the start of a sentence (smooth, not a snap)
SPEAK_CLOSE_S        = 0.22   # ramp-close time at the end of a sentence

# ── Lining the jaw up with the SOUND ─────────────────────────────────────────────
# The two constants below exist because the jaw window used to be timed from the wrong two
# instants, and on a short line that is the whole illusion. Reported on the robot 2026-08-12:
# after "Hey Kai" the mouth moved and the "Yes?" arrived afterwards.
#
# Both errors are at the START of the file and both push the same way, so they add:

# (1) PLAYBACK DOES NOT START WHEN WE ASK IT TO. ai/voice_assistant.py opened the jaw window at the
# instant it called tts.play(), but that only spawns paplay — the first sample is not audible until
# the process has connected to PulseAudio and the stream buffer has filled. Measured on the robot
# 2026-08-12, `pactl list sink-inputs` sampled through a 5 s playback with the same flags ai/tts.py
# uses: Buffer Latency 57-120 ms plus Sink Latency 63-98 ms, i.e. ~190-210 ms of pipeline the first
# sample has to cross, and paplay's own spawn + connect on top (~30 ms warm, ~210 ms on the first
# playback of a process, which also pays apply_output_profile()).
#
# TRACKS TTS_LATENCY_MSEC (200) — the buffer half of that measurement IS that setting. Re-measure
# this if you change it; the same command is in the note there. Set 0.0 to restore the old
# behaviour of opening the jaw at the play() call.
TTS_PLAYBACK_LEAD_S  = 0.22

# (2) THE FILE IS LONGER THAN THE SPEECH. Piper pads leading and trailing silence into every WAV
# (config/filler.py's FILLER_MAX_STALL_S note has been carrying this measurement for a while, and
# calls timing the jaw off SPEECH rather than file length "the honest fix ... its own pass"). This
# is that pass, for the jaw only — FILLER_MAX_STALL_S still measures the file, deliberately, since
# what it bounds is how long the speaker is busy.
#
# Measured on the ack WAV the robot is actually playing (/tmp/kai_ack/kai_canned_ack.wav,
# 2026-08-12): 0.824 s long, of which 0.040 s is leading silence and 0.244 s is trailing silence
# and reverb tail. So barely two thirds of that file is the word "Yes?", and the jaw was miming
# across all of it — holding open for a quarter-second after the sound had stopped.
#
# AND THE PAD IS NOT A FIXED NUMBER, which is why this is a scan and not a constant: the canned
# lines are re-synthesized on every startup, and the very next restart wrote an ack of the same
# 0.824 s with the speech at 0.080-0.620 instead — the same word, 40 ms further into the file.
#
# Set SPEAK_TRIM_SILENCE False to time the jaw to the whole file again.
SPEAK_TRIM_SILENCE   = True
# PEAK sample value that counts as sound, against a 16-bit full scale of 32768 — so 600 is about
# -35 dBFS. Peak rather than RMS because the scan runs immediately before playback and max()/min()
# over an array slice is C-speed (see ai/tts.wav_speech_span).
#
# Swept 50..3000 over the robot's own WAVs 2026-08-12 (the ack, the two canned failures, a stall, two
# 8 s openers and a real reply) rather than picked. There IS a plateau and it is 300..1000: every
# threshold in that band put the ack's speech at 0.040-0.580/0.600 s of its 0.824 s file, i.e. within
# 20 ms. 600 sits in the middle of it. Outside the band it degrades gracefully in both directions —
# at 50-100 TTS_POST_ROOM's reverb tail still reads as speech (the ack's end drifts out to 0.64-0.80 s,
# so the jaw hangs open again); at 2000-3000 it starts clipping quiet word-endings (0.56, 0.54).
#
# The scan costs 1.3 ms on the ack and 1.6 ms on an 8 s line, measured on the Jetson the same day —
# it is on the speech path immediately before playback, so that was worth knowing.
SPEAK_SILENCE_PEAK   = 600
# Scan granularity. 20 ms is finer than the jaw's own ramps (SPEAK_OPEN_S = 0.22) so the precision
# is not the limit here, and it keeps the scan cheap on a 15 s reply.
SPEAK_SILENCE_STEP_S = 0.020

# ── Text-to-speech (Piper) ────────────────────────────────────────────────────────
# Kai speaks its replies aloud through the USB audio dongle + PAM8403 amp (PulseAudio sink
# TTS_SINK). ai/tts.py shells out to Piper (piper-tts, CPU/onnxruntime — no API key, no runtime
# internet, ~tens of MB RAM, so it fits alongside Ollama/Whisper/MediaPipe on the 8GB Jetson) to
# synthesize a WAV, then plays it with paplay. When TTS is unavailable (disabled, engine missing,
# or the voice model absent) the assistant degrades to the silent jaw pantomime above.
TTS_ENABLED      = True
TTS_ENGINE       = "piper"
# How to invoke Piper. `python3 -m piper` matches the installed piper-tts; swap to
# ["/home/devconph/.local/bin/piper"] if the module entry point ever changes. Text is fed on stdin
# and the WAV is written with `-m MODEL -f OUTFILE` (flags verified against piper-tts 1.4.2).
TTS_PIPER_CMD    = ["python3", "-m", "piper"]
# Voice model, relative to the project root (resolved in ai/tts.py). This is the she/they voice —
# swap the file (and download its .onnx + .onnx.json) to change how Kai sounds:
#   python3 -m piper.download_voices <name> --data-dir voices
# Already downloaded here, all interchangeable by editing this one line:
#   en_US-hfc_female-medium   natural US female  — the current pick (0.42x realtime synth)
#   en_GB-jenny_dioco-medium  warm British female                (0.45x)
#   en_US-libritts_r-medium   most expressive of the mediums     (0.59x)
#   en_US-lessac-medium       the previous default               (0.30x)
# AVOID the "-high" models for live conversation: en_US-lessac-high and en_US-ljspeech-high sound
# better but synthesize at ~0.95x realtime on this Jetson's CPU, i.e. a 7 s reply costs ~7 s of dead
# air before it starts. Both are in voices/ if you want to A/B them again (/tmp/kai_voice_ab holds
# the last set of samples).
TTS_VOICE_MODEL  = "voices/en_US-hfc_female-medium.onnx"
TTS_LENGTH_SCALE = 1.0   # Piper phoneme length / speaking rate — >1 slower, <1 faster

# ── Kai breathing between sentences ─────────────────────────────────────────────
# Seconds of silence Piper inserts at each SENTENCE boundary. Until 2026-08-10 this flag was never
# passed, and Piper's default is 0 — so Kai said four sentences in one continuous run, which is a
# thing no person does and one of the plainest "this is a machine" cues in the whole audio path.
#
# MEASURED on the robot, longest interior silences in the RAW pre-sox WAV of a four-sentence reply
# (relative gate — see the note on DELIVERY_PAUSE below for why an absolute one finds nothing):
#
#   setting            the three sentence boundaries
#   0.0 (= the old     0.20, 0.17, 0.14 s     <- not a pause; just a phrase-final decay
#        behaviour)
#   0.3                0.49, 0.47, 0.45 s
#   0.5                0.80, 0.74, 0.67 s
#
# Conversational speech pauses roughly 0.4-0.7 s between sentences, so 0.35 sits at the lower, safer
# end of natural: enough to read as a breath, not enough to read as Kai having stalled. Raise toward
# 0.5 if it still runs together in the room; drop to 0.0 to restore the old behaviour exactly.
#
# Note what this is NOT: ai/delivery.py's DELIVERY_PAUSE buys a 0.156 s breath *inside* a long
# sentence, and was measured and shipped while the boundary a person leans on hardest got nothing.
# The two are complementary, not alternatives.
#
# COSTS, both checked before this shipped:
#   * ~0.3 s per interior boundary, so a four-sentence reply runs ~1 s longer. Kai is deaf while
#     speaking (no echo cancellation), so this lengthens the deaf window slightly. It does NOT delay
#     first audio — synthesis time is unchanged, measured 3.4-3.7 s per long line at every value.
#   * NO trailing padding: measured 0.13-0.18 s of tail silence at every setting, i.e. unchanged. So
#     a one-sentence filler stall keeps its length (1.11 -> 1.14 s, inside run-to-run noise),
#     FILLER_MAX_STALL_S needs no re-deriving, and the jaw cannot be left miming into added silence
#     at all — which matters because the jaw schedule is sized from wav_duration, so any silence added
#     to the end of a WAV is silence the jaw spends mouthing nothing.
# Dashboard-settable so it can be A/B'd mid-conversation; a change re-warms the canned lines.
TTS_SENTENCE_SILENCE_S = 0.35

# ── Piper's noise parameters: plumbed, and deliberately left at the voice's own values ──────────
# Piper is a VITS model and generates prosody by sampling noise. Two parameters control how much it
# varies, and neither was ever passed — so both sat at whatever the voice file shipped:
#
#   noise_scale   acoustic sampling: timbre and pitch contour within a line
#   noise_w       phoneme DURATION variation — the rhythm
#
# Read off voices/*.onnx.json, which are checked in: en_US-hfc_female-medium (shipping),
# en_GB-jenny_dioco and en_US-lessac all ship 0.667 / 0.8 — the stock VITS defaults — while
# en_US-libritts_r-medium ships 0.333 / 0.333. Worth knowing before ever A/B-ing those two voices
# again: the previous comparison (docs/plan/completed/expressive-voice-plan.md) was unknowingly
# comparing two voices at less than half each other's variation.
#
# THESE DEFAULTS ARE THE SHIPPING VOICE'S OWN, so passing them changes no audio. That is not
# timidity, it is what the measurement said. Four repeats per config of the same long line, because
# VITS draws fresh noise every run and one sample per cell cannot tell an effect from a draw:
#
#   config                          p10-p90 range, 4 runs        mean      duration
#   today (no flags)                9.4 / 9.6 / 9.5 / 9.8        9.58 st   16.16 s
#   noise_scale 0.8, noise_w 1.1    9.0 / 9.4 / 9.4 / 9.9        9.43 st   16.80 s
#
# Within-config spread is +-0.4-0.9 st, so the knobs move intonation by LESS THAN THE NOISE FLOOR,
# and the livelier setting measures marginally flatter than today. An 8-cell sweep (noise_w 0.8-1.2
# x noise_scale 0.667/0.8) stayed inside 8.8-10.0 st, the same band. What they do change, outside the
# noise, is timing: +0.64 s on identical text, ~4%.
#
# So they are here to be TRIED BY EAR (both are dashboard-settable, bounded 0-1.5 in settings.py),
# not because a number justifies moving them. If you do: noise_w first and in small steps, and past
# roughly 1.0-1.2 either one starts slurring consonants and wandering off pitch rather than sounding
# expressive. Anything you conclude by ear, write it here with the date — this file is the only place
# that record survives.
TTS_NOISE_SCALE = 0.667
TTS_NOISE_W     = 0.8

# ── Let sentences differ in loudness ────────────────────────────────────────────
# Piper normalises EVERY SENTENCE to full scale on its own. Measured on the robot, per-sentence peaks
# in a four-sentence reply: -0.00, -0.00, -0.00, -0.00 dBFS — a spread of exactly 0.00 dB. No two
# sentences Kai has ever spoken differed in peak level, which is not something a person can do.
#
# False passes --no-normalize and hands that job to the sox chain, which peak-normalises ONCE over the
# whole reply (TTS_POST_EFFECTS ends in `gain -n -1`) and so preserves the relative differences.
#
# BE HONEST ABOUT THE SIZE OF THIS. Measured through the real chain:
#
#   normalize on   raw -0.00 -0.00 -0.00 -0.00  ->  after sox  -1.00 -1.00 -1.00 -1.00   (0.00 dB)
#   normalize off  raw -11.00 -11.83 -13.16 -9.74 -> after sox  -1.00 -1.00 -1.79 -1.00   (0.79 dB)
#
# So 3.42 dB of natural variation arrives as 0.79 dB: the compand's upward compression eats the rest.
# Recovering it would mean softening `compand`, and that is the setting keeping Kai audible in a loud
# room — a venue beats 2.6 dB of sentence dynamics. Not worth it; this is a free 0 -> 0.79, no more.
#
# COUPLED TO THE SOX CHAIN ON PURPOSE (see ai/tts._run_piper): with normalisation off, Piper's raw
# output is ~10 dB quieter, so the flag is only passed when TTS_POST_PROCESS is on. If sox is present
# in config but fails at runtime, _post_process already falls back to the raw WAV with a WARNING —
# that reply is now quiet as well as mono. `tts_volume` on the dashboard is the immediate remedy.
TTS_PIPER_NORMALIZE = False

TTS_VOLUME       = 1.0   # playback volume, applied to paplay's sink input (1.0 = PA_VOLUME_NORM).
                         # Applied at playback, NOT at synthesis: TTS_POST_EFFECTS below ends in
                         # `gain -n -1`, which normalises a synthesis-time gain straight back out
                         # (measured — identical output peak at 0.4 and 1.6, and 1.6 clipped the raw
                         # audio). Dashboard-settable, 0..2.
# PulseAudio sink for playback — the USB dongle/PAM8403. Named (not index 0) so it survives
# re-enumeration. Find it with `pactl list short sinks`.
TTS_SINK         = "alsa_output.usb-C-Media_Electronics_Inc._USB_Audio_Device-00.analog-stereo"

# ── Output card profile (asserted before playback) ──────────────────────────────
# PulseAudio moves this dongle to its DIGITAL (S/PDIF) profile on its own. Observed live: the card
# was on output:analog-stereo, and minutes later — with nothing reconfigured — had flipped to
# output:iec958-stereo. When it flips, TTS_SINK above STOPS EXISTING (pactl answers "No such
# entity"), every paplay exits non-zero, and Kai mimes replies in silence: the samples leave on the
# optical path while the analog jack — the only one with an amp and speaker on it — stays quiet.
# Nothing in the log looks audio-shaped, which is what makes this expensive to diagnose from the
# symptom ("the speaker broke") rather than from the cause.
# So assert the analog profile instead of trusting Pulse to keep it: once before the first reply,
# and again before play()'s retry, since this flip is the likeliest reason a playback just failed.
# Best-effort, exactly like the I2S capture route above — a missing pactl/card only logs.
TTS_ASSERT_CARD_PROFILE = True
TTS_CARD         = "alsa_card.usb-C-Media_Electronics_Inc._USB_Audio_Device-00"  # `pactl list short cards`
# Keep the "+input:mono-fallback" half. This one card carries the dongle's mic as well, and the
# bare "output:analog-stereo" profile drops that capture source entirely — so asserting the short
# form here would fix playback by taking a mic away.
TTS_CARD_PROFILE = "output:analog-stereo+input:mono-fallback"
# Ceiling on the pactl call above, for the same reason MIXER_TIMEOUT_S exists in config/wake.py: an
# unresponsive pulseaudio must not be able to wedge the speak worker thread indefinitely.
TTS_PACTL_TIMEOUT_S = 5.0

# ── Delivery shaping (ai/delivery.py) ───────────────────────────────────────────
# Applied to the SPOKEN text only, on the way to Piper. Read ai/delivery.py's module docstring
# before touching these — in short: 29 voices across 7 engine families were measured on this box
# (docs/plan/completed/expressive-voice-plan.md) and the flat-tone complaint survived all of them, because every
# TTS model that fits beside Ollama here was trained on read-aloud audiobook corpora. The remaining
# lever is DELIVERY, not timbre: breaths, non-uniform pacing, a conversational opening.
#
# Default for the dashboard's "Natural delivery" toggle; the live value is
# settings.get("delivery_shaping"). Turn it off to hear the unshaped voice for comparison — that
# A/B is the only way to judge any of this, and it needs no restart.
DELIVERY_ENABLED = True

# The break inserted before a clause-initial conjunction. A SEMICOLON, and that is a measured choice
# rather than a typographic one — it is never seen, only heard.
#
# MEASURED on this robot (en_US-hfc_female-medium, length-scale 1.0, longest interior silence in the
# RAW pre-sox Piper output, 4 replies at the real insertion points):
#
#   token        avg pause   per-reply
#   none           0.053 s   0.090 0.110 0.070 0.050
#   comma          0.110 s   0.090 0.359 0.120 0.090   <- erratic: twice it bought NOTHING
#   semicolon      0.156 s   0.160 0.289 0.259 0.229   <- consistent, and ~3x "none"
#
# and on a single-sentence sweep of the other candidates: ellipsis 0.110 s, double-comma 0.140 s,
# " --" 0.100 s, period 0.080 s. The comma was the original choice on the reasoning that it cannot
# be mis-voiced — true, but it turned out to be worth ~10 ms in half the sentences, which is no
# breath at all. The semicolon measured equally unvoiced and roughly twice the pause.
#
# espeak voices some strings in ways you cannot predict by reading them (config/thinking.py records
# "Hmmmm..." transcribing back as "H-A-M-A-M-M"), so a change here is verified by synthesizing
# through Piper and running Whisper over the result — checking that the token's NAME ("semicolon",
# "comma") never appears in the transcript. Not by eye, and not by ear alone.
DELIVERY_PAUSE = ";"

# Conjunctions that may earn a breath before them. Multi-word entries are matched with flexible
# internal whitespace and win over the single word they start with.
#
# TAGALOG ADDED 2026-08-11, on the user's approval — this list used to be English-only, so a Tagalog
# reply matched nothing and went out completely unshaped: no breaths at all, in the language the room
# actually speaks. That was the intended degradation while nobody could vet Filipino markers.
#
# Deliberately connectives ONLY, no Tagalog openers. This transform inserts PUNCTUATION and adds no
# words, so it cannot mispronounce anything — which matters here, because the voice is en_US and a
# native speaker has judged Kai's Tagalog pronunciation bad (see docs/plan/wip/natural-audio-plan.md).
# Adding Tagalog *words* for that voice to say (the DELIVERY_OPENERS route) was considered and
# rejected for exactly that reason. Do not quietly add them later.
#
# "at" and "o" are left out on purpose: both are far too common and too short, and a breath before
# every "at" is a stutter rather than a rhythm. "kasi" and "kaya" are the two likeliest to misfire,
# since both sit mid-clause in Tagalog more often than "because"/"so" do in English — the
# DELIVERY_BREATH_MIN_WORDS / _MIN_TAIL_WORDS gates below are what keep that in check. If the Tagalog
# starts sounding choppy, drop those two first, not the whole list.
DELIVERY_BREATH_CONJUNCTIONS = (
    "but", "so", "because", "although", "though", "while", "which",
    "and then", "or", "unless", "whereas",
    "pero", "kasi", "kaya", "tapos", "kung", "habang", "para",
)
# Words that must precede a conjunction before it is worth breathing after, counted from the
# sentence start or the last existing break. The failure mode of this whole transform is
# over-punctuation: a comma every few words is a stutter, not a breath, and sounds worse than the
# flat reading it replaced. Raise this first if the delivery starts sounding choppy.
DELIVERY_BREATH_MIN_WORDS = 6
# ...and words that must FOLLOW it, so a break never strands a two-word tail.
DELIVERY_BREATH_MIN_TAIL_WORDS = 3
# Ceiling per sentence, for the same reason. 0 disables breath insertion entirely.
DELIVERY_BREATH_MAX_PER_SENTENCE = 1

# Discourse markers that may open a reply. This is the highest-value item in the block and the
# riskiest: it ADDS words the LLM did not generate (to the speech only — the dashboard still shows
# the real reply). It earns its place because a listener judges the first second of a turn, and
# right now every turn starts the same way, mid-fact. Keep them short, and keep them the kind of
# word that survives PH English/Tagalog code-switching. Empty tuple disables.
DELIVERY_OPENERS = ("So,", "Well,", "Okay,", "Right,", "Alright,")
# Percentage of replies that get one, chosen by a CRC of the text (stable per reply, evenly spread,
# never random). An opener on EVERY reply becomes its own fixed shape — the exact tic this is meant
# to break up — so this is deliberately well under 100. Lower it to 0 to keep breaths and tempo
# without ever adding a word.
DELIVERY_OPENER_RATE = 35
# Replies shorter than this get none: "Yes, at 9 AM." does not want a preamble.
DELIVERY_OPENER_MIN_WORDS = 8
# First words that already open conversationally, or where a marker would be actively wrong — a
# greeting, an apology, a direct yes/no. Compared lowercased with trailing punctuation stripped.
DELIVERY_OPENER_SKIP_STARTS = frozenset({
    "so", "well", "okay", "ok", "right", "alright", "actually", "sure", "yeah", "yes", "no",
    "hi", "hello", "hey", "sorry", "oh", "hmm", "kumusta", "oo", "hindi", "opo",
})

# Per-reply jitter on Piper's --length-scale, as a fraction: 0.06 = ±6% speaking rate, keyed on the
# text. Aimed at "uniform pacing" — two consecutive replies currently come out at byte-identical
# tempo, which no single-reply improvement can fix. Costs nothing (one CLI argument, not a re-synth).
# Past a few percent it stops reading as natural variation and starts reading as a rate bug. 0 off.
DELIVERY_TEMPO_JITTER = 0.06
# Hard bounds on the result — the same range as the dashboard's Speaking rate slider, so a mis-set
# jitter can never hand Piper a scale that smears the voice.
DELIVERY_TEMPO_MIN = 0.5
DELIVERY_TEMPO_MAX = 2.0

TTS_OUTPUT_DIR   = "/tmp"                 # where the transient reply WAV (kai_tts.wav) is written
# paplay needs XDG_RUNTIME_DIR to find the PulseAudio socket in non-login contexts (the @reboot
# cron autostart, or an SSH session). ai/tts.py forces this in the playback subprocess env.
TTS_XDG_RUNTIME  = "/run/user/1000"
# Ask PulseAudio for a small output buffer instead of its default. Measured on this box:
#   `pactl list sinks` -> "Latency: 1904406 usec, configured 2000000 usec"
# i.e. a TWO SECOND buffer, so paplay exited up to 1.9 s before the speaker actually went quiet.
# That is what made Kai hear his own reply and answer it, and it forced a 2 s deaf window after every
# reply just to cover the drain. At 200 ms the mic can reopen ~0.5 s after Kai stops (see
# TTS_TAIL_MUTE_S in config/wake.py), which is the difference between feeling responsive and feeling
# broken. Raise it if playback ever crackles or underruns; set None to use Pulse's default.
#
# Confirmed again 2026-08-12 with the flag on and off, sampling `pactl list sink-inputs` through a
# 5 s playback: with it, Buffer Latency 57-120 ms and paplay ran 5.46 s for a 5.00 s file (it waits
# for the drain); without it, Buffer Latency 2000 ms and paplay returned in 5.02 s — exiting with
# most of a two-second buffer still to play, exactly as described above.
#
# This is also the buffer half of TTS_PLAYBACK_LEAD_S, which is how long the jaw waits before it
# starts miming. Re-measure that one if you change this.
TTS_LATENCY_MSEC = 200

# ── TTS loudness post-processing ────────────────────────────────────────────────
# Piper emits a quiet, MONO WAV (~6 dB lower RMS than typical playback, and mono routes to only
# part of a stereo speaker path). Pipe it through sox to (a) duplicate to stereo so BOTH amp
# channels are driven, and (b) compress + peak-normalize so speech is as loud as other audio.
# Best-effort: if sox is missing or the filter fails, ai/tts.py falls back to the raw Piper WAV
# unchanged (speech still plays, just quieter/mono). Set TTS_POST_PROCESS False to disable.
TTS_POST_PROCESS  = True
TTS_POST_SOX      = "sox"   # sox binary (on PATH); swap for an absolute path if needed
TTS_POST_CHANNELS = 2       # 2 = duplicate mono -> stereo (drive both speaker channels)
# sox effect chain applied after synthesis: light compression to lift perceived loudness, then
# peak-normalize to -1 dB. Tune here without touching code (verified against sox 14.4.x).
TTS_POST_EFFECTS  = ["compand", "0.3,1", "6:-70,-60,-20", "-5", "-90", "0.2", "gain", "-n", "-1"]

# ── Giving Kai a room to speak in ───────────────────────────────────────────────
# The chain above is LOUDNESS ONLY, and that was the whole of the output path until 2026-08-11: no EQ,
# no space. Piper writes 22050 Hz mono, sox duplicates it to two identical channels, compresses and
# peak-normalises. Bone-dry, dead-centre, flat-EQ speech is itself a synthetic cue, independent of
# which model produced it — a real voice always arrives with a room attached.
#
# These two run either side of the loudness chain (see ai/tts._post_process for the order and why),
# and each is its own constant so loudness, EQ and space revert independently. Empty list = off.

# Everything below ~90 Hz is energy this hardware cannot reproduce: a PAM8403 into a small driver has
# no output down there, so it only eats the headroom the compand then reacts to, and it is what makes
# the chassis buzz on plosives. Applied BEFORE the compand for that reason — removing it afterwards
# would be too late to matter. Mirrors MIC_HIGHPASS_HZ = 80 on the capture side.
TTS_POST_HIGHPASS = ["highpass", "90"]

# A small room, applied AFTER the compand: compress the dry signal, then add the space, which is the
# order a person mixing this would use. sox's `reverb` argument order is
#   reverberance  HF-damping  room-scale  stereo-depth  pre-delay(ms)  wet-gain(dB)
# so this is 18% reverberance, a 28% room and the wet signal 4 dB down — deliberately subtle. The
# stronger variant tried alongside it was `30 50 45 100 0 -2`; both are rendered in
# voice-audition/room-ab/ for comparison.
#
# MEASURED before shipping, because a reverb tail is exactly the kind of thing that breaks other
# contracts here:
#   * Duration is UNCHANGED — 7.84 s for the test line and 1.27 s for a filler stall under every
#     chain tried. sox decays the tail into the silence Piper already pads onto every WAV rather than
#     appending to the file, so FILLER_MAX_STALL_S (1.8) and FILLER_MAX_LINE_S need no re-deriving,
#     and the jaw window (sized from wav_duration) does not change.
#   * Trailing silence SHRINKS, 0.13 s -> 0.10 s on the line, because the tail now occupies it. That
#     is the right direction: it cannot strand the jaw miming into added silence.
#   * Intelligibility: Whisper transcribed all four chains IDENTICALLY on the same English line, so
#     the room costs nothing a decoder can detect.
# What none of that measures is the VENUE. Reverb trades against clarity in ambient noise, and this
# robot has already lost hands-free once to a loud room (see config/wake.py's ambient adaptation). If
# Kai gets harder to understand at an event, set this to [] and keep the high-pass.
TTS_POST_ROOM = ["reverb", "18", "50", "28", "100", "0", "-4"]
