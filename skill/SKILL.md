---
name: stt
description: Speech-to-text pipeline for long-form audio/video on Apple Silicon. Preprocesses audio (denoise + loudnorm), transcribes with whisper.cpp + Core ML (large-v3-turbo, GPU+Neural Engine), diarizes speakers locally (pyannote.audio with non-gated fallback, then Resemblyzer), auto-attributes speaker names from conversation content via LLM, and produces a full set of output files (clean.wav, transcript.txt, attributed.txt, per-speaker txts, json, srt). Use when the user asks to "transcribe", "stt", "speech to text", "convert audio to text", "diarize", or hands over an audio/video file (.m4a, .mp3, .wav, .flac, .mp4, .mov, .webm).
---

# stt — Speech-to-Text Pipeline

Full pipeline: audio/video → preprocessed → transcribed → diarized → speaker-attributed.

Optimized for Apple Silicon (M1/M2/M3/M4/M5). Uses whisper.cpp + Core ML for transcription
(best speed/quality on Mac), pyannote.audio locally for diarization, with a Resemblyzer
fallback if pyannote weights aren't available.

## Pipeline

```
input (audio/video)
    │
    ▼
[ffmpeg] extract audio if video, then:
  highpass 80Hz → arnndn RNN denoise → compressor → loudnorm -16 LUFS → 16kHz mono WAV
    │
    ▼
[whisper.cpp + Core ML, large-v3-turbo]
  GPU (Metal) + Apple Neural Engine, temperature fallback enabled,
  prompt-biased with user-provided proper nouns
    │  → segments with timestamps
    ▼
[diarization]  ← tries in this order:
  1. pyannote.audio 3.1 (if HF_TOKEN set — best)
  2. pyannote.audio with non-gated community model
  3. Resemblyzer + Silero VAD + AgglomerativeClustering (offline fallback)
    │  → speaker turn timeline
    ▼
[merge]  per-segment max-overlap speaker assignment
    │
    ▼
[LLM speaker naming]  Claude reads the attributed transcript, infers speaker
  names from conversational context (e.g. who explains what business, who
  asks questions, who refers to themselves by name), then asks user to confirm
    │
    ▼
outputs in $cwd/<basename>_transcript/:
  - <name>.clean.wav      preprocessed audio (16kHz mono)
  - <name>.transcript.txt  timestamped, generic SPEAKER_NN labels
  - <name>.attributed.txt  timestamped, human names (after confirmation)
  - <name>.<NAME>.txt      per-speaker chronological text
  - <name>.json            structured: language, segments[], speakers[]
  - <name>.srt             subtitles
```

## When invoked

The user invokes this skill via `/stt <path>` or by asking for transcription.
Claude executes these steps:

### Step 1 — Validate input and check tools

Run:
```bash
bash ~/.claude/skills/stt/check_setup.sh
```

This script verifies whisper.cpp build, ggml-large-v3-turbo.bin model, Core ML
encoder, ffmpeg, Python deps (pyannote.audio, resemblyzer, silero-vad). If
anything's missing, it prints exactly what's needed; do NOT attempt to silently
fix more than the script auto-installs.

### Step 2 — Ask user for proper nouns / domain context

Use AskUserQuestion with a single open-ended prompt:

> "Any proper nouns, names, or domain jargon to bias the transcription? E.g. names of speakers, companies, products, technical terms. (Skip to use no biasing.)"

This becomes the `--prompt` for whisper.cpp. If user skips, proceed with a generic conversation prompt.

### Step 3 — Run the pipeline

```bash
python3 ~/.claude/skills/stt/stt.py \
  --input "<absolute path to audio/video>" \
  --output-dir "$(pwd)" \
  --prompt "<user-provided proper nouns>"
```

This script:
- Extracts audio if video input (mp4/mov/webm/mkv)
- Preprocesses with ffmpeg
- Calls whisper.cpp via shell
- Runs diarization (auto-detects best available method)
- Merges speaker turns with whisper segments
- Writes SPEAKER_00/01 labeled transcript + JSON

Run it in the background (long audio takes minutes); use Monitor to watch.

### Step 4 — Auto-attribute speaker names via LLM

After stt.py finishes, the attributed transcript has generic labels. Spawn an
Agent (general-purpose) to read the transcript and infer speaker identities
from conversational context. The agent should:

1. Read `<name>.transcript.txt` in chunks (use offset/limit if >25k tokens)
2. Look for self-introductions, role indicators, who explains what
3. Propose names for SPEAKER_00, SPEAKER_01, etc., with confidence + 2-line justification
4. Return: `{"SPEAKER_00": "Alice (CEO of AcmeCo, explains the business)", "SPEAKER_01": "Bob (asks questions, mentions his own company)"}`

If the agent has low confidence (multiple speakers indistinguishable, or
audio is monologue), it returns `null` for that speaker.

### Step 5 — Confirm with the user

Use AskUserQuestion to confirm/correct the inferred names. Show the agent's
proposal with 1-line justification per speaker. User can:
- Accept inferred names
- Provide corrections
- Keep generic labels (SPEAKER_00, SPEAKER_01)

### Step 6 — Apply final labels and emit outputs

Run:
```bash
python3 ~/.claude/skills/stt/relabel.py \
  --transcript "<name>.transcript.txt" \
  --json "<name>.json" \
  --mapping '{"SPEAKER_00":"Alice","SPEAKER_01":"Bob"}'
```

This writes:
- `<name>.attributed.txt` (timestamps + human names + merged consecutive turns)
- `<name>.<NAME>.txt` per speaker (no timestamps)
- Updates `<name>.json` to include human names

### Step 7 — Report

One-line summary: file paths + segment count + speaker distribution + total
runtime. Use markdown links for files.

## Notes for Claude

- **Default model is large-v3-turbo** via whisper.cpp + Core ML. Don't downgrade
  to medium/tiny without an explicit reason — the quality difference is large.
- **Never pass `--no-fallback`** to whisper-cli — it causes infinite loops on
  silent intros (verified May 2026 by Ken: produced 3215 lines of "Clip.").
- **Always preprocess audio** — adds ~10 sec but consistently improves accuracy.
- **Strip residual silent-intro hallucinations** ("Thank you for watching",
  "*Police*", "Subtitles by..."). Heuristic: any single-segment phrase under
  10 chars in the first 30 sec with no surrounding context is suspect.
- **Speaker diarization clustering is failure-prone.** If pyannote isn't
  available and Resemblyzer produces a wildly skewed distribution (e.g. 95%+
  one speaker for a 2-speaker conversation), fall back to LLM agent-based
  attribution using the transcript content alone.
- **Long audio**: Anything > 30 min, run whisper.cpp in background and use
  Monitor for completion. Do not block on it.

## Args (optional flags via /stt)

`/stt <file> [options]`

- `--no-preprocess`: skip ffmpeg preprocessing (use raw audio)
- `--model {turbo,large-v3,medium,small,base,tiny}`: override default turbo
- `--language CODE`: ISO 639-1 (default: en)
- `--num-speakers N`: force speaker count (default: auto)
- `--no-diarize`: skip diarization, just transcribe
- `--skip-attribution`: keep generic SPEAKER_NN labels, no LLM naming step

## Files in this skill

- `SKILL.md` (this file)
- `stt.py` — main pipeline
- `diarize.py` — diarization with pyannote/Resemblyzer fallback
- `relabel.py` — apply human name mappings to transcript outputs
- `check_setup.sh` — verify dependencies + auto-install where safe
- `requirements.txt` — pip dependencies

## Setup (first run only)

The first invocation triggers `check_setup.sh` which will:
1. Verify ffmpeg is on PATH (`brew install ffmpeg` if missing)
2. Clone + build whisper.cpp at `~/whisper.cpp` if not present
3. Download `ggml-large-v3-turbo.bin` + generate Core ML encoder
4. Install Python deps: `faster-whisper, pyannote.audio, resemblyzer, silero-vad, torch, torchaudio, scikit-learn`
5. Download RNN denoise model to `~/.cache/rnnoise-models/sh.rnnn`

For optimal pyannote diarization, the user can set `HF_TOKEN` env var
(accept terms at hf.co/pyannote/speaker-diarization-3.1 first). Otherwise
falls back gracefully.
