# stt

Fast, local speech-to-text + speaker diarization on Apple Silicon. Optimized for long-form conversation audio (meetings, interviews, podcasts).

- **Transcription**: [whisper.cpp](https://github.com/ggerganov/whisper.cpp) + Core ML + `large-v3-turbo` (GPU + Apple Neural Engine)
- **Diarization**: [pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) on MPS GPU (with auto-fallback to Resemblyzer if no HF token)
- **Preprocessing**: ffmpeg with RNN denoise + loudness normalization
- **Hallucination cleanup**: silent-intro trim, repetition-loop detection, filler-word run collapse

Tested on M5 Pro (48GB). A 51-minute meeting transcribes + diarizes in **~2 minutes** end-to-end.

## Usage

### As a Claude Code skill

Drop the `skill/` directory into `~/.claude/skills/stt/`, then invoke `/stt <file>` in Claude Code. Claude will run setup checks, prompt for proper-noun biasing, run the pipeline in the background, then propose speaker names via an LLM agent and ask you to confirm.

```bash
git clone https://github.com/<you>/stt.git
mkdir -p ~/.claude/skills
ln -s "$(pwd)/stt/skill" ~/.claude/skills/stt
```

### As a standalone CLI

```bash
git clone https://github.com/<you>/stt.git
cd stt
bash skill/check_setup.sh                      # one-time setup
export PATH="$(pwd)/cli:$PATH"                 # or symlink into /usr/local/bin

stt audio.m4a                                  # default
stt video.mp4 --speakers 2 --prompt "Alice, Bob, AcmeCo, term1, term2"
stt podcast.mp3 --speakers 3 -o ./out
stt monologue.wav --no-diarize                 # transcribe only

# After the pipeline outputs SPEAKER_00/01/.../, map them to real names:
stt-relabel out/podcast.json --mapping '{"SPEAKER_00":"Alice","SPEAKER_01":"Bob"}'
```

## Outputs

For input `audio.m4a`, writes to `./audio_transcript/`:

| File | Contents |
|---|---|
| `audio.clean.wav` | Preprocessed 16kHz mono WAV (post-denoise + loudnorm) |
| `audio.trimmed.wav` | Same but with silent intro/outro trimmed (Whisper input) |
| `audio.txt` | Raw whisper.cpp output, no speaker labels |
| `audio.transcript.txt` | Timestamped, `SPEAKER_NN` labels, consecutive turns merged |
| `audio.attributed.txt` | (after `stt-relabel`) — same but with human names |
| `audio.<NAME>.txt` | Per-speaker text, chronological |
| `audio.json` | Structured: language, segments, merged turns, mapping, diarization_method |
| `audio.srt` | Subtitles (SRT format, speaker prefix in each cue) |

## Installation

Requirements:

- macOS on Apple Silicon (M1+) — Linux/Windows technically possible but untested
- Xcode (not just Command Line Tools) — needed for `coremlc` to compile Core ML models
- Homebrew with `ffmpeg` and `cmake`
- Python 3.10+ with pip

```bash
brew install ffmpeg cmake
bash skill/check_setup.sh
```

`check_setup.sh` does the heavy lifting:

1. Clones + builds `whisper.cpp` at `~/whisper.cpp` (with `-DWHISPER_COREML=1`)
2. Downloads `ggml-large-v3-turbo.bin` (~1.6 GB)
3. Converts the Whisper encoder to Core ML and compiles with `coremlc`
4. Downloads the RNN denoise model
5. `pip install`s Python deps (torch, torchaudio, scikit-learn, resemblyzer, pyannote.audio)

First run takes 10–30 min for downloads + Core ML compilation. Subsequent runs are seconds.

## Optional: pyannote 3.1 for best diarization

The Resemblyzer fallback works without authentication but produces less accurate speaker boundaries on in-person meetings. For best quality:

1. Accept terms on each of:
   - https://hf.co/pyannote/speaker-diarization-3.1
   - https://hf.co/pyannote/segmentation-3.0
   - https://hf.co/pyannote/speaker-diarization-community-1 *(pyannote-audio 4.x dependency)*
2. Create a read-only token: https://hf.co/settings/tokens
3. Export it:
   ```bash
   echo 'export HF_TOKEN="hf_..."' >> ~/.zshrc
   source ~/.zshrc
   ```

After that, the skill detects `HF_TOKEN` and uses pyannote 3.1 (138s for 51-min audio on M5 Pro GPU). Weights are cached locally; no internet needed for subsequent runs.

## Why this stack (vs. alternatives)

Benchmarked on a 51-minute meeting recording (May 2026, M5 Pro):

| Pipeline | Runtime | Quality notes |
|---|---|---|
| faster-whisper `medium` / CPU | 75 min | Slow; ECAPA-TDNN clustering collapsed all speakers into one |
| mlx-whisper `large-v3` / GPU | 2:12 | Mangled proper nouns ("Clip My Walls TV"); no diarization |
| **whisper.cpp `large-v3-turbo` + Core ML** | **1:09** | Best proper-noun fidelity, Metal + Neural Engine acceleration |

Key engineering decisions:

- **`large-v3-turbo`** over plain `large-v3`: ~8× faster, ~95% of the quality on English.
- **Core ML encoder + Metal**: uses both GPU and the Apple Neural Engine; meaningfully faster than mlx-only.
- **Never `--no-fallback`**: temperature fallback prevents infinite loops on silent intros (this flag is a footgun that produced 3215 lines of "Clip." in one test).
- **VAD-trim before transcription**: eliminates "*Music*", "Thank you for watching", "*Clap*" hallucinations at the source.
- **pyannote with `min_speakers/max_speakers`** (not `num_speakers`): forcing exactly 2 speakers on in-person meeting audio collapses to ~95/5; letting pyannote find 3-4 clusters then merging tiny ones recovers the real split (validated 65/35 vs 95/5).

## Architecture

```
input (audio/video)
   │
   ▼
[ffmpeg]  extract audio (if video) → highpass 80Hz → arnndn denoise →
          compressor → loudnorm -16 LUFS → 16kHz mono WAV
   │
   ▼
[silero-vad]  trim silent intro/outro (eliminates Whisper hallucinations)
   │
   ▼
[whisper.cpp + Core ML, large-v3-turbo]
          temperature fallback enabled, beam-size 5, best-of 5
   │  → segments
   ▼
[strip_hallucinations]  drop annotations + repetition loops + filler runs
   │
   ▼
[diarize]  pyannote.audio 3.1 (HF_TOKEN) → Resemblyzer fallback
          min/max_speakers heuristic, tiny-cluster merge
   │  → speaker turns
   ▼
[assign_speakers]  max-overlap per segment, nearest-fallback
   │
   ▼
[merge_consecutive]  break long monologue blocks at sentence boundaries
   │
   ▼
outputs (transcript.txt, json, srt, per-speaker, attributed)
```

## File layout

```
stt/
├── README.md               this file
├── LICENSE                 MIT
├── skill/                  the actual pipeline + Claude Code SKILL.md
│   ├── SKILL.md           Claude Code orchestration recipe
│   ├── stt.py             main pipeline
│   ├── diarize.py         pyannote + Resemblyzer fallback
│   ├── relabel.py         apply human speaker names
│   ├── vad_trim.py        silero VAD trim
│   ├── check_setup.sh     install/verify all deps
│   └── requirements.txt   Python deps
├── cli/                    standalone CLI wrappers
│   ├── stt
│   └── stt-relabel
└── examples/               sample audio + expected output
```

## Environment variables

| Var | Purpose |
|---|---|
| `HF_TOKEN` | Unlock pyannote 3.1. Required for highest-quality diarization. |
| `STT_ROOT` | Override the skill scripts directory (the CLI uses this to find `stt.py`). |
| `HF_HUB_OFFLINE=1` | After first run, force everything offline. No internet calls. |

## Contributing

Issues + PRs welcome. The skill itself is small (~1k LOC Python) and aims to stay that way.

Useful experiments if you want to contribute:

- Linux/Windows portability (currently macOS-only due to Core ML)
- WhisperX integration for word-level timestamps
- VAD-based speaker pre-segmentation (improves pyannote on tough audio)
- A "test fixture" of 5–10 short public-domain audio clips + expected outputs

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — Georgi Gerganov
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — Hervé Bredin
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer) — Resemble AI
- [silero-vad](https://github.com/snakers4/silero-vad)
- [rnnoise-models](https://github.com/GregorR/rnnoise-models) — Gregor Richards
