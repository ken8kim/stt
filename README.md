# stt

Fast, local speech-to-text + speaker diarization on Apple Silicon.

```
51-minute meeting → fully transcribed + speaker-attributed in ~2 minutes
```

No cloud uploads. No API bills. No "your audio has been queued." Just a `.m4a` going in and a clean attributed transcript coming out.

---

## Why this exists

I had a 51-minute meeting recording I wanted transcribed and split by speaker. Should be a solved problem in 2026, right?

It wasn't.

The first thing I tried — `faster-whisper medium` on CPU + a speaker-embedding clustering library — ran for **75 minutes** and produced a "diarized" transcript where one speaker got 1292 segments and the other got 3. Useless.

I switched to `mlx-whisper large-v3` on the Mac GPU. 2 minutes 12 seconds — much better. But it mangled domain-specific proper nouns into wrong words ("Sims ESPN," company names hallucinated into other phrases). Speed without accuracy is also useless.

`whisper.cpp` with Core ML and the new `large-v3-turbo` model finally hit the sweet spot — **69 seconds, proper nouns intact** — but I learned the hard way that passing `--no-fallback` causes the model to get stuck on silent intros and produce 3,215 consecutive lines of the word "Clip."

Then there's diarization. `pyannote.audio 3.1` is the gold standard, but if you pass `num_speakers=2` on a typical in-person meeting recording, it dumps 95% of the audio into one cluster (it treats both voices through a single mic as a monologue). The fix is `min_speakers=2, max_speakers=4` and post-hoc merging of tiny clusters. Nobody documents this.

Also: `pyannote.audio 4.x` quietly added a third gated HuggingFace model that the docs don't mention, so even with an HF token you get a 401 unless you've accepted terms on all three model pages.

I'd lost an evening figuring this out. So I packaged it.

## What you get

A pipeline that does the obvious thing:

```bash
stt meeting.m4a --speakers 2 --prompt "Alice, Bob, AcmeCo, dressage"
```

…and 2 minutes later you have a timestamped, speaker-attributed transcript plus a per-speaker text file, plus an SRT, plus structured JSON.

Under the hood:

1. **Preprocess** — ffmpeg with high-pass, RNN-based denoise, dynamic range compression, EBU R128 loudness normalization. Conference-call audio gets a lot of help here.
2. **Trim** — Silero VAD removes silent intro/outro. Eliminates Whisper's "Thank you for watching," "*Music*," "*Clap*" hallucinations at the source.
3. **Transcribe** — `whisper.cpp` with the Core ML encoder running on both the Apple Neural Engine and Metal GPU. `large-v3-turbo` is ~8× faster than `large-v3` and ~95% of the quality on English. Temperature fallback enabled, beam-size 5, best-of 5, prompt biasing for proper nouns.
4. **Clean** — cross-segment dedup catches Whisper's repetition loops ("No problem. No problem. No problem.") that the temperature fallback didn't fully escape. In-segment filler-word run collapse handles "Yeah. Yeah, yeah. Yeah. Yeah" patterns. Word-bigram dedup catches "travel to travel to."
5. **Diarize** — `pyannote.audio 3.1` on MPS GPU if you have an HF token, otherwise `Resemblyzer` with `ward` linkage and an imbalance penalty as a no-auth fallback.
6. **Assign** — each transcript segment gets the speaker with maximum temporal overlap; segments with zero overlap fall back to the nearest-in-time speaker.
7. **Merge + break** — consecutive same-speaker segments get merged into paragraphs, but no single paragraph exceeds 45 seconds (breaks at sentence boundaries).
8. **Output** — clean.wav, transcript.txt, per-speaker .txt, SRT, structured JSON.

## Benchmarks

Same 51-minute meeting recording, M5 Pro 48GB:

| Pipeline | Runtime | Diarization | Verdict |
|---|---|---|---|
| `faster-whisper medium` / CPU + ECAPA agglomerative | **75 min** | 99.8% one speaker | Slow and broken |
| `mlx-whisper large-v3` / GPU + LLM attribution | 2:12 | n/a (LLM) | Fast, mangles proper nouns |
| `whisper.cpp large-v3-turbo` + Core ML, no diarize | 1:09 | n/a | Fastest transcription |
| this repo, Resemblyzer fallback | 2:15 | 75/25 split | No auth needed, decent |
| **this repo, pyannote 3.1** | **3:30** | **65/35 split, 36/36 turns** | Best quality |

## Install

```bash
git clone https://github.com/ken8kim/stt.git
cd stt
brew install ffmpeg cmake
bash skill/check_setup.sh    # builds whisper.cpp, downloads model, compiles Core ML encoder
```

First run takes 10–30 minutes for downloads and Core ML compilation. After that it's all local — disconnect your network and it still works.

Requirements:
- macOS on Apple Silicon (M1 or later)
- Xcode (full install, not just Command Line Tools — `coremlc` lives there)
- Python 3.10+

## Usage

**Standalone CLI:**

```bash
export PATH="$(pwd)/cli:$PATH"

stt audio.m4a                                  # auto-detect speakers
stt video.mp4 --speakers 2 --prompt "..."      # ffmpeg extracts audio first
stt podcast.mp3 --speakers 3 -o ./out
stt monologue.wav --no-diarize                 # transcribe only

# After SPEAKER_NN labels, swap in real names:
stt-relabel out/audio.json --mapping '{"SPEAKER_00":"Alice","SPEAKER_01":"Bob"}'
```

**As a Claude Code skill:**

```bash
mkdir -p ~/.claude/skills
ln -s "$(pwd)/skill" ~/.claude/skills/stt
```

Then invoke `/stt <file>` in any Claude Code session. Claude orchestrates the pipeline, asks you for prompt-biasing terms, runs it in the background, infers speaker names via an LLM agent from the conversation context, and prompts you to confirm before writing the final labeled outputs.

## Outputs

For input `audio.m4a`, writes to `./audio_transcript/`:

| File | Contents |
|---|---|
| `audio.clean.wav` | Preprocessed 16kHz mono WAV |
| `audio.transcript.txt` | Timestamped, `SPEAKER_NN` labels |
| `audio.attributed.txt` | Same but with human names (after `stt-relabel`) |
| `audio.<NAME>.txt` | Per-speaker chronological text |
| `audio.json` | Structured: segments, merged turns, mapping, diarization method |
| `audio.srt` | Subtitles |

## Optional: HF token for best-in-class diarization

The Resemblyzer fallback works without authentication and produces ~75/25 talk-time splits on typical meeting audio. For the cleaner ~65/35 splits and tighter turn boundaries that `pyannote 3.1` produces:

1. Accept terms on **all three** HuggingFace pages:
   - https://hf.co/pyannote/speaker-diarization-3.1
   - https://hf.co/pyannote/segmentation-3.0
   - https://hf.co/pyannote/speaker-diarization-community-1 *(undocumented pyannote-audio 4.x dependency)*
2. Create a read-only token: https://hf.co/settings/tokens
3. Export it:
   ```bash
   echo 'export HF_TOKEN="hf_..."' >> ~/.zshrc && source ~/.zshrc
   ```

The skill detects `HF_TOKEN` and switches to pyannote automatically. Weights are cached after first run — no internet calls after that. Set `HF_HUB_OFFLINE=1` to make this explicit.

## Things that surprised me (so they don't surprise you)

- **Never pass `--no-fallback` to `whisper-cli`.** It disables temperature fallback. On silent intros, the model gets stuck repeating one word forever. Verified: 3,215 lines of "Clip."
- **Pyannote 3.1 with `num_speakers=2` is broken for in-person meeting audio.** Both speakers picked up by the same mic at similar levels confuse the clustering — it labels 95%+ of the audio as one speaker. Use `min_speakers=2, max_speakers=4` and merge tiny clusters in post.
- **Pyannote-audio 4.x silently added a third gated model dependency.** Even with terms accepted on `speaker-diarization-3.1` and `segmentation-3.0`, you'll get 401 on `speaker-diarization-community-1`. The error message doesn't tell you this clearly.
- **`coremlc` is in Xcode, not Command Line Tools.** If you only have the CLT, the Core ML conversion step fails with "utility not found." Either install Xcode or use the GGML model alone (loses Neural Engine acceleration).
- **mlx-whisper without `condition_on_previous_text=False` is a hallucination factory.** With it on, a bad output becomes the prompt for the next chunk and the model spirals. Turn it off.

## Architecture

```
input (audio/video)
   │
   ▼
[ffmpeg]  extract audio → highpass 80Hz → arnndn denoise →
          compressor → loudnorm -16 LUFS → 16kHz mono WAV
   │
   ▼
[silero-vad]  trim silent intro/outro
   │
   ▼
[whisper.cpp + Core ML]  large-v3-turbo, beam 5, best-of 5, temperature fallback
   │
   ▼
[strip_hallucinations]  annotation markers + repetition loops + filler-word runs
   │
   ▼
[diarize]  pyannote 3.1 (HF_TOKEN) OR Resemblyzer (no auth)
   │
   ▼
[assign_speakers]  max-overlap, nearest-in-time fallback
   │
   ▼
[merge_consecutive]  same speaker → paragraph, break at sentence if turn > 45s
   │
   ▼
outputs (txt, json, srt, per-speaker)
```

## Layout

```
stt/
├── README.md
├── LICENSE                 (MIT)
├── CHANGELOG.md
├── skill/                  pipeline + Claude Code SKILL.md
│   ├── SKILL.md
│   ├── stt.py             (main)
│   ├── diarize.py         (pyannote + Resemblyzer fallback)
│   ├── relabel.py         (apply human names)
│   ├── vad_trim.py        (silero VAD)
│   ├── check_setup.sh     (install everything)
│   └── requirements.txt
├── cli/                    standalone wrappers
│   ├── stt
│   └── stt-relabel
└── examples/
    ├── sample.wav          JFK inaugural address (11s, public domain)
    └── expected_output/
```

## Environment variables

| Var | Purpose |
|---|---|
| `HF_TOKEN` | Unlock pyannote 3.1 |
| `STT_ROOT` | Override skill scripts directory (CLI uses this to find `stt.py`) |
| `HF_HUB_OFFLINE=1` | After first run, force everything offline |

## Contributing

Issues + PRs welcome. The whole thing is ~1,000 lines of Python.

Useful directions:

- Linux/Windows portability (currently macOS-only due to Core ML)
- WhisperX integration for word-level timestamps
- VAD-based speaker pre-segmentation (improves pyannote on tough audio)
- A test fixture of 5–10 short public-domain clips with expected outputs

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgements

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) — Georgi Gerganov
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — Hervé Bredin
- [Resemblyzer](https://github.com/resemble-ai/Resemblyzer)
- [silero-vad](https://github.com/snakers4/silero-vad)
- [rnnoise-models](https://github.com/GregorR/rnnoise-models)
- The Anthropic Claude Code team for the skill primitive that made this glue-easy
