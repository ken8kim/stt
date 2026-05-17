# Changelog

## v0.1.0 — Initial release

- whisper.cpp + Core ML + ggml-large-v3-turbo transcription pipeline
- ffmpeg preprocessing: highpass + arnndn denoise + compressor + loudnorm
- Silero VAD trim of silent intro/outro (eliminates "Thank you for watching" hallucinations)
- Hallucination cleanup: annotation markers, repetition loops, filler-word runs
- pyannote.audio 3.1 diarization with min/max_speakers heuristic + tiny-cluster merge
- Resemblyzer fallback when no HF_TOKEN (ward linkage + imbalance penalty)
- Speaker assignment with nearest-fallback for zero-overlap segments
- Sentence-boundary breaks for long merged turns
- Claude Code skill (SKILL.md) + standalone CLI wrappers
- Outputs: transcript.txt, .json, .srt, per-speaker files, attributed.txt
