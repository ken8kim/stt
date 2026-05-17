# Changelog

## v0.2.0 — Cross-platform support

- `platform_detect.py` — auto-detect OS (macOS/Linux/Windows) + GPU (MPS/CUDA/ROCm/Vulkan/CPU)
- Unified `check_setup.sh` (Linux + macOS + WSL) — picks cmake flags + PyTorch wheel index per platform
- New `check_setup.ps1` for native Windows (winget/choco-based installs)
- `diarize.py` now picks `cuda` > `mps` > `cpu` automatically; PyTorch ROCm shares the CUDA code path
- `stt.py` handles `whisper-cli.exe` on Windows and `%LOCALAPPDATA%` for caches
- README rewritten with multi-platform install matrix and per-platform requirements

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
