#!/usr/bin/env python3
"""Find first/last speech onsets via Silero VAD, write a trimmed WAV with
silent intro/outro removed. Eliminates the silent-audio hallucinations
that Whisper produces ("*Clap*", "Thank you for watching", etc.).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--pad-sec", type=float, default=0.5,
                    help="Padding around detected speech (default 0.5s)")
    ap.add_argument("--metadata", type=Path,
                    help="Optional path to write trim offsets JSON")
    args = ap.parse_args()

    import torch
    import torchaudio

    wav, sr = torchaudio.load(str(args.audio))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    ts = get_speech_timestamps(
        wav.squeeze(), model, sampling_rate=sr,
        min_speech_duration_ms=250,
        min_silence_duration_ms=500,
    )
    if not ts:
        print("  ! VAD found no speech; passing through audio unchanged", file=sys.stderr)
        # Fallback: just copy
        torchaudio.save(str(args.output), wav, sr, encoding="PCM_S", bits_per_sample=16)
        if args.metadata:
            args.metadata.write_text(json.dumps({"trim_start_sec": 0.0, "trim_end_sec": 0.0}))
        return

    pad = int(args.pad_sec * sr)
    start = max(0, ts[0]["start"] - pad)
    end = min(wav.shape[1], ts[-1]["end"] + pad)
    trimmed = wav[:, start:end]

    torchaudio.save(str(args.output), trimmed, sr, encoding="PCM_S", bits_per_sample=16)

    trim_start_sec = start / sr
    trim_end_sec = (wav.shape[1] - end) / sr
    print(
        f"  trimmed {trim_start_sec:.1f}s from start, {trim_end_sec:.1f}s from end",
        file=sys.stderr,
    )
    if args.metadata:
        args.metadata.write_text(json.dumps({
            "trim_start_sec": trim_start_sec,
            "trim_end_sec": trim_end_sec,
            "original_duration_sec": wav.shape[1] / sr,
            "trimmed_duration_sec": trimmed.shape[1] / sr,
        }))


if __name__ == "__main__":
    main()
