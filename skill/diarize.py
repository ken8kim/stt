#!/usr/bin/env python3
"""Speaker diarization with graceful fallback.

Tries in order:
  1. pyannote.audio 3.1 (if HF_TOKEN set) — best
  2. pyannote.audio with non-gated community pipeline — if available
  3. Resemblyzer + Silero VAD + AgglomerativeClustering(ward) — pure offline

Output (stdout): JSON {"method": str, "turns": [{"start", "end", "speaker"}, ...]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def try_pyannote_with_token(audio: Path, num_speakers: int | None) -> list[dict] | None:
    """Run pyannote 3.1 if HF_TOKEN is set.

    On in-person meeting recordings (single mic, both speakers similar level),
    pyannote with hard `num_speakers=2` often collapses to ~95/5 — it can't
    separate the two voices and dumps backchannel/silence into the minority
    cluster. Workaround: use `min_speakers=2, max_speakers=N+2`, let pyannote
    find 3-4 clusters, then merge any cluster with <5% of total duration into
    the nearest large cluster.
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return None
    try:
        import torch
        from pyannote.audio import Pipeline
        print(f"  trying pyannote/speaker-diarization-3.1 (HF token detected)...", file=sys.stderr)
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=token
        )
        # Pick best device: CUDA covers NVIDIA + PyTorch ROCm (same API surface).
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            print(f"  → using CUDA/ROCm GPU", file=sys.stderr)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
            print(f"  → using MPS (Apple) GPU", file=sys.stderr)
        else:
            print(f"  → CPU (no GPU acceleration available for PyTorch)", file=sys.stderr)
        if num_speakers:
            kwargs = {"min_speakers": num_speakers, "max_speakers": num_speakers + 2}
        else:
            kwargs = {"min_speakers": 2, "max_speakers": 6}
        result = pipeline(str(audio), **kwargs)
        # 4.x returns DiarizeOutput; 3.x returns Annotation directly.
        annotation = result.speaker_diarization if hasattr(result, "speaker_diarization") else result
        turns = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})

        # Post-process: merge tiny clusters (<5% total duration) into nearest main cluster.
        from collections import Counter
        durs: dict[str, float] = {}
        for t in turns:
            durs[t["speaker"]] = durs.get(t["speaker"], 0) + (t["end"] - t["start"])
        total = sum(durs.values())
        big = {spk for spk, d in durs.items() if d / total >= 0.05}
        if num_speakers:
            # Keep only the N largest clusters
            big = set(sorted(durs, key=durs.get, reverse=True)[:num_speakers])
        if len(big) < len(durs):
            small = set(durs) - big
            print(
                f"  merging {len(small)} tiny clusters into nearest main speaker: "
                f"{sorted(small)}",
                file=sys.stderr,
            )
            big_turns_sorted = sorted([t for t in turns if t["speaker"] in big],
                                       key=lambda t: t["start"])
            def nearest_big(start: float, end: float) -> str:
                mid = (start + end) / 2
                # nearest in time
                best, best_dist = None, float("inf")
                for bt in big_turns_sorted:
                    if bt["end"] < start:
                        d = start - bt["end"]
                    elif bt["start"] > end:
                        d = bt["start"] - end
                    else:
                        d = 0  # overlap
                    if d < best_dist:
                        best, best_dist = bt["speaker"], d
                return best if best else sorted(big)[0]
            for t in turns:
                if t["speaker"] in small:
                    t["speaker"] = nearest_big(t["start"], t["end"])

        return turns
    except Exception as e:
        print(f"  pyannote 3.1 failed: {e}", file=sys.stderr)
        return None


def try_pyannote_community(audio: Path, num_speakers: int | None) -> list[dict] | None:
    """Try non-gated pyannote community pipeline if it exists locally cached."""
    try:
        import torch
        from pyannote.audio import Pipeline
        print(f"  trying pyannote/speaker-diarization-community-1...", file=sys.stderr)
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
        if pipeline is None:
            return None
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pipeline.to(torch.device("mps"))
        kwargs = {"num_speakers": num_speakers} if num_speakers else {}
        diar = pipeline(str(audio), **kwargs)
        turns = []
        for turn, _, speaker in diar.itertracks(yield_label=True):
            turns.append({"start": float(turn.start), "end": float(turn.end), "speaker": speaker})
        return turns
    except Exception as e:
        print(f"  community pyannote unavailable: {e}", file=sys.stderr)
        return None


def diarize_resemblyzer(audio: Path, num_speakers: int | None) -> list[dict]:
    """Pure-offline: Silero VAD → Resemblyzer embeddings → AgglomerativeClustering(ward).

    Designed to avoid the failure mode of SpeechBrain ECAPA + average linkage,
    which collapsed everything into one cluster. Uses:
      - Silero VAD: clean speech boundaries
      - Resemblyzer: 256-d voice fingerprints
      - ward linkage: forces balanced clusters
      - L2-normalize embeddings before clustering for cosine-equivalent geometry
    """
    print(f"  using Resemblyzer fallback (offline)...", file=sys.stderr)
    import numpy as np
    import torch
    import torchaudio
    from resemblyzer import VoiceEncoder, preprocess_wav
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    # 1) Silero VAD — bigger chunks for more reliable speaker fingerprints
    print(f"    silero VAD...", file=sys.stderr)
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    wav_tensor, sr = torchaudio.load(str(audio))
    if wav_tensor.shape[0] > 1:
        wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, 16000)
        sr = 16000
    speech_ts = get_speech_timestamps(
        wav_tensor.squeeze(), vad_model, sampling_rate=sr,
        min_speech_duration_ms=1000,  # ≥1s gives Resemblyzer enough signal
        max_speech_duration_s=10,
        min_silence_duration_ms=400,
    )
    print(f"    {len(speech_ts)} VAD speech chunks", file=sys.stderr)
    if not speech_ts:
        return []

    # 2) Resemblyzer embeddings for each chunk
    print(f"    resemblyzer embeddings...", file=sys.stderr)
    encoder = VoiceEncoder(verbose=False)
    wav_np = wav_tensor.squeeze().numpy()
    embeddings = []
    valid_ts = []
    for ts in speech_ts:
        chunk = wav_np[ts["start"]:ts["end"]]
        chunk = preprocess_wav(chunk, source_sr=sr)
        if len(chunk) < sr * 0.8:  # skip chunks shorter than 0.8s post-preprocessing
            continue
        emb = encoder.embed_utterance(chunk)
        embeddings.append(emb)
        valid_ts.append(ts)
    if not embeddings:
        return []
    X = np.vstack(embeddings)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    # 3) Cluster — try multiple linkages, pick the one whose silhouette is highest
    print(f"    clustering {len(X)} embeddings...", file=sys.stderr)
    def cluster_with(linkage: str, k: int):
        if linkage == "ward":
            return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        return AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage=linkage
        ).fit_predict(X)

    from collections import Counter
    candidates = []
    target_ks = [num_speakers] if (num_speakers and num_speakers >= 2) else range(2, min(7, len(X)))
    for k in target_ks:
        for linkage in ("ward", "complete", "average"):
            try:
                lab = cluster_with(linkage, k)
            except Exception as e:
                print(f"    skip k={k} {linkage}: {e}", file=sys.stderr)
                continue
            if len(set(lab)) < 2:
                continue
            try:
                s = silhouette_score(X, lab, metric="cosine")
            except Exception:
                continue
            counts = Counter(lab)
            max_frac = max(counts.values()) / sum(counts.values())
            # Heavy penalty for imbalance — anything over 80% gets hit hard.
            # Imbalance > 95% is almost certainly an outlier-detection failure (cluster of 1-5 samples).
            imbalance_penalty = max(0.0, max_frac - 0.7) * 2.0
            score = s - imbalance_penalty
            candidates.append((score, s, k, linkage, lab, max_frac))
            print(
                f"    k={k} {linkage}: sil={s:.3f}, max_frac={max_frac:.2%}, score={score:.3f}",
                file=sys.stderr,
            )
    if not candidates:
        labels = np.zeros(len(X), dtype=int)
        print(f"    ! no valid clustering found, using single speaker", file=sys.stderr)
    else:
        candidates.sort(key=lambda c: c[0], reverse=True)
        score, sil, k, linkage, labels, max_frac = candidates[0]
        print(
            f"    → chose k={k}, linkage={linkage}, silhouette={sil:.3f}, "
            f"largest cluster={max_frac:.0%}",
            file=sys.stderr,
        )

    # 4) Temporal smoothing — a single short chunk surrounded by the other speaker
    # is almost certainly mis-clustered. Smooth with a 3-window majority.
    if len(labels) >= 3:
        smoothed = labels.copy()
        for i in range(1, len(labels) - 1):
            if labels[i - 1] == labels[i + 1] and labels[i] != labels[i - 1]:
                smoothed[i] = labels[i - 1]
        labels = smoothed

    # 5) Build turns
    turns = []
    for ts, lab in zip(valid_ts, labels):
        turns.append({
            "start": ts["start"] / sr,
            "end": ts["end"] / sr,
            "speaker": f"SPEAKER_{int(lab):02d}",
        })

    # Sanity check
    from collections import Counter
    c = Counter(t["speaker"] for t in turns)
    total = sum(c.values())
    dom = c.most_common(1)[0]
    if dom[1] / total > 0.9:
        print(
            f"  ! warning: cluster imbalance ({dom[0]}={dom[1]}/{total} = {dom[1]/total:.0%}). "
            f"Likely diarization failure; consider LLM-based attribution.",
            file=sys.stderr,
        )
    return turns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--output", required=True, type=Path,
                    help="Path to write JSON result. (stdout is too noisy with library messages.)")
    args = ap.parse_args()

    audio = args.audio.resolve()
    if not audio.exists():
        sys.exit(f"Audio not found: {audio}")

    # Try pyannote with HF token
    turns = try_pyannote_with_token(audio, args.num_speakers)
    method = "pyannote-3.1"

    # Try community model
    if turns is None:
        turns = try_pyannote_community(audio, args.num_speakers)
        method = "pyannote-community"

    # Fall back to Resemblyzer
    if turns is None:
        turns = diarize_resemblyzer(audio, args.num_speakers)
        method = "resemblyzer"

    args.output.write_text(json.dumps({"method": method, "turns": turns}))
    print(f"  diarization method: {method}, turns: {len(turns)}", file=sys.stderr)


if __name__ == "__main__":
    main()
