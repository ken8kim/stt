#!/usr/bin/env python3
"""stt — full speech-to-text pipeline.

Steps:
1. Extract audio from video if needed (ffmpeg)
2. Preprocess audio: highpass + arnndn denoise + compressor + loudnorm → 16kHz mono WAV
3. Transcribe with whisper.cpp + Core ML (large-v3-turbo)
4. Diarize speakers (pyannote.audio with non-gated/Resemblyzer fallback)
5. Merge whisper segments with speaker turns
6. Strip silent-intro hallucinations
7. Write outputs: <base>.clean.wav, <base>.transcript.txt, <base>.json, <base>.srt

Speaker name attribution (step 7) is handled by Claude in the conversation
loop, not by this script.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

WHISPER_CPP_DIR = Path(os.environ.get("WHISPER_CPP_DIR") or Path.home() / "whisper.cpp")
# whisper-cli on POSIX, whisper-cli.exe on Windows (sometimes under build/bin/Release/)
_WHISPER_BIN_CANDIDATES = [
    WHISPER_CPP_DIR / "build" / "bin" / "whisper-cli",
    WHISPER_CPP_DIR / "build" / "bin" / "whisper-cli.exe",
    WHISPER_CPP_DIR / "build" / "bin" / "Release" / "whisper-cli.exe",
]
WHISPER_BIN = next((p for p in _WHISPER_BIN_CANDIDATES if p.exists()), _WHISPER_BIN_CANDIDATES[0])
WHISPER_MODEL = WHISPER_CPP_DIR / "models" / "ggml-large-v3-turbo.bin"
# RNN cache: ~/.cache on POSIX, %LOCALAPPDATA% on Windows
if sys.platform == "win32":
    _RNN_BASE = Path(os.environ.get("LOCALAPPDATA", str(Path.home())))
else:
    _RNN_BASE = Path.home() / ".cache"
RNN_MODEL = _RNN_BASE / "rnnoise-models" / "sh.rnnn"
DIARIZE_SCRIPT = Path(__file__).parent / "diarize.py"
VAD_TRIM_SCRIPT = Path(__file__).parent / "vad_trim.py"

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".opus", ".aiff"}

# Patterns commonly emitted by Whisper on silent / non-speech audio
HALLUCINATION_PATTERNS = [
    r"^\s*[\*\[]?\s*(thanks?|thank you)( for watching)?\s*[\*\]]?\.?\s*$",
    r"^\s*[\*\[]\s*(music|applause|silence|laughter|silent|clap|claps|clapping|"
    r"applauding|cheering|cheers|noise|background\s+music|inaudible)\s*[\*\]]\.?\s*$",
    r"^\s*[\*\[]?\s*subtitles?(\s+by[^.\]\*]*)?\s*[\*\]]?\.?\s*$",
    r"^\s*[\*\[]\s*(police|sirens?|coughs?|coughing|sighs?|sniffles?)\s*[\*\]]\.?\s*$",
    r"^\s*[\s\W]*$",
]
HALLUCINATION_RE = re.compile("|".join(HALLUCINATION_PATTERNS), re.IGNORECASE)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess; raise on non-zero exit."""
    log(f"$ {' '.join(str(c) for c in cmd[:6])}{'...' if len(cmd) > 6 else ''}")
    return subprocess.run(cmd, check=True, **kwargs)


def extract_audio_if_video(input_path: Path, out_dir: Path) -> Path:
    """If input is a video file, extract audio to a temp wav. Otherwise return input."""
    if input_path.suffix.lower() not in VIDEO_EXTS:
        return input_path
    log(f"Video input detected — extracting audio...")
    extracted = out_dir / f"{input_path.stem}.extracted.wav"
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(extracted),
    ])
    return extracted


def preprocess_audio(input_path: Path, out_dir: Path, basename: str) -> Path:
    """ffmpeg: highpass + RNN denoise + compressor + loudnorm → 16kHz mono WAV."""
    out_path = out_dir / f"{basename}.clean.wav"
    af_chain = "highpass=f=80"
    if RNN_MODEL.exists():
        af_chain += f",arnndn=m={RNN_MODEL}"
    af_chain += ",acompressor=threshold=-22dB:ratio=3:attack=5:release=100"
    af_chain += ",loudnorm=I=-16:LRA=11:TP=-1.5"
    af_chain += ",aresample=16000"
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(input_path),
        "-af", af_chain,
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(out_path),
    ])
    log(f"  → {out_path.name}")
    return out_path


def vad_trim(audio: Path, out_dir: Path, basename: str) -> tuple[Path, float]:
    """Trim leading/trailing silence with Silero VAD. Returns (trimmed_path, leading_trim_sec)."""
    trimmed = out_dir / f"{basename}.trimmed.wav"
    meta = out_dir / f"{basename}.trim_meta.json"
    cmd = [
        sys.executable, str(VAD_TRIM_SCRIPT),
        "--audio", str(audio),
        "--output", str(trimmed),
        "--metadata", str(meta),
        "--pad-sec", "0.5",
    ]
    log("VAD-trimming silent intro/outro...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not trimmed.exists():
        log(f"  ! VAD trim failed; using untrimmed audio: {proc.stderr[-300:]}")
        return audio, 0.0
    if proc.stderr.strip():
        for line in proc.stderr.strip().split("\n")[-2:]:
            log(f"    {line}")
    leading = 0.0
    if meta.exists():
        try:
            leading = float(json.loads(meta.read_text()).get("trim_start_sec", 0.0))
        except Exception:
            leading = 0.0
    return trimmed, leading


def transcribe_whispercpp(
    audio: Path,
    out_dir: Path,
    basename: str,
    prompt: str,
    language: str = "en",
) -> Path:
    """Run whisper.cpp; emits <base>.txt, .json, .srt next to -of prefix."""
    of = out_dir / basename
    cmd = [
        str(WHISPER_BIN),
        "-m", str(WHISPER_MODEL),
        "-f", str(audio),
        "-l", language,
        "-t", "8",
        "--beam-size", "5",
        "--best-of", "5",
        "--temperature", "0",
        "--temperature-inc", "0.2",   # fallback ladder; do NOT --no-fallback
        "--entropy-thold", "2.4",
        "--logprob-thold", "-1.0",
        "--word-thold", "0.01",
        "--output-txt", "--output-json", "--output-srt",
        "-of", str(of),
    ]
    if prompt:
        cmd += ["--prompt", prompt]
    log(f"Transcribing with whisper.cpp + Core ML...")
    t0 = time.time()
    run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    log(f"  → transcription done in {time.time()-t0:.1f}s")
    json_path = out_dir / f"{basename}.json"
    if not json_path.exists():
        sys.exit(f"whisper.cpp did not produce {json_path}")
    return json_path


def strip_hallucinations(segments: list[dict]) -> list[dict]:
    """Drop segments that match known silent-audio hallucination patterns.

    Catches:
    - Annotation markers like "*Clap*", "[Music]", "*applause*"
    - Single-word/silence hallucinations like "Thank you for watching"
    - Repetition loops: if the SAME normalized text appears 3+ times in a 6-segment window,
      keep only the first occurrence (Whisper got stuck briefly).
    - Filler-word runs across segments ("Yeah.", "Yeah, yeah.", "Yeah." → collapse to one).
    """
    def norm(t: str) -> str:
        return re.sub(r"\W+", "", t.lower())

    def filler_key(t: str) -> str | None:
        """If the text is composed entirely of filler words, return the canonical filler
        ('yeah', 'okay', etc.). Otherwise None.
        """
        words = re.findall(r"[A-Za-z]+", t.lower())
        if not words:
            return None
        if all(w in _FILLER_WORDS for w in words):
            # All filler — return the most common one to group "Yeah", "Yeah yeah", "yeah" together
            return words[0]
        return None

    # Pass 1: drop annotation markers + known silence hallucinations
    pass1 = []
    for s in segments:
        text = s.get("text", "").strip()
        if not text:
            continue
        if HALLUCINATION_RE.match(text):
            log(f"  drop hallucination at {s.get('start', 0):.1f}s: {text!r}")
            continue
        pass1.append(s)

    # Pass 2: detect identical-text repetition loops (same phrase ≥3× within next 8 segments)
    cleaned = []
    skip_until = -1
    for i, s in enumerate(pass1):
        if i < skip_until:
            continue
        nt = norm(s["text"])
        if not nt:
            cleaned.append(s)
            continue
        run_end = i
        for j in range(i + 1, min(i + 10, len(pass1))):
            if norm(pass1[j]["text"]) == nt:
                run_end = j
            else:
                break
        run_len = run_end - i + 1
        if run_len >= 3:
            log(
                f"  drop repetition loop at {s.get('start', 0):.1f}s: "
                f"{s['text']!r} × {run_len}"
            )
            # Extend the kept segment's `end` to cover the absorbed segments
            s["end"] = pass1[run_end]["end"]
            cleaned.append(s)
            skip_until = run_end + 1
        else:
            cleaned.append(s)

    # Pass 3: collapse filler-only-segment runs (handles "Yeah.", "Yeah, yeah.", "Yeah." cross-segment)
    pass3 = []
    skip_until = -1
    for i, s in enumerate(cleaned):
        if i < skip_until:
            continue
        fk = filler_key(s["text"])
        if fk is None:
            pass3.append(s)
            continue
        run_end = i
        for j in range(i + 1, min(i + 12, len(cleaned))):
            if filler_key(cleaned[j]["text"]) == fk:
                run_end = j
            else:
                break
        run_len = run_end - i + 1
        if run_len >= 3:
            log(
                f"  collapse filler run at {s.get('start', 0):.1f}s: "
                f"{fk!r} × {run_len} segments"
            )
            # Keep first; extend `end` to absorbed range
            s["end"] = cleaned[run_end]["end"]
            pass3.append(s)
            skip_until = run_end + 1
        else:
            pass3.append(s)

    return pass3


def load_whisper_json(json_path: Path) -> tuple[list[dict], str]:
    """whisper.cpp emits a custom JSON format; normalize to [{start, end, text}]."""
    data = json.loads(json_path.read_text())
    language = data.get("result", {}).get("language", "en")
    raw_segs = data.get("transcription", [])
    segs = []
    for r in raw_segs:
        ts = r.get("offsets", {})
        start = ts.get("from", 0) / 1000.0
        end = ts.get("to", 0) / 1000.0
        text = (r.get("text") or "").strip()
        segs.append({"start": start, "end": end, "text": text})
    return segs, language


def diarize(audio: Path, num_speakers: int | None, out_dir: Path) -> list[tuple[float, float, str]]:
    """Call diarize.py; returns list of (start, end, speaker_label)."""
    diar_json = out_dir / "diarization.json"
    cmd = [
        sys.executable, str(DIARIZE_SCRIPT),
        "--audio", str(audio),
        "--output", str(diar_json),
    ]
    if num_speakers:
        cmd += ["--num-speakers", str(num_speakers)]
    log(f"Diarizing speakers...")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"  ! diarization failed: {proc.stderr[-500:]}")
        return []
    log(f"  → diarization done in {time.time()-t0:.1f}s")
    if proc.stderr.strip():
        for line in proc.stderr.strip().split("\n")[-3:]:
            log(f"    {line}")
    if not diar_json.exists():
        return []
    turns_data = json.loads(diar_json.read_text())
    return [(t["start"], t["end"], t["speaker"]) for t in turns_data["turns"]]


def assign_speakers(segments: list[dict], turns: list[tuple]) -> list[dict]:
    """For each whisper segment, pick speaker with max overlap from turns.
    For segments with zero overlap, fall back to the nearest turn in time."""
    if not turns:
        return [{**s, "speaker": "SPEAKER_00"} for s in segments]
    sorted_turns = sorted(turns, key=lambda t: t[0])

    def nearest_turn_speaker(seg_start: float, seg_end: float) -> str:
        mid = (seg_start + seg_end) / 2
        best, best_dist = sorted_turns[0][2], float("inf")
        for ts, te, spk in sorted_turns:
            if te < seg_start:
                d = seg_start - te
            elif ts > seg_end:
                d = ts - seg_end
            else:
                d = 0
            if d < best_dist:
                best, best_dist = spk, d
                if d == 0:
                    break
        return best

    out = []
    for s in segments:
        best_overlap, best_spk = 0.0, None
        for ts, te, spk in turns:
            ov = max(0.0, min(s["end"], te) - max(s["start"], ts))
            if ov > best_overlap:
                best_overlap, best_spk = ov, spk
        if best_spk is None:
            best_spk = nearest_turn_speaker(s["start"], s["end"])
        out.append({**s, "speaker": best_spk})
    return out


def remap_speaker_labels(segments: list[dict]) -> list[dict]:
    """Normalize speaker labels to SPEAKER_00, SPEAKER_01, ... in order of first appearance."""
    mapping: dict[str, str] = {}
    for s in segments:
        spk = s["speaker"]
        if spk not in mapping:
            mapping[spk] = f"SPEAKER_{len(mapping):02d}"
    return [{**s, "speaker": mapping[s["speaker"]]} for s in segments]


def merge_consecutive(segments: list[dict], max_turn_sec: float = 45.0) -> list[dict]:
    """Merge consecutive same-speaker segments, but break a turn if it would
    exceed `max_turn_sec`. Breaks at sentence boundaries (. ! ?) when possible.
    """
    out = []
    for s in segments:
        if (
            out
            and out[-1]["speaker"] == s["speaker"]
            and (s["end"] - out[-1]["start"]) <= max_turn_sec
        ):
            out[-1]["end"] = s["end"]
            out[-1]["text"] = (out[-1]["text"] + " " + s["text"]).strip()
        elif (
            out
            and out[-1]["speaker"] == s["speaker"]
            and out[-1]["text"].rstrip().endswith((".", "!", "?"))
        ):
            # Same speaker but turn too long; break at sentence boundary
            out.append(dict(s))
        elif out and out[-1]["speaker"] == s["speaker"]:
            # Same speaker, turn too long, but mid-sentence; still merge rather than split awkwardly
            out[-1]["end"] = s["end"]
            out[-1]["text"] = (out[-1]["text"] + " " + s["text"]).strip()
        else:
            out.append(dict(s))
    return out


_FILLER_WORDS = {"yeah", "yes", "no", "okay", "ok", "right", "uh", "um", "hmm", "mhm"}


def _collapse_filler_runs(text: str) -> str:
    """Collapse runs of filler words separated only by punctuation/whitespace.

    Examples:
      "Yeah. Yeah, yeah. Yeah. Yeah."     → "Yeah."
      "No. No. No problem. No. No."        → "No. No problem."
      "Okay. Okay. Okay. So we broadcast..." → "Okay. So we broadcast..."
    """
    tokens = re.split(r"(\s+|[.!?,])", text)
    # Walk through, looking for ≥3 consecutive content tokens that are all the same filler
    out_tokens = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        word = re.sub(r"\W+", "", t.lower())
        if word in _FILLER_WORDS:
            run_words = [t]
            j = i + 1
            while j < len(tokens):
                t2 = tokens[j]
                if t2.strip() == "" or t2 in ".,!?":
                    j += 1
                    continue
                w2 = re.sub(r"\W+", "", t2.lower())
                if w2 == word:
                    run_words.append(t2)
                    j += 1
                else:
                    break
            if len(run_words) >= 3:
                # Keep just the first instance + a period
                out_tokens.append(run_words[0])
                if not out_tokens[-1].endswith((".", "!", "?")):
                    out_tokens.append(".")
                i = j
                continue
        out_tokens.append(t)
        i += 1
    result = "".join(out_tokens)
    result = re.sub(r"\s+", " ", result)
    result = re.sub(r"\s+([.!?,])", r"\1", result)
    return result.strip()


def collapse_in_segment_repetition(segments: list[dict]) -> list[dict]:
    """Collapse repeated phrases/words WITHIN a segment.

    Whisper sometimes produces "Yeah. Yeah. Yeah. Yeah, yeah. Five months." after briefly
    looping. Catches:
    - identical sentences repeated 3+ times
    - runs of filler words ("Yeah. Yeah, yeah. Yeah." → "Yeah.")
    - word bigram immediately repeated ("travel to travel to" → "travel to")
    """
    sent_re = re.compile(r"[^.!?]+[.!?]?\s*")
    for s in segments:
        text = s.get("text", "")
        if not text.strip():
            continue

        # Step 1: collapse identical-sentence runs (≥3×)
        sentences = [m.group(0).strip() for m in sent_re.finditer(text) if m.group(0).strip()]
        cleaned = []
        i = 0
        while i < len(sentences):
            cur = sentences[i]
            cur_norm = re.sub(r"\W+", "", cur.lower())
            j = i + 1
            while j < len(sentences) and re.sub(r"\W+", "", sentences[j].lower()) == cur_norm:
                j += 1
            if (j - i) >= 3 and cur_norm:
                cleaned.append(cur)
            else:
                cleaned.extend(sentences[i:j])
            i = j
        text = " ".join(cleaned)

        # Step 2: collapse filler-word runs
        text = _collapse_filler_runs(text)

        # Step 3: collapse immediate word bigram repetition ("travel to travel to" → "travel to")
        text = re.sub(r"\b(\w+\s+\w+)(\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)

        # Step 4: collapse immediate single-word repetition ("we we we" → "we")
        text = re.sub(r"\b(\w+)(\s+\1\b){2,}", r"\1", text, flags=re.IGNORECASE)

        text = re.sub(r"\s+", " ", text).strip()
        if text and text != s.get("text", "").strip():
            s["text"] = text
    return segments


def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def write_outputs(
    segments: list[dict],
    merged: list[dict],
    out_dir: Path,
    basename: str,
    language: str,
) -> dict:
    transcript_path = out_dir / f"{basename}.transcript.txt"
    json_path = out_dir / f"{basename}.json"
    srt_path = out_dir / f"{basename}.srt"

    # Timestamped transcript with speaker labels
    with transcript_path.open("w") as f:
        for m in merged:
            f.write(
                f"[{fmt_ts(m['start'])} - {fmt_ts(m['end'])}] {m['speaker']}: {m['text']}\n"
            )

    # SRT
    def srt_ts(s):
        h = int(s // 3600); m = int((s % 3600) // 60); sec = s - h*3600 - m*60
        return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")
    with srt_path.open("w") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{srt_ts(seg['start'])} --> {srt_ts(seg['end'])}\n")
            f.write(f"{seg['speaker']}: {seg['text']}\n\n")

    # Structured JSON
    json_path.write_text(json.dumps({
        "language": language,
        "num_speakers": len({s["speaker"] for s in segments}),
        "segments": segments,
        "merged": merged,
    }, indent=2))

    # Per-speaker text files
    by_spk: dict[str, list[str]] = {}
    for m in merged:
        by_spk.setdefault(m["speaker"], []).append(m["text"])
    speaker_files = {}
    for spk, lines in by_spk.items():
        p = out_dir / f"{basename}.{spk}.txt"
        p.write_text("\n\n".join(lines) + "\n")
        speaker_files[spk] = str(p)

    return {
        "transcript": str(transcript_path),
        "json": str(json_path),
        "srt": str(srt_path),
        "per_speaker": speaker_files,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path.cwd())
    ap.add_argument("--prompt", default="")
    ap.add_argument("--language", default="en")
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--no-preprocess", action="store_true")
    ap.add_argument("--no-diarize", action="store_true")
    args = ap.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        sys.exit(f"Input not found: {input_path}")

    basename = input_path.stem
    out_dir = (args.output_dir / f"{basename}_transcript").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output dir: {out_dir}")

    # 1. Video → audio if needed
    audio_in = extract_audio_if_video(input_path, out_dir)

    # 2. Preprocess
    if args.no_preprocess:
        clean = out_dir / f"{basename}.clean.wav"
        run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(audio_in), "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(clean),
        ])
    else:
        clean = preprocess_audio(audio_in, out_dir, basename)

    # 3a. VAD-trim silent intro/outro (eliminates "*Clap*" / "Thank you" hallucinations)
    trimmed_audio, leading_trim = vad_trim(clean, out_dir, basename)

    # 3b. Transcribe
    whisper_json = transcribe_whispercpp(
        trimmed_audio, out_dir, basename, args.prompt, args.language
    )
    segments, language = load_whisper_json(whisper_json)
    log(f"  segments: {len(segments)}, language: {language}")

    # 3c. Shift timestamps back to original timeline (we trimmed `leading_trim` sec off the start)
    if leading_trim > 0:
        for s in segments:
            s["start"] += leading_trim
            s["end"] += leading_trim
        log(f"  shifted timestamps by +{leading_trim:.1f}s to match original timeline")

    # 4. Strip silent-intro hallucinations + collapse in-segment repetition
    segments = strip_hallucinations(segments)
    log(f"  segments after hallucination strip: {len(segments)}")
    segments = collapse_in_segment_repetition(segments)

    # 5. Diarize
    if args.no_diarize:
        log("Skipping diarization (--no-diarize)")
        labeled = [{**s, "speaker": "SPEAKER_00"} for s in segments]
    else:
        turns = diarize(clean, args.num_speakers, out_dir)
        labeled = assign_speakers(segments, turns)

    # 6. Normalize labels + merge consecutive
    labeled = remap_speaker_labels(labeled)
    merged = merge_consecutive(labeled)

    # 7. Write outputs
    out = write_outputs(labeled, merged, out_dir, basename, language)
    speakers = sorted({s["speaker"] for s in labeled})

    log("Done.")
    summary = {
        "input": str(input_path),
        "output_dir": str(out_dir),
        "clean_wav": str(clean),
        "transcript": out["transcript"],
        "json": out["json"],
        "srt": out["srt"],
        "per_speaker": out["per_speaker"],
        "num_segments": len(segments),
        "num_merged_turns": len(merged),
        "speakers": speakers,
        "language": language,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
