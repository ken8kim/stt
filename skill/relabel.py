#!/usr/bin/env python3
"""Apply human speaker name mappings to stt outputs.

Reads the canonical JSON (with SPEAKER_NN labels), substitutes human names,
and writes:
  - <base>.attributed.txt — timestamped, human names, consecutive turns merged
  - <base>.<NAME>.txt    — per-speaker chronological text, no timestamps
Updates the JSON in place with a `name_mapping` field.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def fmt_ts(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def safe_filename_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--mapping", required=True, help='JSON dict, e.g. {"SPEAKER_00":"Alice","SPEAKER_01":"Bob"}')
    ap.add_argument("--keep-old", action="store_true", help="Don't overwrite the SPEAKER_NN transcript.txt")
    args = ap.parse_args()

    json_path = args.json.resolve()
    data = json.loads(json_path.read_text())
    mapping = json.loads(args.mapping)

    # Apply mapping; unmapped labels pass through unchanged
    def rename(spk: str) -> str:
        return mapping.get(spk, spk)

    segments = data.get("segments", [])
    merged = data.get("merged", [])
    for s in segments:
        s["speaker"] = rename(s["speaker"])
    for m in merged:
        m["speaker"] = rename(m["speaker"])

    base_dir = json_path.parent
    base = json_path.stem

    # Write attributed.txt
    attributed_path = base_dir / f"{base}.attributed.txt"
    with attributed_path.open("w") as f:
        for m in merged:
            f.write(f"[{fmt_ts(m['start'])} - {fmt_ts(m['end'])}] {m['speaker']}: {m['text']}\n")

    # Per-speaker
    by_spk: dict[str, list[str]] = {}
    for m in merged:
        by_spk.setdefault(m["speaker"], []).append(m["text"])
    per_speaker_paths = []
    # Delete old SPEAKER_NN.txt files that no longer match new labels
    for old in base_dir.glob(f"{base}.SPEAKER_*.txt"):
        if old.stem.split(".", 1)[-1] not in by_spk:
            old.unlink()
    for spk, lines in by_spk.items():
        p = base_dir / f"{base}.{safe_filename_name(spk)}.txt"
        p.write_text("\n\n".join(lines) + "\n")
        per_speaker_paths.append(str(p))

    # Update JSON in place
    data["name_mapping"] = mapping
    data["segments"] = segments
    data["merged"] = merged
    json_path.write_text(json.dumps(data, indent=2))

    out = {
        "attributed_txt": str(attributed_path),
        "per_speaker": per_speaker_paths,
        "json": str(json_path),
        "applied_mapping": mapping,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
