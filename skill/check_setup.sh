#!/usr/bin/env bash
# Verify everything stt needs is in place. Auto-install where safe; otherwise report.
set -euo pipefail

SKILL_DIR="$HOME/.claude/skills/stt"
WHISPER_DIR="$HOME/whisper.cpp"
RNN_CACHE="$HOME/.cache/rnnoise-models"

green() { printf '\033[32m✓\033[0m %s\n' "$1"; }
yellow() { printf '\033[33m!\033[0m %s\n' "$1"; }
red() { printf '\033[31m✗\033[0m %s\n' "$1"; }

missing=()

# 1) ffmpeg
if command -v ffmpeg >/dev/null 2>&1; then
  green "ffmpeg: $(ffmpeg -version | head -1 | awk '{print $3}')"
else
  red "ffmpeg not found"
  missing+=("brew install ffmpeg")
fi

# 2) whisper.cpp binary
if [ -x "$WHISPER_DIR/build/bin/whisper-cli" ]; then
  green "whisper.cpp: built"
else
  yellow "whisper.cpp not built — building now..."
  if [ ! -d "$WHISPER_DIR" ]; then
    git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git "$WHISPER_DIR"
  fi
  if ! command -v cmake >/dev/null 2>&1; then
    yellow "cmake missing — installing via brew"
    brew install cmake
  fi
  (cd "$WHISPER_DIR" && cmake -B build -DWHISPER_COREML=1 -DCMAKE_BUILD_TYPE=Release >/dev/null)
  (cd "$WHISPER_DIR" && cmake --build build -j --config Release >/dev/null 2>&1)
  green "whisper.cpp built"
fi

# 3) ggml model
if [ -f "$WHISPER_DIR/models/ggml-large-v3-turbo.bin" ]; then
  green "ggml-large-v3-turbo.bin: present ($(du -h "$WHISPER_DIR/models/ggml-large-v3-turbo.bin" | awk '{print $1}'))"
else
  yellow "downloading ggml-large-v3-turbo.bin (~1.6GB, can be slow without HF_TOKEN)..."
  (cd "$WHISPER_DIR" && bash ./models/download-ggml-model.sh large-v3-turbo)
  green "ggml model downloaded"
fi

# 4) Core ML encoder
if [ -d "$WHISPER_DIR/models/ggml-large-v3-turbo-encoder.mlmodelc" ]; then
  green "Core ML encoder: compiled"
else
  yellow "Core ML encoder missing — converting (this can take 10-20 min)..."
  if [ ! -x "/Applications/Xcode.app/Contents/Developer/usr/bin/coremlc" ]; then
    red "Xcode (not just Command Line Tools) required for coremlc compilation."
    red "Install Xcode from the App Store, then re-run /stt."
    missing+=("Install Xcode for Core ML compilation")
  else
    pip3 install -q ane-transformers openai-whisper coremltools 2>/dev/null || true
    (cd "$WHISPER_DIR" && python3 models/convert-whisper-to-coreml.py --model large-v3-turbo --encoder-only True --optimize-ane True)
    (cd "$WHISPER_DIR" && /Applications/Xcode.app/Contents/Developer/usr/bin/coremlc compile models/coreml-encoder-large-v3-turbo.mlpackage models/)
    (cd "$WHISPER_DIR" && rm -rf models/ggml-large-v3-turbo-encoder.mlmodelc && mv models/coreml-encoder-large-v3-turbo.mlmodelc models/ggml-large-v3-turbo-encoder.mlmodelc)
    green "Core ML encoder compiled"
  fi
fi

# 5) RNN denoise model
mkdir -p "$RNN_CACHE"
if [ -f "$RNN_CACHE/sh.rnnn" ]; then
  green "RNN denoise model: present"
else
  yellow "downloading sh.rnnn denoise model..."
  curl -sL -o "$RNN_CACHE/sh.rnnn" \
    https://github.com/GregorR/rnnoise-models/raw/master/somnolent-hogwash-2018-09-01/sh.rnnn
  green "sh.rnnn downloaded"
fi

# 6) Python deps
need_pip=()
for pkg in numpy torch torchaudio scikit-learn; do
  if ! python3 -c "import $pkg" 2>/dev/null; then
    need_pip+=("$pkg")
  fi
done
if ! python3 -c "from resemblyzer import VoiceEncoder" 2>/dev/null; then
  need_pip+=("resemblyzer")
fi
if ! python3 -c "import pyannote.audio" 2>/dev/null; then
  need_pip+=("pyannote.audio")
fi
if [ ${#need_pip[@]} -gt 0 ]; then
  yellow "installing Python deps: ${need_pip[*]}"
  pip3 install --quiet "${need_pip[@]}" || red "pip install failed for some packages"
else
  green "Python deps: ok"
fi

# 7) HF token (optional, for best diarization)
if [ -n "${HF_TOKEN:-}" ] || [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
  green "HF_TOKEN: set (pyannote 3.1 available)"
else
  yellow "HF_TOKEN not set — will use Resemblyzer fallback for diarization."
  yellow "  For best quality: accept terms at https://hf.co/pyannote/speaker-diarization-3.1"
  yellow "  and https://hf.co/pyannote/segmentation-3.0, generate token at https://hf.co/settings/tokens,"
  yellow "  then export HF_TOKEN=hf_xxx"
fi

if [ ${#missing[@]} -gt 0 ]; then
  echo ""
  red "Setup incomplete. Required actions:"
  for m in "${missing[@]}"; do
    echo "  - $m"
  done
  exit 1
fi
echo ""
green "stt setup complete."
