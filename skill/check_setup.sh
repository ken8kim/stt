#!/usr/bin/env bash
# Cross-platform setup for stt. Auto-detects macOS / Linux / Windows-WSL
# and the available GPU (CUDA / ROCm / Vulkan / MPS / CPU), then installs +
# builds whisper.cpp with the right cmake flags and PyTorch with the right
# wheels.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WHISPER_DIR="${WHISPER_CPP_DIR:-$HOME/whisper.cpp}"
RNN_CACHE="$HOME/.cache/rnnoise-models"

green() { printf '\033[32m✓\033[0m %s\n' "$1"; }
yellow() { printf '\033[33m!\033[0m %s\n' "$1"; }
red()    { printf '\033[31m✗\033[0m %s\n' "$1"; }

missing=()

# ---------------------------------------------------------------------------
# Detect platform
# ---------------------------------------------------------------------------
PLATFORM_JSON="$(python3 "$SCRIPT_DIR/platform_detect.py" --json)"
OS=$(echo "$PLATFORM_JSON"          | python3 -c "import sys,json;print(json.load(sys.stdin)['os'])")
GPU=$(echo "$PLATFORM_JSON"         | python3 -c "import sys,json;print(json.load(sys.stdin)['gpu'])")
CMAKE_FLAGS=$(echo "$PLATFORM_JSON"  | python3 -c "import sys,json;print(' '.join(json.load(sys.stdin)['whisper_cmake_flags']))")
TORCH_INDEX=$(echo "$PLATFORM_JSON" | python3 -c "import sys,json;print(json.load(sys.stdin).get('pip_torch_index') or '')")
NEEDS_XCODE=$(echo "$PLATFORM_JSON" | python3 -c "import sys,json;print('1' if json.load(sys.stdin)['needs_xcode'] else '')")

echo "Detected: $OS / GPU=$GPU"
echo "whisper.cpp cmake flags: ${CMAKE_FLAGS:-(CPU only)}"
echo ""

# ---------------------------------------------------------------------------
# System tools (ffmpeg, cmake, git)
# ---------------------------------------------------------------------------
install_pkg() {
    local pkg="$1"
    case "$OS" in
        macos)  brew install "$pkg" ;;
        linux)
            if   command -v apt-get >/dev/null;  then sudo apt-get install -y "$pkg"
            elif command -v dnf     >/dev/null;  then sudo dnf install -y "$pkg"
            elif command -v pacman  >/dev/null;  then sudo pacman -S --noconfirm "$pkg"
            else  yellow "Unsupported Linux distro — install '$pkg' manually."; fi ;;
        windows)
            yellow "On Windows native: install '$pkg' via winget or chocolatey." ;;
    esac
}

for tool in ffmpeg cmake git curl; do
    if command -v "$tool" >/dev/null 2>&1; then
        green "$tool: present"
    else
        yellow "$tool missing — installing..."
        install_pkg "$tool" || missing+=("install $tool manually")
    fi
done

# ---------------------------------------------------------------------------
# Platform-specific: Xcode for macOS Core ML
# ---------------------------------------------------------------------------
if [ -n "$NEEDS_XCODE" ]; then
    if [ -x "/Applications/Xcode.app/Contents/Developer/usr/bin/coremlc" ]; then
        green "coremlc (Xcode): present"
    else
        red "coremlc not found — install full Xcode from the App Store to enable Core ML acceleration"
        missing+=("Install Xcode (App Store) for Core ML compilation")
    fi
fi

# ---------------------------------------------------------------------------
# Platform-specific: CUDA toolkit (Linux/Windows + NVIDIA)
# ---------------------------------------------------------------------------
if [ "$GPU" = "cuda" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        green "CUDA toolkit: $(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | head -1)"
    else
        yellow "nvcc not on PATH. whisper.cpp needs CUDA toolkit headers to build."
        yellow "  Install: https://developer.nvidia.com/cuda-downloads"
        missing+=("Install CUDA toolkit (>= 11.8)")
    fi
fi

# ---------------------------------------------------------------------------
# Platform-specific: ROCm for AMD on Linux
# ---------------------------------------------------------------------------
if [ "$GPU" = "rocm" ]; then
    if command -v hipcc >/dev/null 2>&1; then
        green "ROCm/HIP toolkit: present"
    else
        yellow "hipcc not on PATH. Install ROCm SDK."
        yellow "  https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
        missing+=("Install ROCm SDK (>= 6.0)")
    fi
fi

# ---------------------------------------------------------------------------
# whisper.cpp build
# ---------------------------------------------------------------------------
if [ -x "$WHISPER_DIR/build/bin/whisper-cli" ] || [ -x "$WHISPER_DIR/build/bin/whisper-cli.exe" ]; then
    green "whisper.cpp: built at $WHISPER_DIR"
else
    yellow "Cloning + building whisper.cpp at $WHISPER_DIR"
    if [ ! -d "$WHISPER_DIR" ]; then
        git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git "$WHISPER_DIR"
    fi
    (
        cd "$WHISPER_DIR"
        # shellcheck disable=SC2086
        cmake -B build -DCMAKE_BUILD_TYPE=Release $CMAKE_FLAGS >/dev/null
        cmake --build build -j --config Release >/dev/null 2>&1
    )
    green "whisper.cpp built with flags: ${CMAKE_FLAGS:-(CPU only)}"
fi

# ---------------------------------------------------------------------------
# ggml model
# ---------------------------------------------------------------------------
MODEL="$WHISPER_DIR/models/ggml-large-v3-turbo.bin"
if [ -f "$MODEL" ]; then
    green "ggml-large-v3-turbo.bin: present ($(du -h "$MODEL" | awk '{print $1}'))"
else
    yellow "Downloading ggml-large-v3-turbo.bin (~1.6 GB)..."
    (cd "$WHISPER_DIR" && bash ./models/download-ggml-model.sh large-v3-turbo)
    green "ggml model downloaded"
fi

# ---------------------------------------------------------------------------
# Core ML encoder (macOS Apple Silicon only)
# ---------------------------------------------------------------------------
if [ "$GPU" = "mps" ]; then
    if [ -d "$WHISPER_DIR/models/ggml-large-v3-turbo-encoder.mlmodelc" ]; then
        green "Core ML encoder: compiled"
    elif [ -x "/Applications/Xcode.app/Contents/Developer/usr/bin/coremlc" ]; then
        yellow "Compiling Core ML encoder (10-20 min)..."
        pip3 install -q ane-transformers openai-whisper coremltools 2>/dev/null || true
        (cd "$WHISPER_DIR" && python3 models/convert-whisper-to-coreml.py \
                                --model large-v3-turbo --encoder-only True --optimize-ane True)
        (cd "$WHISPER_DIR" && /Applications/Xcode.app/Contents/Developer/usr/bin/coremlc compile \
                                models/coreml-encoder-large-v3-turbo.mlpackage models/)
        (cd "$WHISPER_DIR" && rm -rf models/ggml-large-v3-turbo-encoder.mlmodelc && \
                                mv models/coreml-encoder-large-v3-turbo.mlmodelc \
                                   models/ggml-large-v3-turbo-encoder.mlmodelc)
        green "Core ML encoder compiled"
    fi
fi

# ---------------------------------------------------------------------------
# RNN denoise model
# ---------------------------------------------------------------------------
mkdir -p "$RNN_CACHE"
if [ -f "$RNN_CACHE/sh.rnnn" ]; then
    green "RNN denoise model: present"
else
    yellow "Downloading sh.rnnn denoise model..."
    curl -sL -o "$RNN_CACHE/sh.rnnn" \
        https://github.com/GregorR/rnnoise-models/raw/master/somnolent-hogwash-2018-09-01/sh.rnnn
    green "sh.rnnn downloaded"
fi

# ---------------------------------------------------------------------------
# PyTorch + Python deps with platform-specific wheels
# ---------------------------------------------------------------------------
need_pip=()
if ! python3 -c "import torch" 2>/dev/null; then need_pip+=("torch"); fi
if ! python3 -c "import torchaudio" 2>/dev/null; then need_pip+=("torchaudio"); fi
if ! python3 -c "import numpy, sklearn" 2>/dev/null; then need_pip+=("numpy" "scikit-learn"); fi
if ! python3 -c "from resemblyzer import VoiceEncoder" 2>/dev/null; then need_pip+=("resemblyzer"); fi
if ! python3 -c "import pyannote.audio" 2>/dev/null; then need_pip+=("pyannote.audio"); fi

if [ ${#need_pip[@]} -gt 0 ]; then
    yellow "Installing Python deps: ${need_pip[*]}"
    torch_deps=()
    other_deps=()
    for dep in "${need_pip[@]}"; do
        if [[ "$dep" == "torch" || "$dep" == "torchaudio" ]]; then
            torch_deps+=("$dep")
        else
            other_deps+=("$dep")
        fi
    done
    if [ ${#torch_deps[@]} -gt 0 ]; then
        if [ -n "$TORCH_INDEX" ]; then
            pip3 install --quiet "${torch_deps[@]}" --index-url "$TORCH_INDEX" \
                || red "PyTorch install failed for $GPU. Try: pip3 install torch torchaudio --index-url $TORCH_INDEX"
        else
            pip3 install --quiet "${torch_deps[@]}" \
                || red "PyTorch install failed."
        fi
    fi
    if [ ${#other_deps[@]} -gt 0 ]; then
        pip3 install --quiet "${other_deps[@]}" || red "pip install failed for: ${other_deps[*]}"
    fi
else
    green "Python deps: ok"
fi

# ---------------------------------------------------------------------------
# HF token (optional but recommended for best diarization)
# ---------------------------------------------------------------------------
if [ -n "${HF_TOKEN:-}" ] || [ -n "${HUGGINGFACE_TOKEN:-}" ]; then
    green "HF_TOKEN: set (pyannote 3.1 available)"
else
    yellow "HF_TOKEN not set — diarization will use Resemblyzer fallback."
    yellow "  Best quality: accept terms on all three:"
    yellow "    https://hf.co/pyannote/speaker-diarization-3.1"
    yellow "    https://hf.co/pyannote/segmentation-3.0"
    yellow "    https://hf.co/pyannote/speaker-diarization-community-1"
    yellow "  Then create a read-only token at https://hf.co/settings/tokens and export HF_TOKEN=hf_xxx"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if [ ${#missing[@]} -gt 0 ]; then
    echo ""
    red "Setup incomplete. Required actions:"
    for m in "${missing[@]}"; do echo "  - $m"; done
    exit 1
fi

echo ""
green "stt setup complete on $OS ($GPU)."
