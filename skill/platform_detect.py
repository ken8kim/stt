#!/usr/bin/env python3
"""Platform + GPU detection for stt.

Detects:
  OS:   macos | linux | windows
  GPU:  mps   | cuda  | rocm | vulkan | cpu

Returns the appropriate whisper.cpp cmake flags, PyTorch device, PyTorch
install index URL, and the binary names (whisper.cpp uses `whisper-cli.exe`
on Windows).

The order of GPU preference is:
  macOS  → mps (Apple)
  linux  → cuda > rocm > vulkan > cpu
  windows → cuda > vulkan > cpu  (no ROCm on Windows as of 2026)
"""
from __future__ import annotations

import json
import os
import platform as _stdplatform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class PlatformInfo:
    os: str                     # "macos" | "linux" | "windows"
    gpu: str                    # "mps" | "cuda" | "rocm" | "vulkan" | "cpu"
    torch_device: str           # "mps" | "cuda" | "cpu"  (rocm uses cuda API)
    whisper_cmake_flags: list[str] = field(default_factory=list)
    pip_torch_index: str | None = None  # extra-index-url for torch install
    whisper_bin_name: str = "whisper-cli"  # ".exe" appended on Windows
    needs_xcode: bool = False
    extra_notes: list[str] = field(default_factory=list)


def _has_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def _try_run(cmd: list[str]) -> bool:
    try:
        subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=5)
        return True
    except Exception:
        return False


def _is_nvidia_available() -> bool:
    if not _has_cmd("nvidia-smi"):
        return False
    return _try_run(["nvidia-smi", "-L"])


def _is_rocm_available() -> bool:
    # ROCm tooling installs rocminfo, hipcc, or rocm-smi
    return any(_has_cmd(x) for x in ("rocminfo", "rocm-smi", "hipcc"))


def _is_vulkan_available() -> bool:
    return _has_cmd("vulkaninfo") or _has_cmd("vkcube")


def _detect_torch_device(gpu: str) -> str:
    """What `torch.device(...)` string should the pipeline pass to PyTorch?

    Note that PyTorch's ROCm build presents the same API as CUDA — both go
    through torch.cuda.is_available(). So ROCm → 'cuda'.
    Vulkan has no first-class PyTorch support → 'cpu' for diarization.
    """
    if gpu == "mps":
        return "mps"
    if gpu in ("cuda", "rocm"):
        return "cuda"
    return "cpu"


def _detect_amdgpu_target() -> str | None:
    """Try to detect the AMD GPU arch (gfx1030, gfx1100, etc.) for ROCm builds."""
    if not _has_cmd("rocminfo"):
        return None
    try:
        out = subprocess.check_output(["rocminfo"], text=True, timeout=10)
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("Name:") and "gfx" in line:
                target = line.split()[1]
                return target
    except Exception:
        pass
    return None


def detect() -> PlatformInfo:
    sys_name = _stdplatform.system()  # 'Darwin', 'Linux', 'Windows'

    if sys_name == "Darwin":
        # macOS — Apple Silicon (MPS) or Intel (CPU)
        is_arm = _stdplatform.machine() == "arm64"
        return PlatformInfo(
            os="macos",
            gpu="mps" if is_arm else "cpu",
            torch_device="mps" if is_arm else "cpu",
            whisper_cmake_flags=["-DWHISPER_COREML=1"] if is_arm else [],
            pip_torch_index=None,  # default wheels work
            needs_xcode=is_arm,  # for coremlc compilation
            extra_notes=(
                ["Apple Silicon: Core ML + Metal acceleration enabled"]
                if is_arm else ["Intel Mac: CPU-only mode"]
            ),
        )

    is_windows = sys_name == "Windows"
    is_linux = sys_name == "Linux"
    bin_name = "whisper-cli.exe" if is_windows else "whisper-cli"

    # NVIDIA first (fastest)
    if _is_nvidia_available():
        return PlatformInfo(
            os="windows" if is_windows else "linux",
            gpu="cuda",
            torch_device="cuda",
            whisper_cmake_flags=["-DGGML_CUDA=1"],
            pip_torch_index="https://download.pytorch.org/whl/cu121",
            whisper_bin_name=bin_name,
            extra_notes=[
                "NVIDIA GPU detected via nvidia-smi",
                "If your CUDA is 11.x, use index 'cu118' instead of 'cu121'",
            ],
        )

    # AMD via ROCm (Linux only)
    if is_linux and _is_rocm_available():
        flags = ["-DGGML_HIPBLAS=1"]
        target = _detect_amdgpu_target()
        if target:
            flags.append(f"-DAMDGPU_TARGETS={target}")
        return PlatformInfo(
            os="linux",
            gpu="rocm",
            torch_device="cuda",  # PyTorch ROCm uses CUDA API
            whisper_cmake_flags=flags,
            pip_torch_index="https://download.pytorch.org/whl/rocm6.0",
            whisper_bin_name=bin_name,
            extra_notes=[
                f"AMD GPU detected via ROCm (target={target or 'auto-detect'})",
                "PyTorch ROCm uses the CUDA API surface (torch.cuda.is_available() returns True)",
            ],
        )

    # Vulkan fallback (cross-platform AMD/Intel iGPU)
    if _is_vulkan_available():
        return PlatformInfo(
            os="windows" if is_windows else "linux",
            gpu="vulkan",
            torch_device="cpu",  # No first-class PyTorch Vulkan support
            whisper_cmake_flags=["-DGGML_VULKAN=1"],
            pip_torch_index=None,
            whisper_bin_name=bin_name,
            extra_notes=[
                "Vulkan detected — accelerates transcription only; diarization runs on CPU",
                "For best AMD performance on Linux, install ROCm SDK instead",
            ],
        )

    # CPU-only fallback
    return PlatformInfo(
        os="windows" if is_windows else "linux",
        gpu="cpu",
        torch_device="cpu",
        whisper_cmake_flags=[],
        pip_torch_index=None,
        whisper_bin_name=bin_name,
        extra_notes=[
            "No GPU acceleration detected. Pipeline will run on CPU only — slow on long audio.",
            "On a 51-min recording, expect 30+ min runtime with the large-v3-turbo model.",
            "Consider using a smaller model (--model small) for CPU-only setups.",
        ],
    )


def print_summary() -> None:
    info = detect()
    d = asdict(info)
    print(json.dumps(d, indent=2))


if __name__ == "__main__":
    if "--json" in sys.argv:
        info = detect()
        print(json.dumps(asdict(info)))
    else:
        info = detect()
        print(f"OS:            {info.os}")
        print(f"GPU:           {info.gpu}")
        print(f"Torch device:  {info.torch_device}")
        print(f"whisper.cpp flags: {' '.join(info.whisper_cmake_flags) or '(none, CPU only)'}")
        if info.pip_torch_index:
            print(f"PyTorch index: {info.pip_torch_index}")
        if info.needs_xcode:
            print("Requires:      Xcode (full install, for coremlc)")
        for n in info.extra_notes:
            print(f"  • {n}")
