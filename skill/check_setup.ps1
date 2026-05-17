# stt setup for Windows (native). Run in PowerShell as a regular user — sudo
# prompts only for system-level installs.
#
#   PS> .\check_setup.ps1
#
# Auto-detects NVIDIA (preferred) or Vulkan (AMD/Intel fallback). ROCm is not
# supported on Windows as of 2026 — use WSL2 + the bash setup for ROCm.

$ErrorActionPreference = "Stop"
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$WhisperDir  = if ($env:WHISPER_CPP_DIR) { $env:WHISPER_CPP_DIR } else { "$env:USERPROFILE\whisper.cpp" }
$RnnCache    = "$env:LOCALAPPDATA\rnnoise-models"
$Missing     = @()

function Ok($msg)    { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Warn($msg)  { Write-Host "[!]    $msg" -ForegroundColor Yellow }
function Fail($msg)  { Write-Host "[FAIL] $msg" -ForegroundColor Red }

# ---------------------------------------------------------------------------
# Detect platform via the shared Python module
# ---------------------------------------------------------------------------
$platformJson = (python "$ScriptDir\platform_detect.py" --json) | ConvertFrom-Json
$os           = $platformJson.os
$gpu          = $platformJson.gpu
$cmakeFlags   = $platformJson.whisper_cmake_flags -join " "
$torchIndex   = $platformJson.pip_torch_index

Write-Host "Detected: $os / GPU=$gpu"
Write-Host "whisper.cpp cmake flags: $(if ($cmakeFlags) { $cmakeFlags } else { '(CPU only)' })"
Write-Host ""

# ---------------------------------------------------------------------------
# System tools: ffmpeg, cmake, git, curl
# ---------------------------------------------------------------------------
function Install-Pkg($name, $wingetId) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Warn "Installing $name via winget..."
        winget install --silent --accept-package-agreements --accept-source-agreements --id $wingetId
    } elseif (Get-Command choco -ErrorAction SilentlyContinue) {
        Warn "Installing $name via chocolatey..."
        choco install -y $name
    } else {
        $script:Missing += "Install $name manually (winget or choco recommended)"
    }
}

$tools = @{
    ffmpeg = "Gyan.FFmpeg"
    cmake  = "Kitware.CMake"
    git    = "Git.Git"
}
foreach ($tool in $tools.Keys) {
    if (Get-Command $tool -ErrorAction SilentlyContinue) {
        Ok "$tool: present"
    } else {
        Install-Pkg $tool $tools[$tool]
    }
}

# ---------------------------------------------------------------------------
# CUDA toolkit (NVIDIA path)
# ---------------------------------------------------------------------------
if ($gpu -eq "cuda") {
    if (Get-Command nvcc -ErrorAction SilentlyContinue) {
        $cudaVer = (nvcc --version | Select-String "release \d+\.\d+").Matches.Value
        Ok "CUDA toolkit: $cudaVer"
    } else {
        Warn "nvcc not on PATH — install CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
        $Missing += "Install CUDA toolkit (>=11.8)"
    }
}

# ---------------------------------------------------------------------------
# whisper.cpp build
# ---------------------------------------------------------------------------
$whisperBin = "$WhisperDir\build\bin\whisper-cli.exe"
if (Test-Path $whisperBin) {
    Ok "whisper.cpp: built at $WhisperDir"
} else {
    Warn "Cloning + building whisper.cpp at $WhisperDir"
    if (-not (Test-Path $WhisperDir)) {
        git clone --depth 1 https://github.com/ggerganov/whisper.cpp.git $WhisperDir
    }
    Push-Location $WhisperDir
    try {
        $cmakeArgs = @("-B", "build", "-DCMAKE_BUILD_TYPE=Release") + ($cmakeFlags -split " " | Where-Object { $_ })
        & cmake @cmakeArgs | Out-Null
        cmake --build build --config Release | Out-Null
    } finally {
        Pop-Location
    }
    Ok "whisper.cpp built with flags: $(if ($cmakeFlags) { $cmakeFlags } else { '(CPU only)' })"
}

# ---------------------------------------------------------------------------
# ggml model
# ---------------------------------------------------------------------------
$model = "$WhisperDir\models\ggml-large-v3-turbo.bin"
if (Test-Path $model) {
    $size = "{0:N1}" -f ((Get-Item $model).Length / 1GB)
    Ok "ggml-large-v3-turbo.bin: present (${size} GB)"
} else {
    Warn "Downloading ggml-large-v3-turbo.bin (~1.6 GB)..."
    $url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
    Invoke-WebRequest -Uri $url -OutFile $model -UseBasicParsing
    Ok "ggml model downloaded"
}

# ---------------------------------------------------------------------------
# RNN denoise model
# ---------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $RnnCache | Out-Null
$rnn = "$RnnCache\sh.rnnn"
if (Test-Path $rnn) {
    Ok "RNN denoise model: present"
} else {
    Warn "Downloading sh.rnnn denoise model..."
    Invoke-WebRequest -Uri "https://github.com/GregorR/rnnoise-models/raw/master/somnolent-hogwash-2018-09-01/sh.rnnn" `
                      -OutFile $rnn -UseBasicParsing
    Ok "sh.rnnn downloaded"
}

# ---------------------------------------------------------------------------
# Python deps
# ---------------------------------------------------------------------------
$needPip = @()
foreach ($mod in @("torch", "torchaudio", "numpy", "sklearn")) {
    $check = python -c "import $mod" 2>$null
    if ($LASTEXITCODE -ne 0) { $needPip += $(if ($mod -eq "sklearn") { "scikit-learn" } else { $mod }) }
}
foreach ($pair in @(@("resemblyzer", "resemblyzer"), @("pyannote.audio", "pyannote.audio"))) {
    $mod = $pair[0]; $pkg = $pair[1]
    $check = python -c "import $($mod -replace '\.', '_' -replace 'pyannote_audio', 'pyannote.audio')" 2>$null
    if ($LASTEXITCODE -ne 0) { $needPip += $pkg }
}
if ($needPip.Count -gt 0) {
    Warn "Installing Python deps: $($needPip -join ', ')"
    $torchDeps = $needPip | Where-Object { $_ -in @("torch", "torchaudio") }
    $otherDeps = $needPip | Where-Object { $_ -notin @("torch", "torchaudio") }
    if ($torchDeps.Count -gt 0) {
        if ($torchIndex) {
            python -m pip install --quiet @torchDeps --index-url $torchIndex
        } else {
            python -m pip install --quiet @torchDeps
        }
    }
    if ($otherDeps.Count -gt 0) {
        python -m pip install --quiet @otherDeps
    }
} else {
    Ok "Python deps: ok"
}

# ---------------------------------------------------------------------------
# HF token check
# ---------------------------------------------------------------------------
if ($env:HF_TOKEN -or $env:HUGGINGFACE_TOKEN) {
    Ok "HF_TOKEN: set (pyannote 3.1 available)"
} else {
    Warn "HF_TOKEN not set — diarization will use Resemblyzer fallback."
    Warn "  Accept terms on all three:"
    Warn "    https://hf.co/pyannote/speaker-diarization-3.1"
    Warn "    https://hf.co/pyannote/segmentation-3.0"
    Warn "    https://hf.co/pyannote/speaker-diarization-community-1"
    Warn "  Then set: [Environment]::SetEnvironmentVariable('HF_TOKEN', 'hf_...', 'User')"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
if ($Missing.Count -gt 0) {
    Write-Host ""
    Fail "Setup incomplete. Required actions:"
    foreach ($m in $Missing) { Write-Host "  - $m" }
    exit 1
}
Write-Host ""
Ok "stt setup complete on $os ($gpu)."
