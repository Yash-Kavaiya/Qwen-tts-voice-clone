#!/usr/bin/env bash
# ============================================================
# Qwen3-TTS Voice Cloning — Environment Setup
# ============================================================
# This script creates a fresh conda environment, installs the
# qwen-tts package with all required dependencies, and
# optionally installs Flash Attention 2 for speed + VRAM savings.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ============================================================

set -euo pipefail

ENV_NAME="qwen3-tts"
PYTHON_VERSION="3.12"

echo "============================================"
echo "  Qwen3-TTS Voice Cloning — Setup"
echo "============================================"
echo ""

# ── 1. Create conda environment ──────────────────────────────
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "⚠  Conda environment '${ENV_NAME}' already exists."
    echo "   Activate it with: conda activate ${ENV_NAME}"
else
    echo "📦 Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    echo "✅ Environment created."
fi

echo ""
echo "🔄 Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# ── 2. Install core dependencies ─────────────────────────────
echo ""
echo "📥 Installing qwen-tts and dependencies..."
pip install -U qwen-tts
pip install -U soundfile numpy gradio

# ── 3. Install Flash Attention 2 (optional but recommended) ──
echo ""
echo "🚀 Installing Flash Attention 2 (for faster inference)..."
echo "   If this fails, the models will still work without it."
echo ""

# Use MAX_JOBS=4 if machine has < 96 GB RAM to prevent OOM during build
TOTAL_RAM_GB=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "0")
if [ "${TOTAL_RAM_GB}" -lt 96 ] 2>/dev/null; then
    echo "   (Detected <96 GB RAM — limiting build parallelism)"
    MAX_JOBS=4 pip install -U flash-attn --no-build-isolation || {
        echo "⚠  Flash Attention 2 installation failed (non-fatal)."
        echo "   You can still use the models without it."
    }
else
    pip install -U flash-attn --no-build-isolation || {
        echo "⚠  Flash Attention 2 installation failed (non-fatal)."
        echo "   You can still use the models without it."
    }
fi

# ── 4. Done ──────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  ✅ Setup complete!"
echo "============================================"
echo ""
echo "  Activate:  conda activate ${ENV_NAME}"
echo ""
echo "  Quick test:"
echo "    python voice_clone.py --help"
echo "    python voice_design.py --help"
echo "    python app.py"
echo ""
echo "  Models will be auto-downloaded on first run."
echo "============================================"
