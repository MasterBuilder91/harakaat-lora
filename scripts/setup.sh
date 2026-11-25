#!/usr/bin/env bash
# setup.sh — WSL2 Ubuntu environment setup for harakaat-lora
# Run from: /mnt/e/harakaat-lora
# Usage: bash scripts/setup.sh

set -euo pipefail

PROJ_DIR="/mnt/e/harakaat-lora"
VENV_DIR="$PROJ_DIR/venv"
CACHE_DIR="$PROJ_DIR/.cache"

echo "=== harakaat-lora environment setup ==="
echo "Project dir: $PROJ_DIR"
echo ""

# 1. System packages
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv git build-essential cmake wget curl

# 2. CUDA library path for WSL2
echo "[2/6] Configuring CUDA WSL2 library path..."
WSL_CUDA="/usr/lib/wsl/lib"
if [ -d "$WSL_CUDA" ]; then
    if ! grep -q "wsl/lib" /etc/ld.so.conf.d/wsl-cuda.conf 2>/dev/null; then
        echo "$WSL_CUDA" | sudo tee /etc/ld.so.conf.d/wsl-cuda.conf
        sudo ldconfig
        echo "  Added $WSL_CUDA to ldconfig"
    else
        echo "  Already configured"
    fi
    # Also export for current session
    export LD_LIBRARY_PATH="$WSL_CUDA:${LD_LIBRARY_PATH:-}"
    # Persist in bashrc
    if ! grep -q "wsl/lib" ~/.bashrc; then
        echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
        echo "  Added LD_LIBRARY_PATH to ~/.bashrc"
    fi
else
    echo "  WARNING: $WSL_CUDA not found — CUDA may not work"
fi

# 3. Python venv
echo "[3/6] Creating Python virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools

# 4. PyTorch with CUDA 12.4
echo "[4/6] Installing PyTorch (CUDA 12.4)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. Training stack
echo "[5/6] Installing training dependencies..."
pip install \
    transformers==4.47.1 \
    datasets \
    peft \
    trl \
    accelerate \
    bitsandbytes \
    sentencepiece \
    protobuf \
    evaluate \
    scipy \
    matplotlib \
    tqdm \
    huggingface_hub

# 6. Configure HuggingFace cache to E: drive
echo "[6/6] Configuring HuggingFace cache directory..."
mkdir -p "$CACHE_DIR/huggingface"
# Persist env vars
if ! grep -q "HF_HOME" ~/.bashrc; then
    echo "export HF_HOME=$CACHE_DIR/huggingface" >> ~/.bashrc
    echo "export HF_DATASETS_CACHE=$CACHE_DIR/huggingface/datasets" >> ~/.bashrc
    echo "  Added HF_HOME to ~/.bashrc"
fi
export HF_HOME="$CACHE_DIR/huggingface"
export HF_DATASETS_CACHE="$CACHE_DIR/huggingface/datasets"

# Verify GPU
echo ""
echo "=== GPU Verification ==="
python - << 'EOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"BF16 support: {torch.cuda.is_bf16_supported()}")
else:
    print("WARNING: CUDA not available. Check LD_LIBRARY_PATH and driver.")
    print("Try: source ~/.bashrc && python -c 'import torch; print(torch.cuda.is_available())'")
EOF

echo ""
echo "=== Setup complete ==="
echo "Activate venv with: source $VENV_DIR/bin/activate"
echo "Then run: python data/download_tashkeela.py"
