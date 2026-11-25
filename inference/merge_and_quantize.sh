#!/usr/bin/env bash
# merge_and_quantize.sh
# Merges the QLoRA adapter into the base model, then quantizes to GGUF (Q4_K_M).
# Requirements: llama.cpp (auto-installed below), Ollama
#
# Usage (from project root, with venv active):
#   bash inference/merge_and_quantize.sh

set -euo pipefail

PROJ_DIR="/mnt/e/harakaat-lora"
ADAPTER_DIR="$PROJ_DIR/outputs/harakaat-adapter"
MERGED_DIR="$PROJ_DIR/outputs/harakaat-merged"
GGUF_DIR="$PROJ_DIR/outputs/harakaat-gguf"
LLAMA_CPP_DIR="$PROJ_DIR/llama.cpp"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"

export HF_HOME="$PROJ_DIR/.cache/huggingface"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

source "$PROJ_DIR/venv/bin/activate"

echo "=== Step 1: Merge LoRA adapter into base model ==="
python - << 'EOF'
import os, sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJ_DIR = Path("/mnt/e/harakaat-lora")
os.environ["HF_HOME"] = str(PROJ_DIR / ".cache" / "huggingface")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_DIR = str(PROJ_DIR / "outputs" / "harakaat-adapter")
MERGED_DIR = str(PROJ_DIR / "outputs" / "harakaat-merged")

print(f"Loading base model: {MODEL_ID}")
# Load in float16 for merging (not 4-bit — merge requires full weights)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu",          # Load to CPU for merge (saves VRAM)
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

print(f"Loading adapter from {ADAPTER_DIR}")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)

print("Merging adapter weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {MERGED_DIR}")
model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
print("Merge complete.")
EOF

echo ""
echo "=== Step 2: Clone / build llama.cpp ==="
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    git clone https://github.com/ggerganov/llama.cpp "$LLAMA_CPP_DIR"
fi

cd "$LLAMA_CPP_DIR"
git pull --ff-only

# Build convert_hf_to_gguf.py deps
pip install -q gguf sentencepiece

echo ""
echo "=== Step 3: Convert to GGUF (F16 first) ==="
mkdir -p "$GGUF_DIR"
F16_PATH="$GGUF_DIR/harakaat-f16.gguf"

python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
    "$MERGED_DIR" \
    --outfile "$F16_PATH" \
    --outtype f16

echo "F16 GGUF saved to $F16_PATH"

echo ""
echo "=== Step 4: Quantize to Q4_K_M ==="
# Build llama.cpp quantize tool if not built
if [ ! -f "$LLAMA_CPP_DIR/build/bin/llama-quantize" ]; then
    mkdir -p "$LLAMA_CPP_DIR/build"
    cd "$LLAMA_CPP_DIR/build"
    cmake .. -DGGML_CUDA=ON 2>/dev/null || cmake .. -DLLAMA_CUBLAS=ON 2>/dev/null || cmake ..
    make -j4 llama-quantize
fi

Q4_PATH="$GGUF_DIR/harakaat-q4_k_m.gguf"
"$LLAMA_CPP_DIR/build/bin/llama-quantize" \
    "$F16_PATH" \
    "$Q4_PATH" \
    Q4_K_M

echo "Q4_K_M GGUF saved to $Q4_PATH"
echo "File size: $(du -sh $Q4_PATH | cut -f1)"

echo ""
echo "=== Step 5: Create Ollama model ==="
# Copy Modelfile with the GGUF path substituted
MODELFILE_PATH="$PROJ_DIR/inference/Modelfile"
sed "s|GGUF_PATH_PLACEHOLDER|$Q4_PATH|g" "$MODELFILE_PATH" > /tmp/harakaat_Modelfile

echo "Creating Ollama model 'harakaat'..."
ollama create harakaat -f /tmp/harakaat_Modelfile

echo ""
echo "=== Done! ==="
echo ""
echo "Run the model with:"
echo "  ollama run harakaat \"أضف التشكيل الكامل للنص التالي: كتب الولد الدرس\""
echo ""
echo "Files:"
echo "  Merged (fp16):  $MERGED_DIR"
echo "  GGUF F16:       $F16_PATH"
echo "  GGUF Q4_K_M:    $Q4_PATH  (this is what Ollama uses)"
