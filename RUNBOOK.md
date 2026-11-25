# RUNBOOK — exact commands to run from scratch

> Copy-paste these in order inside WSL2 Ubuntu terminal.
> Prerequisites: WSL2 running, RTX 3060 driver installed (nvidia-smi should work).

---

## Step 0 — Open WSL and go to project

```bash
# In Windows: open Start → "Ubuntu" or run: wsl -d Ubuntu
cd /mnt/e/harakaat-lora
```

---

## Step 1 — System setup (one-time)

```bash
bash scripts/setup.sh
```

This installs: `pip`, `venv`, PyTorch (CUDA 12.4), transformers, peft, trl, bitsandbytes.
Takes ~5–10 min on first run (downloads ~2 GB).

Then activate the venv:
```bash
source /mnt/e/harakaat-lora/venv/bin/activate
```

Add to `~/.bashrc` for future sessions:
```bash
echo 'source /mnt/e/harakaat-lora/venv/bin/activate' >> ~/.bashrc
echo 'export HF_HOME=/mnt/e/harakaat-lora/.cache/huggingface' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

---

## Step 2 — Verify GPU

```bash
python - << 'EOF'
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU:  {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"BF16: {torch.cuda.is_bf16_supported()}")
EOF
```

Expected output:
```
CUDA: True
GPU:  NVIDIA GeForce RTX 3060
VRAM: 12.0 GB
BF16: True
```

---

## Step 3 — Download Tashkeela dataset

```bash
python data/download_tashkeela.py
```

Downloads ~75 MB. Saves to `data/raw/tashkeela_hf/`.

---

## Step 4 — Prepare dataset

```bash
python data/prepare_dataset.py
python data/verify_dataset.py
```

Creates `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`.

---

## Step 5 — Train (8–14 hours)

```bash
# Run in a tmux/screen session so it survives terminal close
tmux new -s training
source /mnt/e/harakaat-lora/venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export HF_HOME=/mnt/e/harakaat-lora/.cache/huggingface
cd /mnt/e/harakaat-lora
python training/train.py
```

Watch GPU usage in another terminal:
```bash
watch -n 2 nvidia-smi
```

If OOM error:
1. Edit `training/train.py` → reduce `MAX_SEQ_LENGTH = 384`
2. If still OOM → `LORA_R = 8`
3. If still OOM → `MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"`

---

## Step 6 — Evaluate

```bash
python evaluation/evaluate.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter outputs/harakaat-adapter \
  --test-file data/test.jsonl \
  --output results/evaluation_results.json
```

---

## Step 7 — Merge + quantize + run with Ollama

```bash
# Install Ollama if not installed:
curl -fsSL https://ollama.com/install.sh | sh

# Run the merge + GGUF conversion script:
bash inference/merge_and_quantize.sh
```

This will:
1. Merge the LoRA adapter into the base model weights (~30 min on CPU, uses ~15 GB RAM)
2. Clone llama.cpp and convert to GGUF F16
3. Quantize to Q4_K_M (~4.1 GB file)
4. Register as an Ollama model

```bash
# Test it:
ollama run harakaat "أضف التشكيل الكامل للنص التالي: كتب الولد الدرس"
# Expected: كَتَبَ الْوَلَدُ الدَّرْسَ

ollama run harakaat "أضف التشكيل: قرأ العلماء الكتب القديمة"
# Expected: قَرَأَ الْعُلَمَاءُ الْكُتُبَ الْقَدِيمَةَ
```

---

## Disk space by step

| Step | Space needed | Where |
|---|---|---|
| PyTorch + deps | ~4 GB | `/mnt/e/harakaat-lora/venv/` |
| Tashkeela dataset | ~200 MB | `/mnt/e/harakaat-lora/data/` |
| Qwen2.5-7B model (HF) | ~15 GB | `/mnt/e/harakaat-lora/.cache/` |
| LoRA adapter | ~200 MB | `/mnt/e/harakaat-lora/outputs/` |
| Merged model (fp16) | ~15 GB | `/mnt/e/harakaat-lora/outputs/` |
| GGUF Q4_K_M | ~4.5 GB | `/mnt/e/harakaat-lora/outputs/` |
| **Total** | **~39 GB** | all on E: drive |
