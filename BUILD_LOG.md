# BUILD LOG — harakaat-lora

> Every error, root cause, and fix. No failures hidden.

---

## 2026-11-25 — Initial Environment Assessment

**Hardware:** RTX 3060 12 GB, 60 GB RAM, Windows 11 + WSL2 Ubuntu 24.04

**Disk space check:**
- C: drive: 44.7 GB free — **NOT enough** for 60-80 GB of models + data
- E: drive: 1335 GB free — **sufficient**
- **Decision:** All large files (model cache, data, venv) go on E: via `/mnt/e/harakaat-lora`

**WSL2 status:**
- Ubuntu 24.04 LTS, WSL version 3.0.5
- nvidia-smi: RTX 3060 visible, CUDA 13.3 (driver)
- `/dev/dxg` present, CUDA libs at `/usr/lib/wsl/lib/libcuda.so.1.1`
- `/dev/nvidia*` nodes NOT present — this is normal for WSL2; PyTorch finds the GPU via libcuda.so

**pip status:**
- `pip` not installed in WSL Ubuntu 24.04 (Python 3.12 uses `python3 -m ensurepip`)
- **Fix:** Install via `sudo apt install python3-pip python3-venv`

**nvcc status:**
- Not installed — no CUDA toolkit in WSL
- **Not needed:** PyTorch ships with its own CUDA libraries; only the driver libcuda.so is required
- **Verification needed:** Run `import torch; torch.cuda.is_available()` after installing PyTorch

---

## 2026-11-25 — Environment Setup

### Python + venv
```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git
python3 -m venv /mnt/e/harakaat-lora/venv
source /mnt/e/harakaat-lora/venv/bin/activate
```

### PyTorch installation (CUDA 12.4 bundle)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### HuggingFace + training stack
```bash
pip install transformers==4.47.1 datasets peft trl accelerate bitsandbytes
pip install sentencepiece protobuf evaluate
```

**Known issue with bitsandbytes on WSL2:**
bitsandbytes requires CUDA to be found at import time. If `import bitsandbytes` fails:
- Check `LD_LIBRARY_PATH` includes `/usr/lib/wsl/lib`
- Run: `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`
- Permanent fix: add to `~/.bashrc`

---

## 2026-11-25 — Dataset

**Source:** `arbml/tashkeela` on HuggingFace
**License:** MIT ✓

**Arabic diacritic Unicode range:**
- Fathah: U+064E
- Kasra: U+0650
- Dammah: U+064F
- Sukun: U+0652
- Shadda: U+0651
- Tanwin fath: U+064B
- Tanwin damm: U+064C
- Tanwin kasr: U+064D
- Alef above: U+0670 (superscript alef)

**Stripping function:**
```python
import re
HARAKAT = re.compile(r'[ً-ٰٟ]')
def strip_diacritics(text): return HARAKAT.sub('', text)
```

---

## Errors & Fixes (ongoing)

| # | Error | Cause | Fix |
|---|---|---|---|
| 1 | `No module named pip` | Ubuntu 24.04 ships without pip | `sudo apt install python3-pip python3-venv` |
| 2 | (pending) | | |

*Updated as errors are encountered during training.*

---

## Training Run (pending)

- Start time: TBD
- End time: TBD
- Final train loss: TBD
- Final val loss: TBD
- Steps: TBD

---

## OOM Prevention Settings

If CUDA OOM occurs on 12 GB VRAM, try in order:
1. Reduce `max_seq_length` from 512 → 384
2. Increase `gradient_accumulation_steps` (keep effective batch = 16)
3. Reduce LoRA `r` from 16 → 8
4. Disable `fp16`, use `bf16` if supported (Ampere/RTX 30xx supports BF16)
5. Fallback to `Qwen/Qwen2.5-3B-Instruct`

RTX 3060 (Ampere) **does support BF16** — prefer it over FP16 for numerical stability.
