# harakaat-lora

Fine-tuning **Qwen2.5-7B-Instruct** with QLoRA to add full Arabic diacritics (harakaat / تشكيل) to unvocalized classical Arabic text.

> Portfolio project demonstrating: dataset curation, instruction fine-tuning, QLoRA on consumer GPU, GGUF quantization, and local inference with Ollama.

---

## Problem

Classical Arabic is normally written without diacritical marks (harakat: fatḥa, kasra, ḍamma, sukūn, shadda, tanwīn). Correct vocalization requires deep linguistic and contextual knowledge — making it a good test of a language model's Arabic comprehension.

| Input (unvocalized) | Output (vocalized) |
|---|---|
| كتب الولد الدرس | كَتَبَ الْوَلَدُ الدَّرْسَ |
| قرأ العلماء الكتب | قَرَأَ الْعُلَمَاءُ الْكُتُبَ |

---

## Dataset

**Tashkeela** — a freely available, fully vocalized classical Arabic corpus.

| Property | Value |
|---|---|
| Source | [Tashkeela on HuggingFace](https://huggingface.co/datasets/arbml/tashkeela) / [GitHub](https://github.com/Mazzawi/Tashkeela) |
| License | **MIT** |
| Content | ~75 MB of classical Arabic prose with complete diacritics |
| Splits | 90% train / 5% validation / 5% test |
| Format | Instruction pairs: `{input: stripped_text, output: vocalized_text}` |

The dataset was built by stripping all Unicode diacritic characters (U+064B–U+065F, U+0670) from the vocalized text to create inputs, keeping the original as the target.

---

## Model

**[Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)** — Apache 2.0 license.

**Why this model:**
- Multilingual BPE tokenizer with strong Arabic coverage
- 7B parameters fits QLoRA (4-bit) in 12 GB VRAM
- State-of-the-art multilingual instruction following
- Widely tested community QLoRA configs available

**QLoRA config:**
- Quantization: 4-bit NF4 with double quantization
- LoRA rank: 16, alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj
- Max sequence length: 512 tokens
- Batch size: 1 + gradient accumulation 16 (effective batch 16)

---

## Results

| Model | DER (↓) | WER (↓) |
|---|---|---|
| Qwen2.5-7B-Instruct (base, zero-shot) | *TBD* | *TBD* |
| + QLoRA fine-tuned (this repo) | *TBD* | *TBD* |

*Results updated after training completes. See `results/` directory.*

**Metric definitions:**
- **DER (Diacritic Error Rate):** fraction of diacritizable characters where the predicted diacritic differs from gold
- **WER (Word Error Rate):** fraction of words where the full predicted form (with all diacritics) differs from gold

---

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 12 GB VRAM |
| RAM | 60 GB |
| OS | Windows 11 + WSL2 (Ubuntu 24.04) |
| CUDA | 13.3 (via NVIDIA WSL2 driver) |
| Training time | ~*TBD* hours (3 epochs on Tashkeela) |

---

## How to Run

### Prerequisites

- WSL2 with Ubuntu 24.04
- NVIDIA GPU (12 GB+ VRAM recommended)
- ~60 GB free disk space on a non-system drive

### 1. Setup environment

```bash
cd /mnt/e/harakaat-lora
bash scripts/setup.sh
source venv/bin/activate
```

### 2. Download and prepare data

```bash
python data/download_tashkeela.py
python data/prepare_dataset.py
```

### 3. Train

```bash
python training/train.py
# Loss curves saved to results/loss_curve.png
# Adapter saved to outputs/harakaat-adapter/
```

### 4. Evaluate

```bash
python evaluation/evaluate.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter outputs/harakaat-adapter \
  --test-file data/test.jsonl \
  --output results/evaluation_results.json
```

### 5. Merge + quantize to GGUF

```bash
bash inference/merge_and_quantize.sh
```

### 6. Run with Ollama

```bash
ollama create harakaat -f inference/Modelfile
ollama run harakaat "أضف التشكيل الكامل للنص التالي: كتب الولد الدرس"
```

---

## Repository Layout

```
harakaat-lora/
├── README.md
├── BUILD_LOG.md           # Every error, cause, and fix
├── data/
│   ├── download_tashkeela.py
│   ├── prepare_dataset.py
│   └── verify_dataset.py
├── training/
│   ├── train.py           # QLoRA fine-tuning
│   ├── training_config.yaml
│   └── requirements.txt
├── evaluation/
│   ├── evaluate.py        # DER + WER evaluation
│   └── metrics.py         # Metric implementations
├── inference/
│   ├── merge_and_quantize.sh
│   └── Modelfile          # Ollama model definition
├── results/               # Loss curves, eval results (gitignored binaries)
└── scripts/
    └── setup.sh           # WSL environment setup
```

---

## Citations

```bibtex
@inproceedings{mazzawi2017tashkeela,
  title={Tashkeela: Novel corpus of Arabic vocalized texts},
  author={Mazzawi, Husam and Hamou-Lhadj, Abdelwahab},
  booktitle={Proceedings of the Workshop on Arabic Natural Language Processing},
  year={2017}
}

@misc{qwen2.5,
  title={Qwen2.5 Technical Report},
  author={Qwen Team},
  year={2024},
  url={https://huggingface.co/Qwen/Qwen2.5-7B-Instruct}
}
```
