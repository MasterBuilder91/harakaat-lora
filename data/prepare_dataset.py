"""
prepare_dataset.py
Strips diacritics from Tashkeela to build (input, target) pairs.
Splits into train/val/test and saves as JSONL.
"""
import json
import re
import random
import os
from pathlib import Path
from collections import defaultdict

# Arabic diacritic Unicode ranges
HARAKAT = re.compile(r'[ً-ٰٟ]')

# Unicode diacritic set for evaluation
DIACRITIC_CHARS = set('ًٌٍَُِّْٰٕٓٔ')

PROJ_DIR = Path("/mnt/e/harakaat-lora")
DATA_DIR = PROJ_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

os.environ["HF_HOME"] = str(PROJ_DIR / ".cache" / "huggingface")
os.environ["HF_DATASETS_CACHE"] = str(PROJ_DIR / ".cache" / "huggingface" / "datasets")


def strip_diacritics(text: str) -> str:
    return HARAKAT.sub('', text)


def has_diacritics(text: str) -> bool:
    return bool(HARAKAT.search(text))


def count_diacritizable(text: str) -> int:
    """Count Arabic letters that could carry a diacritic."""
    return sum(1 for c in text if '؀' <= c <= 'ۿ' and c not in DIACRITIC_CHARS)


def clean_line(line: str) -> str:
    line = line.strip()
    # Remove XML/HTML tags if any
    line = re.sub(r'<[^>]+>', '', line)
    # Normalize multiple spaces
    line = re.sub(r'  +', ' ', line)
    return line


def build_instruction(input_text: str, output_text: str) -> dict:
    return {
        "instruction": "أضف التشكيل الكامل (الحركات) للنص العربي التالي:",
        "input": input_text,
        "output": output_text,
    }


def load_from_hf() -> list[tuple[str, str]]:
    """Load from saved HuggingFace dataset."""
    from datasets import load_from_disk
    save_path = RAW_DIR / "tashkeela_hf"
    if not save_path.exists():
        raise FileNotFoundError(f"Run download_tashkeela.py first. Expected: {save_path}")

    ds = load_from_disk(str(save_path))
    print(f"Loaded HF dataset: {ds}")

    pairs = []
    split = list(ds.keys())[0]
    data = ds[split]

    # Find the text column
    sample = data[0]
    text_col = None
    for col in ['text', 'sentence', 'content', 'vocalized', 'tashkeel']:
        if col in sample:
            text_col = col
            break
    if text_col is None:
        # Use first string column
        text_col = next(k for k, v in sample.items() if isinstance(v, str))

    print(f"Using column: '{text_col}'")

    for item in data:
        text = item[text_col]
        if not isinstance(text, str):
            continue
        lines = text.split('\n')
        for line in lines:
            line = clean_line(line)
            if len(line) < 10:
                continue
            if not has_diacritics(line):
                continue
            if count_diacritizable(line) < 5:
                continue
            stripped = strip_diacritics(line)
            pairs.append((stripped, line))

    return pairs


def load_from_github() -> list[tuple[str, str]]:
    """Load from raw GitHub text files (fallback)."""
    pairs = []
    text_files = list(RAW_DIR.rglob("*.txt"))
    print(f"Loading from {len(text_files)} text files...")

    for fpath in text_files:
        try:
            text = fpath.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue

        for line in text.split('\n'):
            line = clean_line(line)
            if len(line) < 10:
                continue
            if not has_diacritics(line):
                continue
            if count_diacritizable(line) < 5:
                continue
            stripped = strip_diacritics(line)
            pairs.append((stripped, line))

    return pairs


def split_and_save(pairs: list[tuple[str, str]], seed: int = 42):
    """Split into train/val/test and save as JSONL."""
    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_test = max(500, int(n * 0.05))
    n_val = max(500, int(n * 0.05))
    n_train = n - n_val - n_test

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    for split_name, split_pairs in splits.items():
        out_path = DATA_DIR / f"{split_name}.jsonl"
        with open(out_path, 'w', encoding='utf-8') as f:
            for inp, out in split_pairs:
                record = build_instruction(inp, out)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"  {split_name:5s}: {len(split_pairs):6,} examples → {out_path}")

    return splits


def main():
    print("=== Tashkeela Dataset Preparation ===")
    print()

    # Try HF first, fall back to GitHub files
    pairs = None
    try:
        pairs = load_from_hf()
        print(f"Loaded {len(pairs):,} pairs from HuggingFace dataset")
    except FileNotFoundError:
        print("HF dataset not found, trying raw GitHub files...")
        pairs = load_from_github()
        print(f"Loaded {len(pairs):,} pairs from raw text files")

    if not pairs:
        print("ERROR: No data found. Run download_tashkeela.py first.")
        return

    # Dedup
    seen = set()
    unique = []
    for inp, out in pairs:
        if inp not in seen:
            seen.add(inp)
            unique.append((inp, out))
    print(f"After dedup: {len(unique):,} unique examples")

    # Length filter: keep sequences that fit in 512 tokens approx
    # Rough heuristic: Arabic char ≈ 1.5 tokens on average
    filtered = [(i, o) for i, o in unique if len(o) <= 300]
    print(f"After length filter (≤300 chars): {len(filtered):,} examples")

    # Stats
    lengths = [len(o) for _, o in filtered]
    print(f"Output length — mean: {sum(lengths)/len(lengths):.0f}, "
          f"max: {max(lengths)}, min: {min(lengths)}")

    print()
    print("Creating train/val/test splits...")
    split_and_save(filtered)

    print()
    print("Done. Next: python training/train.py")


if __name__ == "__main__":
    main()
