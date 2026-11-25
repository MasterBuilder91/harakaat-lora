"""
verify_dataset.py — Sanity check on prepared splits.
Run after prepare_dataset.py to confirm data quality.
"""
import json
import re
from pathlib import Path

PROJ_DIR = Path("/mnt/e/harakaat-lora")
DATA_DIR = PROJ_DIR / "data"
HARAKAT = re.compile(r'[ً-ٰٟ]')


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def der_sample(input_text, output_text):
    """Check that stripping output gives back input."""
    stripped = HARAKAT.sub('', output_text)
    return stripped == input_text


for split in ['train', 'val', 'test']:
    path = DATA_DIR / f"{split}.jsonl"
    if not path.exists():
        print(f"Missing: {path}")
        continue

    records = load_jsonl(path)
    n = len(records)
    n_valid = sum(1 for r in records if der_sample(r['input'], r['output']))
    has_diac = sum(1 for r in records if HARAKAT.search(r['output']))
    lengths = [len(r['output']) for r in records]

    print(f"\n{split.upper()} ({n:,} examples)")
    print(f"  Input = strip(output):  {n_valid}/{n} ({100*n_valid/n:.1f}%)")
    print(f"  Has diacritics:          {has_diac}/{n}")
    print(f"  Output length — avg: {sum(lengths)/len(lengths):.0f}, max: {max(lengths)}")
    print(f"  Sample input:  {records[0]['input'][:80]}")
    print(f"  Sample output: {records[0]['output'][:80]}")
