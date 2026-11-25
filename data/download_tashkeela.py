"""
download_tashkeela.py
Downloads the Tashkeela dataset from HuggingFace Hub.
License: MIT (Mazzawi & Hamou-Lhadj, 2017)
"""
import os
import sys
from pathlib import Path

# Point cache to E: drive
PROJ_DIR = Path("/mnt/e/harakaat-lora")
CACHE_DIR = PROJ_DIR / ".cache" / "huggingface"
DATA_DIR = PROJ_DIR / "data" / "raw"

os.environ["HF_HOME"] = str(CACHE_DIR)
os.environ["HF_DATASETS_CACHE"] = str(CACHE_DIR / "datasets")

from datasets import load_dataset
from huggingface_hub import snapshot_download


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading arbml/tashkeela from HuggingFace...")
    print("License: MIT")
    print(f"Saving to: {DATA_DIR}")
    print()

    try:
        ds = load_dataset("arbml/tashkeela", cache_dir=str(CACHE_DIR / "datasets"))
        print(f"Dataset loaded: {ds}")
        print(f"Splits: {list(ds.keys())}")

        # Inspect first example
        split = list(ds.keys())[0]
        first = ds[split][0]
        print(f"\nFirst example fields: {list(first.keys())}")
        for k, v in first.items():
            val = str(v)[:200]
            print(f"  {k}: {val}")

        # Save to disk for offline use
        save_path = DATA_DIR / "tashkeela_hf"
        ds.save_to_disk(str(save_path))
        print(f"\nSaved dataset to {save_path}")

    except Exception as e:
        print(f"ERROR loading arbml/tashkeela: {e}")
        print("\nFallback: trying msamsami/tashkeela...")
        try:
            ds = load_dataset("msamsami/tashkeela", cache_dir=str(CACHE_DIR / "datasets"))
            split = list(ds.keys())[0]
            first = ds[split][0]
            print(f"Dataset fields: {list(first.keys())}")
            save_path = DATA_DIR / "tashkeela_hf"
            ds.save_to_disk(str(save_path))
            print(f"Saved to {save_path}")
        except Exception as e2:
            print(f"ERROR: {e2}")
            print("\nManual fallback: download from GitHub")
            manual_download()
            return

    print("\nDone. Run: python data/prepare_dataset.py")


def manual_download():
    """Download raw Tashkeela files from GitHub as fallback."""
    import urllib.request
    import zipfile

    url = "https://github.com/Mazzawi/Tashkeela/archive/refs/heads/master.zip"
    zip_path = DATA_DIR / "tashkeela_github.zip"

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    print(f"Extracted to {DATA_DIR}")
    zip_path.unlink()

    # Find text files
    text_files = list(DATA_DIR.rglob("*.txt"))
    print(f"Found {len(text_files)} text files")


if __name__ == "__main__":
    main()
