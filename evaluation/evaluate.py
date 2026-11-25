"""
evaluate.py — Compare base model vs QLoRA fine-tuned model on the test set.

Usage:
    python evaluation/evaluate.py \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --adapter outputs/harakaat-adapter \
        --test-file data/test.jsonl \
        --output results/evaluation_results.json

Output:
    - results/evaluation_results.json  (full metrics)
    - prints a results table to stdout
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Optional

PROJ_DIR = Path("/mnt/e/harakaat-lora")
os.environ["HF_HOME"] = str(PROJ_DIR / ".cache" / "huggingface")
sys.path.insert(0, str(PROJ_DIR))

from evaluation.metrics import compute_der, compute_wer

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "أنت نظام متخصص في إضافة التشكيل الكامل للنصوص العربية الفصحى."


def load_test_data(path: str, max_examples: int = 500) -> list[dict]:
    records = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records[:max_examples]


def build_prompt(record: dict) -> str:
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{record['instruction']}\n{record['input']}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    batch_size: int = 4,
) -> list[str]:
    outputs = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
            batch = prompts[i:i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)

            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

            for j, (inp_ids, out_ids) in enumerate(zip(inputs["input_ids"], generated)):
                new_tokens = out_ids[len(inp_ids):]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                text = text.strip().split("<|im_end|>")[0].strip()
                outputs.append(text)

    return outputs


def load_model(model_id: str, adapter_path: Optional[str] = None):
    logger.info(f"Loading {'base' if adapter_path is None else 'fine-tuned'} model: {model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if adapter_path and Path(adapter_path).exists():
        model = PeftModel.from_pretrained(model, adapter_path)
        logger.info(f"Loaded LoRA adapter from {adapter_path}")

    return model, tokenizer


def evaluate_model(
    model,
    tokenizer,
    records: list[dict],
    label: str,
) -> dict:
    prompts = [build_prompt(r) for r in records]
    references = [r["output"] for r in records]

    logger.info(f"Evaluating {label} on {len(records)} examples...")
    predictions = generate_batch(model, tokenizer, prompts)

    der_result = compute_der(predictions, references)
    wer_result = compute_wer(predictions, references)

    # Save sample predictions
    samples = []
    for i in range(min(5, len(records))):
        samples.append({
            "input": records[i]["input"],
            "reference": references[i],
            "prediction": predictions[i],
        })

    return {
        "model": label,
        "n_examples": len(records),
        "DER": der_result["der"],
        "WER": wer_result["wer"],
        "der_errors": der_result["errors"],
        "der_total": der_result["total"],
        "wer_errors": wer_result["errors"],
        "wer_total": wer_result["total"],
        "samples": samples,
    }


def print_results_table(results: list[dict]):
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Model':<35} {'DER':>8} {'WER':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<35} {r['DER']:>7.1%} {r['WER']:>7.1%}")
    print("=" * 60)
    print("\nDER = Diacritic Error Rate (lower is better)")
    print("WER = Word Error Rate      (lower is better)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", default=str(PROJ_DIR / "outputs" / "harakaat-adapter"))
    parser.add_argument("--test-file", default=str(PROJ_DIR / "data" / "test.jsonl"))
    parser.add_argument("--output", default=str(PROJ_DIR / "results" / "evaluation_results.json"))
    parser.add_argument("--max-examples", type=int, default=500)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        logger.error("CUDA required for evaluation. Check LD_LIBRARY_PATH.")
        sys.exit(1)

    records = load_test_data(args.test_file, args.max_examples)
    logger.info(f"Test examples: {len(records)}")

    all_results = []

    # Evaluate base model
    model, tokenizer = load_model(args.base_model, adapter_path=None)
    base_result = evaluate_model(model, tokenizer, records, label="Qwen2.5-7B base (zero-shot)")
    all_results.append(base_result)
    del model
    torch.cuda.empty_cache()

    # Evaluate fine-tuned model
    model, tokenizer = load_model(args.base_model, adapter_path=args.adapter)
    ft_result = evaluate_model(model, tokenizer, records, label="Qwen2.5-7B + harakaat LoRA")
    all_results.append(ft_result)
    del model
    torch.cuda.empty_cache()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {output_path}")

    print_results_table(all_results)
    print(f"\nFull results: {output_path}")

    # Print samples
    print("\n--- Sample predictions (fine-tuned) ---")
    for s in ft_result.get("samples", [])[:3]:
        print(f"  Input:  {s['input'][:60]}")
        print(f"  Gold:   {s['reference'][:60]}")
        print(f"  Pred:   {s['prediction'][:60]}")
        print()


if __name__ == "__main__":
    main()
