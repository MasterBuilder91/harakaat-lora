"""
metrics.py — DER and WER for Arabic diacritization.

DER (Diacritic Error Rate): proportion of diacritizable characters where the
predicted diacritic differs from the gold standard.

WER (Word Error Rate): proportion of words where the full predicted vocalized
form differs from the gold standard word.
"""
import re
from typing import Optional

# Arabic diacritic Unicode range
HARAKAT = re.compile(r'[ً-ٰٟ]')
DIACRITIC_SET = frozenset('ًٌٍَُِّْٰ')

# Arabic letter range (excludes harakat and punctuation)
ARABIC_LETTER = re.compile(r'[ء-غف-يٱ-ڷ]')


def get_diacritics_sequence(text: str) -> list[Optional[str]]:
    """
    For each Arabic letter in text, return the diacritic following it (or None).
    Returns a list aligned to the stripped letter sequence.
    """
    result = []
    chars = list(text)
    i = 0
    while i < len(chars):
        c = chars[i]
        if ARABIC_LETTER.match(c):
            # Collect following diacritics (can be multiple, e.g. shadda + vowel)
            diac = ''
            j = i + 1
            while j < len(chars) and chars[j] in DIACRITIC_SET:
                diac += chars[j]
                j += 1
            result.append(diac if diac else None)
            i = j
        elif c in DIACRITIC_SET:
            # Orphan diacritic (before letter) — skip
            i += 1
        else:
            i += 1
    return result


def compute_der(predictions: list[str], references: list[str]) -> dict:
    """
    Compute Diacritic Error Rate.

    Args:
        predictions: list of predicted vocalized strings
        references:  list of gold vocalized strings

    Returns:
        dict with 'der' (float 0-1), 'errors', 'total'
    """
    total = 0
    errors = 0

    for pred, ref in zip(predictions, references):
        pred_stripped = HARAKAT.sub('', pred)
        ref_stripped = HARAKAT.sub('', ref)

        # Align on stripped form — if they differ the prediction garbled letters
        if pred_stripped != ref_stripped:
            # Count all diacritizable positions as errors
            ref_diacs = get_diacritics_sequence(ref)
            total += len(ref_diacs)
            errors += len(ref_diacs)
            continue

        pred_diacs = get_diacritics_sequence(pred)
        ref_diacs = get_diacritics_sequence(ref)

        for p, r in zip(pred_diacs, ref_diacs):
            total += 1
            if p != r:
                errors += 1

    der = errors / total if total > 0 else 0.0
    return {"der": round(der, 4), "errors": errors, "total": total}


def compute_wer(predictions: list[str], references: list[str]) -> dict:
    """
    Compute Word Error Rate on the vocalized forms.
    Words must match exactly (including all diacritics).
    """
    total_words = 0
    error_words = 0

    for pred, ref in zip(predictions, references):
        pred_words = pred.strip().split()
        ref_words = ref.strip().split()
        # Align by min length (substitution-only WER for simplicity)
        for pw, rw in zip(pred_words, ref_words):
            total_words += 1
            if pw != rw:
                error_words += 1
        # Extra/missing words count as errors
        diff = abs(len(pred_words) - len(ref_words))
        total_words += diff
        error_words += diff

    wer = error_words / total_words if total_words > 0 else 0.0
    return {"wer": round(wer, 4), "errors": error_words, "total": total_words}


if __name__ == "__main__":
    # Quick self-test
    ref = "كَتَبَ الْوَلَدُ الدَّرْسَ"
    pred_good = "كَتَبَ الْوَلَدُ الدَّرْسَ"
    pred_bad = "كِتَبَ الْوَلَدِ الدَّرْسَ"

    print("Self-test:")
    print(f"  Perfect match — DER: {compute_der([pred_good], [ref])['der']}")
    print(f"  Two errors    — DER: {compute_der([pred_bad], [ref])['der']:.4f}")
    print(f"  Perfect match — WER: {compute_wer([pred_good], [ref])['wer']}")
    print(f"  Two errors    — WER: {compute_wer([pred_bad], [ref])['wer']:.4f}")
