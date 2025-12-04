#!/usr/bin/env python3
"""
Build a 'hard slice' challenge set from SNLI train split.

Hard definition:
  - hypothesis has NO negation words
  - lexical overlap(premise, hypothesis) < 0.2
  - hypothesis length > 10 tokens (optional, but used here)

Output JSONL format:
  {"premise": "...", "hypothesis": "...", "label": 0}

Usage:
  python build_snli_hard_challenge.py \
      --output snli_hard_challenge.jsonl \
      --max_per_label 5000 \
      --seed 42
"""

import argparse
import json
import random
import re

import datasets


NEGATION_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing", "nowhere",
    "neither", "nor", "n't", "cant", "can't", "dont", "don't",
}


def simple_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens


def has_negation(text):
    toks = simple_tokenize(text)
    return any(t in NEGATION_WORDS for t in toks)


def lexical_overlap(prem, hyp):
    prem_tokens = set(simple_tokenize(prem))
    hyp_tokens = set(simple_tokenize(hyp))
    if not prem_tokens or not hyp_tokens:
        return 0.0
    inter = len(prem_tokens & hyp_tokens)
    union = len(prem_tokens | hyp_tokens)
    return inter / union


def is_hard_example(ex):
    prem = ex["premise"]
    hyp = ex["hypothesis"]

    if has_negation(hyp):
        return False

    hyp_len = len(simple_tokenize(hyp))
    if hyp_len <= 10:
        return False

    overlap = lexical_overlap(prem, hyp)
    if overlap >= 0.2:
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL path for challenge set.")
    parser.add_argument("--max_per_label", type=int, default=5000,
                        help="Maximum number of examples per label to include.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading SNLI...")
    dataset = datasets.load_dataset("snli")

    # Filter out SNLI examples with label -1 (unlabeled / problematic)
    train = dataset["train"].filter(lambda ex: ex["label"] != -1)

    print(f"Train size after removing unlabeled examples: {len(train)}")

    # Collect hard examples by gold label
    by_label = {0: [], 1: [], 2: []}  # entailment, neutral, contradiction

    for ex in train:
        if is_hard_example(ex):
            lab = ex["label"]
            if lab in by_label:
                by_label[lab].append(ex)

    for lab in sorted(by_label.keys()):
        print(f"Label {lab}: found {len(by_label[lab])} hard examples")

    # Sample up to max_per_label per label to keep set manageable and balanced
    selected = []
    for lab, exs in by_label.items():
        random.shuffle(exs)
        keep = exs[: args.max_per_label]
        selected.extend(keep)
        print(f"Using {len(keep)} examples for label {lab}")

    random.shuffle(selected)
    print(f"Total challenge examples: {len(selected)}")

    # Write JSONL
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in selected:
            out = {
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "label": int(ex["label"]),
            }
            f.write(json.dumps(out))
            f.write("\n")

    print(f"Wrote challenge set to {args.output}")


if __name__ == "__main__":
    main()
