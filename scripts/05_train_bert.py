#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_train_bert.py

Train BERT classifier with ground truth support for:
- Type-based dataset splitting
- Difficulty-based dataset splitting  
- Per-type detection reporting
- Per-difficulty detection reporting

Usage:
    python scripts/05_train_bert.py data1.jsonl data2.jsonl \
        --ground-truth-jsonl ground_truth.jsonl \
        --report-by-type \
        --report-by-difficulty \
        --split-stratify-by-type
"""

import argparse
import json
import os
import pathlib
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed as transformers_set_seed,
)


# =============================================================================
# Utility functions for path normalization
# =============================================================================
def _basename_any(p: str) -> str:
    """Extract basename from path, handling both Unix and Windows separators."""
    if not p:
        return ""
    return os.path.basename(p.replace("\\", "/"))


def _normalize_pkg_name(name: str) -> str:
    """
    Normalize package name for matching.
    Examples:
        aliyundrive-6.0.4__16.tgz -> aliyundrive-6.0.4.tgz
        2025-02-05-genesys-richmedia-v1.0.1.zip -> genesys-richmedia-v1.0.1.zip
    """
    name = _basename_any(name)
    # Remove date prefix like "2025-02-05-"
    name = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", name)
    # Remove trailing __N before extension
    name = re.sub(r"__\d+(?=\.)", "", name)
    return name


# =============================================================================
# Ground truth loading
# =============================================================================
def _load_ground_truth_map(gt_jsonl_path: str, key_mode: str = "malicious_types"):
    """
    Load ground truth JSONL and return mapping from package name to info.
    
    Returns:
        dict[str, dict]: Maps package name -> {"types": set[str], "difficulty": str}
    """
    pkg_to_info = defaultdict(lambda: {"types": set(), "difficulty": None})
    
    with open(gt_jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"Warning: ground_truth.jsonl parse error at line {line_no}: {e}")
                continue
            
            # Get package name from archive_name
            archive_name = obj.get("archive_name", "")
            if not archive_name:
                continue
            
            # Generate multiple key variants for matching
            keys = set()
            keys.add(archive_name)
            keys.add(_basename_any(archive_name))
            keys.add(_normalize_pkg_name(archive_name))
            
            # Extract difficulty from bin_label
            difficulty = obj.get("bin_label")
            
            # Extract types
            types = set()
            annotation = obj.get("annotation", {})
            
            if key_mode in ("malicious_types", "both"):
                mts = annotation.get("malicious_types", [])
                if isinstance(mts, list):
                    for t in mts:
                        if t and str(t).strip():
                            types.add(str(t).strip())
            
            if key_mode in ("type_bucket", "both"):
                tb = obj.get("type_bucket")
                if tb:
                    types.add(str(tb))
            
            # Store info for all key variants
            for k in keys:
                if k:
                    pkg_to_info[k]["types"].update(types)
                    if difficulty is not None:
                        pkg_to_info[k]["difficulty"] = difficulty
    
    print(f"[Ground Truth] Loaded {len(pkg_to_info)} package entries from {gt_jsonl_path}")
    return dict(pkg_to_info)


def _lookup_pkg_info(pkg_name: str, pkg_to_info: dict) -> dict:
    """Look up package info using multiple name variants."""
    if not pkg_to_info:
        return {"types": set(), "difficulty": None}
    
    # Try exact match first
    if pkg_name in pkg_to_info:
        return pkg_to_info[pkg_name]
    
    # Try basename
    basename = _basename_any(pkg_name)
    if basename in pkg_to_info:
        return pkg_to_info[basename]
    
    # Try normalized name
    normalized = _normalize_pkg_name(pkg_name)
    if normalized in pkg_to_info:
        return pkg_to_info[normalized]
    
    return {"types": set(), "difficulty": None}


# =============================================================================
# Per-type and per-difficulty reporting
# =============================================================================
def _per_type_recall_report(
    x_test, y_true, y_pred, pkg_to_info=None,
    title="Per-type recall on MALICIOUS in test set"
):
    """Report detection recall grouped by malicious type."""
    stats = defaultdict(lambda: {"tp": 0, "fn": 0, "n": 0})
    unknown_count = 0
    
    for r, yt, yp in zip(x_test, y_true, y_pred):
        if int(yt) != 1:  # Only count malicious samples
            continue
        
        # Get types from ground truth or row fields
        types = set()
        pkg_name = r.get("pkg", "")
        
        if pkg_to_info:
            info = _lookup_pkg_info(pkg_name, pkg_to_info)
            types = info.get("types", set())
        
        # Fallback to row fields
        if not types:
            mt = r.get("malicious_types")
            if isinstance(mt, list) and mt:
                types = {str(x).strip() for x in mt if str(x).strip()}
            else:
                tb = r.get("type_bucket")
                if tb:
                    types = {str(tb)}
        
        if not types:
            unknown_count += 1
            continue
        
        for t in types:
            stats[t]["n"] += 1
            if int(yp) == 1:
                stats[t]["tp"] += 1
            else:
                stats[t]["fn"] += 1
    
    rows = []
    for t, s in stats.items():
        n = s["n"]
        tp = s["tp"]
        fn = s["fn"]
        recall = tp / n if n else 0.0
        rows.append((t, n, tp, fn, recall))
    
    rows.sort(key=lambda x: (-x[1], x[0]))
    
    print("\n" + "=" * 80)
    print(title)
    if rows:
        df = pd.DataFrame(rows, columns=["type", "n_malicious", "TP", "FN", "recall"])
        pd.set_option("display.max_rows", 500)
        print(df.to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}"))
    else:
        print("No per-type data available.")
    if unknown_count:
        print(f"\nMalicious samples with unknown type: {unknown_count}")
    print("=" * 80 + "\n")


def _per_difficulty_recall_report(
    x_test, y_true, y_pred, pkg_to_info=None,
    title="Per-difficulty recall on MALICIOUS in test set"
):
    """Report detection recall grouped by difficulty level (bin_label)."""
    stats = defaultdict(lambda: {"tp": 0, "fn": 0, "n": 0})
    unknown_count = 0
    benign_fp = 0
    benign_tn = 0
    
    for r, yt, yp in zip(x_test, y_true, y_pred):
        if int(yt) != 1:
            if int(yp) == 1:
                benign_fp += 1
            else:
                benign_tn += 1
            continue
        
        difficulty = None
        pkg_name = r.get("pkg", "")
        
        if pkg_to_info:
            info = _lookup_pkg_info(pkg_name, pkg_to_info)
            difficulty = info.get("difficulty")
        
        # Fallback to row field
        if difficulty is None:
            difficulty = r.get("bin_label") or r.get("difficulty")
        
        if difficulty is None:
            unknown_count += 1
            difficulty = "unknown"
        
        stats[difficulty]["n"] += 1
        if int(yp) == 1:
            stats[difficulty]["tp"] += 1
        else:
            stats[difficulty]["fn"] += 1
    
    benign_total = benign_fp + benign_tn
    rows = []
    for d, s in stats.items():
        n = s["n"]
        tp = s["tp"]
        fn = s["fn"]
        recall = tp / n if n else 0.0
        if benign_total:
            precision = tp / (tp + benign_fp) if (tp + benign_fp) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            rows.append((d, n, tp, fn, recall, precision, f1))
        else:
            rows.append((d, n, tp, fn, recall))
    
    # Sort by difficulty name (try to sort p0-X, p34-X etc. logically)
    def sort_key(x):
        d = str(x[0])
        # Try to extract numeric prefix for sorting
        m = re.match(r"p(\d+)", d)
        if m:
            return (0, int(m.group(1)), d)
        return (1, 0, d)
    
    rows.sort(key=sort_key)
    
    print("\n" + "=" * 80)
    print(title)
    if rows:
        if benign_total:
            df = pd.DataFrame(
                rows,
                columns=["difficulty", "n_malicious", "TP", "FN", "recall", "precision", "f1"],
            )
        else:
            df = pd.DataFrame(rows, columns=["difficulty", "n_malicious", "TP", "FN", "recall"])
        pd.set_option("display.max_rows", 500)
        print(df.to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}"))
    else:
        print("No per-difficulty data available.")
    if unknown_count:
        print(f"\nMalicious samples with unknown difficulty: {unknown_count}")
    if benign_total:
        print(f"\nBenign samples used for precision/f1: {benign_total} (FP={benign_fp}, TN={benign_tn})")
    print("=" * 80 + "\n")


# =============================================================================
# Stratification helpers
# =============================================================================
def split_key(row, pkg_to_info=None, use_difficulty=False):
    """
    Generate a stratification key for a row.
    
    Args:
        row: Data row dict
        pkg_to_info: Ground truth mapping (optional)
        use_difficulty: If True, include difficulty in key
    
    Returns:
        str: Stratification key like "benign", "mal:type", "mal:type:difficulty"
    """
    if int(row.get("label", 0)) != 1:
        return "benign"
    
    types = set()
    difficulty = None
    pkg_name = row.get("pkg", "")
    
    # Try to get info from ground truth
    if pkg_to_info:
        info = _lookup_pkg_info(pkg_name, pkg_to_info)
        types = info.get("types", set())
        difficulty = info.get("difficulty")
    
    # Fallback to row fields
    if not types:
        tb = row.get("type_bucket")
        if tb:
            types = {str(tb)}
        else:
            mt = row.get("malicious_types")
            if isinstance(mt, list) and mt:
                types = {str(x).strip() for x in mt if str(x).strip()}
    
    if difficulty is None:
        difficulty = row.get("bin_label") or row.get("difficulty")
    
    # Build key
    if types:
        type_str = "+".join(sorted(types))
    else:
        type_str = "unknown"
    
    if use_difficulty and difficulty:
        return f"mal:{type_str}:{difficulty}"
    else:
        return f"mal:{type_str}"


def ensure_min_type_count(rows, pkg_to_info=None, min_count=2, use_difficulty=False):
    """Duplicate samples to ensure minimum count per stratification group."""
    groups = {}
    for r in rows:
        key = split_key(r, pkg_to_info, use_difficulty)
        groups.setdefault(key, []).append(r)
    
    added = 0
    for k, items in groups.items():
        if len(items) < min_count:
            need = min_count - len(items)
            rows.extend(items * need)
            added += need
    
    if added:
        print(f"[Stratification] Duplicated {added} samples to satisfy min per-group count.")


def build_stratify(rows, fallback_labels, pkg_to_info=None, use_difficulty=False):
    """Build stratification keys, falling back to labels if groups are too small."""
    keys = [split_key(r, pkg_to_info, use_difficulty) for r in rows]
    counts = Counter(keys)
    
    if counts and min(counts.values()) >= 2:
        return keys
    
    print("[Stratification] Warning: groups too small for stratified split; falling back to label.")
    return fallback_labels


# =============================================================================
# Dataset class
# =============================================================================
class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, max_length=256):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        enc_out = self.tokenizer(
            r["sequence_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors=None,
        )
        item = {k: torch.tensor(v) for k, v in enc_out.items()}
        item["labels"] = torch.tensor(r["label"], dtype=torch.long)
        return item


# =============================================================================
# Metrics
# =============================================================================
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    y = p.label_ids
    acc = accuracy_score(y, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": pr, "recall": rc, "f1": f1}


def _stable_row_key(row: dict) -> tuple[str, int, str, str]:
    return (
        str(row.get("pkg", "")),
        int(row.get("label", 0)),
        str(row.get("sequence_text", "")),
        json.dumps(row, sort_keys=True, ensure_ascii=False),
    )


def load_jsonl_rows(path_str: str) -> list[dict]:
    fpath = pathlib.Path(path_str)
    if not fpath.exists():
        raise FileNotFoundError(f"Data file {path_str} not found")
    rows = [
        json.loads(line)
        for line in fpath.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return sorted(rows, key=_stable_row_key)


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    transformers_set_seed(seed)


def split_train_eval_rows(
    rows: list[dict],
    labels: list[int],
    stratify_keys,
    eval_fraction: float,
    seed: int,
):
    if len(rows) < 2 or eval_fraction <= 0:
        return rows, list(rows)

    test_size = min(max(eval_fraction, 1 / len(rows)), 0.5)
    stratify_arg = stratify_keys if len(set(stratify_keys)) > 1 else None
    try:
        x_train, x_eval, _, _ = train_test_split(
            rows,
            labels,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=stratify_arg,
        )
    except ValueError:
        x_train, x_eval, _, _ = train_test_split(
            rows,
            labels,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )
    return x_train, x_eval


# =============================================================================
# Main
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train BERT classifier with ground truth support"
    )
    parser.add_argument(
        "data_files", nargs="*",
        help="Input JSONL data files"
    )
    parser.add_argument("--train-jsonl", default=None, help="Explicit training JSONL.")
    parser.add_argument("--eval-jsonl", default=None, help="Optional explicit eval JSONL.")
    parser.add_argument("--test-jsonl", default=None, help="Explicit test JSONL.")
    parser.add_argument(
        "--ground-truth-jsonl", default=None,
        help="Path to ground_truth.jsonl with type/difficulty info"
    )
    parser.add_argument(
        "--gt-key-mode", choices=["type_bucket", "malicious_types", "both"],
        default="malicious_types",
        help="Field for type grouping (default: malicious_types)"
    )
    parser.add_argument(
        "--report-by-type", action="store_true",
        help="Output per-type recall statistics on test set"
    )
    parser.add_argument(
        "--report-by-difficulty", action="store_true",
        help="Output per-difficulty recall statistics on test set"
    )
    parser.add_argument(
        "--split-stratify-by-type", action="store_true",
        help="Stratify train/test split by malicious type"
    )
    parser.add_argument(
        "--split-stratify-by-difficulty", action="store_true",
        help="Include difficulty in stratification (requires --split-stratify-by-type)"
    )
    parser.add_argument(
        "--output-dir", default="./outputs/models/bert",
        help="Model output directory"
    )
    parser.add_argument(
        "--final-model-dir", default="./outputs/models/bert-final",
        help="Final model save directory"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-length", type=int, default=256,
        help="Maximum sequence length for tokenizer"
    )
    parser.add_argument(
        "--eval-fraction", type=float, default=0.1,
        help="Eval fraction when using --train-jsonl without --eval-jsonl"
    )
    oversample_group = parser.add_mutually_exclusive_group()
    oversample_group.add_argument(
        "--oversample", dest="oversample", action="store_true",
        help="Enable oversampling of minority class"
    )
    oversample_group.add_argument(
        "--no-oversample", dest="oversample", action="store_false",
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--metrics-out", default=None,
        help="Optional path to write machine-readable test metrics JSON"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/eval split, oversampling, and model training"
    )
    parser.set_defaults(oversample=False)
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    # Load ground truth if provided
    pkg_to_info = None
    if args.ground_truth_jsonl:
        if not pathlib.Path(args.ground_truth_jsonl).exists():
            print(f"Error: Ground truth file {args.ground_truth_jsonl} not found")
            sys.exit(1)
        pkg_to_info = _load_ground_truth_map(args.ground_truth_jsonl, args.gt_key_mode)
    
    # Stratification settings
    use_stratify = args.split_stratify_by_type
    use_difficulty = args.split_stratify_by_difficulty

    explicit_mode = bool(args.train_jsonl or args.test_jsonl)
    if explicit_mode:
        if not (args.train_jsonl and args.test_jsonl):
            print("Error: --train-jsonl and --test-jsonl must be provided together")
            sys.exit(1)
        if args.data_files:
            print("Error: positional data_files cannot be mixed with --train-jsonl/--test-jsonl")
            sys.exit(1)

        train_rows = load_jsonl_rows(args.train_jsonl)
        x_test = load_jsonl_rows(args.test_jsonl)
        if use_stratify:
            ensure_min_type_count(
                train_rows,
                pkg_to_info,
                min_count=2,
                use_difficulty=use_difficulty,
            )
        train_labels_all = [r["label"] for r in train_rows]
        if args.eval_jsonl:
            x_train = train_rows
            x_eval = load_jsonl_rows(args.eval_jsonl)
        else:
            if use_stratify:
                stratify_keys = build_stratify(
                    train_rows, train_labels_all, pkg_to_info, use_difficulty
                )
            else:
                stratify_keys = train_labels_all
            x_train, x_eval = split_train_eval_rows(
                train_rows,
                train_labels_all,
                stratify_keys,
                args.eval_fraction,
                args.seed,
            )
        print(
            f"[Data] Loaded explicit splits: train={len(x_train)}, eval={len(x_eval)}, test={len(x_test)}"
        )
    else:
        rows = []
        for path_str in args.data_files:
            try:
                rows.extend(load_jsonl_rows(path_str))
            except FileNotFoundError as exc:
                print(f"Error: {exc}")
                sys.exit(1)

        print(f"[Data] Loaded {len(rows)} samples from {len(args.data_files)} file(s)")

        if use_stratify:
            ensure_min_type_count(rows, pkg_to_info, min_count=2, use_difficulty=use_difficulty)

        y = [r["label"] for r in rows]
        if use_stratify:
            stratify_keys = build_stratify(rows, y, pkg_to_info, use_difficulty)
        else:
            stratify_keys = y

        x_train, x_temp, y_train, y_temp = train_test_split(
            rows, y, test_size=0.2, random_state=args.seed, shuffle=True, stratify=stratify_keys
        )

        if use_stratify:
            ensure_min_type_count(x_temp, pkg_to_info, min_count=2, use_difficulty=use_difficulty)
            if len(x_temp) != len(y_temp):
                y_temp = [r["label"] for r in x_temp]
            stratify_keys_2 = build_stratify(x_temp, y_temp, pkg_to_info, use_difficulty)
        else:
            stratify_keys_2 = y_temp

        x_eval, x_test, y_eval, y_test = train_test_split(
            x_temp, y_temp, test_size=0.5, random_state=args.seed, shuffle=True, stratify=stratify_keys_2
        )

        print(f"[Split] train: {len(x_train)}, eval: {len(x_eval)}, test: {len(x_test)}")
    
    # Oversampling on training set
    if args.oversample:
        df_train = pd.DataFrame(x_train)
        ros = RandomOverSampler(random_state=args.seed)
        X_res, y_res = ros.fit_resample(df_train[['sequence_text']], df_train['label'])
        
        # Rebuild rows with all original fields preserved where possible
        x_train_new = []
        for i, (s, l) in enumerate(zip(X_res['sequence_text'], y_res)):
            # Find original row with this text
            orig_row = next((r for r in x_train if r['sequence_text'] == s), None)
            if orig_row:
                new_row = orig_row.copy()
                new_row['label'] = int(l)
                x_train_new.append(new_row)
            else:
                x_train_new.append({"sequence_text": s, "label": int(l)})
        x_train = x_train_new
        
        print(f"[Oversample] train after oversampling: {len(x_train)}")
    
    # Tokenizer and model
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    
    # Datasets
    train_ds = TokenizedDataset(x_train, tok, args.max_length)
    eval_ds = TokenizedDataset(x_eval, tok, args.max_length)
    test_ds = TokenizedDataset(x_test, tok, args.max_length)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=args.learning_rate,
        seed=args.seed,
        data_seed=args.seed,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        compute_metrics=compute_metrics,
    )
    
    # Train and evaluate
    trainer.train()
    trainer.evaluate()
    trainer.save_model(args.final_model_dir)
    print(f"[Model] Saved to {args.final_model_dir}")
    
    # Predict on test set
    preds_output = trainer.predict(test_ds)
    y_true = preds_output.label_ids
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    
    # Standard metrics
    print("\n" + "=" * 80)
    print("Confusion Matrix:")
    labels = np.unique(y_true)
    print(pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=labels, columns=labels
    ))
    print("\nClassification Report:")
    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["benign", "malicious"],
        digits=4,
        zero_division=0,
    )
    print(report_text)
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("=" * 80)

    metrics_payload = {
        "split_mode": "explicit" if explicit_mode else "resplit",
        "seed": args.seed,
        "counts": {
            "train": len(x_train),
            "eval": len(x_eval),
            "test": len(x_test),
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["benign", "malicious"],
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
    }
    if args.metrics_out:
        pathlib.Path(args.metrics_out).write_text(
            json.dumps(metrics_payload, indent=2),
            encoding="utf-8",
        )
    
    # Per-type reporting
    if args.report_by_type:
        _per_type_recall_report(
            x_test, y_true, y_pred, pkg_to_info,
            title=f"Per-type recall on MALICIOUS in test set (gt_key_mode={args.gt_key_mode})"
        )
    
    # Per-difficulty reporting
    if args.report_by_difficulty:
        _per_difficulty_recall_report(
            x_test, y_true, y_pred, pkg_to_info,
            title="Per-difficulty recall on MALICIOUS in test set"
        )
    
    # Legacy per-type output (for backwards compatibility)
    if not args.report_by_type:
        type_stats = {}
        unknown_mal = 0
        for r, yt, yp in zip(x_test, y_true, y_pred):
            if int(yt) != 1:
                continue
            
            types = []
            if pkg_to_info:
                info = _lookup_pkg_info(r.get("pkg", ""), pkg_to_info)
                types = list(info.get("types", []))
            
            if not types:
                mt = r.get("malicious_types")
                if isinstance(mt, list) and mt:
                    types = [str(x) for x in mt if str(x).strip()]
                else:
                    tb = r.get("type_bucket")
                    types = [str(tb)] if tb else []
            
            if not types:
                unknown_mal += 1
                continue
            
            for tp in types:
                stat = type_stats.setdefault(tp, {"total": 0, "hit": 0})
                stat["total"] += 1
                if int(yp) == 1:
                    stat["hit"] += 1
        
        if type_stats:
            print("\nDetection Rate by Malicious Type (test set):")
            for tp, stat in sorted(type_stats.items(), key=lambda x: x[1]["total"], reverse=True):
                total = stat["total"]
                hit = stat["hit"]
                rate = (hit / total) if total else 0.0
                print(f"{tp}: {rate:.4f} ({hit}/{total})")
        if unknown_mal:
            print(f"\nMalicious samples missing type info in test set: {unknown_mal}")


if __name__ == "__main__":
    main()
