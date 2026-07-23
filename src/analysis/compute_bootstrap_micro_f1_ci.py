#!/usr/bin/env python
"""Bootstrap 95% CIs for semantic micro-F1 from per-sample CXR label outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("outputs/eval_quantization_qwen")
DEFAULT_OUT_FULL = Path("outputs/bootstrap_micro_f1_ci.csv")
DEFAULT_OUT_PAPER = Path("outputs/bootstrap_micro_f1_ci_paper_ready.csv")
GROUP_COLS = ["model_name", "mode", "prompt_mode"]


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV-like content, with a delimiter-sniff fallback for mislabeled files."""
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python")

    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=None, engine="python")
    return df


def is_binary_pair(df: pd.DataFrame, gt_col: str, pred_col: str) -> bool:
    gt = pd.to_numeric(df[gt_col], errors="coerce")
    pred = pd.to_numeric(df[pred_col], errors="coerce")
    if gt.isna().any() or pred.isna().any():
        return False
    gt_values = set(gt.unique().tolist())
    pred_values = set(pred.unique().tolist())
    return gt_values.issubset({0, 1}) and pred_values.issubset({0, 1})


def detect_labels(df: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    for col in df.columns:
        if not col.startswith("gt_"):
            continue
        if col.startswith("gt_score_"):
            continue
        label = col[3:]
        pred_col = f"pred_{label}"
        if pred_col not in df.columns:
            continue
        if is_binary_pair(df, col, pred_col):
            labels.append(label)
    return labels


def path_allowed(path: Path) -> tuple[bool, str]:
    text = str(path).replace("\\", "/")
    lower = text.lower()
    lower_parts = [part.lower() for part in path.parts]
    if any(part == "old" or part.startswith("old_") or part.startswith("old-") for part in lower_parts):
        return False, "excluded old/OLD path"
    if "epoch2" in lower:
        return False, "excluded epoch2 path"
    if "epoch" in lower and "epoch1" not in lower:
        return False, "excluded non-epoch1 path"
    return True, "included"


def discover_files(root: Path) -> tuple[list[Path], list[tuple[Path, str]]]:
    skipped: list[tuple[Path, str]] = []
    if not root.exists():
        return [], [(root, "root does not exist")]

    candidates = sorted(root.rglob("per_sample_metrics_symmetric.csv"))
    allowed: list[Path] = []
    for path in candidates:
        ok, reason = path_allowed(path)
        if ok:
            allowed.append(path)
        else:
            skipped.append((path, reason))

    canonical_prompted = [
        p
        for p in allowed
        if "prompted" in str(p).lower() and "canonical" in str(p).lower()
    ]
    if canonical_prompted:
        filtered: list[Path] = []
        for path in allowed:
            lower = str(path).lower()
            if "prompted" in lower and "canonical" not in lower:
                skipped.append((path, "excluded prompted non-canonical path because canonical prompted outputs exist"))
            else:
                filtered.append(path)
        allowed = filtered

    return allowed, skipped


def safe_metrics(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def row_counts(df: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt = df[[f"gt_{label}" for label in labels]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.int8)
    pred = df[[f"pred_{label}" for label in labels]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.int8)
    tp = ((gt == 1) & (pred == 1)).sum(axis=1).astype(np.int32)
    fp = ((gt == 0) & (pred == 1)).sum(axis=1).astype(np.int32)
    fn = ((gt == 1) & (pred == 0)).sum(axis=1).astype(np.int32)
    return tp, fp, fn


def bootstrap_f1(
    row_tp: np.ndarray,
    row_fp: np.ndarray,
    row_fn: np.ndarray,
    rng: np.random.Generator,
    b: int,
    chunk_size: int = 250,
) -> tuple[float, float]:
    n = len(row_tp)
    values = np.empty(b, dtype=np.float64)
    start = 0
    while start < b:
        end = min(start + chunk_size, b)
        idx = rng.integers(0, n, size=(end - start, n))
        tp = row_tp[idx].sum(axis=1)
        fp = row_fp[idx].sum(axis=1)
        fn = row_fn[idx].sum(axis=1)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp, dtype=np.float64), where=(tp + fn) > 0)
        values[start:end] = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision, dtype=np.float64),
            where=(precision + recall) > 0,
        )
        start = end
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def model_label(model_name: str, mode: str) -> str:
    if pd.isna(mode) or str(mode).strip() == "":
        return str(model_name)
    return f"{model_name} ({mode})"


def pct(x: float) -> float:
    return round(100.0 * float(x), 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--B", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-full", type=Path, default=DEFAULT_OUT_FULL)
    parser.add_argument("--out-paper", type=Path, default=DEFAULT_OUT_PAPER)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    files, skipped = discover_files(args.root)

    print("\nDetected files:")
    if files:
        for path in files:
            print(f"  INCLUDED: {path}")
    else:
        print("  None")

    rows: list[dict[str, object]] = []
    labels_by_file: dict[str, list[str]] = {}

    for path in files:
        try:
            df = read_table(path)
        except Exception as exc:
            skipped.append((path, f"read failed: {exc}"))
            continue

        labels = detect_labels(df)
        labels_by_file[str(path)] = labels
        if not labels:
            skipped.append((path, "no binary gt_/pred_ label pairs detected"))
            continue
        if len(labels) != 14:
            skipped.append((path, f"warning: detected {len(labels)} labels, expected 14"))

        missing_group_cols = [col for col in GROUP_COLS if col not in df.columns]
        for col in missing_group_cols:
            df[col] = "unknown"

        df["_source_path"] = str(path)
        group_cols = GROUP_COLS + ["_source_path"]
        for keys, group in df.groupby(group_cols, dropna=False, sort=True):
            model_name, mode, prompt_mode, source_path = keys
            row_tp, row_fp, row_fn = row_counts(group, labels)
            tp = int(row_tp.sum())
            fp = int(row_fp.sum())
            fn = int(row_fn.sum())
            precision, recall, f1 = safe_metrics(tp, fp, fn)
            ci_low, ci_high = bootstrap_f1(row_tp, row_fp, row_fn, rng, args.B)
            rows.append(
                {
                    "model_name": model_name,
                    "mode": mode,
                    "prompt_mode": prompt_mode,
                    "source_path": source_path,
                    "n": int(len(group)),
                    "n_labels": int(len(labels)),
                    "labels": ";".join(labels),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "micro_precision": precision,
                    "micro_recall": recall,
                    "micro_f1": f1,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "B": args.B,
                    "seed": args.seed,
                }
            )

    print("\nDetected labels:")
    if labels_by_file:
        for path, labels in labels_by_file.items():
            print(f"  {path}")
            print(f"    {len(labels)} labels: {', '.join(labels)}")
    else:
        print("  None")

    result = pd.DataFrame(rows)
    if result.empty:
        print("\nBootstrap CI table:")
        print("  No results.")
    else:
        result = result.sort_values(["model_name", "mode", "prompt_mode", "source_path"]).reset_index(drop=True)
        args.out_full.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.out_full, index=False)

        paper = pd.DataFrame(
            {
                "model_label": [model_label(r.model_name, r.mode) for r in result.itertuples(index=False)],
                "prompt_mode": result["prompt_mode"],
                "n": result["n"],
                "micro_f1_percent": result["micro_f1"].map(pct),
                "ci95_low_percent": result["ci95_low"].map(pct),
                "ci95_high_percent": result["ci95_high"].map(pct),
                "micro_precision_percent": result["micro_precision"].map(pct),
                "micro_recall_percent": result["micro_recall"].map(pct),
                "source_path": result["source_path"],
            }
        )
        paper.to_csv(args.out_paper, index=False)

        print("\nBootstrap CI table:")
        display_cols = [
            "model_name",
            "mode",
            "prompt_mode",
            "n",
            "tp",
            "fp",
            "fn",
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "ci95_low",
            "ci95_high",
            "source_path",
        ]
        print(result[display_cols].to_string(index=False))
        print(f"\nSaved full CSV: {args.out_full}")
        print(f"Saved paper-ready CSV: {args.out_paper}")

    print("\nSkipped files and reasons:")
    if skipped:
        for path, reason in skipped:
            print(f"  {path}: {reason}")
    else:
        print("  None")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
