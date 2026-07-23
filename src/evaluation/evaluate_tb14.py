#!/usr/bin/env python
"""
Symmetric TB-14 evaluation for Qwen outputs.

Fixes:
- Keeps expertise_en from original Qwen CSV for BLEU/ROUGE.
- Uses original prediction_nlp as GT-side NLP scores.
- Uses VLM NLP scorer output as VLM-side NLP scores.
- Avoids duplicated/confusing prediction_nlp_x/y columns.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


TB14_MAPPING = [
    ("apical_changes",        "apical_changes",                 "DIRECT"),
    ("atelectasis",           "atelectasis",                    "DIRECT"),
    ("bronchial_changes",     "bronchial_changes",              "DIRECT"),
    ("bulla",                 "bulla",                          "DIRECT"),
    ("cardiomegaly",          "cardiomegaly",                   "DIRECT"),
    ("cavity",                "cavity",                         "DIRECT"),
    ("consolidation",         "consolidation",                  "DIRECT"),
    ("mediastinum_displaced", "mediastinum_displaced",          "DIRECT"),
    ("pleural_abnormality",   "pleural_abnormality",            "DIRECT"),
    ("pleural_effusion",      "pleural_effusion",               "DIRECT"),
    ("pneumothorax",          "pneumothorax",                   "DIRECT"),
    ("hilar_enlargement",     "paratracheal_hilar_enlargement", "RENAME"),
    ("fibrosis",              "interstitial_shadowing",         "RENAME"),
    ("infiltration",          "parenchymal_lesion",             "RENAME"),
]
XRAYDAR_COLS = [xr for _, xr, _ in TB14_MAPPING]

def normalize_filename_value(x: Any) -> str:
    """Normalize filename/path for safe joining: basename only, .json -> .dcm."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    if not s:
        return ""
    s = Path(s).name
    if s.lower().endswith(".json"):
        s = s[:-5] + ".dcm"
    return s


def add_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cv_filename" in df.columns:
        df["_join_cv_filename"] = df["cv_filename"].apply(normalize_filename_value)
    else:
        df["_join_cv_filename"] = ""

    if "dicom_path" in df.columns:
        df["_join_dicom_path"] = df["dicom_path"].apply(normalize_filename_value)
    else:
        df["_join_dicom_path"] = ""

    # Best filename key: cv_filename first, fallback to dicom_path basename.
    df["_join_filename"] = df["_join_cv_filename"]
    empty = df["_join_filename"].astype(str).str.len().eq(0)
    df.loc[empty, "_join_filename"] = df.loc[empty, "_join_dicom_path"]
    return df



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--vlm-nlp-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def parse_dict_like(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if x is None:
        return {}
    if isinstance(x, float) and math.isnan(x):
        return {}
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return {}
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def score_from_dict_cell(cell: Any, label: str) -> float:
    d = parse_dict_like(cell)
    if "AI_prediction" in d and isinstance(d["AI_prediction"], dict):
        d = d["AI_prediction"]
    try:
        return float(d.get(label, 0.0) or 0.0)
    except Exception:
        return 0.0


def tokenize(x: Any) -> List[str]:
    if pd.isna(x):
        return []
    return re.findall(r"[a-z0-9]+", " ".join(str(x).lower().strip().split()))


def ngrams(tokens: List[str], n: int):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]


def bleu_n(reference: Any, prediction: Any, n: int = 1) -> float:
    ref = tokenize(reference)
    pred = tokenize(prediction)
    if not ref or not pred or len(pred) < n:
        return 0.0
    ref_ngrams = ngrams(ref, n)
    pred_ngrams = ngrams(pred, n)
    ref_counts = {}
    for g in ref_ngrams:
        ref_counts[g] = ref_counts.get(g, 0) + 1
    used = {}
    matched = 0
    for g in pred_ngrams:
        used[g] = used.get(g, 0) + 1
        if used[g] <= ref_counts.get(g, 0):
            matched += 1
    precision = matched / max(len(pred_ngrams), 1)
    bp = 1.0 if len(pred) > len(ref) else np.exp(1 - len(ref) / max(len(pred), 1))
    return float(bp * precision)


def rouge_n_f1(reference: Any, prediction: Any, n: int = 1) -> float:
    ref = ngrams(tokenize(reference), n)
    pred = ngrams(tokenize(prediction), n)
    if not ref or not pred:
        return 0.0
    ref_counts = {}
    for g in ref:
        ref_counts[g] = ref_counts.get(g, 0) + 1
    used = {}
    matched = 0
    for g in pred:
        used[g] = used.get(g, 0) + 1
        if used[g] <= ref_counts.get(g, 0):
            matched += 1
    p = matched / len(pred)
    r = matched / len(ref)
    return 0.0 if p + r == 0 else float(2 * p * r / (p + r))


def lcs_len(a: List[str], b: List[str]) -> int:
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            cur[j] = prev[j-1] + 1 if a[i-1] == b[j-1] else max(prev[j], cur[j-1])
        prev = cur
    return prev[-1]


def rouge_l_f1(reference: Any, prediction: Any) -> float:
    ref = tokenize(reference)
    pred = tokenize(prediction)
    if not ref or not pred:
        return 0.0
    lcs = lcs_len(ref, pred)
    p = lcs / len(pred)
    r = lcs / len(ref)
    return 0.0 if p + r == 0 else float(2 * p * r / (p + r))


def f1_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0
    return tp, fp, fn, tn, precision, recall, f1, accuracy


def choose_join_cols(base: pd.DataFrame, vlm: pd.DataFrame) -> List[str]:
    """
    Prefer filename-based join because source_row_index may be regenerated by
    aligned inference scripts and is not a stable methodological key.
    """
    candidates = [
        ["_join_filename", "prompt_mode", "mode"],
        ["_join_filename", "prompt_mode"],
        ["cv_filename", "prompt_mode", "mode"],
        ["cv_filename", "prompt_mode"],
        ["dicom_path", "prompt_mode", "mode"],
        ["dicom_path", "prompt_mode"],
    ]

    for cols in candidates:
        if all(c in base.columns for c in cols) and all(c in vlm.columns for c in cols):
            left_dups = int(base.duplicated(cols).sum())
            right_dups = int(vlm.duplicated(cols).sum())
            print(f"Candidate join columns {cols}: left_dups={left_dups}, right_dups={right_dups}")
            if left_dups == 0 and right_dups == 0:
                return cols

    raise ValueError(
        "Cannot find unique safe join columns. Check duplicated cv_filename/prompt_mode/mode "
        "in input CSV and VLM NLP CSV."
    )


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.input_csv)
    vlm = pd.read_csv(args.vlm_nlp_csv)

    # Add stable normalized filename keys for joining.
    base = add_join_keys(base)
    vlm = add_join_keys(vlm)

    if "success" in base.columns:
        base = base[base["success"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()

    for col in ["model_name", "mode", "prompt_mode"]:
        if col not in base.columns:
            base[col] = "unknown"

    required = ["expertise_en", "prediction", "prediction_nlp"]
    missing = [c for c in required if c not in base.columns]
    if missing:
        raise ValueError(f"input-csv missing columns: {missing}")

    join_cols = choose_join_cols(base, vlm)
    print("Join columns:", join_cols)

    keep = list(join_cols)
    if "prediction_nlp" in vlm.columns:
        keep.append("prediction_nlp")
    for c in XRAYDAR_COLS:
        if c in vlm.columns:
            keep.append(c)
    keep = list(dict.fromkeys(keep))

    vlm_scores = vlm[keep].copy()
    rename = {"prediction_nlp": "vlm_prediction_nlp"}
    for c in XRAYDAR_COLS:
        if c in vlm_scores.columns:
            rename[c] = f"vlm_score_{c}"
    vlm_scores = vlm_scores.rename(columns=rename)
    vlm_scores = vlm_scores.drop_duplicates(subset=join_cols, keep="first")

    merged = base.merge(vlm_scores, on=join_cols, how="left", validate="one_to_one")
    print("Merged rows:", len(merged))

    sample_rows = []
    keep_cols = [
        "source_row_index", "split_id", "model_name", "mode", "prompt_mode",
        "prompt_template", "cv_filename", "dicom_path", "study_id", "other_id",
        "gpu_name", "latency_ms", "peak_gpu_mem_allocated_gb", "peak_gpu_mem_reserved_gb",
        "output_words", "model_load_time_s", "expertise_en", "prediction",
    ]

    for _, r in merged.iterrows():
        rec = {c: r.get(c) for c in keep_cols if c in r.index}
        ref = r.get("expertise_en", "")
        pred = r.get("prediction", "")

        rec.update({
            "bleu1": bleu_n(ref, pred, 1),
            "bleu2": bleu_n(ref, pred, 2),
            "bleu4": bleu_n(ref, pred, 4),
            "rouge1_f1": rouge_n_f1(ref, pred, 1),
            "rouge2_f1": rouge_n_f1(ref, pred, 2),
            "rougeL_f1": rouge_l_f1(ref, pred),
            "gt_prediction_nlp": r.get("prediction_nlp", ""),
            "vlm_prediction_nlp": r.get("vlm_prediction_nlp", ""),
        })

        for tb_label, xr_col, status in TB14_MAPPING:
            gt_score = score_from_dict_cell(r.get("prediction_nlp", ""), xr_col)
            vlm_score_col = f"vlm_score_{xr_col}"
            if vlm_score_col in r.index and pd.notna(r.get(vlm_score_col)):
                try:
                    pred_score = float(r.get(vlm_score_col))
                except Exception:
                    pred_score = 0.0
            else:
                pred_score = score_from_dict_cell(r.get("vlm_prediction_nlp", ""), xr_col)

            rec[f"gt_score_{tb_label}"] = gt_score
            rec[f"pred_score_{tb_label}"] = pred_score
            rec[f"gt_{tb_label}"] = int(gt_score >= args.threshold)
            rec[f"pred_{tb_label}"] = int(pred_score >= args.threshold)

        sample_rows.append(rec)

    sdf = pd.DataFrame(sample_rows)
    per_sample_path = out_dir / "per_sample_metrics_symmetric.csv"
    sdf.to_csv(per_sample_path, index=False)

    group_cols = ["model_name", "mode", "prompt_mode"]
    summary_rows = []
    label_rows = []

    for key, g in sdf.groupby(group_cols, dropna=False):
        key_dict = dict(zip(group_cols, key))
        metric = dict(key_dict)
        metric.update({
            "n": len(g),
            "bleu1_mean": g["bleu1"].mean(),
            "bleu2_mean": g["bleu2"].mean(),
            "bleu4_mean": g["bleu4"].mean(),
            "rouge1_f1_mean": g["rouge1_f1"].mean(),
            "rouge2_f1_mean": g["rouge2_f1"].mean(),
            "rougeL_f1_mean": g["rougeL_f1"].mean(),
            "semantic_threshold": args.threshold,
            "semantic_method": "symmetric_xraydar_robertax_nlp_vs_nlp",
        })
        for eff_col in ["latency_ms", "peak_gpu_mem_allocated_gb", "peak_gpu_mem_reserved_gb", "output_words"]:
            if eff_col in g.columns:
                metric[f"{eff_col}_mean"] = pd.to_numeric(g[eff_col], errors="coerce").mean()
                metric[f"{eff_col}_median"] = pd.to_numeric(g[eff_col], errors="coerce").median()

        all_true, all_pred, per_label_f1 = [], [], []
        for tb_label, xr_col, status in TB14_MAPPING:
            y_true = g[f"gt_{tb_label}"].astype(int).values
            y_pred = g[f"pred_{tb_label}"].astype(int).values
            tp, fp, fn, tn, p, r_, f1, acc = f1_counts(y_true, y_pred)
            support = int(y_true.sum())
            pred_pos = int(y_pred.sum())
            per_label_f1.append(f1)
            all_true.extend(y_true.tolist())
            all_pred.extend(y_pred.tolist())
            lr = dict(key_dict)
            lr.update({
                "tb_label": tb_label,
                "label": tb_label,
                "xraydar_column": xr_col,
                "mapping_status": status,
                "support": support,
                "n_pos_gt": support,
                "n_pos_pred": pred_pos,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "precision": p, "recall": r_, "f1": f1, "accuracy": acc,
            })
            label_rows.append(lr)

        tp, fp, fn, tn, p, r_, micro, acc = f1_counts(all_true, all_pred)
        metric.update({
            "semantic_micro_precision": p,
            "semantic_micro_recall": r_,
            "semantic_micro_f1": micro,
            "semantic_macro_f1": float(np.mean(per_label_f1)) if per_label_f1 else 0.0,
            "semantic_micro_accuracy": acc,
            "semantic_tp": tp,
            "semantic_fp": fp,
            "semantic_fn": fn,
            "semantic_tn": tn,
        })
        summary_rows.append(metric)

    summary_path = out_dir / "summary_by_model_mode_prompt_symmetric.csv"
    label_path = out_dir / "tb14_per_class_symmetric.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(label_rows).to_csv(label_path, index=False)

    print("DONE")
    print(f"Per-sample: {per_sample_path}")
    print(f"Summary:    {summary_path}")
    print(f"Per-label:  {label_path}")


if __name__ == "__main__":
    main()
