#!/usr/bin/env python
"""
Export generated Qwen/VLM reports into a clean CSV for the same X-raydar NLP
labeler used to generate semantic GT labels.

Purpose
-------
Prepare generated radiology reports for downstream Xraydar NLP scoring.

This script creates a stable export table compatible with:
- FP16 / INT8 / INT4 / ONNX
- native / prompted modes
- multi-hardware experiments
- symmetric TB-14 evaluation pipeline

Key design decisions
--------------------
1. Uses `_join_filename` as the canonical join key.
2. Preserves prompt_mode + mode for safe downstream joins.
3. Avoids unstable legacy fields such as source_row_index.
4. Keeps hardware metadata (gpu_name) for cross-device analysis.
5. Exports only successful rows by default.

Example
-------
python e0_export_qwen_predictions_for_xraydar_nlp.py \
  --input-csv /path/to/qwen_int4_fold0_l4.csv \
  --output-csv /path/to/nlp_inputs/qwen_int4_fold0_l4_for_xraydar.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--text-col",
        default="prediction",
        help="Column containing generated report text."
    )
    return parser.parse_args()


def normalize_success_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only successful inference rows when success column exists.
    """
    if "success" not in df.columns:
        return df

    ok_values = {"true", "1", "yes"}
    mask = (
        df["success"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(ok_values)
    )
    return df[mask].copy()


def ensure_required_columns(df: pd.DataFrame, text_col: str):
    required = [text_col]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )


def build_export_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Build clean export table for downstream Xraydar NLP scoring.
    """

    preferred_cols = [
        "_join_filename",
        "model_name",
        "mode",
        "prompt_mode",
        "gpu_name",
        "cv_fold",
        "split_name",
        "study_id",
        "other_id",
        "cv_filename",
        "dicom_path",
        "expertise_en",
        text_col,
    ]

    keep_cols = [
        c for c in preferred_cols
        if c in df.columns
    ]

    out = df[keep_cols].copy()

    out = out.rename(columns={
        text_col: "text_for_nlp"
    })

    out["text_for_nlp"] = (
        out["text_for_nlp"]
        .fillna("")
        .astype(str)
    )

    # Compatibility aliases for downstream NLP pipelines
    out["report"] = out["text_for_nlp"]
    out["generated_report"] = out["text_for_nlp"]

    return out


def main():
    args = parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)

    ensure_required_columns(df, args.text_col)

    df = normalize_success_column(df)

    out = build_export_dataframe(df, args.text_col)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    out.to_csv(output_path, index=False)

    print(f"Saved NLP input: {output_path}")
    print(f"Rows: {len(out)}")
    print(f"Columns: {list(out.columns)}")

    print("\nNext step:")
    print("Run Xraydar NLP scoring on column 'text_for_nlp'.")


if __name__ == "__main__":
    main()