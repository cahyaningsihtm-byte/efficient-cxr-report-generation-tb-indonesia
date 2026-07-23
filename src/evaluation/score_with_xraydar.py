#!/usr/bin/env python
"""
Score generated CXR reports with the existing X-raydar RoBERTaX predict.py,
without importing simpletransformers.classification.

This avoids compatibility errors from old simpletransformers classification modules.
It uses the exact repo functions:
  - doc_to_torch()
  - build_model()
  - load_list_nlp_labels()

The X-Raydar NLP repository is an external dependency and is not bundled here.
Pass its local source directory explicitly with --repo-dir.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument(
        "--repo-dir",
        required=True,
        help="Local path to the compatible X-Raydar NLP source directory containing predict.py.",
    )
    p.add_argument("--text-col", default="text_for_nlp")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def load_predict_module(repo_dir: Path):
    predict_path = repo_dir / "predict.py"
    if not predict_path.exists():
        raise FileNotFoundError(f"predict.py not found: {predict_path}")

    sys.path.insert(0, str(repo_dir))
    spec = importlib.util.spec_from_file_location("xraydar_predict", str(predict_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def score_batch(texts, predict_mod, model, tokenizer, device):
    input_ids, attention_masks = predict_mod.doc_to_torch(texts, tokenizer)
    input_ids = input_ids.to(device)
    attention_masks = attention_masks.to(device)

    with torch.no_grad():
        logits = model(input_ids, token_type_ids=None, attention_mask=attention_masks)[0]
        probs = torch.sigmoid(logits).detach().cpu().numpy()

    labels = predict_mod.load_list_nlp_labels()
    rows = []
    for row in probs:
        rows.append({label: float(row[i]) for i, label in enumerate(labels)})
    return rows


def main():
    args = parse_args()
    repo_dir = Path(args.repo_dir).resolve()

    if not Path(args.input_csv).exists():
        raise FileNotFoundError(args.input_csv)

    df = pd.read_csv(args.input_csv)
    if args.text_col not in df.columns:
        raise ValueError(f"text column '{args.text_col}' not found. Available: {list(df.columns)}")

    # build_model() in predict.py uses relative paths "./model/..."
    # so we must run from repo_dir.
    old_cwd = os.getcwd()
    os.chdir(str(repo_dir))

    try:
        predict_mod = load_predict_module(repo_dir)
        labels = predict_mod.load_list_nlp_labels()
        print(f"Loaded labels: {len(labels)}")
        print(f"First labels: {labels[:10]}")

        model, tokenizer = predict_mod.build_model()

        use_cuda = (not args.cpu) and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        model = model.to(device)
        model.eval()

        texts = df[args.text_col].fillna("").astype(str).tolist()
        print(f"Scoring rows: {len(texts)} | batch_size={args.batch_size} | device={device}")

        pred_dicts = []
        for start in range(0, len(texts), args.batch_size):
            end = min(start + args.batch_size, len(texts))
            batch = texts[start:end]
            pred_dicts.extend(score_batch(batch, predict_mod, model, tokenizer, device))
            print(f"  scored {end}/{len(texts)}", flush=True)

    finally:
        os.chdir(old_cwd)

    out = df.copy()
    out["prediction_nlp"] = [json.dumps(d, ensure_ascii=False) for d in pred_dicts]

    for label in labels:
        out[label] = [d.get(label, 0.0) for d in pred_dicts]
        out[f"{label}_bin"] = (out[label].astype(float) >= args.threshold).astype(int)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print(f"Saved NLP scores: {args.output_csv}")
    print(f"Rows: {len(out)}")


if __name__ == "__main__":
    main()
