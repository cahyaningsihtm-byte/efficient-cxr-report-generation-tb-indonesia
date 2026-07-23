#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def norm_cv(x):
    return Path(str(x).strip()).stem

def ensure_cv_filename(df, role):
    if "cv_filename" in df.columns:
        return df

    if "filename" in df.columns:
        df["cv_filename"] = df["filename"].astype(str).map(lambda x: Path(x).name)
        return df

    if "id" in df.columns:
        df["cv_filename"] = df["id"].astype(str).map(lambda x: f"{Path(x).stem}.dcm")
        return df

    if "image" in df.columns:
        df["cv_filename"] = df["image"].astype(str).map(lambda x: f"{Path(x).stem}.dcm")
        return df

    if "dicom_path" in df.columns:
        df["cv_filename"] = df["dicom_path"].astype(str).map(lambda x: Path(x).name)
        return df

    raise ValueError(
        f"{role} CSV must contain cv_filename or one of filename/id/image/dicom_path. "
        f"Columns: {list(df.columns)}"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csvs", nargs="+", required=True)
    ap.add_argument("--master-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    dfs = []
    for p in args.input_csvs:
        df = pd.read_csv(p)
        df["source_csv"] = str(p)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    master = pd.read_csv(args.master_csv)

    df = ensure_cv_filename(df, "Inference")
    master = ensure_cv_filename(master, "Master")

    keep = ["cv_filename"]
    for c in ["expertise_en", "prediction_nlp", "prediction_cv", "dicom_path"]:
        if c in master.columns:
            keep.append(c)

    df["cv_key"] = df["cv_filename"].map(norm_cv)
    master["cv_key"] = master["cv_filename"].map(norm_cv)

    keep = ["cv_key"] + [c for c in keep if c != "cv_filename"]
    master = master[keep].drop_duplicates("cv_key")
    df = df.merge(master, on="cv_key", how="left", suffixes=("", "_master"))

    for c in ["expertise_en", "prediction_nlp", "prediction_cv", "dicom_path"]:
        cm = f"{c}_master"
        if cm in df.columns:
            if c in ["prediction_nlp", "prediction_cv"]:
                df[c] = df[cm]
            elif c not in df.columns:
                df[c] = df[cm]
            else:
                df[c] = df[c].where(df[c].notna() & (df[c].astype(str).str.strip() != ""), df[cm])
            df.drop(columns=[cm], inplace=True)

    required = ["cv_filename", "model_name", "mode", "prompt_mode", "expertise_en", "prediction", "prediction_nlp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n_missing_ref_nlp = (
        df["prediction_nlp"].isna()
        | df["prediction_nlp"].astype(str).str.strip().isin(["", "nan", "None"])
    ).sum()
    if n_missing_ref_nlp:
        raise ValueError(
            f"prediction_nlp is missing/blank for {n_missing_ref_nlp}/{len(df)} rows after merge. "
            "Use the approved master metadata CSV that contains the reference NLP labels."
        )

    if "success" in df.columns:
        df = df[df["success"].astype(str).str.lower().isin(["true", "1", "yes", "ok"])].copy()

    df = df[df["prediction"].notna() & (df["prediction"].astype(str).str.strip() != "")].copy()

    dup = df.duplicated(["cv_filename", "prompt_mode", "mode"]).sum()
    if dup:
        print(f"WARNING duplicates: {dup}")

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("Saved:", out)
    print("Rows:", len(df))
    print("Modes:", sorted(df["mode"].unique()))
    print("Prompt modes:", sorted(df["prompt_mode"].unique()))

if __name__ == "__main__":
    main()
