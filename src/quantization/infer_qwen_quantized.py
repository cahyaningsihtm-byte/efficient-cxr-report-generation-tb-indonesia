#!/usr/bin/env python
"""
Unified Qwen VLM inference for CXR-RAIT quantization experiments.

Purpose
-------
Run Qwen inference in fp16, int8, or int4 mode on a stable fold definition
from the master split workflow.

This script is intended as the final inference entry point for:
- canonical Fold 0 inference
- future Fold 1 / Fold 2 runs
- downstream Xraydar NLP export
- symmetric TB-14 evaluation
- clean comparison across FP16 / INT8 / INT4

Key design decisions
--------------------
1. No legacy row-order split logic.
   The old modulo-based split is removed.

2. Fold selection is based on:
   - a plain-text fold ID list (recommended for immediate use), or
   - the master split CSV.

3. `_join_filename` is created as the canonical join key for downstream
   combine/evaluation.

4. The exported CSV is intentionally clean:
   - no full prompt text
   - no source_row_index
   - no prompt_template
   - no legacy split_id

5. Resume safety is preserved via JSONL.
   Internal `run_key` is based on:
   `_join_filename + prompt_mode + mode`

Example
-------
python src/quantization/infer_qwen_quantized.py \
  --mode int4 \
  --input-csv /path/to/approved_data/master_metadata.csv \
  --dicom-root /path/to/approved_data/dicom_root \
  --model-path /path/to/verified_qwen_checkpoint \
  --output-dir /path/to/local_outputs/qwen_quantization \
  --output-name qwen_int4_fold0 \
  --fold-ids-txt /path/to/approved_data/fold0_ids.txt \
  --fold-id 0 \
  --prompt-modes native prompted \
  --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None


NATIVE_PROMPT = "Describe the findings in this chest X-ray."
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

TB14_PROMPT = """Analyze this chest X-ray and write a radiology report with FINDINGS and IMPRESSION sections.

In FINDINGS, systematically evaluate and describe each of the following. For each, state whether it is present, absent, or uncertain, and if present, describe the location (left/right, upper/middle/lower zone) and severity (mild/moderate/severe):

1. Infiltration
2. Apical changes
3. Cardiomegaly
4. Pleural effusion
5. Hilar enlargement
6. Pleural abnormality
7. Consolidation
8. Cavity
9. Fibrosis
10. Atelectasis
11. Bronchial changes
12. Bulla
13. Pneumothorax
14. Mediastinum displaced

In IMPRESSION, summarize the key findings and assess the likelihood of tuberculosis.

FINDINGS:"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Final unified Qwen inference script for FP16 / INT8 / INT4"
    )
    parser.add_argument("--mode", required=True, choices=["fp16", "int8", "int4"])
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--dicom-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)

    parser.add_argument(
        "--master-split-csv",
        default=None,
        help="Optional master split CSV or manifest CSV containing fold membership.",
    )
    parser.add_argument(
        "--fold-id",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Fold ID to run. Used with --master-split-csv or as a label with --fold-ids-txt.",
    )
    parser.add_argument(
        "--fold-ids-txt",
        default=None,
        help="Optional plain-text file containing one case ID per line, e.g. fold0_ids.txt",
    )

    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--prompt-modes",
        nargs="+",
        default=["native", "prompted"],
        choices=["native", "prompted", "prompted_tb14"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def validate_args(args):
    if args.master_split_csv and args.fold_ids_txt:
        raise ValueError("Use either --master-split-csv or --fold-ids-txt, not both.")

    if args.master_split_csv and args.fold_id is None:
        raise ValueError("--fold-id is required when using --master-split-csv.")

    if (args.fold_id is not None) and (not args.master_split_csv) and (not args.fold_ids_txt):
        raise ValueError("--fold-id alone is not enough. Provide --master-split-csv or --fold-ids-txt.")


def normalize_colname(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def normalize_case_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return Path(text).stem


def normalize_prompt_mode(value: str) -> str:
    value = str(value).strip().lower()
    if value in {"prompted", "prompted_tb14"}:
        return "prompted"
    if value == "native":
        return "native"
    raise ValueError(f"Unsupported prompt mode: {value}")


def get_prompt_text(prompt_mode: str) -> str:
    if prompt_mode == "native":
        return NATIVE_PROMPT
    if prompt_mode == "prompted":
        return TB14_PROMPT
    raise ValueError(f"Unsupported normalized prompt mode: {prompt_mode}")


def deduplicate_prompt_modes(prompt_modes: List[str]) -> List[str]:
    normalized = []
    seen = set()
    for item in prompt_modes:
        x = normalize_prompt_mode(item)
        if x not in seen:
            normalized.append(x)
            seen.add(x)
    return normalized


def extract_info_from_filename(filename: str) -> Optional[Dict[str, int]]:
    """
    Parse DICOM filename pattern:
    S YY MM NNN - accession

    Example:
    SYYMMNNN-ACCESSION.dcm -> year=20YY, month=MM, filename_no=NNN
    """
    stem = Path(str(filename)).stem
    match = re.match(r"S(\d{2})(\d{2})(\d{3})-(.*)$", stem)
    if not match:
        return None
    return {
        "year": 2000 + int(match.group(1)),
        "month": int(match.group(2)),
        "filename_no": int(match.group(3)),
    }


def resolve_cv_filename(row: pd.Series) -> Optional[str]:
    """
    Resolve the DICOM filename from available filename-like columns.
    Preference:
    1) cv_filename
    2) filename
    3) dcm_fn
    4) dicom_path basename
    """
    for col in ["cv_filename", "filename", "dcm_fn"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            filename = str(row[col]).strip()
            if filename.endswith(".json"):
                filename = Path(filename).stem + ".dcm"
            return filename

    if "dicom_path" in row and pd.notna(row["dicom_path"]) and str(row["dicom_path"]).strip():
        dicom_path = str(row["dicom_path"]).strip()
        if dicom_path.endswith(".dcm"):
            return Path(dicom_path).name

    return None


def resolve_join_filename(row: pd.Series) -> str:
    """
    Canonical join key used downstream.
    Always stored without extension.
    """
    filename = resolve_cv_filename(row)
    if filename:
        return normalize_case_id(filename)

    if "cv_filename" in row and pd.notna(row["cv_filename"]):
        return normalize_case_id(row["cv_filename"])
    if "filename" in row and pd.notna(row["filename"]):
        return normalize_case_id(row["filename"])
    if "dicom_path" in row and pd.notna(row["dicom_path"]):
        return normalize_case_id(Path(str(row["dicom_path"])).name)
    return ""


def build_dicom_path(row: pd.Series, dicom_root: str) -> str:
    """
    Resolve image path.

    If dicom_path already exists in the row and points to a .dcm file,
    use it directly. For public examples, cv_filename/dicom_path may point to a
    metadata-stripped PNG/JPEG relative to dicom_root. Otherwise reconstruct
    DICOM paths using cv_filename-derived year/month.
    """
    if "dicom_path" in row and pd.notna(row["dicom_path"]):
        dicom_path = str(row["dicom_path"]).strip()
        suffix = Path(dicom_path).suffix.lower()
        if suffix == ".dcm" or suffix in IMAGE_EXTS:
            p = Path(dicom_path)
            return str(p if p.is_absolute() else Path(dicom_root) / p)

    filename = resolve_cv_filename(row)
    if not filename:
        raise ValueError("Cannot resolve image filename from cv_filename / filename / dcm_fn / dicom_path")

    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_EXTS:
        p = Path(filename)
        return str(p if p.is_absolute() else Path(dicom_root) / p)

    info = extract_info_from_filename(filename)
    if info is None:
        raise ValueError(f"Cannot parse year/month from filename: {filename}")

    return str(Path(dicom_root) / str(info["year"]) / str(info["month"]) / filename)


def resolve_gt_text(row: pd.Series) -> str:
    """
    Output field name is standardized to 'expertise_en',
    but source data may use 'ekspertise_en'.
    """
    for col in ["ekspertise_en", "expertise_en"]:
        if col in row and pd.notna(row[col]):
            return str(row[col])
    return ""


def load_case_ids_from_txt(path: str) -> Set[str]:
    case_ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = normalize_case_id(line.strip())
            if x:
                case_ids.add(x)
    return case_ids


def load_case_ids_from_master(master_split_csv: str, fold_id: int) -> Tuple[Set[str], str]:
    """
    Load case IDs from master split file.
    Supports either:
    - qwen_master_3fold_split.csv
    - qwen_master_3fold_manifest.csv
    """
    df = pd.read_csv(master_split_csv)
    df = df.rename(columns={c: normalize_colname(c) for c in df.columns})

    if "cv_fold" not in df.columns:
        raise ValueError("master split CSV must contain column 'cv_fold'")

    candidate_id_cols = ["normalized_case_id", "_join_filename", "cv_filename", "filename"]
    id_col = None
    for col in candidate_id_cols:
        if col in df.columns:
            id_col = col
            break

    if id_col is None:
        raise ValueError(
            "master split CSV must contain one of: normalized_case_id, _join_filename, cv_filename, filename"
        )

    subset = df[df["cv_fold"] == int(fold_id)].copy()
    if subset.empty:
        raise ValueError(f"No rows found for fold {fold_id} in {master_split_csv}")

    case_ids = {
        normalize_case_id(x)
        for x in subset[id_col].tolist()
        if normalize_case_id(x)
    }

    split_name = f"fold{fold_id}"
    if "split_name" in subset.columns:
        unique_names = [str(x) for x in subset["split_name"].dropna().unique().tolist()]
        if len(unique_names) == 1:
            split_name = unique_names[0]

    return case_ids, split_name


def load_rows(
    input_csv: str,
    dicom_root: str,
    selected_case_ids: Optional[Set[str]],
    fold_id: Optional[int],
    split_name: str,
    start: int,
    limit: Optional[int],
) -> List[Dict]:
    df = pd.read_csv(input_csv)
    df = df.rename(columns={c: normalize_colname(c) for c in df.columns})

    df["_join_filename"] = df.apply(resolve_join_filename, axis=1)
    df["cv_filename_resolved"] = df.apply(resolve_cv_filename, axis=1)

    df = df[df["_join_filename"] != ""].copy()

    if selected_case_ids is not None:
        df = df[df["_join_filename"].isin(selected_case_ids)].copy()

    df = df.sort_values("_join_filename").reset_index(drop=True)

    if start:
        df = df.iloc[start:].copy()
    if limit is not None:
        df = df.iloc[:limit].copy()

    rows = []
    for _, row in df.iterrows():
        record = row.to_dict()
        record["dicom_path"] = build_dicom_path(row, dicom_root)
        record["cv_fold"] = fold_id if fold_id is not None else record.get("cv_fold", "")
        record["split_name"] = split_name
        record["expertise_en"] = resolve_gt_text(row)
        rows.append(record)

    return rows


def load_dicom_as_pil(dcm_path: str):
    suffix = Path(dcm_path).suffix.lower()
    if suffix in IMAGE_EXTS:
        image = Image.open(dcm_path).convert("RGB")
        return image, {
            "source_format": suffix.lstrip("."),
            "image_size": list(image.size),
        }

    ds = pydicom.dcmread(dcm_path, force=True)
    pixel_array = ds.pixel_array.astype(np.float32)

    photometric = getattr(ds, "PhotometricInterpretation", "UNKNOWN")
    if photometric == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array

    p1, p99 = np.percentile(pixel_array, [1, 99])
    pixel_array = np.clip(pixel_array, p1, p99)

    pixel_min = float(pixel_array.min())
    pixel_max = float(pixel_array.max())

    if pixel_max - pixel_min < 1e-8:
        pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
    else:
        pixel_array = (pixel_array - pixel_min) / (pixel_max - pixel_min)
        pixel_array = (pixel_array * 255.0).clip(0, 255).astype(np.uint8)

    image = Image.fromarray(pixel_array).convert("RGB")
    meta = {
        "photometric_interpretation": photometric,
        "image_size": list(image.size),
        "pixel_min": pixel_min,
        "pixel_max": pixel_max,
    }
    return image, meta


def load_model_and_processor(model_path: str, mode: str):
    print(f"Loading processor: {model_path}", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print(f"Loading Qwen model mode={mode}: {model_path}", flush=True)
    t0 = time.time()

    kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
    )

    if mode == "fp16":
        kwargs["torch_dtype"] = torch.float16

    elif mode == "int8":
        if BitsAndBytesConfig is None:
            raise RuntimeError("BitsAndBytesConfig unavailable. Please install/update transformers and bitsandbytes.")
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    elif mode == "int4":
        if BitsAndBytesConfig is None:
            raise RuntimeError("BitsAndBytesConfig unavailable. Please install/update transformers and bitsandbytes.")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    model.eval()

    model_load_time_s = time.time() - t0
    print(f"Model loaded in {model_load_time_s:.1f}s", flush=True)
    return processor, model, model_load_time_s


def move_inputs_to_device(inputs: Dict, model) -> Dict:
    """
    Move processor inputs to the first active device.
    This matches the previously working device_map='auto' workflow.
    """
    target_device = getattr(model, "device", None)

    if target_device is None:
        try:
            target_device = next(model.parameters()).device
        except Exception:
            target_device = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        key: value.to(target_device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def run_inference(model, processor, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
    )
    inputs = move_inputs_to_device(inputs, model)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
        )

    input_len = inputs["input_ids"].shape[1]
    output_text = processor.batch_decode(
        output_ids[:, input_len:],
        skip_special_tokens=True,
    )[0].strip()

    return output_text


def build_run_key(join_filename: str, prompt_mode: str, mode: str) -> str:
    return f"{join_filename}::{prompt_mode}::{mode}"


def load_done_keys(jsonl_path: Path) -> Set[str]:
    done = set()
    if not jsonl_path.exists():
        return done

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("success", True):
                    run_key = rec.get("run_key")
                    if run_key:
                        done.add(run_key)
            except Exception:
                pass
    return done


def clean_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the clean downstream columns.
    Intentionally excludes:
    - run_key
    - full prompt text
    - source_row_index
    - prompt_template
    - legacy split_id
    """
    desired_columns = [
        "_join_filename",
        "cv_filename",
        "dicom_path",
        "cv_fold",
        "split_name",
        "study_id",
        "other_id",
        "expertise_en",
        "prediction_nlp",
        "prediction_cv",
        "model_name",
        "gpu_name",
        "mode",
        "prompt_mode",
        "prediction",
        "success",
        "error",
        "model_load_time_s",
        "latency_ms",
        "peak_gpu_mem_allocated_gb",
        "peak_gpu_mem_reserved_gb",
        "output_words",
        "photometric_interpretation",
        "image_size",
        "pixel_min",
        "pixel_max",
    ]

    for col in desired_columns:
        if col not in df.columns:
            df[col] = ""

    return df[desired_columns].copy()


def jsonl_to_clean_csv(jsonl_path: Path, csv_path: Path):
    records = []
    if not jsonl_path.exists():
        return

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    if not records:
        return

    df = pd.DataFrame(records)
    df = clean_csv_columns(df)
    df.to_csv(csv_path, index=False)
    print(f"CSV saved: {csv_path} ({len(df)} rows)", flush=True)


def main():
    args = parse_args()
    validate_args(args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / f"{args.output_name}.jsonl"
    csv_path = out_dir / f"{args.output_name}.csv"

    selected_case_ids = None
    split_name = "all_rows"

    if args.fold_ids_txt:
        selected_case_ids = load_case_ids_from_txt(args.fold_ids_txt)
        split_name = Path(args.fold_ids_txt).stem
        print(f"Loaded case IDs from TXT: {len(selected_case_ids)} | source={args.fold_ids_txt}", flush=True)

    elif args.master_split_csv:
        selected_case_ids, split_name = load_case_ids_from_master(args.master_split_csv, args.fold_id)
        print(
            f"Loaded case IDs from master split: {len(selected_case_ids)} | fold={args.fold_id} | split_name={split_name}",
            flush=True,
        )

    prompt_modes = deduplicate_prompt_modes(args.prompt_modes)
    print(f"Prompt modes: {prompt_modes}", flush=True)

    rows = load_rows(
        input_csv=args.input_csv,
        dicom_root=args.dicom_root,
        selected_case_ids=selected_case_ids,
        fold_id=args.fold_id,
        split_name=split_name,
        start=args.start,
        limit=args.limit,
    )
    print(
        f"Loaded rows: {len(rows)} | fold_id={args.fold_id} | start={args.start} | limit={args.limit}",
        flush=True,
    )

    processor, model, model_load_time_s = load_model_and_processor(args.model_path, args.mode)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    done = load_done_keys(jsonl_path)
    print(f"Resume keys already done: {len(done)}", flush=True)

    for idx, row in enumerate(rows, start=1):
        join_filename = str(row.get("_join_filename", "")).strip()
        cv_filename = str(row.get("cv_filename_resolved") or row.get("cv_filename") or row.get("filename") or "")

        for prompt_mode in prompt_modes:
            run_key = build_run_key(join_filename, prompt_mode, args.mode)
            if run_key in done:
                print(f"SKIP done [{idx}/{len(rows)}] {join_filename} | {args.mode} | {prompt_mode}", flush=True)
                continue

            print(f"[{idx}/{len(rows)}] {join_filename} | {args.mode} | {prompt_mode}", flush=True)

            rec = {
                "run_key": run_key,
                "_join_filename": join_filename,
                "cv_filename": cv_filename,
                "dicom_path": row.get("dicom_path", ""),
                "cv_fold": row.get("cv_fold", ""),
                "split_name": row.get("split_name", ""),
                "study_id": row.get("study_id", ""),
                "other_id": row.get("other_id", ""),
                "expertise_en": row.get("expertise_en", ""),
                "prediction_nlp": row.get("prediction_nlp", ""),
                "prediction_cv": row.get("prediction_cv", ""),
                "model_name": Path(args.model_path).name,
                "gpu_name": gpu_name,
                "mode": args.mode,
                "prompt_mode": prompt_mode,
                "prediction": "",
                "success": False,
                "error": "",
                "model_load_time_s": round(model_load_time_s, 2),
                "latency_ms": None,
                "peak_gpu_mem_allocated_gb": None,
                "peak_gpu_mem_reserved_gb": None,
                "output_words": 0,
                "photometric_interpretation": "",
                "image_size": "",
                "pixel_min": None,
                "pixel_max": None,
            }

            try:
                dicom_path = rec["dicom_path"]
                if not dicom_path or not os.path.exists(dicom_path):
                    raise FileNotFoundError(dicom_path)

                image, meta = load_dicom_as_pil(dicom_path)
                rec.update(meta)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                t0 = time.perf_counter()
                prediction = run_inference(
                    model=model,
                    processor=processor,
                    image=image,
                    prompt=get_prompt_text(prompt_mode),
                    max_new_tokens=args.max_new_tokens,
                )

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                latency_ms = (time.perf_counter() - t0) * 1000.0

                allocated_gb = (
                    torch.cuda.max_memory_allocated() / (1024 ** 3)
                    if torch.cuda.is_available() else 0.0
                )
                reserved_gb = (
                    torch.cuda.max_memory_reserved() / (1024 ** 3)
                    if torch.cuda.is_available() else 0.0
                )

                rec.update({
                    "prediction": prediction,
                    "success": True,
                    "latency_ms": round(latency_ms, 2),
                    "peak_gpu_mem_allocated_gb": round(allocated_gb, 3),
                    "peak_gpu_mem_reserved_gb": round(reserved_gb, 3),
                    "output_words": len(str(prediction).split()),
                })

                print(
                    f"  OK latency={latency_ms:.1f} ms | reserved={reserved_gb:.2f} GB | words={rec['output_words']}",
                    flush=True,
                )

            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                print(f"  ERROR: {rec['error']}", flush=True)

            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if rec["success"]:
                done.add(run_key)

    jsonl_to_clean_csv(jsonl_path, csv_path)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
