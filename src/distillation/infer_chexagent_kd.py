#!/usr/bin/env python
from __future__ import annotations

import argparse, csv, json, os, re, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


NORMALIZATION_MAP = {
    "ild": "interstitial lung disease",
    "nodule/mass": "pulmonary nodule or mass",
    "pleural thickening": "pleural thickening",
    "pleural effusion": "pleural effusion",
    "consolidation": "consolidation",
    "pulmonary fibrosis": "pulmonary fibrosis",
    "infiltration": "infiltration",
}


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


def dcm_to_pil(dcm_path: str) -> Image.Image:
    ds = pydicom.dcmread(dcm_path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
    arr = arr.astype(np.uint8)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = 255 - arr
    return Image.fromarray(arr).convert("RGB")


def extract_findings(raw_text: str) -> list[str]:
    text = str(raw_text).strip()
    findings = []
    findings.extend(re.findall(r"<\|ref\|>\s*(.*?)\s*</ref>", text, flags=re.I))
    findings.extend(re.findall(r"([A-Za-z][A-Za-z/\-\s]+?)\s*</ref>", text, flags=re.I))

    known_terms = [
        "Consolidation", "Pleural effusion", "Pleural thickening",
        "Peribronchovascular interstitial opacity", "ILD", "Pulmonary fibrosis",
        "Infiltration", "Nodule/Mass", "Aortic enlargement",
        "Cardiomegaly", "Atelectasis", "Pneumothorax",
    ]
    for term in known_terms:
        if re.search(rf"\b{re.escape(term)}\b", text, flags=re.I):
            findings.append(term)

    cleaned, seen = [], set()
    for f in findings:
        f = re.sub(r"\s+", " ", f.strip().strip("\"'").strip(" ,;:."))
        if not f:
            continue
        key = NORMALIZATION_MAP.get(f.lower(), f.lower())
        if key not in seen:
            seen.add(key)
            cleaned.append(key)
    return cleaned


def format_findings_list(findings: list[str]) -> str:
    if not findings:
        return "no clear finding extracted"
    if len(findings) == 1:
        return findings[0]
    if len(findings) == 2:
        return f"{findings[0]} and {findings[1]}"
    return ", ".join(findings[:-1]) + f", and {findings[-1]}"


def generate_pseudo_report(findings: list[str]) -> str:
    if not findings:
        return "No clear radiographic abnormality could be summarized from the model output. Manual review is recommended."

    fset = set(findings)
    s = []

    if "consolidation" in fset:
        s.append("There is focal air-space consolidation.")
    if "pleural effusion" in fset:
        s.append("A pleural effusion is present.")
    if "pleural thickening" in fset:
        s.append("Pleural thickening is present.")
    if "interstitial lung disease" in fset:
        s.append("Interstitial lung disease-like opacity is noted.")
    if "pulmonary fibrosis" in fset:
        s.append("Pulmonary fibrotic change is noted.")
    if "infiltration" in fset:
        s.append("Patchy pulmonary infiltration is present.")
    if "pulmonary nodule or mass" in fset:
        s.append("A nodular or mass-like pulmonary opacity is identified.")
    if "cardiomegaly" in fset:
        s.append("Cardiomegaly is present.")
    if "atelectasis" in fset:
        s.append("Atelectatic change is present.")
    if "pneumothorax" in fset:
        s.append("Pneumothorax is present.")

    if not s:
        s.append(f"Extracted radiographic findings include {format_findings_list(findings)}.")

    s.append(f"Impression: Extracted findings suggest {format_findings_list(findings)}.")
    return " ".join(s)


def load_items(test_json, manifest_csv):
    items = json.load(open(test_json, "r", encoding="utf-8"))
    man = pd.read_csv(manifest_csv)

    if "cv_filename" in man.columns:
        id_col = "cv_filename"
    elif "id" in man.columns:
        id_col = "id"
        man["cv_filename"] = man["id"].astype(str).map(lambda x: f"{Path(x).stem}.dcm")
    elif "image" in man.columns:
        id_col = "image"
        man["cv_filename"] = man["image"].astype(str).map(lambda x: f"{Path(x).stem}.dcm")
    elif "dicom_path" in man.columns:
        id_col = "dicom_path"
        man["cv_filename"] = man["dicom_path"].astype(str).map(lambda x: Path(x).name)
    else:
        raise ValueError(f"Cannot infer id column. Manifest columns: {list(man.columns)}")

    man["_id"] = man[id_col].astype(str).map(lambda x: Path(x).stem)
    meta = {r["_id"]: r.to_dict() for _, r in man.iterrows()}
    return items, meta


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--test-json", required=True)
    ap.add_argument("--train-fold", default="NA")
    ap.add_argument("--test-fold", default="0")
    ap.add_argument("--manifest-csv", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--prompt-mode", choices=["native", "prompted"], required=True)
    ap.add_argument("--mode", default="chexagent_baseline_bf16")
    ap.add_argument("--model-name", default="CheXagent")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # Cache locations are optional and must be supplied by the local user.
    # No institution-specific server paths are embedded in this public script.
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(hf_home) / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(hf_home) / "hub"))

    print("Loading tokenizer:", args.model_path, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("Loading model:", args.model_path, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        trust_remote_code=True,
    )
    model = model.to(torch.bfloat16)

    print("Loading adapter:", args.adapter_path, flush=True)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.to(torch.bfloat16)

    model.eval()

    #total_params = count_params(model)

    total_params = count_params(model)
    print(f"total_params={total_params:,}", flush=True)
    base_params = count_params(model.base_model.model)
    adapter_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    adapter_param_pct = adapter_params / total_params * 100
    trainable_params_for_training = adapter_params
    trainable_param_pct = adapter_param_pct

    items, meta = load_items(args.test_json, args.manifest_csv)
    if args.limit:
        items = items[: args.limit]

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "cv_filename", "dicom_path", "image", "train_fold", "test_fold",
        "model_name", "mode", "prompt_mode",
        "expertise_en", "prediction_nlp", "prediction_cv",
        "prediction", "report_raw", "findings_clean",
        "latency_ms", "output_words",
        "peak_gpu_mem_allocated_gb", "peak_gpu_mem_reserved_gb",
        "total_params", "base_params", "adapter_params", "adapter_param_pct",
        "trainable_params_for_training", "trainable_param_pct",
        "success", "error",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, item in enumerate(items, 1):
            cid = Path(str(item["id"])).stem
            m = meta.get(cid, {})
            dcm_path = m.get("dicom_path", "")
            cv_filename = m.get("cv_filename", f"{cid}.dcm")
            image_name = m.get("image", item.get("image", ""))

            rec = {
                "cv_filename": cv_filename,
                "dicom_path": dcm_path,
                "image": image_name,
                "train_fold": args.train_fold,
                "test_fold": args.test_fold,
                "model_name": args.model_name,
                "mode": args.mode,
                "prompt_mode": args.prompt_mode,
                "expertise_en": m.get("expertise_en", m.get("report", "")),
                "prediction_nlp": m.get("prediction_nlp", ""),
                "prediction_cv": m.get("prediction_cv", ""),
                "total_params": total_params,
                "base_params": base_params,
                "adapter_params": adapter_params,
                "adapter_param_pct": adapter_param_pct,
                "trainable_params_for_training": trainable_params_for_training,
                "trainable_param_pct": trainable_param_pct,
                "success": False,
                "error": "",
            }

            temp_png = None
            try:
                img = dcm_to_pil(dcm_path)
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp") as tmp:
                    temp_png = tmp.name
                img.save(temp_png)

                prompt = TB14_PROMPT if args.prompt_mode == "prompted" else "Describe the findings in this chest X-ray."

                query = tokenizer.from_list_format([
                    {"image": temp_png},
                    {"text": prompt},
                ])

                conv = [
                    {"from": "system", "value": "You are a helpful assistant."},
                    {"from": "human", "value": query},
                ]

                input_ids = tokenizer.apply_chat_template(
                    conv,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )

                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()

                with torch.no_grad():
                    output = model.generate(
                        input_ids.to("cuda"),
                        do_sample=False,
                        num_beams=1,
                        temperature=1.0,
                        top_p=1.0,
                        use_cache=True,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                    )[0]

                latency_ms = (time.perf_counter() - t0) * 1000
                alloc_gb = torch.cuda.max_memory_allocated() / 1024**3
                reserved_gb = torch.cuda.max_memory_reserved() / 1024**3

                raw = tokenizer.decode(
                    output[input_ids.size(1):-1],
                    skip_special_tokens=False,
                ).strip()

                findings = extract_findings(raw)
                pred = generate_pseudo_report(findings)

                rec.update({
                    "prediction": pred,
                    "report_raw": raw,
                    "findings_clean": "; ".join(findings),
                    "latency_ms": latency_ms,
                    "output_words": len(pred.split()),
                    "peak_gpu_mem_allocated_gb": alloc_gb,
                    "peak_gpu_mem_reserved_gb": reserved_gb,
                    "success": True,
                    "error": "",
                })

                print(
                    f"[{i}/{len(items)}] OK {cv_filename} | mode={args.mode} | prompt={args.prompt_mode} "
                    f"| latency={latency_ms:.1f} ms | reserved={reserved_gb:.2f} GB | words={rec['output_words']}",
                    flush=True,
                )

            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                print(f"[{i}/{len(items)}] ERROR {cv_filename}: {rec['error']}", flush=True)

            finally:
                if temp_png and os.path.exists(temp_png):
                    os.remove(temp_png)

            writer.writerow(rec)
            f.flush()

    print("Saved:", out_path, flush=True)


if __name__ == "__main__":
    main()
