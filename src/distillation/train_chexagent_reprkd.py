#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


NATIVE_PROMPT = "Describe the findings in this chest X-ray."

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="Base CheXAgent path")
    ap.add_argument("--adapter-path", required=True, help="Existing report-KD LoRA adapter path to continue from")
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--teacher-repr-cache", required=True)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--prompt-mode", choices=["native", "prompted"], required=True)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--projection-lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--gt-weight", type=float, default=0.4)
    ap.add_argument("--teacher-weight", type=float, default=0.5)
    ap.add_argument("--repr-weight", type=float, default=0.1)
    ap.add_argument("--student-hidden-size", type=int, default=None)
    ap.add_argument("--max-grad-norm", type=float, default=0.3)
    return ap.parse_args()


def get_prompt(prompt_mode):
    return TB14_PROMPT if prompt_mode == "prompted" else NATIVE_PROMPT


def make_query(tokenizer, image_path, prompt):
    return tokenizer.from_list_format([
        {"image": str(image_path)},
        {"text": prompt},
    ])


def build_batch(tokenizer, image_path, prompt, answer, device):
    query = make_query(tokenizer, image_path, prompt)

    conv_prompt = [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": query},
    ]

    prompt_ids = tokenizer.apply_chat_template(
        conv_prompt,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    answer_ids = tokenizer(
        str(answer).strip(),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"]

    input_ids = torch.cat(
        [prompt_ids, answer_ids.to(prompt_ids.device)],
        dim=1,
    )

    labels = input_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100

    valid_tokens = int((labels != -100).sum().item())
    if valid_tokens == 0:
        raise RuntimeError("No valid label tokens after appending answer_ids.")

    return {
        "input_ids": input_ids.to(device),
        "labels": labels.to(device),
        "prompt_len": int(prompt_ids.shape[1]),
        "full_len": int(input_ids.shape[1]),
        "answer_len": int(answer_ids.shape[1]),
    }


def model_inputs(batch):
    return {"input_ids": batch["input_ids"], "labels": batch["labels"]}


def get_hidden_states(outputs):
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states
    if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states is not None:
        return outputs.decoder_hidden_states
    raise RuntimeError("Student output does not contain hidden_states.")


def infer_hidden_size(model, explicit):
    if explicit:
        return explicit

    cfg = getattr(model, "config", None)
    candidates = [
        "hidden_size",
        "n_embd",
        "d_model",
    ]
    for name in candidates:
        value = getattr(cfg, name, None)
        if isinstance(value, int) and value > 0:
            return value

    for nested_name in ["text_config", "llm_config", "language_config"]:
        nested = getattr(cfg, nested_name, None)
        if nested is None:
            continue
        for name in candidates:
            value = getattr(nested, name, None)
            if isinstance(value, int) and value > 0:
                return value

    raise RuntimeError(
        "Could not infer student hidden size from config. "
        "Pass --student-hidden-size explicitly."
    )


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_teacher_repr(path):
    obj = safe_torch_load(path)
    if isinstance(obj, dict) and "repr" in obj:
        repr_map = obj["repr"]
        metadata = obj.get("metadata", {})
    else:
        repr_map = obj
        metadata = {}
    if not repr_map:
        raise RuntimeError(f"No teacher representations found in {path}")
    return repr_map, metadata


def main():
    args = parse_args()

    # Cache locations are optional and must be supplied by the local user.
    # No institution-specific server paths are embedded in this public script.
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(hf_home) / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(hf_home) / "hub"))

    rows = json.load(open(args.train_json, "r", encoding="utf-8"))
    if args.max_samples:
        rows = rows[: args.max_samples]

    teacher_repr, teacher_meta = load_teacher_repr(args.teacher_repr_cache)
    first_repr = next(iter(teacher_repr.values()))
    teacher_dim = int(first_repr.numel())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Teacher repr cache:", args.teacher_repr_cache, flush=True)
    print("Teacher repr rows:", len(teacher_repr), flush=True)
    print("Teacher repr dim:", teacher_dim, flush=True)
    print("Teacher repr metadata:", teacher_meta, flush=True)

    print("Loading tokenizer:", args.model_path, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("Loading base model:", args.model_path, flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model = base_model.to(torch.bfloat16)
    base_model.config.use_cache = False

    print("Loading trainable adapter:", args.adapter_path, flush=True)
    model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=True)
    model = model.to(torch.bfloat16)
    model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled", flush=True)

    model.train()
    model.print_trainable_parameters()

    student_dim = infer_hidden_size(model, args.student_hidden_size)
    print("Student hidden dim:", student_dim, flush=True)

    device = next(model.parameters()).device
    projection = torch.nn.Linear(student_dim, teacher_dim).to(device=device, dtype=torch.float32)
    projection.train()

    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr},
            {"params": projection.parameters(), "lr": args.projection_lr},
        ]
    )

    prompt = get_prompt(args.prompt_mode)
    valid = 0
    skipped_missing_repr = 0
    opt.zero_grad()

    print(
        f"Rows={len(rows)} prompt={args.prompt_mode} "
        f"gt_w={args.gt_weight} teacher_w={args.teacher_weight} repr_w={args.repr_weight}",
        flush=True,
    )

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}", flush=True)

        for step, item in enumerate(rows, 1):
            cid = str(item["id"])
            try:
                if cid not in teacher_repr:
                    skipped_missing_repr += 1
                    print(f"step={step} SKIP missing teacher repr: {cid}", flush=True)
                    continue

                image_path = Path(args.image_folder) / item["image"]
                gt_report = str(item["gt_report"]).strip()
                teacher_report = str(item["teacher_report"]).strip()

                gt_batch = build_batch(tokenizer, image_path, prompt, gt_report, device)
                teacher_batch = build_batch(tokenizer, image_path, prompt, teacher_report, device)

                gt_outputs = model(**model_inputs(gt_batch), return_dict=True)
                teacher_outputs = model(
                    **model_inputs(teacher_batch),
                    output_hidden_states=True,
                    return_dict=True,
                )

                gt_loss = gt_outputs.loss
                teacher_loss = teacher_outputs.loss

                hidden = get_hidden_states(teacher_outputs)[-1][0]
                start = teacher_batch["prompt_len"]
                end = teacher_batch["full_len"]
                student_answer_hidden = hidden[start:end]
                if student_answer_hidden.numel() == 0:
                    print(f"step={step} SKIP empty student answer hidden: {cid}", flush=True)
                    continue

                student_repr = student_answer_hidden.float().mean(dim=0)
                projected = projection(student_repr)

                target_repr = teacher_repr[cid].to(device=device, dtype=torch.float32).view(-1)
                repr_loss = F.mse_loss(
                    F.normalize(projected, dim=0),
                    F.normalize(target_repr, dim=0),
                )

                raw_loss = (
                    args.gt_weight * gt_loss
                    + args.teacher_weight * teacher_loss
                    + args.repr_weight * repr_loss
                )

                if torch.isnan(raw_loss):
                    print("NAN DETECTED")
                    print("gt_loss:", float(gt_loss.detach().cpu()))
                    print("teacher_loss:", float(teacher_loss.detach().cpu()))
                    print("repr_loss:", float(repr_loss.detach().cpu()))
                    print("cid:", cid)
                    continue

                loss = raw_loss / args.grad_accum
                loss.backward()
                valid += 1

                print(
                    f"step={step} cid={cid} "
                    f"gt={gt_loss.item():.4f} teacher={teacher_loss.item():.4f} "
                    f"repr={repr_loss.item():.6f} total={raw_loss.item():.4f}",
                    flush=True,
                )

                if valid % args.grad_accum == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        lora_params + list(projection.parameters()),
                        args.max_grad_norm,
                    )
                    opt.step()
                    opt.zero_grad()
                    print(f"optimizer step | grad_norm={float(grad_norm):.4f}", flush=True)

            except Exception as e:
                print(f"step={step} ERROR {type(e).__name__}: {e}", flush=True)
                opt.zero_grad()

    if valid == 0:
        raise RuntimeError("No valid training steps completed.")

    if valid % args.grad_accum != 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            lora_params + list(projection.parameters()),
            args.max_grad_norm,
        )
        opt.step()
        opt.zero_grad()
        print(f"final optimizer step | grad_norm={float(grad_norm):.4f}", flush=True)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(
        {
            "projection_state_dict": projection.state_dict(),
            "student_hidden_dim": student_dim,
            "teacher_hidden_dim": teacher_dim,
            "teacher_repr_cache": args.teacher_repr_cache,
            "loss_weights": {
                "gt_weight": args.gt_weight,
                "teacher_weight": args.teacher_weight,
                "repr_weight": args.repr_weight,
            },
        },
        out_dir / "representation_projection.pt",
    )
    (out_dir / "stage2_training_summary.json").write_text(
        json.dumps(
            {
                "base_model": args.model_path,
                "input_adapter": args.adapter_path,
                "train_json": args.train_json,
                "teacher_repr_cache": args.teacher_repr_cache,
                "output_dir": str(out_dir),
                "prompt_mode": args.prompt_mode,
                "rows": len(rows),
                "valid_steps": valid,
                "skipped_missing_repr": skipped_missing_repr,
                "loss_weights": {
                    "gt_weight": args.gt_weight,
                    "teacher_weight": args.teacher_weight,
                    "repr_weight": args.repr_weight,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved adapter:", out_dir, flush=True)
    print("Saved projection:", out_dir / "representation_projection.pt", flush=True)


if __name__ == "__main__":
    main()
