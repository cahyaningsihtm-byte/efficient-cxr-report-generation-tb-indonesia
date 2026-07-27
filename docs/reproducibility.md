# Reproducibility Guide

## Scope

This package supports code-level and pipeline-level reproduction of the computational workflow behind the camera-ready paper on an appropriately approved local dataset. It does not make the clinical dataset, patient split definitions, case-level outputs, or full model checkpoints public, so it is not a full numerical reproduction package. The documented adapter-only LoRA artefact is the approved public exception.

## Prerequisites

1. A compatible Linux GPU environment. The reported deployment measurements used an NVIDIA L4 GPU.
2. The exported Conda specifications described in [../environments/README.md](../environments/README.md). The Qwen environment records Python 3.10.20, PyTorch 2.7.1+cu118, Transformers 5.8.0, PEFT 0.18.1, and bitsandbytes 0.49.2.
3. Approved local access to the image/report dataset and any required model checkpoints. The released de-identified PNG examples can be used only for smoke testing the public inference path; they are not sufficient to reproduce paper metrics.
4. A compatible local checkout of the external X-Raydar NLP implementation for semantic scoring.

## Configuration

Copy the path template locally:

~~~bash
cp configs/paths.example.env .env
~~~

Replace the example path values with approved local directories. The `.env` file is ignored by Git and must never be committed. Public, non-sensitive experiment metadata is summarised in `configs/paper_experiment.example.yaml`.

## Completing Model Metadata

Use the exact local checkpoint and adapter directories from the paper runs. For each model directory, inspect `config.json`, `tokenizer_config.json`, and any Hugging Face snapshot metadata available in the local cache. Record the upstream model ID, local checkpoint directory name, architecture, model type, and exact snapshot commit or revision when available.

For PEFT/LoRA adapters, inspect `adapter_config.json`. Record the base model path or ID, PEFT type, LoRA rank, alpha, dropout, target modules, and whether the adapter weights are approved for public release. If the adapter is not approved for release, keep only the descriptive metadata and do not include the weight files.

## Pipeline

1. **Teacher inference.** Run `src/quantization/infer_qwen_quantized.py` separately for FP16, INT8, and INT4 and for native and structured-prompted inference. Inputs are an approved metadata CSV, an approved local DICOM root, a Qwen checkpoint path, and fold membership. Outputs are local inference CSV/JSONL files that must remain outside Git.
2. **Teacher output preparation and scoring.** Use `src/evaluation/prepare_xraydar_input.py`, then `src/evaluation/score_with_xraydar.py --repo-dir /path/to/xraydar/src`. Inputs are generated report CSVs; outputs are local NLP-score CSVs.
3. **TB-14 evaluation.** Use `src/evaluation/evaluate_tb14.py` to create aggregate summary and per-label scores. Outputs include summary tables and per-sample metrics; do not commit per-sample output.
4. **Student distillation.** Use `src/distillation/train_chexagent_reprkd.py` with the approved CheXAgent base model, stage-1 adapter, training JSON, teacher representation cache, and image folder.
5. **Student inference.** Use `src/distillation/infer_chexagent_kd.py`; then use `src/evaluation/build_clean_eval_csv.py` and repeat steps 2-3.
6. **Paper analyses.** Use `src/analysis/compute_bootstrap_micro_f1_ci.py` for confidence intervals and `src/analysis/create_tb14_kd_delta_chart.py` for the per-abnormality figure.

## Public example smoke test

The released LoRA adapter can be run on the two de-identified PNG examples after the authorised CheXAgent base model is available locally:

~~~bash
python src/distillation/infer_chexagent_kd.py \
  --model-path /path/to/StanfordAIMI-CheXagent-2-3b \
  --adapter-path models/adapters/qwen_fp16_report_reprkd_prompted_l4_epoch1 \
  --test-json data/examples/deidentified_chexagent_items.json \
  --manifest-csv data/examples/deidentified_chexagent_manifest.csv \
  --output-csv outputs/example_chexagent_lora_predictions.csv \
  --prompt-mode prompted \
  --mode chex_qf16_rrkd_example \
  --model-name CheX-QF16-RRKD \
  --limit 2
~~~

This smoke test checks the public adapter loading and image-inference path. It does not reproduce the paper metrics because the private fold definitions, case-level reference labels, and full institutional dataset are not public.

## Required reporting controls

- Record the exact model identifier, revision/commit, library versions, GPU, precision, prompt mode, decoding settings, batch size, and random seed.
- Report latency as median seconds per report and state whether model-load time is excluded.
- Report the teacher as `Qwen3.5-9B`, a multimodal Qwen vision-language model corresponding to the verified local `qwen3.5-9b` checkpoint and upstream model `Qwen/Qwen3.5-9B`; avoid the historical `Qwen-VL` name because it refers to a different model family.
- Keep raw reports, identifiers, per-case predictions, and paths outside Git.

## Safe local validation

Before sharing a derivative release or contributing changes, run the checks below and manually inspect every non-ignored file. A clean Git status is not proof that a file is safe to publish.

~~~bash
python -m compileall -q src
git status --ignored
git check-ignore -v data/example_file.dcm checkpoints/model.safetensors .env
~~~
