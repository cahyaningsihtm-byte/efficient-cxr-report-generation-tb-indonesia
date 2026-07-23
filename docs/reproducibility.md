# Reproducibility Guide

## Scope

This package reproduces the computational workflow behind the camera-ready paper on an appropriately approved local dataset. It does not make the clinical dataset or trained weights public.

## Prerequisites

1. A compatible Linux GPU environment. The reported deployment measurements used an NVIDIA L4 GPU.
2. The exported Conda specifications described in [../environments/README.md](../environments/README.md).
3. Approved local access to the image/report dataset and any required model checkpoints.
4. A compatible local checkout of the external X-Raydar NLP implementation for semantic scoring.

## Configuration

Copy the path template locally:

~~~bash
cp configs/paths.example.env .env
~~~

Replace all placeholder values. The `.env` file is ignored by Git and must never be committed. Then complete the model IDs, revision, adapter location, and hardware information in `configs/paper_experiment.example.yaml`.

## Pipeline

1. **Teacher inference.** Run `src/quantization/infer_qwen_quantized.py` separately for FP16, INT8, and INT4 and for native and structured-prompted inference. Inputs, output directories, and fold membership must remain outside this repository.
2. **Teacher output preparation and scoring.** Use `src/evaluation/prepare_xraydar_input.py`, then `src/evaluation/score_with_xraydar.py --repo-dir /path/to/xraydar/src`.
3. **TB-14 evaluation.** Use `src/evaluation/evaluate_tb14.py` to create aggregate summary and per-label scores. Do not commit its per-sample output.
4. **Student distillation.** Use `src/distillation/train_chexagent_reprkd.py` with the approved CheXAgent base model, stage-1 adapter, training JSON, teacher representation cache, and image folder.
5. **Student inference.** Use `src/distillation/infer_chexagent_kd.py`; then use `src/evaluation/build_clean_eval_csv.py` and repeat steps 2-3.
6. **Paper analyses.** Use `src/analysis/compute_bootstrap_micro_f1_ci.py` for confidence intervals and `src/analysis/create_tb14_kd_delta_chart.py` for the per-abnormality figure.

## Required reporting controls

- Record the exact model identifier, revision/commit, library versions, GPU, precision, prompt mode, decoding settings, batch size, and random seed.
- Report latency as median seconds per report and state whether model-load time is excluded.
- Keep the trained model family and checkpoint name unambiguous. Do not describe the teacher as `Qwen-VL` unless that exact, historical model was actually used.
- Keep raw reports, identifiers, per-case predictions, and paths outside Git.

## Validation before release

Run the checks below before making the repository public:

~~~bash
python -m compileall -q src
git status --ignored
git check-ignore -v data/example_file.dcm checkpoints/model.safetensors .env
~~~

Review every non-ignored file manually. A clean Git status is not proof that a file is safe to publish.
