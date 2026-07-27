# Qwen FP16 Report+Representation KD LoRA Adapter

This directory contains the study-produced CheXAgent LoRA adapter used for the report-plus-representation knowledge-distillation experiment in the EMA MICCAI 2026 paper.

## Contents

| File | Description | Approximate size |
| --- | --- | --- |
| `adapter_config.json` | Sanitised PEFT LoRA adapter configuration. The base model is listed as `StanfordAIMI/CheXagent-2-3b`. | 0.008 MiB |
| `adapter_model.safetensors` | PEFT LoRA adapter weights. | 9.795 MiB / 10.27 MB |
| `representation_projection.pt` | Representation-alignment projection used by the report-plus-representation KD workflow. | 40.018 MiB |

The original local training paths and run-specific private data paths are not included.

## Intended use

Use this adapter with the CheXAgent base model. This repository does not redistribute the CheXAgent base weights; users must obtain the base model from its authorised source and comply with the upstream licence and access conditions.

This adapter is provided for research reproducibility only. It is not intended for clinical use, diagnosis, triage, or patient management.

## What is not included

The merged checkpoint directory `qwen_fp16_report_reprkd_prompted_l4_epoch1_merged/` is not included. That artefact was produced by merging the LoRA adapter into the CheXAgent base model for deployment/compression experiments and is distinct from the adapter-only release here.
