# Accelerating Chest X-ray Report Generation for Tuberculosis Screening in Indonesia: An Efficiency-Aware Benchmark of Quantized and Distilled Vision-Language Models

This repository is the curated reproducibility package for the EMA MICCAI 2026 paper above. The repository name is shortened for readability, but the full paper title is used here as the authoritative title.

The project benchmarks a Qwen3.5-9B multimodal vision-language teacher under FP16, INT8, and INT4 inference and distils report-level and representation-level supervision to a CheXAgent student. It evaluates TB-14 clinical-semantic fidelity together with latency and allocated VRAM.

## Model identity

The teacher model used for the Qwen experiments is `Qwen3.5-9B` with a 9B-parameter model family, loaded in the experimental environment as `unsloth/Qwen3.5-9B`. This is a multimodal Qwen vision-language model and should not be described as the historical `Qwen-VL` model. The Qwen teacher was selected for distillation because it achieved the strongest prompted micro-F1 among the candidate VLM baselines reported in the aggregate results.

## Data source and institutional collaboration

The study dataset comprises chest X-ray images and paired radiology reports from the tuberculosis registry of Cipto Mangunkusumo Hospital (RSCM), Jakarta. Access to the anonymised data was provided through the BDC IMERI Idealab repository at the Indonesian Medical Education and Research Institute (IMERI), covering cases from 2015 and 2018-2019. The underlying clinical records are not distributed through this repository.

## What is included

- Qwen FP16/INT8/INT4 inference code.
- CheXAgent report-plus-representation knowledge-distillation code.
- Data-cleaning, X-Raydar preparation/scoring, TB-14 evaluation, bootstrap, and figure-generation code.
- Portable configuration templates and privacy-preserving documentation.
- Model source links and experiment configuration templates.
- Two synthetic format examples showing the expected CSV structure without redistributing clinical records.

## What is not included

- The institutional CXR-report dataset, DICOM files, raw reports, and patient-level metadata.
- Per-case predictions, evaluation files, scheduler logs, model caches, or checkpoints.
- Third-party model weights and the external X-Raydar NLP source code.

See [data/README.md](data/README.md) for the data policy and [models/README.md](models/README.md) for model and weight handling.

## Quick start

1. Create the required Conda environment using the specifications in [environments/README.md](environments/README.md).
2. Copy `configs/paths.example.env` to a local `.env` file and provide the paths for the approved local data and model directories.
3. Review `configs/paper_experiment.example.yaml` and set the experiment-specific parameters.
4. Follow the pipeline in [docs/reproducibility.md](docs/reproducibility.md).

## Clinical-use statement

This research is for retrospective benchmarking and human-verifiable reporting support. It is **not** an autonomous diagnostic system and must not be used for clinical decision-making without appropriate prospective validation, local governance approval, and clinician oversight.

## Acknowledgements

This work was supported by the MICCAI Society 2026 Project Climate and Environmental Health award.
