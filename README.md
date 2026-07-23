# Efficient CXR Report Generation for TB Screening in Indonesia

This repository is the curated reproducibility package for the EMA MICCAI 2026 paper:

> **Accelerating Chest X-ray Report Generation for Tuberculosis Screening in Indonesia: An Efficiency-Aware Benchmark of Quantized and Distilled Vision-Language Models**

The project benchmarks a multimodal Qwen teacher under FP16, INT8, and INT4 inference and distils report-level and representation-level supervision to a CheXAgent student. It evaluates TB-14 clinical-semantic fidelity together with latency and allocated VRAM.

## Data source and institutional collaboration

The study dataset comprises chest X-ray images and paired radiology reports from the tuberculosis registry of Cipto Mangunkusumo Hospital (RSCM), Jakarta. Access to the anonymised data was provided through the BDC IMERI Idealab repository at the Indonesian Medical Education and Research Institute (IMERI), covering cases from 2015 and 2018-2019. The underlying clinical records are not distributed through this repository.

## Repository status

This is a **private preparation repository**. It contains only the selected, sanitised code and documentation needed to reproduce the published workflow on an appropriately approved dataset. It does not contain patient images, reports, identifiers, per-case predictions, checkpoints, or API tokens.

The public release is prepared following a separate internal privacy and reproducibility review. The published repository will contain only the reviewed materials described below.

## What is included

- Qwen FP16/INT8/INT4 inference code.
- CheXAgent report-plus-representation knowledge-distillation code.
- Data-cleaning, X-Raydar preparation/scoring, TB-14 evaluation, bootstrap, and figure-generation code.
- Portable configuration templates and privacy-preserving documentation.

## What is not included

- The institutional CXR-report dataset, DICOM files, raw reports, and patient-level metadata.
- Per-case predictions, evaluation files, scheduler logs, model caches, or checkpoints.
- Third-party model weights and the external X-Raydar NLP source code.

See [data/README.md](data/README.md) for the data policy and [models/README.md](models/README.md) for model and weight handling.

## Quick start

1. Obtain approval to use an appropriate local dataset; do not copy it into this repository.
2. Export the required Conda environments from the original compute system as described in [environments/README.md](environments/README.md).
3. Copy `configs/paths.example.env` to a local `.env` file and replace every placeholder.
4. Complete `configs/paper_experiment.example.yaml` from the manifest of the final experimental run.
5. Follow the pipeline in [docs/reproducibility.md](docs/reproducibility.md).

## Clinical-use statement

This research is for retrospective benchmarking and human-verifiable reporting support. It is **not** an autonomous diagnostic system and must not be used for clinical decision-making without appropriate prospective validation, local governance approval, and clinician oversight.

## Acknowledgements

This work was supported by the MICCAI Society 2026 Project Climate and Environmental Health award.

## Citation and authorship

Citation metadata, author affiliations, and licensing information will be provided in the final archival release.
