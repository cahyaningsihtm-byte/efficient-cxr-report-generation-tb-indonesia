# Software Environments

This directory provides Conda environment specifications for the released workflow. The files describe software dependencies only; they do not contain the Conda installation itself, clinical data, model weights, access tokens, or machine-specific paths.

## Tested Core Setup

The Qwen inference environment exported from the paper workflow records:

- OS: Linux GPU environment.
- Python: 3.10.20.
- PyTorch: 2.7.1+cu118.
- CUDA runtime packages: cu118.
- Transformers: 5.8.0.
- PEFT: 0.18.1.
- Quantization library: bitsandbytes 0.49.2.
- Additional quantization/deployment libraries present in the export: torchao 0.15.0, optimum 2.1.0, onnxruntime-gpu 1.23.2.
- Reported GPU: NVIDIA L4.
- Approximate allocated VRAM in the aggregate results: about 19 GB for Qwen FP16 and about 9 GB for Qwen INT4 on L4.

| File | Purpose |
| --- | --- |
| `qwen.yml` | Minimal environment specification for Qwen inference. |
| `qwen-lock.yml` | Resolved Qwen environment specification used for more closely matched installations. |
| `chexagent.yml` | Environment specification for CheXAgent distillation and inference. |
| `xraydar_nlp.yml` | Environment specification for X-Raydar-based clinical-semantic evaluation. |

Create an environment with Conda, for example:

~~~bash
conda env create -f environments/qwen.yml
conda activate qwen
~~~

The environment specifications are intended to support software setup on a compatible CUDA-enabled system. Exact runtime behaviour can vary with the GPU driver, CUDA version, upstream model revision, and third-party package availability. Model acquisition and access requirements are described in [../models/README.md](../models/README.md).
