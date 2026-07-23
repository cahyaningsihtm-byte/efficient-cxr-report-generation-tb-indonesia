# Software Environments

This directory provides Conda environment specifications for the released workflow. The files describe software dependencies only; they do not contain the Conda installation itself, clinical data, model weights, access tokens, or machine-specific paths.

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
