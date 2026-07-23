# Model And Weight Availability

The teacher runtime used in the paper experiments was loaded from the official Hugging Face repository as `Qwen/Qwen3.5-9B` and mirrored locally as `qwen3.5-9b`, with architecture `Qwen3_5ForConditionalGeneration`. The exact snapshot revision was not recoverable from the local checkpoint copy. This is the Qwen3.5-9B multimodal vision-language model and not the historical `Qwen-VL` model.

The exact revision or commit used for the final paper tables should be recorded in `configs/paper_experiment.example.yaml` from the local checkpoint or run manifest.

| Role | Model identifier used or to verify | Public source |
| --- | --- | --- |
| Qwen teacher | `Qwen/Qwen3.5-9B` | <https://huggingface.co/Qwen/Qwen3.5-9B> |
| CheXAgent student/base baseline | `StanfordAIMI/CheXagent-2-3b` | <https://huggingface.co/StanfordAIMI/CheXagent-2-3b> |
| MedGemma baseline | `google/medgemma-1.5-4b-it` | <https://huggingface.co/google/medgemma-1.5-4b-it> |
| CXR-LLaVA-v2 baseline | `ECOFRI/CXR-LLAVA-v2` | <https://huggingface.co/ECOFRI/CXR-LLAVA-v2> |
| VILA-M3-3B baseline | `MONAI/Llama3-VILA-M3-3B` | <https://huggingface.co/MONAI/Llama3-VILA-M3-3B> |

The project does not redistribute Qwen, CheXAgent, X-Raydar, or other third-party model files. Users must obtain each upstream model from its authorised source and comply with its licence and access conditions.

Trained LoRA adapter weights are not included in this repository. Any future distribution of adapters is subject to upstream licences, institutional governance, and explicit release approval.
