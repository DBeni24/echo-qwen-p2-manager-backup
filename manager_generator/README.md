# Echo / BeniQwen Manager Dataset Generator

Portable manager-policy dataset generator.

## Release candidate

- Runtime: V05_SAFE_POLICY_V21_V20_QUALITY_RUNTIME_RC1
- Quality: V05_SAFE_POLICY_V20_QUALITY_EXTENSION
- Repair: V05_SAFE_POLICY_V21_REPAIR_CORE_V20
- Generation and repair prompts are frozen.

## Requirements

- Python 3.11 or 3.12
- NVIDIA CUDA GPU
- torch
- transformers
- accelerate
- safetensors

The Qwen model weights are not included.
Datasets, reject logs, checkpoints and secrets must not be committed.

Set ECHO_MODEL_PATH to the local Qwen3.5-9B directory.
Set ECHO_RUN_ROOT to the desired output directory.

Linux: run ./run_generator.sh
Windows PowerShell: run .\run_generator.ps1

Run a smoke test and kill/resume test before a long generation run.
