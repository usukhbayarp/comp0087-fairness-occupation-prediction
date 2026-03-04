# Pythia Checkpoint (LoRA / QLoRA)

This folder stores the **fine-tuned adapter checkpoints** for ** Pythia (160M, 410M, 1.4B)** on the occupation classification task.

## What’s inside

Typical structure:

- `best/` — **final checkpoint** used for evaluation + prediction (recommended)
- `checkpoint-<step>/` — intermediate checkpoints (for resuming training)

Common files:
- `adapter_model.safetensors` — LoRA weights
- `adapter_config.json` — LoRA config (r/alpha/target modules)
- `label_meta.json` — label mapping (must match your JSONL output)
- `trainer_state.json`, `optimizer.pt`, `scheduler.pt` — only in `checkpoint-*` (for resuming)

## Which one to use?

- **Evaluation / dumping predictions:** use `best/`
- **Resume training:** use the latest `checkpoint-<step>/`

## Minimal load example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_ID = "EleutherAI/pythia-1.4b"
CKPT_DIR = "./checkpoints/pythia-1.4b/best"

tok = AutoTokenizer.from_pretrained(CKPT_DIR, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, CKPT_DIR).eval()