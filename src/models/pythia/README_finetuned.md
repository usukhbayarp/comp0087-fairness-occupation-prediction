# Part 3: Pythia Fine-tuning & Evaluation Pipeline

Fine-tune Pythia models (LoRA / QLoRA / full) on the Bias-in-Bios dataset and evaluate the resulting checkpoints. The pipeline consists of three scripts that run in sequence:

1. **`export_finetune.py`** -- export the HuggingFace dataset to JSONL splits
2. **`pythia_finetune.py`** -- fine-tune a Pythia model on the training split
3. **`pythia_eval.py`** -- evaluate the fine-tuned checkpoint on the test split

---

## Dependencies

```bash
pip install transformers==4.40.0 peft==0.10.0 accelerate bitsandbytes scikit-learn tqdm
```

> `bitsandbytes` is only required when using `--use_4bit` (QLoRA).

---

## 1. export_finetune.py

Export the Bias-in-Bios dataset (loaded via `src/data/data.py`) into per-split JSONL files for fine-tuning and evaluation.

### How to Run

```bash
python src/models/pythia/export_finetune.py \
    --output_dir processed \
    --top_n 20
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output_dir` | str | `processed` | Directory to save JSONL files. The relative path `processed` maps to `<repo>/data/processed` |
| `--top_n` | int | `20` | Number of top occupations to retain |
| `--mask_gender` | flag | `False` | Whether to apply gender masking to text |

### Output

```
<output_dir>/
├── train.jsonl
├── dev.jsonl
├── test.jsonl
└── candidate_labels.txt    ← space-separated list of label names
```

Each JSONL line follows this schema:

```json
{"id": 0, "text": "...", "label": "professor", "gender": "M"}
```

---

## 2. pythia_finetune.py

Fine-tune a Pythia model on the training split using LoRA (default), QLoRA (`--use_4bit`), or full fine-tuning (`--full_finetune`).

### How to Run

```bash
python src/models/pythia/pythia_finetune.py \
    --model_size 160m \
    --train_batch_size 64 \
    --data_dir data/processed \
    --output_dir checkpoints
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_size` | str | `410m` | Pythia size: `70m`, `160m`, `410m`, `1b`, `1.4b`, `2.8b` |
| `--data_dir` | str | *(required)* | Folder containing `train.jsonl` and `dev.jsonl` from `export_finetune.py`. Relative path `processed` maps to `<repo>/data/processed` |
| `--output_dir` | str | `./checkpoints` | Root directory for checkpoint output |
| `--num_epochs` | int | `3` | Number of training epochs |
| `--train_batch_size` | int | `16` | Per-device training batch size |
| `--eval_batch_size` | int | `16` | Per-device evaluation batch size |
| `--grad_accum` | int | `1` | Gradient accumulation steps |
| `--lr` | float | `2e-5` | Learning rate |
| `--max_length` | int | `256` | Maximum token length; longer texts are truncated |
| `--max_train_samples` | int | `None` | Cap training samples (useful for quick testing) |
| `--use_4bit` | flag | `False` | Enable QLoRA 4-bit quantisation (recommended for 1.4b+; ignored when `--full_finetune` is set) |
| `--full_finetune` | flag | `False` | Full fine-tuning of all parameters; overrides LoRA/QLoRA |
| `--seed` | int | `42` | Random seed for reproducibility |

### Input

Two JSONL files produced by `export_finetune.py`:

```
<data_dir>/
├── train.jsonl
└── dev.jsonl
```

### Output

```
<output_dir>/pythia-<size>/
├── checkpoint-*/              ← per-epoch checkpoints
└── best/                      ← best checkpoint (by macro F1 on dev)
    ├── adapter_model.safetensors   (LoRA/QLoRA) or model.safetensors (full)
    ├── adapter_config.json         (LoRA/QLoRA only)
    ├── tokenizer.json
    └── label_meta.json             ← label2id / id2label mappings
```

### Notes

- The LoRA adapter targets Pythia's merged QKV projection (`query_key_value`) in every attention layer, with `r=16`, `lora_alpha=32`, `lora_dropout=0.1`.
- Early stopping is enabled with patience of 2 epochs, monitored on `macro_f1`.
- BF16 mixed precision is used automatically when the GPU supports it.
- `torch.compile` is enabled for faster training.

---

## 3. pythia_eval.py

Evaluate a fine-tuned Pythia checkpoint (LoRA, QLoRA, or full) on the test split and save predictions in the shared JSONL schema.

### How to Run

```bash
python src/models/pythia/pythia_eval.py \
    --model_size 160m \
    --checkpoint_dir checkpoints/pythia-160m/best \
    --data_dir data/processed \
    --output_dir results/pythia_finetuned \
    --batch_size 64
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_size` | str | *(required)* | Pythia size string, e.g. `410m`, `1.4b` -- used only for output filenames |
| `--checkpoint_dir` | str | *(required)* | Path to the `best/` directory saved by `pythia_finetune.py` |
| `--data_dir` | str | *(required)* | Folder containing `test.jsonl` |
| `--output_dir` | str | `./results` | Directory for all output files |
| `--batch_size` | int | `32` | Inference batch size |
| `--max_length` | int | `256` | Maximum token length; longer texts are truncated |
| `--suffix` | str | `finetuned` | Suffix for output filenames, e.g. `QLoRA` or `full` |

### Input

Two sources are required:

**1. From `pythia_finetune.py` -- the `best/` checkpoint directory:**

```
<checkpoint_dir>/
├── adapter_model.safetensors   (LoRA/QLoRA) or model.safetensors (full)
├── adapter_config.json         (LoRA/QLoRA only)
├── tokenizer.json
└── label_meta.json             ← required; contains label2id / id2label mappings
```

**2. From `export_finetune.py` -- test split:**

```
<data_dir>/
└── test.jsonl
```

### Output

Three files are saved under `--output_dir`:

```
<output_dir>/
├── preds_pythia_<size>_<suffix>.jsonl    ← per-sample predictions
├── metrics_pythia_<size>_<suffix>.json   ← aggregated metrics
└── clf_report_pythia_<size>_<suffix>.txt ← per-class classification report
```

#### preds_pythia_\<size\>_\<suffix\>.jsonl

One JSON record per test sample:

```json
{
  "id": 1,
  "text": "...",
  "label_true": "physician",
  "label_pred": "nurse",
  "gender": "F",
  "model": "pythia-410m",
  "regime": "finetuned",
  "score": -0.47,
  "conf": 0.62
}
```

`conf` is the softmax probability of the predicted class; `score` is its log-probability.

#### metrics_pythia_\<size\>_\<suffix\>.json

```json
{
  "model": "pythia-410m",
  "regime": "finetuned",
  "accuracy": 0.88,
  "macro_f1": 0.85,
  "num_samples": 95468,
  "gender_accuracy": { "M": 0.871, "F": 0.880 }
}
```

#### clf_report_pythia_\<size\>_\<suffix\>.txt

Standard sklearn classification report with per-class precision, recall, F1, and support.

### Notes

- The script auto-detects whether the checkpoint is LoRA/QLoRA or full fine-tuned by checking for the presence of `adapter_config.json`.
- For LoRA checkpoints, `merge_and_unload()` is called to merge adapter weights into the base model before inference for faster execution.
- Test samples whose label is absent from `label_meta.json` are silently skipped and a count is printed.
- `--model_size` does not affect which model is loaded -- the base model name is read from `adapter_config.json` (LoRA) or the checkpoint itself (full). It is used only to name the output files.

---

## End-to-End Example

```bash
# Step 1: Export dataset
python src/models/pythia/export_finetune.py --output_dir processed --top_n 20

# Step 2: Fine-tune (LoRA, pythia-160m)
python src/models/pythia/pythia_finetune.py \
    --model_size 160m \
    --train_batch_size 64 \
    --data_dir data/processed \
    --output_dir checkpoints

# Step 3: Evaluate
python src/models/pythia/pythia_eval.py \
    --model_size 160m \
    --checkpoint_dir checkpoints/pythia-160m/best \
    --data_dir data/processed \
    --output_dir results/pythia_finetuned \
    --batch_size 64
```
