# pythia_eval.py

Evaluate a fine-tuned Pythia LoRA checkpoint on the test split and save predictions in the shared JSONL schema. 

## Dependencies

Same environment as `pythia_finetune.py`. If not already installed:

```bash
!pip install transformers==4.40.0 peft==0.10.0 accelerate scikit-learn tqdm
```

## Input

Two sources are required:

**1. From `pythia_finetune.py` — the `best/` checkpoint directory:**

```
<checkpoint_dir>/
├── adapter_model.safetensors
├── adapter_config.json
├── tokenizer.json
└── label_meta.json          ← required; contains label2id / id2label mappings
```

**2. From Part 1 — test split:**

```
<data_dir>/
└── test.jsonl
```

Each line in `test.jsonl` follows this schema:

```json
{"id": 0, "text": "...", "label": "professor", "gender": "M"}
```

## How to Run

```bash
!python pythia_eval.py \
    --model_size 410m \
    --checkpoint_dir ./checkpoints/pythia-410m/best \
    --data_dir ./processed \
    --output_dir ./results
```

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_size` | str | *(required)* | Pythia size string, e.g. `410m`, `1.4b` — used only for output filenames |
| `--checkpoint_dir` | str | *(required)* | Path to the `best/` directory saved by `pythia_finetune.py` |
| `--data_dir` | str | *(required)* | Folder containing `test.jsonl` |
| `--output_dir` | str | `./results` | Directory for all output files |
| `--batch_size` | int | `32` | Inference batch size |
| `--max_length` | int | `256` | Maximum token length; longer texts are truncated |

## Output

Three files are saved under `--output_dir`:

```
<output_dir>/
├── preds_pythia_<size>_finetuned.jsonl    ← per-sample predictions
├── metrics_pythia_<size>_finetuned.json   ← aggregated metrics
└── clf_report_pythia_<size>_finetuned.txt ← per-class classification report
```

### preds_pythia_\<size\>_finetuned.jsonl

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

### metrics_pythia_\<size\>_finetuned.json

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

### clf_report_pythia_\<size\>_finetuned.txt

Standard sklearn classification report with per-class precision, recall, F1, and support.

## Notes

- The script calls `merge_and_unload()` on the LoRA adapter before inference, merging the adapter weights back into the base model. This produces the same outputs as keeping them separate, but runs faster.
- Test samples whose label is absent from `label_meta.json` are silently skipped and a count is printed.
- `--model_size` does not affect which model is loaded — the base model name is read directly from `adapter_config.json` inside `--checkpoint_dir`. It is used only to name the output files.
