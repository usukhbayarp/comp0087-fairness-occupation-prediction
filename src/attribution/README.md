# Attribution and Proxy Bias Analysis Pipeline

This module implements the attribution-based analysis used in this project to study **proxy bias in occupation classification models**.

The pipeline focuses on three core questions:

1. Which tokens most strongly influence model predictions?
2. How do these proxy tokens differ between masked and unmasked training regimes?
3. Do the identified tokens meaningfully affect predictions (faithfulness)?

The implementation is designed to be modular, reproducible, and compatible with standard Hugging Face transformer models.

---

# Overview of the Pipeline

The workflow consists of four main stages:

## 1. Attribution (Integrated Gradients)

Compute token-level attribution scores using Integrated Gradients for each example.

**Input:**
- Prediction JSONL file (with text + predictions)
- Fine-tuned encoder checkpoint

**Output:**
- Attribution JSONL with:
  - token-level scores
  - merged tokens
  - top-k influential tokens per example

Core files:
- `scripts/run_attribution_encoder.py`
- `src/attribution/attribution_encoder.py`
- `src/attribution/token_utils.py`

---

## 2. Proxy Token Aggregation

Aggregate top attributed tokens into structured proxy-word tables.

**Grouping:**
- By profession
- Optional: by profession × gender

**Metrics:**
- `count_topk`
- `doc_freq`
- `mean_attr`
- `weighted_score`

Core files:
- `scripts/run_proxy_audit.py`
- `src/attribution/proxy_audit.py`

---

## 3. Masked vs Unmasked Comparison

Compare proxy-token distributions between masked and unmasked models.

**Outputs:**
- Overlap summaries (shared vs unique tokens)
- Token-level shifts (importance changes)
- Explicit gender-token analysis
- Optional profession × gender comparisons

Core file:
- `scripts/compare_masked_unmasked_proxies.py`

---

## 4. Result Summarization

Generate compact summaries and a readable report.

**Outputs:**
- Most/least stable professions
- Strongest token increases/decreases
- Total shift per profession
- Gender-conditioned summaries
- Text report for interpretation

Core file:
- `scripts/summarize_proxy_results.py`

---

## 5. Erasure Faithfulness (Validation)

Evaluate whether top-attributed tokens affect model predictions.

**Method:**
- Remove top-K tokens from input text
- Re-run model
- Measure probability drop

**Output:**
- Per-example faithfulness scores
- Aggregated summaries

Core file:
- `scripts/erasure_faithfulness.py`

---

# File Structure

## Scripts

- `scripts/run_attribution_encoder.py`  
  Runs attribution on prediction JSONL.

- `scripts/run_proxy_audit.py`  
  Aggregates attribution outputs into proxy-token tables.

- `scripts/compare_masked_unmasked_proxies.py`  
  Compares masked vs unmasked proxy patterns.

- `scripts/summarize_proxy_results.py`  
  Produces final summaries and reports.

- `scripts/erasure_faithfulness.py`  
  Runs erasure-based faithfulness evaluation.

---

## Source Code

- `src/attribution/attribution_encoder.py`  
  Integrated Gradients implementation.

- `src/attribution/proxy_audit.py`  
  Token aggregation and filtering.

- `src/attribution/token_utils.py`  
  Subword merging and token processing.

---

# Expected Inputs

## 1. Prediction JSONL

Each row should include:

- `id`
- `text`
- `label_pred`

Optional (recommended):
- `label_true`
- `gender`
- `model`
- `regime`
- `conf`

---

## 2. Model Checkpoints

A fine-tuned encoder model (e.g., DistilBERT, RoBERTa).

The same checkpoint should be used consistently across:
- attribution
- erasure faithfulness

---

# How to Run (from Project Root)

## Step 1 — Attribution

```bash
python scripts/run_attribution_encoder.py \
  --input_jsonl results/preds/distilbert_unmasked.jsonl \
  --model_path checkpoints/distilbert_unmasked \
  --output_jsonl results/attribution/distilbert_unmasked_attr.jsonl \
  --limit 3000 \
  --max_length 256 \
  --n_steps 32 \
  --top_k 5
```

Repeat for masked model:

```bash
python scripts/run_attribution_encoder.py \
  --input_jsonl results/preds/distilbert_masked.jsonl \
  --model_path checkpoints/distilbert_masked \
  --output_jsonl results/attribution/distilbert_masked_attr.jsonl \
  --limit 3000 \
  --max_length 256 \
  --n_steps 32 \
  --top_k 5
```

### Running with different encoder models

The same pipeline can be applied to other encoder models such as RoBERTa by changing the input files and checkpoint paths.

Example (RoBERTa):

```bash
python scripts/run_attribution_encoder.py \
  --input_jsonl results/preds/roberta_unmasked.jsonl \
  --model_path checkpoints/roberta_unmasked \
  --output_jsonl results/attribution/roberta_unmasked_attr.jsonl \
  --limit 3000 \
  --max_length 256 \
  --n_steps 32 \
  --top_k 5
```

Repeat for the masked model:

```bash
python scripts/run_attribution_encoder.py \
  --input_jsonl results/preds/roberta_masked.jsonl \
  --model_path checkpoints/roberta_masked \
  --output_jsonl results/attribution/roberta_masked_attr.jsonl \
  --limit 3000 \
  --max_length 256 \
  --n_steps 32 \
  --top_k 5
```

## Step 2 — Proxy Aggregation

```bash
python scripts/run_proxy_audit.py \
  --input_jsonl results/attribution/distilbert_unmasked_attr.jsonl \
  --output_csv results/proxy/unmasked_profession.csv \
  --output_gender_csv results/proxy/unmasked_profession_gender.csv
```

Repeat for masked model:
```bash
python scripts/run_proxy_audit.py \
  --input_jsonl results/attribution/distilbert_masked_attr.jsonl \
  --output_csv results/proxy/masked_profession.csv \
  --output_gender_csv results/proxy/masked_profession_gender.csv
```

## Step 3 — Masked vs Unmasked Comparison

```bash
python scripts/compare_masked_unmasked_proxies.py \
  --masked_csv results/proxy/masked_profession.csv \
  --unmasked_csv results/proxy/unmasked_profession.csv \
  --out_summary_csv results/compare/summary.csv \
  --out_token_shift_csv results/compare/token_shift.csv \
  --out_gender_token_csv results/compare/gender_tokens.csv \
  --masked_gender_csv results/proxy/masked_profession_gender.csv \
  --unmasked_gender_csv results/proxy/unmasked_profession_gender.csv \
  --out_gender_conditioned_summary_csv results/compare/gender_summary.csv
```

## Step 4 — Summarization

```bash
python scripts/summarize_proxy_results.py \
  --model_name distilbert \
  --summary_csv results/compare/summary.csv \
  --token_shift_csv results/compare/token_shift.csv \
  --profession_gender_summary_csv results/compare/gender_summary.csv \
  --gender_tokens_csv results/compare/gender_tokens.csv \
  --out_dir results/compare/final
```

## Step 5 — Erasure Faithfulness

```bash
python scripts/erasure_faithfulness.py \
  --input_jsonl results/attribution/distilbert_unmasked_attr.jsonl \
  --model_path checkpoints/distilbert_unmasked \
  --output_csv results/faithfulness/unmasked.csv \
  --top_k_erase 5
```

Repeat for masked model.

```bash
python scripts/erasure_faithfulness.py \
  --input_jsonl results/attribution/distilbert_masked_attr.jsonl \
  --model_path checkpoints/distilbert_masked \
  --output_csv results/faithfulness/masked.csv \
  --top_k_erase 5
```

## Reproducibility Notes
- Run all commands from the repository root.
- Ensure input JSONL files are consistent and validated.
- Explicitly set --limit for full runs (default is small for testing).
- Device is automatically selected (cuda, mps, or cpu).
- Minor numerical differences may occur across hardware.

## Interpretation Notes
- Attribution scores reflect associations, not causal effects.
- Erasure faithfulness provides a useful validation signal, but remains an approximation.
- Masking may reduce explicit gender tokens while leaving other proxy features active.

## Output Structure

```bash
results/
  attribution/
  proxy/
  compare/
  faithfulness/
```

## Summary

This pipeline provides a structured framework for:

- extracting token-level attributions from encoder models
- identifying proxy features associated with predictions
- comparing masked and unmasked training regimes
- analysing stability and shifts in proxy-token distributions
- validating attribution signals through erasure-based testing

The outputs support empirical analysis of how bias manifests and persists in occupation classification models.