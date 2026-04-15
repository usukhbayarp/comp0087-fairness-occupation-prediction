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

## 6. Counterfactual Pronoun Swap (Causal Intervention)

Measure how much occupation predictions depend directly on gendered
pronouns by swapping them counterfactually (M→F or F→M) while leaving
all other text unchanged.

**Input:**
- Prediction JSONL (fields: `id`, `text`, `label_pred`, `gender`)
- Fine-tuned encoder checkpoint

**Method:**
- For each biography, swap gender-matched pronouns and titles:
  - Male bios: `he→she`, `him→her`, `his→her`, `himself→herself`, `mr→ms`
  - Female bios: `she→he`, `her→his`, `hers→his`, `herself→himself`, `ms/mrs→mr`
- Case is preserved for each individual match (Title, UPPER, lower)
- Re-run model on swapped text with a single forward pass
- Record whether the predicted occupation flips

**Note on ambiguity:** `her` is mapped uniformly to `his` regardless of
grammatical role (object vs possessive), following standard practice in
the counterfactual data augmentation literature. Examples where no
pronouns are found are skipped.

**Output:**
- Per-example: stored label, fresh original label (from this checkpoint),
  stored/fresh agreement flag, swapped label, flip flag, n tokens swapped,
  probability of original class on original text, probability of original
  class after swap, probability drop
- `_by_profession.csv`: flip rate, mean probability after swap, mean
  probability drop per profession (grouped by fresh original label)
- `_by_profession_gender.csv`: same metrics split by profession × gender

Core files:
- `src/attribution/counterfactual_swap.py`
- `scripts/run_counterfactual_swap.py`

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

- `scripts/run_counterfactual_swap.py`  
  Runs counterfactual pronoun swap experiment.

---

## Source Code

- `src/attribution/attribution_encoder.py`  
  Integrated Gradients implementation.

- `src/attribution/proxy_audit.py`  
  Token aggregation and filtering.

- `src/attribution/counterfactual_swap.py`  
  Counterfactual pronoun swap logic and experiment runner.

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
- counterfactual swap

**Important:** the prediction JSONL supplied to the counterfactual swap script
must have been produced by the same checkpoint passed to `--model_path`.
If they differ, flip rates are unreliable. The script reports a
`Stored==fresh agree` rate as a sanity check; values below ~95% suggest
a checkpoint/JSONL mismatch.

---

# How to Run (from Project Root)

> **Important:** All scripts must be run as Python modules from the project root, not as direct script paths.
>
> ```bash
> # ❌ Do NOT run — causes ModuleNotFoundError: No module named 'src'
> python scripts/run_attribution_encoder.py
>
> # ✅ Correct usage
> python -m scripts.run_attribution_encoder ...
> ```

## Step 1 — Attribution

### DistilBERT unmasked

```bash
python -m scripts.run_attribution_encoder \
  --input_jsonl results/predictions/distilbert_unmasked.jsonl \
  --model_path checkpoints/distilbert_unmasked \
  --output_jsonl results/attribution/distilbert_unmasked_attr.jsonl \
  --limit 3000 \
  --max_length 256 \
  --n_steps 32 \
  --top_k 5
```

### DistilBERT masked

```bash
python -m scripts.run_attribution_encoder \
  --input_jsonl results/predictions/distilbert_masked.jsonl \
  --model_path checkpoints/distilbert_masked \
  --output_jsonl results/attribution/distilbert_masked_attr.jsonl \
  --limit 3000 \
  --max_length 256 \
  --n_steps 32 \
  --top_k 5
```

### RoBERTa unmasked

```bash
python -m scripts.run_attribution_encoder \
  --input_jsonl results/predictions/roberta_unmasked.jsonl \
  --model_path checkpoints/roberta_unmasked \
  --output_jsonl results/attribution/roberta_unmasked_attr.jsonl \
  --limit 3000 \
  --max_length 256 \
  --n_steps 32 \
  --top_k 5
```

### RoBERTa masked

```bash
python -m scripts.run_attribution_encoder \
  --input_jsonl results/predictions/roberta_masked.jsonl \
  --model_path checkpoints/roberta_masked \
  --output_jsonl results/attribution/roberta_masked_attr.jsonl \
  --limit 3000 \
  --max_length 256 \
  --n_steps 32 \
  --top_k 5
```

## Step 2 — Proxy Aggregation

### DistilBERT unmasked

```bash
python -m scripts.run_proxy_audit \
  --input_jsonl results/attribution/distilbert_unmasked_attr.jsonl \
  --output_csv results/proxy/distilbert_unmasked_profession.csv \
  --output_gender_csv results/proxy/distilbert_unmasked_profession_gender.csv \
  --top_n_print 10 \
  --min_token_len 3 \
  --min_count 1 \
  --min_doc_freq 1
```

### DistilBERT masked

```bash
python -m scripts.run_proxy_audit \
  --input_jsonl results/attribution/distilbert_masked_attr.jsonl \
  --output_csv results/proxy/distilbert_masked_profession.csv \
  --output_gender_csv results/proxy/distilbert_masked_profession_gender.csv \
  --top_n_print 10 \
  --min_token_len 3 \
  --min_count 1 \
  --min_doc_freq 1
```

### RoBERTa unmasked

```bash
python -m scripts.run_proxy_audit \
  --input_jsonl results/attribution/roberta_unmasked_attr.jsonl \
  --output_csv results/proxy/roberta_unmasked_profession.csv \
  --output_gender_csv results/proxy/roberta_unmasked_profession_gender.csv \
  --top_n_print 10 \
  --min_token_len 3 \
  --min_count 1 \
  --min_doc_freq 1
```

### RoBERTa masked

```bash
python -m scripts.run_proxy_audit \
  --input_jsonl results/attribution/roberta_masked_attr.jsonl \
  --output_csv results/proxy/roberta_masked_profession.csv \
  --output_gender_csv results/proxy/roberta_masked_profession_gender.csv \
  --top_n_print 10 \
  --min_token_len 3 \
  --min_count 1 \
  --min_doc_freq 1
```

## Step 3 — Masked vs Unmasked Comparison

### DistilBERT

```bash
python -m scripts.compare_masked_unmasked_proxies \
  --masked_csv results/proxy/distilbert_masked_profession.csv \
  --unmasked_csv results/proxy/distilbert_unmasked_profession.csv \
  --out_summary_csv results/compare/distilbert_summary.csv \
  --out_token_shift_csv results/compare/distilbert_token_shift.csv \
  --out_gender_token_csv results/compare/distilbert_gender_tokens.csv \
  --masked_gender_csv results/proxy/distilbert_masked_profession_gender.csv \
  --unmasked_gender_csv results/proxy/distilbert_unmasked_profession_gender.csv \
  --out_gender_conditioned_summary_csv results/compare/distilbert_gender_summary.csv \
  --top_n 10
```

### RoBERTa

```bash
python -m scripts.compare_masked_unmasked_proxies \
  --masked_csv results/proxy/roberta_masked_profession.csv \
  --unmasked_csv results/proxy/roberta_unmasked_profession.csv \
  --out_summary_csv results/compare/roberta_summary.csv \
  --out_token_shift_csv results/compare/roberta_token_shift.csv \
  --out_gender_token_csv results/compare/roberta_gender_tokens.csv \
  --masked_gender_csv results/proxy/roberta_masked_profession_gender.csv \
  --unmasked_gender_csv results/proxy/roberta_unmasked_profession_gender.csv \
  --out_gender_conditioned_summary_csv results/compare/roberta_gender_summary.csv \
  --top_n 10
```

## Step 4 — Summarization

### DistilBERT

```bash
python -m scripts.summarize_proxy_results \
  --model_name distilbert \
  --summary_csv results/compare/distilbert_summary.csv \
  --token_shift_csv results/compare/distilbert_token_shift.csv \
  --profession_gender_summary_csv results/compare/distilbert_gender_summary.csv \
  --gender_tokens_csv results/compare/distilbert_gender_tokens.csv \
  --out_dir results/compare/distilbert_final
```

### RoBERTa

```bash
python -m scripts.summarize_proxy_results \
  --model_name roberta \
  --summary_csv results/compare/roberta_summary.csv \
  --token_shift_csv results/compare/roberta_token_shift.csv \
  --profession_gender_summary_csv results/compare/roberta_gender_summary.csv \
  --gender_tokens_csv results/compare/roberta_gender_tokens.csv \
  --out_dir results/compare/roberta_final
```

## Step 5 — Erasure Faithfulness

### DistilBERT unmasked

```bash
python -m scripts.erasure_faithfulness \
  --input_jsonl results/attribution/distilbert_unmasked_attr.jsonl \
  --model_path checkpoints/distilbert_unmasked \
  --output_csv results/faithfulness/distilbert_unmasked.csv \
  --top_k_erase 5 \
  --max_length 256
```

### DistilBERT masked

```bash
python -m scripts.erasure_faithfulness \
  --input_jsonl results/attribution/distilbert_masked_attr.jsonl \
  --model_path checkpoints/distilbert_masked \
  --output_csv results/faithfulness/distilbert_masked.csv \
  --top_k_erase 5 \
  --max_length 256
```

### RoBERTa unmasked

```bash
python -m scripts.erasure_faithfulness \
  --input_jsonl results/attribution/roberta_unmasked_attr.jsonl \
  --model_path checkpoints/roberta_unmasked \
  --output_csv results/faithfulness/roberta_unmasked.csv \
  --top_k_erase 5 \
  --max_length 256
```

### RoBERTa masked

```bash
python -m scripts.erasure_faithfulness \
  --input_jsonl results/attribution/roberta_masked_attr.jsonl \
  --model_path checkpoints/roberta_masked \
  --output_csv results/faithfulness/roberta_masked.csv \
  --top_k_erase 5 \
  --max_length 256
```

## Step 6 — Counterfactual Pronoun Swap

Takes the raw **prediction JSONL** (not attribution JSONL) as input.

> **Only run on unmasked conditions.** Masked prediction JSONLs have all
> gendered pronouns already replaced with `[MASK]`, so there are no tokens
> to swap and the script will skip all examples and produce no output.

### DistilBERT unmasked

```bash
python -m scripts.run_counterfactual_swap \
  --input_jsonl results/predictions/distilbert_unmasked.jsonl \
  --model_path checkpoints/distilbert_unmasked \
  --output_csv results/counterfactual/distilbert_unmasked.csv \
  --max_length 256
```

### RoBERTa unmasked

```bash
python -m scripts.run_counterfactual_swap \
  --input_jsonl results/predictions/roberta_unmasked.jsonl \
  --model_path checkpoints/roberta_unmasked \
  --output_csv results/counterfactual/roberta_unmasked.csv \
  --max_length 256
```

---

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
- Counterfactual swap flip rates provide limited interventional evidence:
  a low flip rate suggests proxy tokens (not explicit pronouns) drive predictions,
  consistent with the IG attribution analysis. Results should be interpreted
  alongside the masked/unmasked EO gap comparison.

## Output Structure

```bash
results/
  attribution/
  proxy/
  compare/
  faithfulness/
  counterfactual/
```

## Summary

This pipeline provides a structured framework for:

- extracting token-level attributions from encoder models
- identifying proxy features associated with predictions
- comparing masked and unmasked training regimes
- analysing stability and shifts in proxy-token distributions
- validating attribution signals through erasure-based testing

The outputs support empirical analysis of how bias manifests and persists in occupation classification models.