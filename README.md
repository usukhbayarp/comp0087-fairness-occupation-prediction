# COMP0087 Group Project

This repository contains the code for the COMP0087 (Statistical Natural Language Processing) group project.  
The project investigates **fairness, proxy bias, and model scaling effects** in occupation prediction using the **Bias-in-Bios** dataset.

Our primary focus is on comparing models of different capacities—particularly within the **Pythia model family**—across **zero-shot, few-shot, and fine-tuned** settings, and evaluating whether increased model size leads to improved fairness or merely conceals bias.

Beyond aggregate fairness metrics, we apply **gradient-based attribution** (Integrated Gradients) to identify which tokens drive occupation predictions, how proxy token importance shifts under gender masking, and whether these signals are functionally load-bearing—validated through erasure faithfulness testing and counterfactual pronoun swap experiments.

---

## Repository Structure

The repository is organized to cleanly separate datasets, code, and results.

```text
|-- data/                          # Only data artifacts and metrics (Part 1)
|   |-- raw/                       # Raw Bias-in-Bios datasets
|   |-- processed/                 # Processed data outputs
|   |-- stats/                     # Dataset scale, stats, and plots (dataset_stats.json)
|   |-- pythia_finetuned/          # Finetuned Pythia models
|
|-- notebooks/                     # Exploratory analysis and debugging notebooks
|
|-- results/                       # Experiment predictions, tables, and figures
|   |-- predictions/               # JSONL files for encoder and masked/unmasked predictions (Part 4)
|   |-- pythia/                    # Zero-shot and few-shot Pythia predictions (Part 2)
|   |-- pythia_finetuned/          # Fine-tuned Pythia predictions (masked and unmasked) (Part 3)
|   |-- attribution/               # Token-level attribution JSONL files (Part 6)
|   |-- proxy/                     # Aggregated proxy-token CSVs per profession (Part 6)
|   |-- compare/                   # Masked vs. unmasked comparison and summary CSVs (Part 6)
|   |-- faithfulness/              # Erasure faithfulness scores (Part 6)
|   |-- counterfactual/            # Counterfactual pronoun-swap results (Part 6)
|   |-- tables/                    # CSV results: summary_results.csv, detailed_fairness.csv (Parts 5)
|   |-- figures/                   # Visualizations (Part 5)
|       |-- Correlation Plots/     # Sub-folders for bias analysis
|           |-- Amplification/     # Gender Ratio vs. EO Gap correlation plots
|           |-- Delta/             # Mitigation success (Unmasked EO - Masked EO) plots
|       |-- job_bias_comparison.png
|       |-- pareto_frontier.png
|       |-- scaling_performance.png
|       |-- scaling_fairness.png
|       |-- scaling_fairness_breakdown.png
|
|-- scripts/                       # Entry-point scripts for running pipelines
|   |-- evaluate.py                # Single command evaluation harness (Part 5)
|   |-- export_dataset_jsonl.py    # Exports the dataset to JSONL format (Part 1-2)
|   |-- make_dataset_stats.py      # Computes statistics about the dataset (Part 1)
|   |-- run_attribution_encoder.py # Runs Integrated Gradients attribution (Part 6)
|   |-- run_proxy_audit.py         # Aggregates top tokens into proxy tables (Part 6)
|   |-- compare_masked_unmasked_proxies.py  # Compares proxy patterns across regimes (Part 6)
|   |-- summarize_proxy_results.py # Produces final proxy summaries (Part 6)
|   |-- erasure_faithfulness.py    # Erasure-based faithfulness validation (Part 6)
|   |-- run_counterfactual_swap.py # Counterfactual pronoun swap experiment (Part 6)
|
|-- src/                           # Reusable source code modules
|   |-- data/                      # data.py, masking.py (Part 1)
|   |-- models/
|   |   |-- pythia/                # prompts.py, pythia_zerofew.py, pythia_finetune.py, pythia_eval.py (Parts 2-3)
|   |   |-- encoders/              # train_encoder.py, eval_encoder.py (Part 4)
|   |-- evaluation/                # Implementation of fairness, plots (Part 5)
|   |-- attribution/               # attribution_encoder.py, token_utils.py, proxy_audit.py,
|   |                              #   counterfactual_swap.py (Part 6)
```

Each subdirectory contains a `README.md` describing its purpose in more detail.


## Setup

We recommend using a Python virtual environment.

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### PyTorch Setup (Only for Training purposes)

Depending on your operating system and hardware, you may need to install a specific version of PyTorch to enable hardware acceleration (CUDA for Windows/Linux or MPS for macOS).

**Windows (CUDA)**
If you have an NVIDIA GPU, you should install the CUDA-enabled version of PyTorch. First, [install the NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) compatible with your GPU. Then, run the following command (check the [PyTorch website](https://pytorch.org/get-started/locally/) for the exact command corresponding to your CUDA version, e.g., CUDA 11.8 or 12.1):

To check your CUDA version, run the following command:
```bash
nvcc --version
```

```bash
# Example for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**macOS (Apple Silicon / MPS)**
If you are using a Mac with Apple Silicon (M1/M2/M3 chips), PyTorch provides Metal Performance Shaders (MPS) support out of the box. The standard PyTorch installation from `requirements.txt` should be sufficient, but you can also install it manually:
```bash
pip install torch torchvision torchaudio
```

## Execution Guide

### Part 1: Data Pipeline
> See [src/data/README.md](src/data/README.md) for full details.

Export the unmasked and masked datasets to JSONL (Top-20 occupations):

```bash
# Unmasked (baseline)
python -m scripts.export_dataset_jsonl \
    --top_n 20 \
    --output_dir data/processed/unmasked

# Masked (gender-controlled)
python -m scripts.export_dataset_jsonl \
    --top_n 20 --mask_gender \
    --output_dir data/processed/masked
```

Compute dataset statistics:

```bash
python scripts/make_dataset_stats.py
```

Outputs are saved to `data/processed/` (JSONL splits) and `data/stats/` (statistics and plots).

### Part 2: Pythia Zero-shot / Few-shot Inference
> See [src/models/pythia/README.md](src/models/pythia/README.md) for full details.

To generate predictions for Pythia evaluating on the text prompts, run:
```bash
bash scripts/run_pythia_zerofew.sh
```
The results will be securely serialized to `results/pythia`.

### Part 3: Pythia Finetuned Inference
> See [src/models/pythia/README_finetuned.md](src/models/pythia/README_finetuned.md) for full details.

Export dataset:

```bash
python src/models/pythia/export_finetune.py --output_dir processed --top_n 20
```

Finetuning:
```bash
python src/models/pythia/pythia_finetune.py \
    --model_size 160m \
    --train_batch_size 64 \
    --data_dir data/processed \
    --output_dir checkpoints
```

Evaluation:
```bash
python src/models/pythia/pythia_eval.py \
    --model_size 160m \
    --checkpoint_dir checkpoints/pythia-160m/best \
    --data_dir data/processed \
    --output_dir results/pythia_finetuned \
    --batch_size 64
```

Results will be stored in `results/pythia_finetuned`.

### Part 4: Encoder-based Models (DistilBERT & RoBERTa)
> See [src/models/encoders/README.md](src/models/encoders/README.md) for full details.

Train encoder models (DistilBERT and RoBERTa) on both unmasked and masked data regimes:

```bash
python -m src.models.encoders.train_encoder --model_name distilbert-base-uncased --data_regime unmasked
python -m src.models.encoders.train_encoder --model_name distilbert-base-uncased --data_regime masked
python -m src.models.encoders.train_encoder --model_name roberta-base --data_regime unmasked
python -m src.models.encoders.train_encoder --model_name roberta-base --data_regime masked
```

Best checkpoints (by Macro-F1) will be saved to `checkpoints/`. Pretrained checkpoints can also be downloaded from [OneDrive](https://liveuclac-my.sharepoint.com/:f:/g/personal/zcabkam_ucl_ac_uk/IgB1sMM6KvDkRJinN4I-5d-fAUxZA_lAs4SteWG-cKuAbZU?e=zyG7xK).

Evaluate and export predictions:

```bash
python -m src.models.encoders.eval_encoder \
    --model_dir checkpoints/distilbert_unmasked \
    --model_tag distilbert-ft \
    --data_regime unmasked \
    --out_jsonl results/predictions/distilbert_unmasked.jsonl

python -m src.models.encoders.eval_encoder \
    --model_dir checkpoints/distilbert_masked \
    --model_tag distilbert-ft \
    --data_regime masked \
    --out_jsonl results/predictions/distilbert_masked.jsonl

python -m src.models.encoders.eval_encoder \
    --model_dir checkpoints/roberta_unmasked \
    --model_tag roberta-ft \
    --data_regime unmasked \
    --out_jsonl results/predictions/roberta_unmasked.jsonl

python -m src.models.encoders.eval_encoder \
    --model_dir checkpoints/roberta_masked \
    --model_tag roberta-ft \
    --data_regime masked \
    --out_jsonl results/predictions/roberta_masked.jsonl
```

Results will be stored in `results/predictions/`.

### Part 5: Fairness Evaluation & Visualisation
> See [src/evaluation/README.md](src/evaluation/README.md) for full details.

Run the evaluation harness to compute fairness metrics (Demographic Parity, Equal Opportunity gaps) across all models:

```bash
python -m scripts.evaluate
```

Generate plots (Pareto frontier, scaling laws, job bias comparison, correlation plots):

```bash
python -m src.evaluation.plots
```

Results will be stored in `results/tables/` and `results/figures/`.

### Part 6: Attribution & Proxy Bias Analysis
> See [src/attribution/README.md](src/attribution/README.md) for full commands and details.

**Step 1 — Attribution (run for both unmasked and masked):**

```bash
python -m scripts.run_attribution_encoder \
  --input_jsonl results/predictions/distilbert_unmasked.jsonl \
  --model_path checkpoints/distilbert_unmasked \
  --output_jsonl results/attribution/distilbert_unmasked_attr.jsonl \
  --limit 3000 --max_length 256 --n_steps 32 --top_k 5

python -m scripts.run_attribution_encoder \
  --input_jsonl results/predictions/distilbert_masked.jsonl \
  --model_path checkpoints/distilbert_masked \
  --output_jsonl results/attribution/distilbert_masked_attr.jsonl \
  --limit 3000 --max_length 256 --n_steps 32 --top_k 5
```

**Step 2 — Proxy aggregation (run for both unmasked and masked):**

```bash
python -m scripts.run_proxy_audit \
  --input_jsonl results/attribution/distilbert_unmasked_attr.jsonl \
  --output_csv results/proxy/distilbert_unmasked_profession.csv \
  --output_gender_csv results/proxy/distilbert_unmasked_profession_gender.csv \
  --top_n_print 10 --min_token_len 3 --min_count 1 --min_doc_freq 1

python -m scripts.run_proxy_audit \
  --input_jsonl results/attribution/distilbert_masked_attr.jsonl \
  --output_csv results/proxy/distilbert_masked_profession.csv \
  --output_gender_csv results/proxy/distilbert_masked_profession_gender.csv \
  --top_n_print 10 --min_token_len 3 --min_count 1 --min_doc_freq 1
```

**Step 3 — Masked vs unmasked comparison (requires both sides from Steps 1-2):**

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

**Step 4 — Summarization:**

```bash
python -m scripts.summarize_proxy_results \
  --model_name distilbert \
  --summary_csv results/compare/distilbert_summary.csv \
  --token_shift_csv results/compare/distilbert_token_shift.csv \
  --profession_gender_summary_csv results/compare/distilbert_gender_summary.csv \
  --gender_tokens_csv results/compare/distilbert_gender_tokens.csv \
  --out_dir results/compare/distilbert_final
```

**Step 5 — Erasure faithfulness validation:**

```bash
python -m scripts.erasure_faithfulness \
  --input_jsonl results/attribution/distilbert_unmasked_attr.jsonl \
  --model_path checkpoints/distilbert_unmasked \
  --output_csv results/faithfulness/distilbert_unmasked.csv \
  --top_k_erase 5 --max_length 256
```

**Step 6 — Counterfactual pronoun swap (unmasked only):**

```bash
python -m scripts.run_counterfactual_swap \
  --input_jsonl results/predictions/distilbert_unmasked.jsonl \
  --model_path checkpoints/distilbert_unmasked \
  --output_csv results/counterfactual/distilbert_unmasked.csv \
  --max_length 256
```

Repeat all steps above for RoBERTa (`roberta_unmasked` / `roberta_masked`). Results will be stored in `results/attribution/`, `results/proxy/`, `results/compare/`, `results/faithfulness/`, and `results/counterfactual/`.
