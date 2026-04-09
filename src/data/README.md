# Description
Using the Bias-in-Bios dataset, accessed via the Hugging Face Datasets library (LabHC/bias_in_bios). The dataset contains short biographies annotated with a profession label and a binary gender attribute. It is split into three predefined splits: train, dev, and test.

Each raw example contains the following fields:

**hard_text**: the biography text

**profession**: an integer-encoded occupation label

**gender**: a binary-encoded gender attribute

# Unified Data Schema

To ensure consistency across all experiments, standardize the dataset into a unified schema with the following fields:

**id**: unique integer identifier (generated if not provided)

**text**: biography text

**label**: profession label (kept as stringified occupation ID)

**label_id**: integer label index derived from the selected label vocabulary

**gender**: normalized gender label in {M, F}

This standardized format is shared across all subsequent modeling and evaluation components.

# Label Vocabulary (Top-N Occupations)

The original dataset contains a large number of profession categories with a highly skewed distribution. For feasibility and stability of evaluation, restrict the label space to the Top-20 most frequent professions, computed from the training split only.

All examples whose profession labels fall outside the Top-20 set are filtered out from train, dev, and test splits. A label2id / id2label mapping is constructed based on this restricted label set and reused throughout the project.

# Gender Normalization

The gender field in this dataset is encoded numerically. Since the dataset does not expose a predefined label mapping, infer the encoding directly from the training data. The observed values are {0, 1}, which map to:

0 → M

1 → F

This mapping is explicitly recorded in the dataset metadata to ensure transparency and reproducibility. After normalization, no examples remain with unknown gender labels.

# Preprocessing and Masking Utilities

Providing a reusable preprocessing pipeline that supports:

optional lowercasing of biography text

optional masking of explicit gendered terms (e.g., he, she, his, her) via a dedicated masking utility

Masking is disabled by default but can be enabled by downstream experiments to study counterfactual or debiased settings.

# Dataset Statistics

Computing and export comprehensive dataset statistics, including:

total number of examples per split

overall gender distribution

profession frequency distribution (class imbalance)

per-profession gender breakdown

These statistics are saved in dataset_stats.json and accompanied by simple visualizations (gender distribution and Top-20 profession counts). The final processed dataset contains 381,807 examples across all splits, with an overall gender distribution of approximately 53.8% male and 46.2% female.

# Outputs

Part 1 produces the following artifacts:

standardized dataset loader (data.py)

gender masking utility (masking.py)

dataset statistics (dataset_stats.json)

## Output Files

Each run writes three JSONL splits:

- `data/exports/clean_train.jsonl`
- `data/exports/clean_dev.jsonl`
- `data/exports/clean_test.jsonl`

Masked preprocessing writes to:

- `data/exports/masked_train.jsonl`
- `data/exports/masked_dev.jsonl`
- `data/exports/masked_test.jsonl`

summary tables and plots for class and gender distributions

# How to run

## Export dataset to JSONL files
```bash
python -m src.data.export_dataset_jsonl --output_dir processed

--top_n {20}

### Unmasked Dataset (used for baseline encoder + Pythia runs)
```bash
python -m scripts.export_dataset_jsonl \
    --top_k 20 \
    --mask_gender false \
    --output_dir data/processed/unmasked
```
### Masked Dataset (used for fairness-controlled runs)
```bash
python -m scripts.export_dataset_jsonl \
    --top_k 20 \
    --mask_gender true \
    --mask_titles true \
    --mask_gendered_nouns true \
    --mask_label_leakage true \
    --output_dir data/processed/masked
```
```
**Note**: edit the parameters in the {} to your needs.


## Compute dataset statistics
```bash
python make_dataset_stats.py --output_dir stats
```
