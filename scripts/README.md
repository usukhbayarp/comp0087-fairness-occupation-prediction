# Introduction
This directory contains the entry-point scripts for running the data processing, model training, and evaluation pipelines.

# Script Structure

- `make_dataset_stats.py`: Computes statistics about the dataset.
- `export_dataset_jsonl.py`: Exports the dataset to JSONL format in order to be used in the next steps.
- `evaluate.py`: Evaluates the models on the dataset covering DP and EO gaps across all valid `results/predictions` and `results/pythia*` jsonls.
- `run_attribution_encoder.py`: Runs Integrated Gradients attribution.
- `run_proxy_audit.py`: Aggregates top tokens into proxy tables.
- `compare_masked_unmasked_proxies.py`: Compares proxy patterns across regimes.
- `summarize_proxy_results.py`: Produces final proxy summaries.
- `erasure_faithfulness.py`: Erasure-based faithfulness validation.
- `run_counterfactual_swap.py`: Counterfactual pronoun swap experiment.
