# Introduction
This directory contains the entry-point scripts for running the data processing, model training, and evaluation pipelines.

# Script Structure

- `make_dataset_stats.py`: Computes statistics about the dataset. (Part 1)
- `export_dataset_jsonl.py`: Exports the dataset to JSONL format in order to be used in the next steps. (Part 1-2)
- `evaluate.py`: Evaluates the models on the dataset. (Part 5)
- `run_pythia_eval.sh`: Computes the predictions for Pythia models (160m-1.4b) in zero-shot and few-shot regimes, results saved in `./results/pythia` directory. (Part 2)
