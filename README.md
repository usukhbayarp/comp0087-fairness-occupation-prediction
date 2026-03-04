# COMP0087 Group Project

This repository contains the code for the COMP0087 (Statistical Natural Language Processing) group project.  
The project investigates **fairness, proxy bias, and model scaling effects** in occupation prediction using the **Bias-in-Bios** dataset.

Our primary focus is on comparing models of different capacities—particularly within the **Pythia model family**—across **zero-shot, few-shot, and fine-tuned** settings, and evaluating whether increased model size leads to improved fairness or merely conceals bias.

---

## Repository Structure

Based on our coding split tasks, the repository is organized (or should be organized) to cleanly separate datasets from code and results. 

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
|   |-- predictions/               # JSONL files for predictions (Parts 2-4)
|   |-- tables/                    # results_table.csv, proxy_words_by_profession.csv (Parts 5-6)
|   |-- figures/                   # p   areto.png, scaling.png (Part 5)
|
|-- scripts/                       # Entry-point scripts for running pipelines
|   |-- evaluate.py                # Single command evaluation harness (Part 5)
|   |-- export_dataset_jsonl.py    # Exports the dataset to JSONL format for the next steps. (Part 1-2)
|   |-- make_dataset_stats.py      # Computes statistics about the dataset. (Part 1)
|
|-- src/                           # Reusable source code modules
|   |-- data/                      # data.py, masking.py (Part 1)
|   |-- models/
|   |   |-- pythia/                # prompts.py, pythia_zerofew.py, pythia_finetune.py, pythia_eval.py (Parts 2-3)
|   |   |-- encoders/              # train_encoder.py, eval_encoder.py (Part 4)
|   |-- evaluation/                # Implementation of fairness, plots (Part 5)
|   |-- attribution/               # attribution_encoder.py, proxy_audit.py, erasure_faithfulness.py (Part 6)
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
