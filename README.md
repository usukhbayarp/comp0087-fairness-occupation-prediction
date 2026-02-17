# COMP0087 Group Project

This repository contains the code for the COMP0087 (Statistical Natural Language Processing) group project.  
The project investigates **fairness, proxy bias, and model scaling effects** in occupation prediction using the **Bias-in-Bios** dataset.

Our primary focus is on comparing models of different capacities—particularly within the **Pythia model family**—across **zero-shot, few-shot, and fine-tuned** settings, and evaluating whether increased model size leads to improved fairness or merely conceals bias.

---

## Repository Structure
src/ # Model training, inference, evaluation, and analysis code
data/ # Dataset loading, preprocessing, and masking utilities
results/ # Generated tables and figures (not raw experiment dumps)
notebooks/ # Exploratory analysis and debugging notebooks
scripts/ # Entry-point scripts for experiments

Each subdirectory contains a `README.md` describing its purpose in more detail.

---

## Setup

We recommend using a Python virtual environment.

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
