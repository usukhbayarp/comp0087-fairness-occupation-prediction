# Introduction

This directory contains the `pythia_zerofew.py` script for evaluating EleutherAI's Pythia models in zero-shot and few-shot regimes. 

## Script Structure and Logic (`pythia_zerofew.py`)

The evaluation script is designed to compute the probability of a model predicting a set of candidate professions given a prompt. 

### How Log-Likelihood is Computed
The script calculates the probability of each occupational candidate label for a given input. Instead of just picking the highest probability token, it calculates the **log-likelihood** of the fully tokenized candidate string given the prompt.
1. The model takes a concatenated sequence of `<prompt> <candidate>`.
2. The logits for the candidate tokens are shifted to align with the labels they are trying to predict (the $i$-th token predicts the $i+1$-th token).
3. We compute the Cross-Entropy loss for these tokens without reduction (`reduction='none'`), and aggregate them to obtain the total log-likelihood for that specific candidate.
4. Finally, a softmax is applied over all candidate scores to get pseudo-probabilities, and the highest scoring candidate is selected as the prediction.

### Sampling Technique
The dataset contains over 38,000 datapoints in the test set, which made the inference time extremely long (for 3070 NVidia it took me 2s per datapoint, which estimates to 76k seconds for the entire dataset). To mitigate this, the script bounds the evaluation to a subset of **3000 downsampled rows**. 
It utilizes **Stratified Proportional Sampling** based on the profession labels. This guarantees that the smaller 3000-row dataset maintains the exact same percentage distribution of occupations as the original, much larger test set, preventing representation skew while saving immense amounts of computation time.

### Batch Size & Memory Management
The evaluation runs in batches (`--batch_size`) rather than row-by-row. Prompts are dynamically padded (with left padding, since we are predicting suffixes) and grouped into a single tensor, speeding up inference drastically. The script expects the user/runner to tune the batch size based on the model cardinality to prevent CUDA Out-Of-Memory (OOM) errors (e.g., using a batch size of 32 for the 160m and 410m models, but downsizing to 16 for the 1.4b model).# Setting Up
There are a couple of requirements that can make your life easier:
* Install CUDA in relation to your pc's specs and check if your pc can run cuda (makes training WAY faster) [check setup here](/README.md#setup)
    * Example: without cuda (it took me 16s/datapoint and I had to run over 38k datapoints... so I just gave up and installed cuda) and with cuda (it took me 2s/datapoint) 
* If not installed already, install venv (with the required dependencies)

# Running Inference Script (./pythia_zerofew.py)
**Run from root**:
```bash
{./venv/Scripts/python.exe or python} src/models/pythia/pythia_zerofew.py --model_size {160m/410m/1.4b} --regime {zeroshot/fewshot} --data_path {data/processed/*.jsonl} --output_dir {results/pythia}

```
**Note**: in data/processed there are 3 files: train.jsonl, dev.jsonl, test.jsonl. You can use any of them as input data.


## Example of running the script for all models and regimes
**Run from root**:
```bash
for model_size in 160m 410m 1.4b; do
    for regime in zeroshot fewshot; do
        {./venv/Scripts/python.exe or just python} src/models/pythia/pythia_zerofew.py --model_size $model_size --regime $regime --data_path data/processed/dev.jsonl --output_dir results/pythia
    done
done
```