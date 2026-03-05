**Description**

This part trains and evaluates encoder-based sequence classification models (DistilBERT and RoBERTa) on the standardised Bias-in-Bios dataset. The models take a biography (text) and predict a profession label (label_id). The goal is to produce prediction JSONL files for the evaluation pipeline.

The encoder approach differs from the prompt-based Pythia experiments: here we update model weights using supervised learning.

**Inputs**

This code assumes the project uses the unified dataset schema from data.py (shared across all experiments):

id: unique integer identifier (generated if not provided)

text: biography text

label: profession label (kept as stringified occupation ID)

label_id: integer label index derived from the selected label vocabulary

gender: normalized gender label in {M, F}

The dataset is loaded using load_bios() from src/data/data.py, which also returns:

label2id and id2label mappings (shared across the project)

**train_encoder.py (Finetuning encoders)**

File: src/models/encoders/train_encoder.py 

**What it does**

Loads the dataset with load_bios()

Tokenizes biographies into transformer inputs (input_ids, attention_mask)

Loads a pretrained encoder (distilbert-base-uncased or roberta-base)

Adds a classification head with num_labels = number_of_professions

Finetunes on the training split and evaluates on the dev split

Tracks Macro-F1 and Accuracy

Saves the best checkpoint (based on Macro-F1). Macro-F1 is used for “best checkpoint” because the dataset is class-imbalanced.
As a result, accuracy can look high if the model predicts the majority classes.
Macro-F1 computes F1 per class and averages them equally, so minority professions still matter.

**Outputs**

A finetuned checkpoint directory containing:

model weights + config

tokenizer files

Saved under checkpoints/

Note: Model checkpoints are not included in the repository due to their large size.
They can be reproduced by running train_encoder.py.

**eval_encoder.py (Inference + JSONL export)**

File: src/models/encoders/eval_encoder.py

**What it does**

Loads the finetuned model checkpoint (from train_encoder.py)

Runs inference on the test split

Computes and prints:

Macro-F1

Accuracy

Writes per-example predictions into a JSONL file (one JSON object per line)

**JSONL output schema**

Each line corresponds to one example and includes these fields:

id: example id

label_true: ground-truth label string

label_pred: predicted label string

gender: gender attribute {M, F}

model: model tag (distilbert-ft, roberta-ft)

regime: finetuned

score: scalar confidence proxy 

conf: probability of predicted class (max softmax prob)

**Outputs**

JSONL predictions written to results/predictions

**How to run**

1) Install dependencies

    pip install -r requirements.txt

2) Train encoder models

    python -m src.models.encoders.train_encoder --model_name distilbert-base-uncased

    python -m src.models.encoders.train_encoder --model_name roberta-base

3) Evaluate distilbert encoder

    python -m src.models.encoders.eval_encoder \
    --model_dir checkpoints/distilbert-base-uncased \
    --model_tag distilbert-ft \
    --out_jsonl results/predictions/preds_distilbert_ft.jsonl

4) Evaluate roberta encoder
    python -m src.models.encoders.eval_encoder \
    --model_dir checkpoints/roberta-base \
    --model_tag roberta-ft \
    --out_jsonl results/predictions/preds_roberta_ft.jsonl



**OS notes (Windows/macOS/Linux)**

Use forward slashes in commands (src/models/...) even on Windows

If running on Windows and you get path issues, run from repo root and avoid relative cd assumptions.

GPU is optional: scripts fall back to CPU automatically.

