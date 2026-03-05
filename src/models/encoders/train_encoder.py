import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data"))
import numpy as np
from typing import Dict, Any, List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.metrics import f1_score, accuracy_score

from src.data.data import load_bios

#The whole file fine tunes the encoder for multi class sequence classification on the BIOS dataset


#computes evaluation metrics for model predictions
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1) #picks class index with the highest logit
    return {
        #balanced measure of precision+recall averaged equally across classes
        "macro_f1": f1_score(labels, preds, average="macro"),
        #% of predictions that are exactly correct
        "accuracy": accuracy_score(labels, preds),
    }

#convert raw text into model inputs
def tokenize_batch(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
    )


def main():
    parser = argparse.ArgumentParser()
    #run command: python train_encoder.py --model_name distilbert-base-uncased/roberta-base --output_dir path and so on
    #ecoder
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["distilbert-base-uncased", "roberta-base"])
    #save dir
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    #max tokens per input
    parser.add_argument("--max_length", type=int, default=256)
    #no of epochs
    parser.add_argument("--epochs", type=int, default=3)
    #batch size per cpu/gpu device
    parser.add_argument("--batch_size", type=int, default=16)
    #learning rate. step size for gradient upds
    parser.add_argument("--lr", type=float, default=2e-5)
    #seed
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    #load dataset + label vocab
    ds, label2id, id2label, meta = load_bios()

    #makes an ordered list of labels by id
    label_list = [id2label[i] for i in range(len(id2label))]

    #downloads/loads the tokenizer that matches the chosen model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    #rename label_id to labels for HF Trainer
    for split in ds:
        ds[split] = ds[split].rename_column("label_id", "labels")

    #adds tokenized fields to the dataset
    ds = ds.map(lambda ex: tokenize_batch(ex, tokenizer, args.max_length), batched=True)

    #input_ids: token ids representing the text
    #attention_mask: 1 for real tokens, 0 for padding
    #labels:gold class id
    columns_to_keep = ["input_ids", "attention_mask", "labels"]


    #token type ids is in bert but not in roberta
    if "token_type_ids" in ds["train"].column_names:
        columns_to_keep.append("token_type_ids")
    #trainer doesn't need some columns so drop them
    ds = ds.remove_columns([c for c in ds["train"].column_names if c not in columns_to_keep])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        label2id=label2id,
        id2label=id2label,
    )

    #pads all sequences to the longest one in that batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    run_name = args.model_name.replace("/", "_")
    out_dir = os.path.join(args.output_dir, run_name)

    #handles compatibility between different transformers versions (some use evaluation_strategy, some use eval_strategy
    eval_kw = "evaluation_strategy"
    try:
        TrainingArguments(output_dir="tmp", evaluation_strategy="no")
    except TypeError:
        eval_kw = "eval_strategy"

    training_args = TrainingArguments(
        output_dir=out_dir,
        #passes either evaluation_strategy="epoch" or eval_strategy="epoch" depending on transformers version
        **{eval_kw: "epoch"},
        save_strategy="epoch",
        load_best_model_at_end=True,
        # f1 is a better metric because accuracy might be misleading for imbalanced classes
        # model could predict only the majority class and still achieve high accuracy but f1 accounts for it
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=1,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["dev"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"Saved best model to: {out_dir}")


if __name__ == "__main__":
    main()