import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
import random
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prompts import format_prompt

def load_data(data_path):
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def get_candidate_log_likelihood_batched(model, tokenizer, prompts, candidate, device, max_tokens=256):
    # Candidate tokenization
    candidate_ids = tokenizer.encode(" " + candidate, return_tensors="pt").to(device)
    cand_seq_len = candidate_ids.shape[1]
    
    # Prompt tokenization with left padding
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    encoded_prompts = tokenizer(
        prompts, 
        padding=True, 
        truncation=True, 
        max_length=max_tokens - cand_seq_len, 
        return_tensors="pt"
    )
    
    prompt_ids = encoded_prompts["input_ids"].to(device)
    attention_mask = encoded_prompts["attention_mask"].to(device)
    
    batch_size = prompt_ids.shape[0]
    
    cand_ids_batch = candidate_ids.expand(batch_size, -1)
    cand_mask_batch = torch.ones((batch_size, cand_seq_len), dtype=attention_mask.dtype, device=device)
    
    input_ids = torch.cat([prompt_ids, cand_ids_batch], dim=1)
    input_attention_mask = torch.cat([attention_mask, cand_mask_batch], dim=1)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=input_attention_mask)
        logits = outputs.logits # [batch, seq_len, vocab_size]
        
    prompt_len = prompt_ids.shape[1]
    # The last token of the prompt predicts the first token of the candidate
    shift_logits = logits[:, prompt_len - 1 : prompt_len + cand_seq_len - 1, :].contiguous()
    shift_labels = cand_ids_batch.contiguous()
    
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction='none')
    loss = loss.view(batch_size, cand_seq_len)
    
    log_likelihoods = (-loss.sum(dim=1)).tolist()
    
    return log_likelihoods

def main():
    parser = argparse.ArgumentParser(description="Pythia Zero-shot and Few-shot Inference with candidate scoring")
    parser.add_argument("--model_size", type=str, required=True, help="Pythia model size (e.g. 160m, 410m, 1.4b)")
    parser.add_argument("--regime", type=str, choices=["zeroshot", "fewshot"], required=True, help="Inference regime")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input JSONL data")
    parser.add_argument("--candidate_labels", type=str, nargs='+', required=False, help="List of candidate occupations (if not provided, will look for candidate_labels.txt in data_path dir)")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output JSONL")
    
    # New arguments for batching, sampling, and max tokens
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate on (for testing/debugging)")
    parser.add_argument("--sampling_method", type=str, choices=["random", "stratified"], default="stratified", help="Sampling method if num_samples is set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum number of tokens for prompt + candidate")
    parser.add_argument("--apply_mask", action="store_true", help="Apply masking by using 'masked_text' instead of 'text' if available")
    
    args = parser.parse_args()
    
    if not args.candidate_labels:
        labels_path = os.path.join(os.path.dirname(args.data_path), "candidate_labels.txt")
        if os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                args.candidate_labels = f.read().strip().split()
        else:
            raise ValueError(f"candidate_labels not provided and not found at {labels_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = f"EleutherAI/pythia-{args.model_size}"
    
    print(f"Loading tokenizer and model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    if device == "cpu":
        model.to(device)
        
    model.eval()
    
    data = load_data(args.data_path)
    
    # Filter data to only include candidate labels (Top 20 professions)
    data = [item for item in data if item.get("label", "") in args.candidate_labels]
    
    # Stratified Sampling
    if args.num_samples is not None and args.num_samples < len(data):
        random.seed(args.seed)
        if args.sampling_method == "random":
            data = random.sample(data, args.num_samples)
        elif args.sampling_method == "stratified":
            label_to_items = defaultdict(list)
            for item in data:
                label_to_items[item.get("label", "")].append(item)
            
            sampled = []
            total_items = sum(len(items) for items in label_to_items.values())
            for label, items in label_to_items.items():
                k = int(round(len(items) / total_items * args.num_samples))
                k = min(k, len(items))
                sampled.extend(random.sample(items, k))
                
            if len(sampled) < args.num_samples:
                remaining = [item for item in data if item not in sampled]
                sampled.extend(random.sample(remaining, args.num_samples - len(sampled)))
            elif len(sampled) > args.num_samples:
                sampled = random.sample(sampled, args.num_samples)
            
            random.shuffle(sampled)
            data = sampled
        print(f"Sampled {len(data)} items using {args.sampling_method} sampling.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    mask_str = "_masked" if args.apply_mask else ""
    output_filename = os.path.join(args.output_dir, f"preds_pythia_{args.model_size}{mask_str}_{args.regime}.jsonl")
    
    print(f"Running batched inference (batch_size={args.batch_size}) on {len(data)} examples. Masked: {args.apply_mask}")
    with open(output_filename, "w", encoding='utf-8') as out_f:
        for i in tqdm(range(0, len(data), args.batch_size)):
            batch_items = data[i:i + args.batch_size]
            prompts = []
            for item in batch_items:
                if args.apply_mask:
                    text = item.get("masked_text", item.get("text", ""))
                else:
                    text = item.get("text", "")
                    
                prompt = format_prompt(text, regime=args.regime)
                prompts.append(prompt)
                
            # Compute log likelihoods for each candidate across the batch
            batch_scores_by_candidate = []
            for candidate in args.candidate_labels:
                scores = get_candidate_log_likelihood_batched(model, tokenizer, prompts, candidate, device, args.max_tokens)
                batch_scores_by_candidate.append(scores)
                
            # Transpose to [batch_size, num_candidates]
            batch_scores = list(map(list, zip(*batch_scores_by_candidate)))
            
            for j, item in enumerate(batch_items):
                scores_tensor = torch.tensor(batch_scores[j])
                probs = F.softmax(scores_tensor, dim=0)
                
                best_idx = torch.argmax(scores_tensor).item()
                label_pred = args.candidate_labels[best_idx]
                
                conf = probs[best_idx].item()
                score_val = batch_scores[j][best_idx]
                
                pred_record = {
                    "id": item.get("id", ""),
                    "label_true": item.get("label", ""),
                    "label_pred": label_pred,
                    "gender": item.get("gender", ""),
                    "model": f"pythia-{args.model_size}",
                    "regime": args.regime,
                    "score": score_val,
                    "conf": conf
                }
                
                out_f.write(json.dumps(pred_record) + "\n")
            out_f.flush()
            
    print(f"Predictions saved to {output_filename}")

if __name__ == "__main__":
    main()
