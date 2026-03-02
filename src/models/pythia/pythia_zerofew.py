import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
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

def get_candidate_log_likelihood(model, tokenizer, prompt, candidate, device):
    # Tokenize the prompt and candidate separately
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    candidate_ids = tokenizer.encode(" " + candidate, return_tensors="pt").to(device) # [batch_size, seq_len]
    candidate_seq_len = candidate_ids.shape[1]

    # Concatenate prompt and candidate tokens
    input_ids = torch.cat([prompt_ids, candidate_ids], dim=1)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits # logits shape: (1, seq_len, vocab_size) because batch size is 1
    

    # First token of the candidate is predicted by the last token of the prompt
    # Last token of the candidate is predicted by the second to last token overall
    start_idx = input_ids.shape[1] - candidate_seq_len - 1
    end_idx = input_ids.shape[1] - 1

    # Logits for the candidate tokens
    shift_logits = logits[0, start_idx:end_idx, :].contiguous()
    # Candidate tokens
    shift_labels = candidate_ids[0].contiguous()
    
    # Calculate log likelihoods via CrossEntropy with reduction='none'
    loss = F.cross_entropy(shift_logits, shift_labels, reduction='none')
    
    return -loss.sum().item()

def main():
    parser = argparse.ArgumentParser(description="Pythia Zero-shot and Few-shot Inference with candidate scoring")
    parser.add_argument("--model_size", type=str, required=True, help="Pythia model size (e.g., 70m, 160m, 410m, 1.4b)")
    parser.add_argument("--regime", type=str, choices=["zeroshot", "fewshot"], required=True, help="Inference regime")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input JSONL data")
    parser.add_argument("--candidate_labels", type=str, nargs='+', required=False, help="List of candidate occupations (if not provided, will look for candidate_labels.txt in data_path dir)")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output JSONL")
    
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
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    if device == "cpu":
        model.to(device)
        
    model.eval()
    
    data = load_data(args.data_path)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = os.path.join(args.output_dir, f"preds_pythia_{args.model_size}_{args.regime}.jsonl")
    
    print(f"Running inference on {len(data)} examples with candidates: {args.candidate_labels}")
    with open(output_filename, "w", encoding='utf-8') as out_f:
        for item in tqdm(data):
            text = item.get("masked_text", item.get("text", "")) # checks for masked_text first, then text
            
            prompt = format_prompt(text, regime=args.regime)
            
            candidate_scores = []
            
            for candidate in args.candidate_labels:
                score = get_candidate_log_likelihood(model, tokenizer, prompt, candidate, device)
                candidate_scores.append(score)
                
            scores_tensor = torch.tensor(candidate_scores)
            probs = F.softmax(scores_tensor, dim=0)
            
            best_idx = torch.argmax(scores_tensor).item()
            label_pred = args.candidate_labels[best_idx]
            
            conf = probs[best_idx].item()
            score_val = candidate_scores[best_idx]
            
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
