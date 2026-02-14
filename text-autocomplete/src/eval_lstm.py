"""
Evaluation utilities for LSTM model using ROUGE metrics.
"""

import torch
from rouge_score import rouge_scorer


def evaluate_lstm(model, data_loader, itos, eos_idx, pad_idx=0, device='cpu', 
                  prefix_ratio=0.75, num_samples=None, use_greedy=True):
    """
    Evaluate LSTM model on a dataset using ROUGE metrics.
    
    Strategy: Take 75% of each sequence as prefix, generate the rest, compare with ground truth.
    
    Args:
        model: trained LSTM model
        data_loader: DataLoader with validation data
        itos: index to string mapping
        eos_idx: index of EOS token
        pad_idx: index of PAD token
        device: torch device
        prefix_ratio: fraction of sequence to use as prefix (default 0.75)
        num_samples: limit number of samples to evaluate (None = all)
        use_greedy: if True, use greedy decoding; if False, use sampling
        
    Returns:
        dict with average ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
        use_stemmer=True
    )
    
    model.eval()
    
    scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': [],
        'rougeLsum': []
    }
    
    total_evaluated = 0
    
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            B, T = x.shape
            
            split = int(T * prefix_ratio)
            
            for b in range(B):
                if num_samples and total_evaluated >= num_samples:
                    break
                
                seq = x[b]
                seq_len = (seq != pad_idx).sum().item()
    
                if seq_len < 4:
                    continue
                
                actual_split = int(seq_len * prefix_ratio)
                
                prefix_tokens = seq[:actual_split].tolist()
                target_tokens = seq[actual_split:seq_len].tolist()
                
                if use_greedy:
                    generated = model.generate_greedy(
                        start_tokens=prefix_tokens,
                        max_new_tokens=len(target_tokens),
                        eos_idx=eos_idx,
                        device=device
                    )
                else:
                    generated = model.generate(
                        start_tokens=prefix_tokens,
                        max_new_tokens=len(target_tokens),
                        eos_idx=eos_idx,
                        device=device,
                        temperature=0.8
                    )
                
                generated_tail = generated[len(prefix_tokens):]
                
                gen_text = ' '.join([
                    itos[t] for t in generated_tail 
                    if t not in [pad_idx, eos_idx]
                ])
                
                target_text = ' '.join([
                    itos[t] for t in target_tokens 
                    if t not in [pad_idx, eos_idx]
                ])
                
                if not gen_text or not target_text:
                    continue
                
                result = scorer.score(target_text, gen_text)
                
                for k in scores:
                    scores[k].append(result[k].fmeasure)
                
                total_evaluated += 1
            
            if num_samples and total_evaluated >= num_samples:
                break
    
    return {k: sum(v) / len(v) if v else 0.0 for k, v in scores.items()}


def print_examples(model, data_loader, stoi, itos, eos_idx, device='cpu', 
                   num_examples=5, prefix_ratio=0.75, show_both=True):
    """
    Print some generation examples for visual inspection.
    
    Args:
        model: trained LSTM model
        data_loader: DataLoader with data
        stoi: string to index mapping
        itos: index to string mapping
        eos_idx: index of EOS token
        device: torch device
        num_examples: number of examples to print
        prefix_ratio: fraction of sequence to use as prefix
        show_both: if True, show both greedy and sampled outputs
    """
    model.eval()
    
    pad_idx = stoi['<pad>']
    examples_printed = 0
    
    print("\n" + "="*70)
    print("GENERATION EXAMPLES")
    print("="*70)
    
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            
            for i in range(len(x)):
                if examples_printed >= num_examples:
                    return
                
                seq = x[i]
                seq_len = (seq != pad_idx).sum().item()
                
                if seq_len < 4:
                    continue
                
                split = int(seq_len * prefix_ratio)
                prefix = seq[:split].tolist()
                target = seq[split:seq_len].tolist()
                
                gen_greedy = model.generate_greedy(
                    start_tokens=prefix,
                    max_new_tokens=len(target),
                    eos_idx=eos_idx,
                    device=device
                )
                
                greedy_tail = gen_greedy[len(prefix):]
                
                prefix_text = ' '.join([itos[t] for t in prefix if t not in [pad_idx, eos_idx]])
                target_text = ' '.join([itos[t] for t in target if t not in [pad_idx, eos_idx]])
                greedy_text = ' '.join([itos[t] for t in greedy_tail if t not in [pad_idx, eos_idx]])
                
                print(f"\nExample {examples_printed + 1}:")
                print(f"Prefix:    {prefix_text}")
                print(f"Target:    {target_text}")
                print(f"Greedy:    {greedy_text}")
                
                if show_both:
                    gen_sample = model.generate(
                        start_tokens=prefix,
                        max_new_tokens=len(target),
                        eos_idx=eos_idx,
                        device=device,
                        temperature=0.8
                    )
                    sample_tail = gen_sample[len(prefix):]
                    sample_text = ' '.join([itos[t] for t in sample_tail if t not in [pad_idx, eos_idx]])
                    print(f"Sampled:   {sample_text}")
                
                examples_printed += 1
            
            if examples_printed >= num_examples:
                break
    
    print("="*70)