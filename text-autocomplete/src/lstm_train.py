"""
Training loop for LSTM language model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_epoch(model, data_loader, optimizer, criterion, device, clip_grad=1.0):

    model.train()
    total_loss = 0
    num_batches = 0
    
    for x, y in tqdm(data_loader, desc="Training"):
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        
        logits = model(x)  
        
        B, L, V = logits.shape
        logits = logits.view(B * L, V)
        y = y.view(B * L)
        
        loss = criterion(logits, y)
        
        loss.backward()
        
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def quick_rouge1(model, data_loader, stoi, itos, eos_idx, device, num_samples=32):
    model.eval()
    
    pad_idx = stoi['<pad>']
    scores = []
    evaluated = 0
    
    for x, _ in data_loader:
        x = x.to(device)
        
        for i in range(len(x)):
            if evaluated >= num_samples:
                break
            
            seq = x[i]
            seq_len = (seq != pad_idx).sum().item()
            
            if seq_len < 4:
                continue
            
            # 75/25
            split = int(seq_len * 0.75)
            prefix = seq[:split].tolist()
            target = seq[split:seq_len].tolist()
            
            generated = model.generate_greedy(
                start_tokens=prefix,
                max_new_tokens=len(target),
                eos_idx=eos_idx,
                device=device
            )
            
            gen_tail = generated[len(prefix):]
            
            pred_words = set([itos[t] for t in gen_tail if t not in [pad_idx, eos_idx]])
            gold_words = set([itos[t] for t in target if t not in [pad_idx, eos_idx]])
            
            if pred_words and gold_words:
                precision = len(pred_words & gold_words) / len(pred_words)
                recall = len(pred_words & gold_words) / len(gold_words)
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    scores.append(f1)
            
            evaluated += 1
        
        if evaluated >= num_samples:
            break
    
    return sum(scores) / len(scores) if scores else 0.0


def train(model, train_loader, val_loader, stoi, itos, eos_idx,
          epochs=10, lr=1e-3, device='cpu', print_examples_every=5):

    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    criterion = nn.CrossEntropyLoss(ignore_index=stoi['<pad>'])
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        val_rouge = quick_rouge1(model, val_loader, stoi, itos, eos_idx, device)
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | Val ROUGE-1: {val_rouge:.4f}")
        
        if (epoch + 1) % print_examples_every == 0:
            print("\n" + "-"*70)
            print(f"Examples after epoch {epoch+1}:")
            print("-"*70)
            
            model.eval()
            pad_idx = stoi['<pad>']
            
            x, _ = next(iter(val_loader))
            seq = x[0].to(device)
            seq_len = (seq != pad_idx).sum().item()
            split = int(seq_len * 0.75)
            
            prefix = seq[:split].tolist()
            target = seq[split:seq_len].tolist()
            
            gen_greedy = model.generate_greedy(prefix, len(target), eos_idx, device)
            gen_sample = model.generate(prefix, len(target), eos_idx, device, temperature=0.8)
            
            prefix_text = ' '.join([itos[t] for t in prefix if t not in [pad_idx, eos_idx]])
            target_text = ' '.join([itos[t] for t in target if t not in [pad_idx, eos_idx]])
            greedy_text = ' '.join([itos[t] for t in gen_greedy[len(prefix):] if t not in [pad_idx, eos_idx]])
            sample_text = ' '.join([itos[t] for t in gen_sample[len(prefix):] if t not in [pad_idx, eos_idx]])
            
            print(f"Prefix:    {prefix_text}")
            print(f"Target:    {target_text}")
            print(f"Greedy:    {greedy_text}")
            print(f"Sampled:   {sample_text}")
            print("-"*70 + "\n")
    
    print("="*70)
    print("TRAINING COMPLETE")
    print("="*70)