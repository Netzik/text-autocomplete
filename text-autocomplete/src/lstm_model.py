"""
LSTM-based language model for text autocomplete.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    """
    LSTM language model that predicts next token given a sequence.
    
    Architecture:
        Embedding -> LSTM -> Linear -> Logits
    """
    
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.1):
        
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embed_size, 
            hidden_size, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0  
        )
        
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        embeds = self.embed(x)
        
        lstm_out, _ = self.lstm(embeds)
        
        logits = self.fc(lstm_out)
        
        return logits
    
    @torch.no_grad()
    def generate(self, start_tokens, max_new_tokens, eos_idx, device='cpu', temperature=1.0, top_k=None):
    
        self.eval()
        
        if isinstance(start_tokens, torch.Tensor):
            tokens = start_tokens.cpu().tolist()
        else:
            tokens = list(start_tokens)
        
        for _ in range(max_new_tokens):
            x = torch.tensor([tokens], dtype=torch.long, device=device)
            
            logits = self.forward(x)  
            logits = logits[0, -1, :]  
            
            logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float('inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == eos_idx:
                break
            
            tokens.append(next_token)
        
        return tokens
    
    @torch.no_grad()
    def generate_greedy(self, start_tokens, max_new_tokens, eos_idx, device='cpu'):

        self.eval()
        
        if isinstance(start_tokens, torch.Tensor):
            tokens = start_tokens.cpu().tolist()
        else:
            tokens = list(start_tokens)
        
        for _ in range(max_new_tokens):
            x = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = self.forward(x)
            next_token = logits[0, -1].argmax().item()
            
            if next_token == eos_idx:
                break
                
            tokens.append(next_token)
        
        return tokens


# ============================================================================
# Helper func
# ============================================================================

def create_model(vocab_size, embed_size=128, hidden_size=256, num_layers=2, dropout=0.1):
    """
    Factory function to create LSTM model with given hyperparameters.
    """
    model = LSTMModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Created LSTM model with {n_params:,} parameters")
    
    return model