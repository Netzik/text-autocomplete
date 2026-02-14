"""
PyTorch Dataset for next token prediction task.
"""

import torch
from torch.utils.data import Dataset


from src.data_utils import tokenize, EOS

class NextTokenDataset(Dataset):
    
    def __init__(self, texts, stoi, max_len=50):

        self.data = []
        
        for text in texts:
            tokens = tokenize(text, stoi)[:max_len] + [stoi[EOS]]
            
            if len(tokens) < 2:
                continue
            
            x = tokens[:-1]
            y = tokens[1:]
            
            self.data.append((x, y))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.LongTensor(x), torch.LongTensor(y)
    
    @staticmethod
    def collate_fn(batch, pad_idx):
        xs, ys = zip(*batch)
        
        x_batch = torch.nn.utils.rnn.pad_sequence(
            xs, batch_first=True, padding_value=pad_idx
        )
        y_batch = torch.nn.utils.rnn.pad_sequence(
            ys, batch_first=True, padding_value=pad_idx
        )
        
        return x_batch, y_batch