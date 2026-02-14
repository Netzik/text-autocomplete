"""
Loading, cleaning, tokenization and vocabulary building.
"""

import re
import random
from collections import Counter

PAD = '<pad>'
UNK = '<unk>'
EOS = '<eos>'

def load_texts(path, n=None):
    """
    Load texts from file, one text per line.
    
    """
    texts = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                texts.append(line)
    return texts

def clean(text):
    """
    Clean text: lowercase, remove URLs/mentions, keep only letters and spaces.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_vocab(texts, vocab_size):
    """
    Build vocabulary from texts, keeping most common words.
    
    """
    words = []
    for text in texts:
        words.extend(text.split())
    
    most_common = Counter(words).most_common(vocab_size - 3)  
    
    stoi = {PAD: 0, UNK: 1, EOS: 2}
    for i, (word, _) in enumerate(most_common):
        stoi[word] = i + 3
    
    itos = {i: w for w, i in stoi.items()}
    
    return stoi, itos

def tokenize(text, stoi):
    """
    Convert text to list of token indices.
    
    """
    return [stoi.get(word, stoi[UNK]) for word in text.split()]

def split_dataset(texts, train_ratio=0.8, val_ratio=0.1):
    """
    Split texts into train/val/test sets.
    
    """
    random.seed(42)
    shuffled = texts.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    
    return train, val, test