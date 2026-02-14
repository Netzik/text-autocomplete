"""
Evaluation pipeline for pre-trained transformer models (DistilGPT2).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from rouge_score import rouge_scorer


class TransformerEvaluator:
    """
    Wrapper for evaluating pre-trained transformer models on text completion task.
    Uses DistilGPT2 by default.
    """
    
    def __init__(self, model_name='distilgpt2', device=None):
    
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model has {n_params:,} parameters")
    
    @torch.no_grad()
    def generate(self, prefix_text, max_new_tokens=20, temperature=1.0, 
                 top_k=50, top_p=0.95, do_sample=True):
        """
        Generate continuation for a given prefix.
        
        Args:
            prefix_text: input text string
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling parameter
            top_p: nucleus sampling parameter
            do_sample: if True, use sampling; if False, use greedy
            
        Returns:
            generated text (only the new part, without prefix)
        """
        inputs = self.tokenizer(prefix_text, return_tensors='pt').to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=1
        )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        generated_text = full_text[len(prefix_text):].strip()
        
        return generated_text
    
    def evaluate(self, texts, prefix_ratio=0.75, num_samples=None, 
                 max_new_tokens=20, temperature=1.0, top_k=50, top_p=0.95):
        """
        Evaluate model on a dataset using ROUGE metrics.
        
        Strategy: Take 75% of each text as prefix, generate the rest, compare with ground truth.
        
        Args:
            texts: list of text strings
            prefix_ratio: fraction of text to use as prefix
            num_samples: limit number of samples (None = all)
            max_new_tokens: max tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
            top_p: nucleus sampling
            
        Returns:
            dict with average ROUGE scores
        """
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
            use_stemmer=True
        )
        
        scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': [],
            'rougeLsum': []
        }
        
        if num_samples:
            texts = texts[:num_samples]
        
        print(f"\nEvaluating on {len(texts)} samples...")
        
        for text in tqdm(texts, desc="Evaluating"):
            words = text.split()
            if len(words) < 4:
                continue
            
            split_idx = int(len(text) * prefix_ratio)
            prefix = text[:split_idx].strip()
            target = text[split_idx:].strip()
            
            if not target:
                continue
            
            generated = self.generate(
                prefix_text=prefix,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True
            )
            
            if not generated:
                continue
            
            result = scorer.score(target, generated)
            
            for k in scores:
                scores[k].append(result[k].fmeasure)
        
        return {k: sum(v) / len(v) if v else 0.0 for k, v in scores.items()}
    
    def print_examples(self, texts, num_examples=10, prefix_ratio=0.75, 
                      max_new_tokens=20, temperature=1.0, top_k=50, top_p=0.95):
        print("\n" + "="*70)
        print("GENERATION EXAMPLES (DistilGPT2)")
        print("="*70)
        
        examples_printed = 0
        
        for text in texts:
            if examples_printed >= num_examples:
                break
            
            words = text.split()
            if len(words) < 4:
                continue
            
            split_idx = int(len(text) * prefix_ratio)
            prefix = text[:split_idx].strip()
            target = text[split_idx:].strip()
            
            if not target:
                continue
            
            gen_greedy = self.generate(
                prefix_text=prefix,
                max_new_tokens=max_new_tokens,
                do_sample=False  
            )
            
            gen_sample = self.generate(
                prefix_text=prefix,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True
            )
            
            print(f"\nExample {examples_printed + 1}:")
            print(f"Prefix:    {prefix}")
            print(f"Target:    {target}")
            print(f"Greedy:    {gen_greedy}")
            print(f"Sampled:   {gen_sample}")
            
            examples_printed += 1
        
        print("="*70)


# ============================================================================
# Helper 
# ============================================================================

def evaluate_distilgpt2(texts, num_samples=500, device=None):
    evaluator = TransformerEvaluator(model_name='distilgpt2', device=device)
    
    scores = evaluator.evaluate(
        texts=texts,
        prefix_ratio=0.75,
        num_samples=num_samples,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
        top_p=0.95
    )
    
    return evaluator, scores