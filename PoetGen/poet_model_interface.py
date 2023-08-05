
from transformers import AutoTokenizer
import torch

class PoetModelInterface(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, *args, **kwargs):
        raise NotImplementedError()
    
    def generate_forced(self, prompt:str, tokenizer: AutoTokenizer):
        raise NotImplementedError()

    @staticmethod
    def rhyme_like(rhyme:str):
        return rhyme.isupper() and len(rhyme) == 4
    
    def save_LM(self, LM_path):
        raise NotImplementedError()