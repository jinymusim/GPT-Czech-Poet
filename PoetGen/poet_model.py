from transformers import  AutoModelForCausalLM
import transformers
import torch
import os
import argparse


class PoetModel(torch.nn.Module):
    def __init__(self, pretrainedModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = AutoModelForCausalLM.from_pretrained(pretrainedModel, output_hidden_states=True)
        self.vowels_regressor = torch.nn.Linear(768,1) # Number of Emmbedings of gpt2 is 768, we want 1 num out
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, vowel_count=None, rhyme=None):
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        last_hidden = outputs['hidden_states'][-1]
        vowel_regression = self.vowels_regressor(last_hidden[:,0,:].view(-1, 768))
        
        vowel_loss = None
        if vowel_count is not None:
            loss_fct = torch.nn.MSELoss()
            vowel_loss = loss_fct(vowel_regression.view(-1, 1), vowel_count.view(-1, 1))
        
        return {"model_output" : outputs, 
                "vowel_regression_output": vowel_regression, "vowel_regression_loss": vowel_loss,}
    
    def save_LM(self, LM_path):
        self.model.save_pretrained(LM_path)