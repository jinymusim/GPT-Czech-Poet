import torch 
from transformers import GPT2Config, GPT2Model

class ContextModule(torch.nn.Module):
    
    def __init__(self, block_count, input_size, output_size,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = GPT2Config(n_positions=input_size, n_embd=input_size, n_layer=block_count, output_hidden_states=True)
        self.context_model = GPT2Model(self.config)
        self.linear_downscale = torch.nn.Linear(input_size, output_size)
        
    def forward(self, input_ids, context_ids, context_attention_mask,*args, **kwargs):
        model_output = self.context_model.forward(input_ids=context_ids, attention_mask=context_attention_mask)
        down = self.linear_downscale.forward(model_output["hidden_states"][-1])
        return  input_ids + down
        
