import torch 
from transformers import GPT2Config, GPT2Model
from poet_constants import poet_types

class ContextModule(torch.nn.Module):
    
    def __init__(self, block_count, input_size, n_embd ,output_size,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = GPT2Config(n_positions=input_size, n_head=(n_embd//(768//12)),n_embd=n_embd, 
                                 n_layer=block_count, output_hidden_states=True,  output_attentions =True)
        self.context_model = GPT2Model(self.config)
        self.linear_downscale = torch.nn.Linear(n_embd, output_size)
        self.input_size = input_size
        self.n_embd = n_embd
        self.output_size = output_size
        self.context_ids = None
        self.context_attention_mask = None
    
    # Context is getting injected from Top
    def forward(self, hidden_states,layer_past=None,*args, **kwargs):
        down = torch.zeros_like(hidden_states)
        model_output = None
        if self.context_ids != None:
            model_output = self.context_model.forward(input_ids=self.context_ids, attention_mask=self.context_attention_mask)
            down = self.linear_downscale.forward(model_output["hidden_states"][-1][:,0,:].view(-1, self.n_embd))[:, None, :]
        # torch.zeros( base n_head ,  ,base n_embd // base n_head))
        return  (hidden_states + down,
                 down[None, :, :, :],
                 (None if model_output == None else model_output["attentions"], 
                None))
        
class PoetTypeMoldule(torch.nn.Module):
    
    def __init__(self, block_count, input_size, output_size,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = GPT2Config(n_positions=input_size,n_head=(input_size//(768//12)), n_embd=input_size, 
                                 n_layer=block_count, output_hidden_states=True, output_attentions =True)
        self.type_model = GPT2Model(self.config)
        self.type_predict = torch.nn.Linear(input_size, len(poet_types))
        self.softmax = torch.nn.Softmax()
        self.linear_scale = torch.nn.Linear(len(poet_types), output_size)
        self.input_size = input_size
        self.output_size = output_size
        self.context_ids = None
        self.context_attention_mask = None
        self.type_labels=None
    
    # Context And type labels are to be injected to bypass GPT2Blocks 
    def forward(self, hidden_states,layer_past=None,*args, **kwargs):
        type_prob = torch.zeros((hidden_states.size[0], len(poet_types)))
        model_output = None
        if self.context_ids != None:
            model_output = self.type_model.forward(input_ids=self.context_ids, attention_mask=self.context_attention_mask)
            poet_type = self.type_predict.forward(model_output["hidden_states"][-1][:,0,:].view(-1, self.input_size))
            type_prob = self.softmax.forward(poet_type)
        if self.type_labels != None:
            type_prob = self.type_labels.type(torch.FloatTensor)
        linear_up = self.linear_scale.forward(type_prob)
        return (hidden_states + linear_up[:, None, :],
                linear_up[None, :, None, :], 
                (None if model_output == None else model_output["attentions"], 
                None))
            
        
        