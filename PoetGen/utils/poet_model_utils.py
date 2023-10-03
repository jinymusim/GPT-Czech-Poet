import torch

class PoetModelInterface(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, *args, **kwargs):
        raise NotImplementedError()
    
    def generate_forced(self,  *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def rhyme_like(rhyme:str):
        return rhyme.isupper() and len(rhyme) in [4,6]
    
    def save_LM(self, LM_path):
        raise NotImplementedError()
    

from transformers import GPT2Config, GPT2Model
from .poet_utils import POET_YEARS_BUCKETS

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
        
class PoetTypeModule(torch.nn.Module):
    
    def __init__(self, block_count, input_size, n_embd,output_size,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = GPT2Config(n_positions=input_size, n_head=(n_embd//(768//12)),n_embd=n_embd, 
                                 n_layer=block_count, output_hidden_states=True,  output_attentions =True)
        self.type_model = GPT2Model(self.config)
        self.type_predict = torch.nn.Linear(n_embd, len(POET_YEARS_BUCKETS))
        self.softmax = torch.nn.Softmax()
        self.linear_scale = torch.nn.Linear(len(POET_YEARS_BUCKETS), output_size)
        self.input_size = input_size
        self.n_embd = n_embd
        self.output_size = output_size
        self.context_ids = None
        self.context_attention_mask = None
        self.type_labels=None
        # Store for loss for model itself
        self.indiv_loss=None
    
    # Context And type labels are to be injected to bypass GPT2Blocks 
    def forward(self, hidden_states,layer_past=None,*args, **kwargs):
        type_prob = torch.zeros((hidden_states.shape[0], len(POET_YEARS_BUCKETS))).to("cuda" if torch.cuda.is_available() else "cpu")
        model_output = None
        if self.context_ids != None:
            model_output = self.type_model.forward(input_ids=self.context_ids, attention_mask=self.context_attention_mask)
            poet_type = self.type_predict.forward(model_output["hidden_states"][-1][:,0,:].view(-1, self.n_embd))
            type_prob = self.softmax.forward(poet_type) 
        if self.type_labels != None:
            loss_fct = torch.nn.CrossEntropyLoss()
            self.indiv_loss = loss_fct(type_prob, self.type_labels)
            type_prob = (self.type_labels.type(torch.FloatTensor)).to("cuda" if torch.cuda.is_available() else "cpu")
        linear_up = self.linear_scale.forward(type_prob)
        return (hidden_states + linear_up[:, None, :],
                linear_up[None, :, None, :], 
                (None if model_output == None else model_output["attentions"], 
                None))
            
        