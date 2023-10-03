import torch
from transformers import  GPT2Model, GPT2Config
from .poet_utils import RHYME_SCHEMES, METER_TYPES

class ValidatorInterface(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, input_ids=None, attention_mask=None, *args, **kwargs):
        raise NotImplementedError()
    
    def predict(self, input_ids=None, *args, **kwargs):
        raise NotImplementedError()
    
    def validate(self, input_ids=None, *args, **kwargs):
        raise NotImplementedError()
    
    
class RhymeValidator(ValidatorInterface):
    def __init__(self,block_count, input_size, n_embd , *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = GPT2Config(n_positions=input_size, n_head=(n_embd//(768//12)),n_embd=n_embd, 
                                 n_layer=block_count, output_hidden_states=True)
        self.model = GPT2Model(self.config)
        
        self.model_size = n_embd
        
        self.rhyme_regressor = torch.nn.Linear(self.model_size, len(RHYME_SCHEMES)) # Rhyme Type
        
    def forward(self, input_ids=None, attention_mask=None, rhyme=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        last_hidden = outputs['hidden_states'][-1]
        
        rhyme_regression = self.rhyme_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(rhyme_regression, dim=1)
        loss_fct = torch.nn.CrossEntropyLoss()
        rhyme_loss = loss_fct(softmaxed, rhyme)
        
        return {"model_output" : softmaxed,
                "loss": rhyme_loss}
        
    def predict(self, input_ids=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids)
        
        last_hidden = outputs['hidden_states'][-1]
        
        rhyme_regression = self.rhyme_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(rhyme_regression, dim=1)
        
        return softmaxed
    
    def validate(self, input_ids=None, rhyme=None,*args, **kwargs):
        outputs = self.model(input_ids=input_ids)
        
        last_hidden = outputs['hidden_states'][-1]
        
        rhyme_regression = self.rhyme_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(rhyme_regression, dim=1)
        
        _true_val = (torch.argmax(rhyme, dim=1) == torch.argmax(softmaxed, dim=1)).float().sum().numpy()
        
        return _true_val
    
    
class MeterValidator(ValidatorInterface):
    def __init__(self,block_count, input_size, n_embd , *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = GPT2Config(n_positions=input_size, n_head=(n_embd//(768//12)),n_embd=n_embd, 
                                 n_layer=block_count, output_hidden_states=True)
        self.model = GPT2Model(self.config)
        
        self.model_size = n_embd
        
        self.meter_regressor = torch.nn.Linear(self.model_size, len(METER_TYPES)) # Meter Type
        
    def forward(self, input_ids=None, attention_mask=None, metre=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        last_hidden = outputs['hidden_states'][-1]
        
        meter_regression = self.meter_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(meter_regression, dim=1)
        loss_fct = torch.nn.CrossEntropyLoss()
        meter_loss = loss_fct(softmaxed, metre)
        
        return {"model_output" : softmaxed,
                "loss": meter_loss}
        
    def predict(self, input_ids=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids)
        
        last_hidden = outputs['hidden_states'][-1]
        
        meter_regression = self.meter_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(meter_regression, dim=1)
        
        return softmaxed
    
    def validate(self, input_ids=None, metre=None,*args, **kwargs):
        outputs = self.model(input_ids=input_ids)
        
        last_hidden = outputs['hidden_states'][-1]
        
        meter_regression = self.meter_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(meter_regression, dim=1)
        
        _true_val = (torch.argmax(metre, dim=1) == torch.argmax(softmaxed, dim=1)).float().sum().numpy()
        
        return _true_val
    

        
        