import torch
from transformers import  AutoModelForMaskedLM
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
    def __init__(self,hidden_layers, input_size, hidden_size ,raw_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.hidden_layers_count = hidden_layers - 1
        self.input_size = input_size
        self.model_size = hidden_size
        self.raw_size = raw_size
        
        
        self.input_layer: torch.nn.Linear = torch.nn.Linear(self.input_size, self.model_size)
        self.hidden_layers=torch.nn.ModuleList([torch.nn.Linear(self.model_size, self.model_size) for i in range(self.hidden_layers_count)])
        self.batch_norms = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.model_size) for j in range(self.hidden_layers_count)])
        self.relu = torch.nn.ReLU()
        
        
        self.rhyme_regressor = torch.nn.Linear(self.model_size, len(RHYME_SCHEMES)) # Common Rhyme Type
        
        self.loss_fnc = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        
    def forward(self, input_ids=None, attention_mask=None, rhyme=None, *args, **kwargs):
        
        hidden = self.input_layer(input_ids)
        for layer, norm in zip(self.hidden_layers, self.batch_norms):
            hidden = norm(hidden)
            hidden = self.relu(hidden)    
            hidden = layer(hidden)
  
        rhyme_regression = self.rhyme_regressor(hidden)
            
        softmaxed = torch.softmax(rhyme_regression, dim=1)
        rhyme_loss = self.loss_fnc(softmaxed, rhyme)
        
        return {"model_output" : softmaxed,
                "loss": rhyme_loss}
        
    def predict(self, input_ids=None, *args, **kwargs):
        
        hidden = self.input_layer(input_ids)
        for layer, norm in zip(self.hidden_layers, self.batch_norms):
            hidden = norm(hidden)
            hidden = self.relu(hidden)    
            hidden = layer(hidden)
        hidden = self.relu(hidden)
  
        rhyme_regression = self.rhyme_regressor(hidden)
            
        softmaxed = torch.softmax(rhyme_regression, dim=1)
        
        return softmaxed
    
    def validate(self, input_ids=None, rhyme=None,*args, **kwargs):
        outputs = self.forward(input_ids=input_ids, rhyme=rhyme)['model_output']
        
        _true_val = (torch.argmax(rhyme, dim=1) == torch.argmax(outputs, dim=1)).float().sum().numpy()
        
        return _true_val
    
    
class MeterValidator(ValidatorInterface):
    def __init__(self, pretrained_model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = AutoModelForMaskedLM.from_pretrained(pretrained_model, output_hidden_states=True)
        
        self.config = self.model.config
        
        self.model_size = self.config.hidden_size
        
        self.meter_regressor = torch.nn.Linear(self.model_size, len(METER_TYPES)) # Meter Type
        
        self.loss_fnc = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        
    def forward(self, input_ids=None, attention_mask=None, metre=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids.type(torch.LongTensor))
        
        last_hidden = outputs['hidden_states'][-1]
        
        meter_regression = self.meter_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(meter_regression, dim=1)
        meter_loss = self.loss_fnc(softmaxed, metre)
        
        return {"model_output" : softmaxed,
                "loss": meter_loss + outputs.loss}
        
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
    

        
        