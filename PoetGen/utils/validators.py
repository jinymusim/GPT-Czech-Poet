import torch
from transformers import  AutoModelForMaskedLM
from .poet_utils import RHYME_SCHEMES, METER_TYPES

class ValidatorInterface(torch.nn.Module):
    """Pytorch Model Interface. Abstract class for all validators

    Args:
        torch (_type_): Is child of torch.nn.Module for integration with torch and huggingface 
    """
    def __init__(self, *args, **kwargs) -> None:
        """ Constructor. As child Class needs to construct Parent
        """
        super().__init__(*args, **kwargs)
        
    def forward(self, input_ids=None, attention_mask=None, *args, **kwargs):
        """Compute model output and model loss

        Args:
            input_ids (_type_, optional): Model inputs. Defaults to None.
            attention_mask (_type_, optional): Attention mask where padding starts. Defaults to None.

        Raises:
            NotImplementedError: Abstract class
        """
        raise NotImplementedError()
    
    def predict(self, input_ids=None, *args, **kwargs):
        """Compute model outputs

        Args:
            input_ids (_type_, optional): Model inputs. Defaults to None.

        Raises:
            NotImplementedError: Abstract class
        """
        raise NotImplementedError()
    
    def validate(self, input_ids=None, *args, **kwargs):
        """Validate model given some labels, Doesn't use loss

        Args:
            input_ids (_type_, optional): Model inputs. Defaults to None.

        Raises:
            NotImplementedError: Abstract class
        """
        raise NotImplementedError()
    
    
class RhymeValidator(ValidatorInterface):
    def __init__(self, pretrained_model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = AutoModelForMaskedLM.from_pretrained(pretrained_model, output_hidden_states=True)
        
        self.config = self.model.config
        
        self.model_size = self.config.hidden_size 
        
        self.rhyme_regressor = torch.nn.Linear(self.model_size, len(RHYME_SCHEMES)) # Common Rhyme Type
        
        self.loss_fnc = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        
    def forward(self, input_ids=None, attention_mask=None, rhyme=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids.type(torch.LongTensor))
        
        last_hidden = outputs['hidden_states'][-1]
        
        rhyme_regression = self.rhyme_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(rhyme_regression, dim=1)
        rhyme_loss = self.loss_fnc(softmaxed, rhyme)
        
        return {"model_output" : softmaxed,
                "loss": rhyme_loss + outputs.loss}
        
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
    def __init__(self, pretrained_model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = AutoModelForMaskedLM.from_pretrained(pretrained_model, output_hidden_states=True)
        
        self.config = self.model.config
        
        self.model_size = self.config.hidden_size
        
        self.meter_regressor = torch.nn.Linear(self.model_size, len(METER_TYPES)) # Meter Type
        
        self.loss_fnc = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        
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
    

        
        