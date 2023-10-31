import torch
import transformers
from tqdm import tqdm
from transformers import  AutoModelForMaskedLM
from .poet_utils import RHYME_SCHEMES, METER_TYPES

from torch.utils.data import DataLoader, Dataset
from pytorch_optimizer import SAM,GSAM, ProportionScheduler, AdamP

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
    

class ValidatorTrainer:
    def __init__(self, model: ValidatorInterface, args: dict, train_dataset: Dataset, data_collator, device):
        self.model = model
        self.args = args
        self.epochs = 1 if "epochs" not in args.keys() else args["epochs"]
        self.batch_size = 1 if "batch_size" not in args.keys() else args["batch_size"]
        self.lr = 3e-4 if "lr" not in args.keys() else args["lr"]
        self.weight_decay = 0.0 if "weight_decay" not in args.keys() else args['weight_decay']
        
        self.train_loader = DataLoader(train_dataset, self.batch_size, True, collate_fn=data_collator)
        
        # SAM Values
        #self.device = device      
        #self.optimizer = SAM(self.model.parameters(), torch.optim.AdamW, lr=self.lr, weight_decay=self.weight_decay)
        #self.scheduler = transformers.get_constant_schedule_with_warmup(self.optimizer, len(train_dataset)//self.batch_size)
        
        # GSAM Value
        self.device = device
        self.base_optim =  AdamP(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = transformers.get_constant_schedule_with_warmup(self.base_optim, len(train_dataset)//self.batch_size)
        self.rho_scheduler=  ProportionScheduler( self.scheduler, max_lr=self.lr)
        self.optimizer = GSAM(self.model.parameters(),self.base_optim, self.model, self.rho_scheduler, alpha=0.05)
      
    def train(self):
        for epoch in  tqdm(range(self.epochs)):
            self.model.train()
            
            # SAM Attempt
            
            #for step, batch in enumerate(self.train_loader):
            #    # First Pass
            #    loss = self.model(input_ids=batch["input_ids"].to(self.device), attention_mask=batch["attention_mask"].to(self.device),
            #                      rhyme = None if batch["rhyme"] == None else batch["rhyme"].to(self.device),
            #                      metre = None if batch["metre"] == None else batch["metre"].to(self.device))['loss']
            #    loss.backward()          
            #    self.optimizer.first_step(zero_grad=True)
            #    # Second Pass
            #    loss = self.model(input_ids=batch["input_ids"].to(self.device), attention_mask=batch["attention_mask"].to(self.device),
            #                          rhyme = None if batch["rhyme"] == None else batch["rhyme"].to(self.device),
            #                          metre = None if batch["metre"] == None else batch["metre"].to(self.device))['loss']
            #    loss.backward()
            #    self.optimizer.second_step(zero_grad=True)
            #    self.scheduler.step()
           
            # GSAM Attempt 
                 
            for step, batch in enumerate(self.train_loader):
                def closure():
                    self.optimizer.base_optimizer.zero_grad()
                    with torch.enable_grad():
                        outputs = self.model(input_ids=batch["input_ids"].to(self.device), attention_mask=batch["attention_mask"].to(self.device),
                                  rhyme = None if batch["rhyme"] == None else batch["rhyme"].to(self.device),
                                  metre = None if batch["metre"] == None else batch["metre"].to(self.device))
                        loss = torch.nn.functional.cross_entropy(outputs['model_output'],batch['rhyme'] if isinstance(self.model, RhymeValidator) else batch['metre'])
                    loss.backward()
                    return outputs['model_output'], loss.detach()
                predictions, loss = self.optimizer.step(closure)
                self.scheduler.step()
                self.optimizer.update_rho_t()
                
                if step % 100 == 0:
                    print(f'Step {step},  loss : {loss.item()}')
    