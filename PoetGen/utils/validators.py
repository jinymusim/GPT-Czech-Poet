import torch
import transformers
import jellyfish
from tqdm import tqdm
from transformers import  AutoModelForMaskedLM
from transformers.utils import ModelOutput
import numpy as np
from .poet_utils import StropheParams

from torch.utils.data import DataLoader, Dataset
from pytorch_optimizer import SAM

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
    
    def predict_state(self, input_ids=None, *args, **kwargs):
        """Compute model outputs

        Args:
            input_ids (_type_, optional): Model inputs. Defaults to None.

        Raises:
            NotImplementedError: Abstract class
        """
        raise NotImplementedError()
    
    def validate_model(self, input_ids=None, *args, **kwargs):
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
        
        self.rhyme_regressor = torch.nn.Linear(self.model_size, len(StropheParams.RHYME)) # Common Rhyme Type
        
        self.loss_fnc = torch.nn.CrossEntropyLoss(label_smoothing=0.0, weight=torch.tensor([1, 1, 1.5, 1.5, 1.5, 1.5, 
                                                                                 2, 2,   2,   3,   3,   3, 
                                                                                 3, 3,   3,   3,   4,   4, 
                                                                                 5, 5,   5,   5,   7,   7, 
                                                                                 7, 7,   7,   8,   8,   8,
                                                                                 9, 9,   9,  10,  10,  10,
                                                                                 12,12, 12,  12,  12,  12,
                                                                                 15,15,1.5]) )
                                                        
    def forward(self, input_ids=None, attention_mask=None, rhyme=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids.type(torch.LongTensor))
        
        last_hidden = outputs['hidden_states'][-1]
        
        rhyme_regression = self.rhyme_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(rhyme_regression, dim=1)
        rhyme_loss = self.loss_fnc(softmaxed, rhyme)
        
        return ModelOutput(loss=rhyme_loss + outputs.loss, model_output=softmaxed)
        
    def predict_state(self, input_ids=None, *args, **kwargs):
        
        outputs = self.model(input_ids=input_ids)
        
        last_hidden = outputs['hidden_states'][-1]
        
        rhyme_regression = self.rhyme_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(rhyme_regression, dim=1)
        
        return softmaxed
    
    def validate_model(self, input_ids=None, rhyme=None, k:int = 2,*args, **kwargs):
        outputs = self.model(input_ids=input_ids)
        
        last_hidden = outputs['hidden_states'][-1]
        
        rhyme_regression = self.rhyme_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(rhyme_regression, dim=1)
        
        softmaxed = softmaxed.flatten().cpu()
        
        predicted_val = torch.argmax(softmaxed)
        
        predicted_top_k = torch.topk(softmaxed, k).indices
        
        label_val = torch.argmax(rhyme.flatten())
        
        validation_true_val = (label_val == predicted_val).float().sum().numpy()
        top_k_presence = 0
        if label_val in predicted_top_k:
            top_k_presence = 1
            
        levenshtein = jellyfish.levenshtein_distance(StropheParams.RHYME[predicted_val] if StropheParams.RHYME[predicted_val] != None else "", StropheParams.RHYME[label_val] if  StropheParams.RHYME[label_val] != None else "")
        
        hit_pred = softmaxed[label_val].detach().numpy()
        
        return {"acc" : validation_true_val,
                "top_k" : top_k_presence,
                "lev_distance": levenshtein,
                "predicted_label" : hit_pred
        }
         
    
    
class MeterValidator(ValidatorInterface):
    def __init__(self, pretrained_model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = AutoModelForMaskedLM.from_pretrained(pretrained_model, output_hidden_states=True)
        
        self.config = self.model.config
        
        self.model_size = self.config.hidden_size
        
        self.meter_regressor = torch.nn.Linear(self.model_size, len(StropheParams.METER)) # Meter Type
        
        self.loss_fnc = torch.nn.CrossEntropyLoss(label_smoothing=0.0, weight=torch.tensor([1, 1.5, 5, 10, 10, 20, 5, 20, 20, 0]))
        
    def forward(self, input_ids=None, attention_mask=None, metre_ids=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids.type(torch.LongTensor))
        
        last_hidden = outputs['hidden_states'][-1]
        
        meter_regression = self.meter_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(meter_regression, dim=1)
        meter_loss = self.loss_fnc(softmaxed, metre_ids)
        
        return ModelOutput(loss=meter_loss + outputs.loss, model_output=softmaxed)
        
    def predict_state(self, input_ids=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids)
        
        last_hidden = outputs['hidden_states'][-1]
        
        meter_regression = self.meter_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(meter_regression, dim=1)
        
        return softmaxed
    
    def validate_model(self, input_ids=None, metre_ids=None, attention_mask=None, k: int=2,*args, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask )
        
        last_hidden = outputs['hidden_states'][-1]
        
        meter_regression = self.meter_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(meter_regression, dim=1)
        
        softmaxed = softmaxed.flatten().cpu()
        
        predicted_val = torch.argmax(softmaxed)
        
        predicted_top_k = torch.topk(softmaxed, k).indices
        
        label_val = torch.argmax(metre_ids.flatten())
        
        validation_true_val = (label_val == predicted_val).float().sum().numpy()
        top_k_presence = 0
        if label_val in predicted_top_k:
            top_k_presence = 1
        
        hit_pred = softmaxed[label_val].detach().numpy()
        
        return {"acc" : validation_true_val,
                "top_k" : top_k_presence,
                "predicted_label" : hit_pred
        }
        
class YearValidator(ValidatorInterface):
    def __init__(self, pretrained_model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = AutoModelForMaskedLM.from_pretrained(pretrained_model, output_hidden_states=True)
        
        self.config = self.model.config
        
        self.model_size = self.config.hidden_size
        
        self.year_era = torch.nn.Linear(self.model_size, len(StropheParams.YEAR))
        self.softmax = torch.nn.Softmax(dim=-1)
        
        self.year_val = torch.nn.Linear(self.model_size, 1) # Year Value     
        
        
        self.loss_fnc_era = torch.nn.CrossEntropyLoss(label_smoothing=0.0,weight=torch.tensor([10, 5, 3, 3, 1, 1, 1.5, 2, 5, 0]))
        
        self.loss_fnc_val = torch.nn.L1Loss()
        
    def forward(self, input_ids=None, attention_mask=None, year_bucket=None, year=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids.type(torch.LongTensor))
        
        last_hidden = outputs['hidden_states'][-1]
        
        
        year_val = self.year_val((last_hidden[:,0,:].view(-1, self.model_size)))
        year_val_loss = self.loss_fnc_val(year_val, year)
        
        year_era = self.year_era((last_hidden[:,0,:].view(-1, self.model_size)))
        year_era = self.softmax(year_era)
        year_era_loss =  self.loss_fnc_era(year_era, year_bucket)
        
        return ModelOutput(loss=year_val_loss + year_era_loss  + outputs.loss, model_output=(year_val, year_era))
        
    def predict_state(self, input_ids=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids)
        
        last_hidden = outputs['hidden_states'][-1]
        
        year_val = self.year_val((last_hidden[:,0,:].view(-1, self.model_size)))
        
        return year_val
    
    def validate_model(self, input_ids=None, year_bucket=None, k: int=2,*args, **kwargs):
        
        outputs = self.model(input_ids=input_ids)
        
        last_hidden = outputs['hidden_states'][-1]
        
        year_val = self.year_val((last_hidden[:,0,:].view(-1, self.model_size)))
        if hasattr(self, 'year_era'):
            year_era = self.year_era((last_hidden[:,0,:].view(-1, self.model_size)))
            year_era = self.softmax(year_era)
        
        year_val = year_val.detach().flatten().cpu().numpy()
        if hasattr(self, 'year_era'):
            year_era = year_era.detach().flatten().cpu().numpy()
        
        publish_vector  = [1/(1 + abs(year - year_val[0])) for year in StropheParams.YEAR[:-1]] + [0]
        publish_vector = np.asarray(publish_vector)/np.sum(publish_vector)
        # Adding era prediction
        if hasattr(self, 'year_era'):
            publish_vector+= year_era
        publish_vector = torch.tensor( np.asarray(publish_vector)/np.sum(publish_vector))
        
        
        predicted_val = torch.argmax(publish_vector)
        
        predicted_top_k = torch.topk(publish_vector, k).indices
        
        label_val = torch.argmax(year_bucket.flatten())
        
        validation_true_val = (label_val == predicted_val).float().sum().numpy()
        top_k_presence = 0
        if label_val in predicted_top_k:
            top_k_presence = 1
        
        hit_pred = publish_vector[label_val].detach().numpy()
        
        distance = abs(label_val.numpy() - predicted_val.numpy())
        
        return {"acc" : validation_true_val,
                "top_k" : top_k_presence,
                "predicted_label" : hit_pred,
                "distance" : distance
        }       
    
    

class ValidatorTrainer:
    def __init__(self, model: ValidatorInterface, args: dict, train_dataset: Dataset, data_collator, device):
        self.model = model
        self.args = args
        self.epochs = 1 if "epochs" not in args.keys() else args["epochs"]
        self.batch_size = 1 if "batch_size" not in args.keys() else args["batch_size"]
        self.lr = 5e-5 if "lr" not in args.keys() else args["lr"]
        self.weight_decay = 0.0 if "weight_decay" not in args.keys() else args['weight_decay']
        
        self.train_loader = DataLoader(train_dataset, self.batch_size, True, collate_fn=data_collator)
        
        # SAM Values
        self.device = device      
        self.optimizer = SAM(self.model.parameters(), torch.optim.AdamW, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = transformers.get_constant_schedule_with_warmup(self.optimizer, 4 * len(train_dataset)//self.batch_size)
        
        # GSAM Value
        #self.device = device
        #self.base_optim =  AdamP(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        #self.scheduler = transformers.get_constant_schedule_with_warmup(self.base_optim, len(train_dataset)//self.batch_size)
        #self.rho_scheduler=  ProportionScheduler( self.scheduler, max_lr=self.lr)
        #self.optimizer = GSAM(self.model.parameters(),self.base_optim, self.model, self.rho_scheduler, alpha=0.05)
      
    def train(self):
        for epoch in  tqdm(range(self.epochs)):
            self.model.train()
            
            # SAM Attempt
            
            for step, batch in enumerate(self.train_loader):
                # First Pass
                loss = self.model(input_ids=batch["input_ids"].to(self.device), attention_mask=batch["attention_mask"].to(self.device),
                                  rhyme = None if batch["rhyme"] == None else batch["rhyme"].to(self.device),
                                  metre_ids = None if batch["metre_ids"] == None else batch["metre_ids"].to(self.device),
                                  year_bucket = None if batch["year_bucket"] == None else batch["year_bucket"].to(self.device),
                                  year = None if batch["year"] == None else batch["year"].to(self.device))['loss']
                loss.backward()          
                self.optimizer.first_step(zero_grad=True)
                # Second Pass
                loss = self.model(input_ids=batch["input_ids"].to(self.device), attention_mask=batch["attention_mask"].to(self.device),
                                      rhyme = None if batch["rhyme"] == None else batch["rhyme"].to(self.device),
                                      metre_ids = None if batch["metre_ids"] == None else batch["metre_ids"].to(self.device),
                                      year_bucket = None if batch["year_bucket"] == None else batch["year_bucket"].to(self.device),
                                      year = None if batch["year"] == None else batch["year"].to(self.device))['loss']
                
                loss.backward()
                self.optimizer.second_step(zero_grad=True)
                self.scheduler.step()
           
            # GSAM Attempt 
                 
            #for step, batch in enumerate(self.train_loader):
            #    def closure():
            #        self.optimizer.base_optimizer.zero_grad()
            #        with torch.enable_grad():
            #            outputs = self.model(input_ids=batch["input_ids"].to(self.device), attention_mask=batch["attention_mask"].to(self.device),
            #                      rhyme = None if batch["rhyme"] == None else batch["rhyme"].to(self.device),
            #                      metre = None if batch["metre"] == None else batch["metre"].to(self.device))
            #            loss = torch.nn.functional.cross_entropy(outputs['model_output'].to(self.device),batch['rhyme'].to(self.device) if isinstance(self.model, RhymeValidator) else batch['metre'].to(self.device))
            #        loss.backward()
            #        return outputs['model_output'], loss.detach()
            #    predictions, loss = self.optimizer.step(closure)
            #    self.scheduler.step()
            #    self.optimizer.update_rho_t()
            #    
                if step % 100 == 0:
                    print(f'Step {len(self.train_loader) * epoch + step},  loss : {loss.item()}', flush=True)
    