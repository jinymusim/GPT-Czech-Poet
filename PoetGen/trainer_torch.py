import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler

class Trainer:
    
    def __init__(self, model: PreTrainedModel, epochs: int, optimizer: Optimizer, scheduler: lr_scheduler._LRScheduler, 
                 dataloader: DataLoader, consistency_task: bool, masking_rate: float) -> None:
        self.model = model
        self.device = model.device
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.consistency_task = consistency_task
        self.mask_rate = masking_rate
        
    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            
            for step, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                label = batch["input_ids"].type(torch.LongTensor)
                inputs: torch.Tensor = batch['input_ids']
                mask = torch.rand(inputs.shape) < 1 -self.mask_rate
                inputs_masked = inputs * mask.int()
                
                out = self.model(input_ids=inputs_masked.to(self.device), labels=label.to(self.device), 
                                 attention_mask=batch['attention'].to(self.device))
                out.loss.backward()
                if self.consistency_task:
                    label_words = batch["last"].type(torch.LongTensor)
                    out_consistency = self.model(input_ids=batch["last"].to(self.device), labels=label_words.to(self.device), 
                                 attention_mask=batch['attention_last'].to(self.device))
                    out_consistency.loss.backward()
                self.optimizer.step()
                    
                self.scheduler.step()
                
                output = {'loss' : out.loss.item()}
                if step % 500 == 0:
                    print(f'Step {step},  loss : {output["loss"]}')
                
                

