import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler

class Trainer:
    
    def __init__(self, model: PreTrainedModel, epochs: int, optimizer: Optimizer, scheduler: lr_scheduler._LRScheduler, dataloader: DataLoader) -> None:
        self.model = model
        self.device = model.device
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        
    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            
            for step, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                label = batch["input_ids"].type(torch.LongTensor)
                out = self.model(input_ids=batch["input_ids"].to(self.device), labels=label.to(self.device), 
                                 attention_mask=batch['attention'].to(self.device))
                out.loss.backward()
                self.optimizer.step()
                    
                self.scheduler.step()
                
                output = {'loss' : out.loss.item()}
                if step % 500 == 0:
                    print(f'Step {step},  loss : {output["loss"]}')
                
                

