import transformers
from transformers import PreTrainedModel
import torch
from torch.optim import Optimizer, lr_scheduler
import os
import sys

class Trainer:
    
    def __init__(self, model: PreTrainedModel, 
                 epochs: int, optimizer: Optimizer, 
                 scheduler: lr_scheduler._LRScheduler,
                 dataloader) -> None:
        self.model = model
        self.epochs = epochs
        self.device = model.device
        self.optimizer = optimizer
        self.scheduler =scheduler
        self.dataloader = dataloader
        
    def train(self):      
        for epoch in range(self.epochs):
            
            self.model.train()
            
            for step, batch in enumerate(self.dataloader):
                
                inputs = batch['input']
                labels = batch['label']
                
                self.optimizer.zero_grad()
                out = self.model(input_ids=inputs.to(self.device), labels=labels.to(self.device), attention_mask=batch['attention_mask'].to(self.device))
                out.loss.backward()
                self.optimizer.step()
                    
                self.scheduler.step()
                
                output = {'loss' : out.loss.item()}
                if step % 500 == 0:
                    print(f'Step {step},  loss : {output["loss"]}')
                
            