import os
import math
import torch
import logging
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.utils import ModelOutput
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
import torch.nn as nn

class Trainer:
    
    def __init__(self, model: PreTrainedModel, epochs: int, optimizer: Optimizer, scheduler: lr_scheduler._LRScheduler, dataloader: DataLoader, half_precision: bool = False) -> None:
        self.model = model
        self.device = model.device
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.half_precision = half_precision
        if self.half_precision:
            self.model.cpu().half().to(self.device)
        
    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            
            for step, batch in enumerate(self.dataloader):
                if self.half_precision:
                    batch.half()
                labels = batch.type(torch.LongTensor)
                self.optimizer.zero_grad() 
                out = self.model(input_ids=batch.to(self.device), labels=labels.to(self.device))
                out.loss.backward()
                self.optimizer.step()
                
                # Move inside so the warmup is more smoothed
                self.scheduler.step()
                
                output = {'loss' : out.loss.item()}
                if step % 500 == 0:
                    print(f'Step {step},  loss : {output["loss"]}')
                
                

