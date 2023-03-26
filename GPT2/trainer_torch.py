import os
import math
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.utils import ModelOutput
from torch.optim import Optimizer, lr_scheduler
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Trainer:
    
    def __init__(self, model: PreTrainedModel, epochs: int, optimizer: Optimizer, scheduler: lr_scheduler._LRScheduler) -> None:
        self.model = model
        self.device = model.device
        self.epochs = epochs
        self.optimezer = optimizer
        self.scheduler = scheduler
        
    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            
            for step, batch in ...:
                
                self.optimizer.zero_grad() 
                out = self.model(input_ids=batch['input_ids'].to(device),  attention_mask=batch['attention_mask'].to(device), labels=labels.to(device))
                out.loss.backward()
                self.optimizer.step()
                
                # Move inside so the warmup is more smoothed
                self.scheduler.step()

