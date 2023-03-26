import os
import math
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.utils import ModelOutput
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
            
            for batch in self.dataloader:
                
                self.optimizer.zero_grad() 
                out = self.model(input_ids=batch.to(device), labels=batch.to(device))
                out.loss.backward()
                self.optimizer.step()
                
                # Move inside so the warmup is more smoothed
                self.scheduler.step()

