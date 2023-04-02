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
            self.scaler = torch.cuda.amp.grad_scaler.GradScaler()
        
    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            
            for step, batch in enumerate(self.dataloader):
                labels = batch.type(torch.LongTensor)
                self.optimizer.zero_grad() 
                if self.half_precision:
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        out = self.model(input_ids=batch.to(self.device), labels=labels.to(self.device))
                    self.scaler.scale(out.loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    out = self.model(input_ids=batch.to(self.device), labels=labels.to(self.device))
                    out.loss.backward()
                    self.optimizer.step()
                    
                    self.scheduler.step()
                
                output = {'loss' : out.loss.item()}
                if step % 500 == 0:
                    print(f'Step {step},  loss : {output["loss"]}')
                
                

