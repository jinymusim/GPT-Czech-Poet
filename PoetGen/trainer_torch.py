import torch
from poet_model import PoetModel
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler

class Trainer:
    
    def __init__(self, model: PoetModel, device,epochs: int, optimizer: Optimizer, scheduler: lr_scheduler._LRScheduler, 
                 dataloader: DataLoader, consistency_task: bool, masking_rate: float) -> None:
        self.model = model
        self.device = device
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
                if self.consistency_task:
                    mask = torch.rand(inputs.shape) < 1 -self.mask_rate
                    inputs = inputs * mask.int()
                
                out = self.model(input_ids=inputs.to(self.device), labels=label.to(self.device), 
                                 attention_mask=batch['attention'].to(self.device), vowel_count=(batch['nums'].type(torch.FloatTensor)).to(self.device))
                out['model_output'].loss.backward(retain_graph=True)             
                if self.consistency_task:
                    out['vowel_regression_loss'].backward()
                    
                self.optimizer.step()
                    
                self.scheduler.step()
                
                output = {'loss' : out['model_output'].loss.item()}
                if step % 500 == 0:
                    print(f'Step {step},  loss : {output["loss"]}')
                
                

