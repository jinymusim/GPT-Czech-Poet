import torch
import os
import json
import csv
import random
import numpy as np
from torch.utils.data import Dataset
from functools import partial

from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments, AutoTokenizer


import argparse

class SentimentModel(torch.nn.Module):
    def __init__(self, pretrained_model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = AutoModelForMaskedLM.from_pretrained(pretrained_model, output_hidden_states=True)
        self.sentiment_regressor = torch.nn.Linear(self.model.config.hidden_size, 2)
        self.loss_fc = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.model_acc = 0
        
    def forward(self, input_ids, attention_mask, sentiment, *args, **kwargs):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids.type(torch.LongTensor))
        last_hidden = out['hidden_states'][-1]
        
        sentiment_regressor = self.sentiment_regressor(last_hidden[:,0,:].view(-1, self.model.config.hidden_size))
        
        softmaxed = torch.softmax(sentiment_regressor, dim=1)
        sentiment_loss = self.loss_fc(softmaxed, sentiment)
        
        return {"model_output": softmaxed,
                "loss": sentiment_loss + out.loss}
        
    def predict(self, input_ids=None, *args, **kwargs):
        
        outputs = self.model(input_ids=input_ids)
        
        last_hidden = outputs['hidden_states'][-1]
        
        sentiment_regressor = self.sentiment_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(sentiment_regressor, dim=1)
        
        return softmaxed
    
    def validate(self, input_ids=None, sentiment=None,*args, **kwargs):
        outputs = self.model(input_ids=input_ids)
        
        last_hidden = outputs['hidden_states'][-1]
        
        sentiment_regressor = self.sentiment_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
            
        softmaxed = torch.softmax(sentiment_regressor, dim=1)
        
        softmaxed = softmaxed.flatten().cpu()
        
        predicted_val = torch.argmax(softmaxed)
        
        label_val = torch.argmax(sentiment.flatten())
        
        validation_true_val = (label_val == predicted_val).float().sum().numpy()        
        
        return {"acc" : validation_true_val}
    
    
class MovieReviewDataset(Dataset):
    def __init__(self, data_path, cache_dir = os.path.abspath(os.path.dirname(__file__)), val_data_rate = 0.1) -> None:
        self.data_path = data_path
        self.cache_dir = cache_dir
        self.val_data_rate = val_data_rate
        if os.path.exists(os.path.join(cache_dir, "train_data.json")) and os.path.exists(os.path.join(cache_dir, "val_data.json")):
            self.train_data = json.load(open(os.path.join(cache_dir, "train_data.json"), 'r'))
            self.val_data = json.load(open(os.path.join(cache_dir, "val_data.json"), 'r'))
        else:
            self.load_csv()
            json.dump(self.train_data ,open(os.path.join(cache_dir, "train_data.json"), 'w+'), indent=6)
            json.dump(self.val_data ,open(os.path.join(cache_dir, "val_data.json"), 'w+'), indent=6)
            
        super().__init__()
        
        
    def load_csv(self):
        self.train_data = []
        self.val_data = []
        csv_reader = csv.reader(open(self.data_path, 'r', encoding='utf-8'))
        for row in csv_reader:
            if random.random() >= self.val_data_rate:
                self.train_data.append(
                    {
                        "input_ids" : row[0],
                        "sentiment" : [1,0] if row[1] == 'positive' else [0,1]
                    }
                )
            else:
                self.val_data.append(
                    {
                        "input_ids" : row[0],
                        "sentiment" : [1,0] if row[1] == 'positive' else [0,1]
                    }
                )
                
    def __len__(self):
        """Return length of training data
        
        Returns:
            int: length of training data
        """
        return len(self.train_data)
        
    def __getitem__(self, index):
        """return indexed item
        
        Args:
            index (int): index from where to return
            
        Returns:
            dict: dict with indexed data
        """
        return self.train_data[index]
    
    @staticmethod
    def collate(batch, tokenizer):
        
        tokenized = tokenizer([text['input_ids'] + tokenizer.eos_token for text in batch],return_tensors='pt', truncation=True, padding=True)
        input_ids = tokenized['input_ids']
        attention = tokenized["attention_mask"]
        
        sentiment = torch.tensor(np.asarray([text['sentiment'] for text in batch], dtype=np.int32), dtype=torch.float32)
        
        return {
            "input_ids" : input_ids,
            "attention_mask" : attention,
            "sentiment": sentiment
        }
                
        
    
    
parser = argparse.ArgumentParser()

parser.add_argument("--model", default='roberta-base', type=str, help="Default model")

parser.add_argument("--epochs", default=4, type=int, help="Number of epochs")
parser.add_argument("--batch_size", default=1, type=int, help="Batch size")

parser.add_argument("--data_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'imdb_dataset.csv')), type=str, help="Path to dataset")
parser.add_argument("--model_save", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "sentiment_model")), type=str, help="Where to save the model")

def validate(model: SentimentModel, dataset, tokenizer, device):
    acc = 0
    for datum in dataset:
        data = MovieReviewDataset.collate([datum], tokenizer)
        acc += model.validate(data['input_ids'].to(device), data['sentiment'])['acc']
        
    print(f"Model Accuracy: {acc/len(dataset)}")
    
    return acc


def main(args):
    # Device for validation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SentimentModel(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    train_data = MovieReviewDataset(args.data_path)
    
    collate = partial(MovieReviewDataset.collate, tokenizer=tokenizer)
    
    tokenizer.model_max_length = 512
    training_args = TrainingArguments(
        save_strategy='no',
        warmup_steps= 4 * len(train_data) / args.batch_size,
        logging_steps=100,
        weight_decay = 0.0,
        num_train_epochs = args.epochs,
        learning_rate = 5e-5,
        fp16 = True if torch.cuda.is_available() else False,
        ddp_backend = "nccl",
        lr_scheduler_type="cosine",
        logging_dir = './logs',
        output_dir = './results',
        per_device_train_batch_size = args.batch_size    
    )
    
    Trainer(model=model, args=training_args,train_dataset=train_data, data_collator=collate).train()
    
    model = model.to(device)
    
    val_acc = validate(model, train_data.val_data, tokenizer, device)
    model.model_acc = val_acc
    
    torch.save(model, args.model_save)
    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
