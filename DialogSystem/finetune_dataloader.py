import os
from transformers import PreTrainedTokenizer
import numpy as np
import datasets
import pickle
import torch
from special_tokens import SPECIAL_TOKENS
class DialogDataset:
    
    def __init__(self, dataset, split: str,tokenizer: PreTrainedTokenizer, cache_dir='./') -> None:
        self.dataset = dataset
        self.split = split
        self.tokenizer = tokenizer
        self.file_name = os.path.join(cache_dir, f"{split}_data.json")
        if os.path.isfile(self.file_name):
            data = pickle.load(open(self.file_name, 'rb'))
        else:
            dataset = datasets.load_dataset(path=dataset, split=split, ignore_verifications=True, streaming=True)
            data = []
            for idx, dialogue in enumerate(dataset):
                if idx % 500 == 0:
                    print(f'Processing dialogue {idx + 1}')
                data.extend(self.parse(dialogue))
            pickle.dump(data, open(self.file_name, 'wb+'))
        self.data = data
                
    def parse(self, dialogue):
        
        dialog_data = []
        context = []
        
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS})

        
        
        for part,act in enumerate(dialogue['dialog']):
            if part % 2 == 0:
                context.append("<|user|> " + act)
            else:
                context.append("<|system|> " + act)
                current_act = {
                    "utterance" : self.tokenizer.encode(act, return_tensors='np', truncation=True)[0],
                    "context": self.tokenizer.encode(" ".join(context), return_tensors='np', truncation=True)[0],
                }
            dialog_data.append(current_act)
            
            
        return dialog_data
    
    @staticmethod
    def collate(batch):
        max_len = max([len(text['context']) for text in batch])
        attention = np.zeros((len(batch), max_len), dtype=np.uint8)
        labels = np.ones((len(batch), max_len), dtype=np.int32) * -100
        for pos, text in enumerate(batch):
            attention[pos,:len(text['context'])] = 1
            labels[pos,len(text['context']) - len(text['utterance']): len(text['context'])] = text['utterance']
        padded_array_context = np.asarray([np.append(text['context'], [0]*(max_len -len(text['context']))) for text in batch], dtype=np.int32)
        
        
        return {
                    "input": torch.tensor(padded_array_context,  dtype=torch.int32),
                    'label' : (torch.tensor(labels, dtype=torch.int32)).type(torch.LongTensor),
                    'attention_mask': torch.tensor(attention, dtype=torch.bool)
            }