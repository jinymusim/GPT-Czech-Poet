import os
from transformers import PreTrainedTokenizer
import numpy as np
import datasets
import pickle
import torch
from special_tokens import SPECIAL_TOKENS, END_OF_TEXT
class DialogDataset:
    
    def __init__(self, dataset, split: str,tokenizer: PreTrainedTokenizer, cache_dir='./') -> None:
        self.dataset = dataset
        self.split = split
        self.tokenizer = tokenizer
        self.file_name = os.path.join(cache_dir, f"{dataset}_{split}_data.json")
        if os.path.isfile(self.file_name):
            data = pickle.load(open(self.file_name, 'rb'))
        else:
            dataset = datasets.load_dataset(path=dataset, split=split, ignore_verifications=True, streaming=True)
            data = []
            for idx, dialogue in enumerate(dataset):
                if idx % 500 == 0:
                    print(f'Processing dialogue {idx + 1}')
                if self.dataset == "daily_dialog":
                    data.extend(self.parse_daily(dialogue))
                elif self.dataset == "multi_woz_v22":
                    data.extend(self.parse_multi(dialogue))
            pickle.dump(data, open(self.file_name, 'wb+'))
        self.data = data
        
        
    def parse_multi(self,dialogue):
        dialog_data = []
        context = []
        
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS})
        for speaker, utt, state in zip(dialogue['turns']['speaker'],dialogue['turns']['utterance'],dialogue['turns']['frames']):
            if speaker == 1:
                belief_state = state["state"]
                if len(belief_state) >= 1:
                    belief_state = belief_state[0]
                else:
                    belief_state = '\{\}'
                current_state = "<|belive|> " + str(belief_state) + " <|endoftext|> "
                current_act = {
                    "utterance" : self.tokenizer.encode(current_state + utt + " <|endoftext|>", return_tensors='np', truncation=True)[0],
                    "context": self.tokenizer.encode(" ".join(context) +  " <|endoftext|> "  + current_state + utt + " <|endoftext|>" + current_state, return_tensors='np', truncation=True)[0],
                }
                dialog_data.append(current_act)
                
                context.append("<|system|> " + utt)
            else:
                context.append("<|user|> " + utt)
        
        return dialog_data
                
    def parse_daily(self, dialogue):
        
        dialog_data = []
        context = []
        
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS})
                
        for part,act in enumerate(dialogue['dialog']):
            if part % 2 == 0:
                context.append("<|user|> " + act)
            else:              
                current_act = {
                    "utterance" : self.tokenizer.encode(act + " <|endoftext|>", return_tensors='np', truncation=True)[0],
                    "context": self.tokenizer.encode(" ".join(context) +  " <|endoftext|> <|system|> "  + act + " <|endoftext|>", return_tensors='np', truncation=True)[0],
                }
                dialog_data.append(current_act)
                
                context.append("<|system|> " + act)
            
            
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