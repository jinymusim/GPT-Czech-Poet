import os
import torch
import datasets
import pickle
import json

class DialogDataset:
    
    def __init__(self, dataset, split,cache_dir='./') -> None:
        self.dataset = dataset
        self.split = split
        self.file_name = os.path.join(cache_dir, f"{split}_data.json")
        if os.path.isfile(self.file_name):
            data = pickle.load(open(self.file_name), 'rb')
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
        
        for act in dialogue['dialog']:
            current_act = {
                "utterance" : act,
                "context": context[:]
            }
            dialog_data.append(current_act)
            context.append(act)
            
        return dialog_data