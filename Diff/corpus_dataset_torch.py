import os
import json
import numpy as np
import torch
import pickle
from transformers import GPT2Tokenizer


class CorpusDatasetPytorch:
    
    class Dataset:
        
        def __init__(self, data_file_paths, tokenizer ,shuffle_bool:bool= True, seed:int= 42):
            self._data_file_paths = data_file_paths
            self._tokenizer = tokenizer
            self._size = len(self._data_file_paths)
            
        @property
        def size(self):
            return self._size
        
        
        def gen_files(self):
            for filename in self._data_file_paths:
                 yield open(filename, 'r')
    
        @property
        def raw_data_gen(self):
            for filename in self._data_file_paths:
                with open(filename, 'r') as file:
                    datum = json.load(file)
                yield datum
                
    def load_json_filenames(self):
        data_filenames = os.listdir(self.data_dir)
        data_by_files = []
        for filename in data_filenames:
            file_path = os.path.join(self.data_dir, filename)
            data_by_files.append(file_path)
        self.dataset = CorpusDatasetPytorch.Dataset(data_by_files, self.tokenizer)
        
    @staticmethod
    def collate(batch):
        max_len = np.max([len(text['input_ids']) for text in batch])
        attention = np.zeros((len(batch), max_len), dtype=np.uint8)
        for pos, text in enumerate(batch):
            attention[pos,:len(text['input_ids'])] = 1
        padded_batch = np.asarray([np.append(text['input_ids'], [0] *(max_len - len(text['input_ids'])))  for text in batch], dtype=np.int32)
        return {
            "input_ids": torch.tensor(padded_batch,  dtype=torch.int32),
            "attention": torch.tensor(attention, dtype=torch.bool)
            }
    
    
    def __init__(self,tokenizer,  data_dir = "PoetGen\corpusCzechVerse-master\ccv", cache_dir='./'):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.load_json_filenames()
        