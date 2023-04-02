import os
import sys
import json
import numpy as np
import torch
from transformers import GPT2Tokenizer
from torch.utils import data


class CorpusDatasetPytorch:
    
    class DatasetBody(data.IterableDataset):
        def __init__(self, generator):
            self.generator = generator

        def __iter__(self):
            return self.generator()
    
    class DatasetText(data.IterableDataset):
        def __init__(self, generator):
            self.generator = generator

        def __iter__(self):
            return self.generator()
        
    class DatasetPart(data.IterableDataset):
        def __init__(self, generator):
            self.generator = generator

        def __iter__(self):
            return self.generator()
    
    class Dataset:
        
        def __init__(self, data_file_paths, tokenizer ,shuffle_bool:bool= True, seed:int= 42):
            self._data_file_paths = data_file_paths
            self._tokenizer = tokenizer
            self._size = len(self._data_file_paths)
            
        @property
        def size(self):
            return self._size
    
        @property
        def raw_data_gen(self):
            for filename in self._data_file_paths:
                with open(filename, 'r') as file:
                    datum = json.load(file)
                yield datum
        
        def data_text_line_gen(self):
            for filename in self._data_file_paths:
                with open(filename, 'r') as file:
                    datum = json.load(file)
                for data_line in datum:
                    for part_line in data_line['body']:
                        for text_line in part_line:
                            tokenized = self._tokenizer.encode(text_line['text'], return_tensors="np", truncation=True)[0]
                            yield tokenized
                            
        def data_part_gen(self):
            for filename in self._data_file_paths:
                with open(filename, 'r') as file:
                    datum = json.load(file)
                for data_line in datum:
                    part = []
                    for part_line in data_line['body']:
                        body = []
                        for text_line in part_line:
                            body.append(text_line['text'])
                        part.append("\n".join(body))
                    tokenized = self._tokenizer.encode("\n".join(part), return_tensors="np", truncation=True)[0]
                    yield tokenized
                    
        def data_body_gen(self):
            for filename in self._data_file_paths:
                with open(filename, 'r') as file:
                    datum = json.load(file)
                for data_line in datum:
                    for part_line in data_line['body']:
                        body = []
                        for text_line in part_line:
                            body.append(text_line['text'])
                        tokenized = self._tokenizer.encode("\n".join(body), return_tensors="np", truncation=True)[0]
                        yield tokenized
    
    def load_json_filenames(self):
        data_filenames = os.listdir(self.data_dir)
        data_by_files = []
        for filename in data_filenames:
            file_path = os.path.join(self.data_dir, filename)
            data_by_files.append(file_path)
        self.dataset = CorpusDatasetPytorch.Dataset(data_by_files, self.tokenizer)
        self.pytorch_dataset_part = CorpusDatasetPytorch.DatasetPart(self.dataset.data_part_gen)
        self.pytorch_dataset_body = CorpusDatasetPytorch.DatasetBody(self.dataset.data_body_gen)
        self.pytorch_dataset_text = CorpusDatasetPytorch.DatasetText(self.dataset.data_text_line_gen)
        
    @staticmethod
    def collate(batch):
        max_len = np.max([len(text) for text in batch])
        padded_batch = np.asarray([np.append(text, [-100] *(max_len - len(text)))  for text in batch], dtype=np.int32)
        return torch.tensor(padded_batch,  dtype=torch.int32)
    
    def __init__(self,tokenizer,  data_dir = "GPT2\corpusCzechVerse-master\ccv"):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.load_json_filenames()