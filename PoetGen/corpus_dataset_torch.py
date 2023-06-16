import os
import json
import numpy as np
import torch
import re
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
        
        def data_text_line_gen(self):
            data = []
            for step,file in enumerate(self.gen_files()):
                if step % 500 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                for data_line in datum:
                    for part_line in data_line['body']:
                        for text_line in part_line:
                            tokenized = self._tokenizer.encode(text_line['text'], return_tensors="np", truncation=True)[0]
                            last_words =self._tokenizer.encode(re.sub(r"[,.?!-]", "", text_line['text']).split()[-1], return_tensors="np", truncation=True)[0]
                            data.append({"input_ids" : tokenized,
                                     "last": last_words,
                                     "num_vowels": [len(re.findall("a|e|i|o|u", text_line['text']))]})
            return data
                            
        def data_part_gen(self):
            data = []
            for step,file in enumerate(self.gen_files()):
                if step % 500 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                for data_line in datum:
                    part = []
                    num_vowels = []
                    last_words = []
                    for part_line in data_line['body']:
                        body = []
                        for text_line in part_line:
                            body.append(text_line['text'])
                            last_words.append(re.sub(r"[,.?!-]", "", text_line['text']).split()[-1])
                            num_vowels.append(len(re.findall("a|e|i|o|u", text_line['text'])))
                        part.append("\n".join(body))
                    tokenized = self._tokenizer.encode("\n".join(part), return_tensors="np", truncation=True)[0]
                    last_words =self._tokenizer.encode(" ".join(last_words), return_tensors="np", truncation=True)[0]
                    data.append({"input_ids" : tokenized,
                                     "last": last_words,
                                     "num_vowels": [sum(num_vowels)]})
            return data
                    
        def data_body_gen(self):
            data = []
            for step,file in enumerate(self.gen_files()):
                if step % 500 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                for data_line in datum:
                    for part_line in data_line['body']:
                        body = []
                        for text_line in part_line:
                            body.append(text_line['text'])
                        tokenized = self._tokenizer.encode("\n".join(body), return_tensors="np", truncation=True)[0]
                        last_words = self._tokenizer.encode(" ".join([re.sub(r"[,.?!-]", "", words).split()[-1] for words in body]), return_tensors="np", truncation=True)[0]
                        data.append({"input_ids" : tokenized,
                                     "last": last_words,
                                     "num_vowels": [sum([len(re.findall("a|e|i|o|u",words)) for words in body])]})
            return data
    
    def load_json_filenames(self):
        data_filenames = os.listdir(self.data_dir)
        data_by_files = []
        for filename in data_filenames:
            file_path = os.path.join(self.data_dir, filename)
            data_by_files.append(file_path)
        self.dataset = CorpusDatasetPytorch.Dataset(data_by_files, self.tokenizer)
        self.pytorch_dataset_part = self.dataset.data_part_gen()    
        self.pytorch_dataset_body = self.dataset.data_body_gen()
        self.pytorch_dataset_text = self.dataset.data_text_line_gen()
        
    @staticmethod
    def collate(batch):
        max_len = np.max([len(text['input_ids']) for text in batch])
        attention = np.zeros((len(batch), max_len), dtype=np.uint8)
        for pos, text in enumerate(batch):
            attention[pos,:len(text['input_ids'])] = 1
        padded_batch = np.asarray([np.append(text['input_ids'], [0] *(max_len - len(text['input_ids'])))  for text in batch], dtype=np.int32)
        
        max_len_words = np.max([len(text['last']) for text in batch])
        attention_words = np.zeros((len(batch), max_len_words), dtype=np.uint8)
        for pos, text in enumerate(batch):
            attention_words[pos,:len(text['last'])] = 1
        padded_batch_words = np.asarray([np.append(text['last'], [0] *(max_len_words - len(text['last'])))  for text in batch], dtype=np.int32)
        

        nums = np.asarray([text['num_vowels']for text in batch], dtype=np.int32)
        
        return {
            "input_ids": torch.tensor(padded_batch,  dtype=torch.int32),
            "attention": torch.tensor(attention, dtype=torch.bool),
            "last": torch.tensor(padded_batch_words, dtype=torch.int32),
            "attention_last" : torch.tensor(attention_words, dtype=torch.bool),
            "nums" :  torch.tensor(nums, dtype=torch.int32)
            }
    
    
    def __init__(self,tokenizer,  data_dir = "PoetGen\corpusCzechVerse-master\ccv", cache_dir='./'):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        if os.path.isfile(os.path.join(cache_dir, "part_poet_data.json")) and os.path.isfile(os.path.join(cache_dir, "body_poet_data.json")) and os.path.isfile(os.path.join(cache_dir, "text_poet_data.json")):
            self.pytorch_dataset_part = pickle.load( open( os.path.join(cache_dir, "part_poet_data.json"), 'rb'))
            self.pytorch_dataset_body = pickle.load( open( os.path.join(cache_dir, "body_poet_data.json"), 'rb'))
            self.pytorch_dataset_text = pickle.load( open( os.path.join(cache_dir, "text_poet_data.json"), 'rb'))
        else:
            self.load_json_filenames()
            pickle.dump(self.pytorch_dataset_part, open( os.path.join(cache_dir, "part_poet_data.json"), 'wb+'))
            pickle.dump(self.pytorch_dataset_body, open( os.path.join(cache_dir, "body_poet_data.json"), 'wb+'))
            pickle.dump(self.pytorch_dataset_text, open( os.path.join(cache_dir, "text_poet_data.json"), 'wb+'))
        