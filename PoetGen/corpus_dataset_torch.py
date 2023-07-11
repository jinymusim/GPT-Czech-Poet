import os
import json
import numpy as np
import torch
import re
import pickle
import constants
from transformers import GPT2Tokenizer


class CorpusDatasetPytorch:
    
    class Dataset:
        
        def __init__(self, data_file_paths, tokenizer ,shuffle_bool:bool= True, seed:int= 42, prompt_length=True, prompt_ending=True, prompt_verse=True, verse_len=4):
            self._data_file_paths = data_file_paths
            self._tokenizer = tokenizer
            self._size = len(self._data_file_paths)
            self.prompt_length = prompt_length
            self.prompt_ending = prompt_ending
            self.prompt_verse = prompt_verse
            self.verse_len = verse_len
            
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
                            data.append({"input_ids" : tokenized,
                                     "num_vowels": [len(re.findall("a|e|i|o|u|y", text_line['text']))]})
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
                    
                    for part_line in data_line['body']:
                        
                        body = []
                        
                        for text_line in part_line:
                            
                            num_str = f"{len(re.findall('a|e|i|o|u|y', text_line['text']))} " if self.prompt_length else ""
                            verse_ending = f"{re.sub(r'[,.?!-„“’]+', '', text_line['text']).strip()[-3:]} # " if self.prompt_ending else ""
                        
                            body.append(num_str +  verse_ending + text_line['text'])
                            
                            num_vowels.append(len(re.findall("a|e|i|o|u|y", text_line['text'])))
                        part.append("\n".join(body))
                    tokenized = self._tokenizer.encode("\n".join(part), return_tensors="np", truncation=True)[0]
                    data.append({"input_ids" : tokenized,
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
                        rhyme= ""
                        i = 0
                        for text_line in part_line:
                            
                            rhyme += "A" if  text_line["rhyme"] == 1 else ("B" if text_line["rhyme"] == 2 else "C")
                            
                            num_str = f"{len(re.findall('a|e|i|o|u|y', text_line['text']))} " if self.prompt_length else ""
                            verse_ending = f"{re.sub(r'[,.?!-„“’]+', '', text_line['text']).strip()[-3:]} # " if self.prompt_ending else ""
                            
                            body.append( num_str + verse_ending  + text_line['text'])
                            
                            i+=1
                            
                            if i == self.verse_len:
                                break
                            
                        
                            
                        tokenized = self._tokenizer.encode(f"{rhyme}\n" +  "\n".join(body), return_tensors="np", truncation=True)[0]
                        data.append({"input_ids" : tokenized,                                    
                                     "num_vowels": [sum([len(re.findall("a|e|i|o|u|y",words)) for words in body])]})
            return data
    
    def load_json_filenames(self, prompt_length, prompt_ending, prompt_verse, verse_len=4):
        data_filenames = os.listdir(self.data_dir)
        data_by_files = []
        for filename in data_filenames:
            file_path = os.path.join(self.data_dir, filename)
            data_by_files.append(file_path)
        self.dataset = CorpusDatasetPytorch.Dataset(data_by_files, self.tokenizer, prompt_ending=prompt_ending, 
                                                    prompt_length=prompt_length, prompt_verse=prompt_verse, verse_len=verse_len)
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
        
        

        nums = np.asarray([text['num_vowels']for text in batch], dtype=np.int32)
        
        return {
            "input_ids": torch.tensor(padded_batch,  dtype=torch.int32),
            "attention": torch.tensor(attention, dtype=torch.bool),
            "nums" :  torch.tensor(nums, dtype=torch.int32)
            }
    
    #TODO: Finish Rhyme Prompting
    def __init__(self,tokenizer,  data_dir = "PoetGen\corpusCzechVerse-master\ccv", cache_dir='./', prompt_length=True, prompt_ending=True, prompt_verse=True):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        if os.path.isfile(os.path.join(cache_dir, "part_poet_data.json")) and os.path.isfile(os.path.join(cache_dir, "body_poet_data.json")) and os.path.isfile(os.path.join(cache_dir, "text_poet_data.json")):
            self.pytorch_dataset_part = pickle.load( open( os.path.join(cache_dir, "part_poet_data.json"), 'rb'))
            self.pytorch_dataset_body = pickle.load( open( os.path.join(cache_dir, "body_poet_data.json"), 'rb'))
            self.pytorch_dataset_text = pickle.load( open( os.path.join(cache_dir, "text_poet_data.json"), 'rb'))
        else:
            self.load_json_filenames(prompt_length, prompt_ending, prompt_verse)
            pickle.dump(self.pytorch_dataset_part, open( os.path.join(cache_dir, "part_poet_data.json"), 'wb+'))
            pickle.dump(self.pytorch_dataset_body, open( os.path.join(cache_dir, "body_poet_data.json"), 'wb+'))
            pickle.dump(self.pytorch_dataset_text, open( os.path.join(cache_dir, "text_poet_data.json"), 'wb+'))
        