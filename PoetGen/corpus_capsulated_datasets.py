import os
import json
import numpy as np
import torch
import re
import pickle
from poet_constants import rhyme_schemes, verse_ending
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset


class CorpusDatasetPytorch:
    
    class TextDataset(Dataset):
        
        def __init__(self, data_file_paths, tokenizer ,shuffle_bool:bool= True, seed:int= 42, prompt_length=True, prompt_ending=True):
            self._data_file_paths = data_file_paths
            self._tokenizer = tokenizer
            self.prompt_length = prompt_length
            self.prompt_ending = prompt_ending
            
            self.data = self.data_text_line_gen()
         
         
        def gen_files(self):
            for filename in self._data_file_paths:
                 yield open(filename, 'r')    
            
            
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
                            sub = re.sub(r'[\,\.\?\!–\„\“\’\;\:()\]\[\_\*\‘\”\'0-9\-\—\"]+', '', text_line['text'])
                            ending = sub.strip()[-2:].lower()
                            data.append({"input_ids" : tokenized,
                                     "num_vowels": [len(re.findall("a|e|i|o|u|á|é|í|ú|ů|ó|ě|y|ý", text_line['text']))],
                                     "verse_end": [1 if ending == verse_ending[i] or (verse_ending[i] == None and ending not in verse_ending) else 0 for i in range(len(verse_ending)) ]})
            return data
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            return self.data[index]
        
    class BodyDataset(Dataset):
        def __init__(self, data_file_paths, tokenizer , shuffle_bool:bool= True, seed:int= 42, prompt_length=True, prompt_ending=True, prompt_verse=True, verse_len=[4,6]):
            self._data_file_paths = data_file_paths
            self._tokenizer = tokenizer
            self.prompt_length = prompt_length
            self.prompt_ending = prompt_ending
            self.prompt_verse = prompt_verse
            self.verse_len = verse_len
            
            self.data = self.data_body_gen()       
        
        def gen_files(self):
            for filename in self._data_file_paths:
                 yield open(filename, 'r')
        
        @staticmethod
        def rhyme_sec(rhyme_ref, current_rhyme):
            rhyme_pos = ["A", "B", "C", "D", "E", "F", "G", "H"]
            return "X" if current_rhyme == None or current_rhyme < rhyme_ref or current_rhyme >= rhyme_ref + len(rhyme_pos) else rhyme_pos[current_rhyme - rhyme_ref]
                                                           
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
                        rhyme_sequence = -1
                        i = 0
                        for text_line in part_line:
                            if rhyme_sequence == -1 and text_line["rhyme"] != None:
                                rhyme_sequence = text_line["rhyme"]
                            rhyme += self.rhyme_sec(rhyme_sequence, text_line["rhyme"])
                            
                            num_str = f"{len(re.findall('a|e|i|o|u|á|é|í|ú|ů|ó|ě|y|ý', text_line['text']))} " if self.prompt_length else ""
                            sub = re.sub(r'[\,\.\?\!–\„\“\’\;\:()\]\[\_\*\‘\”\'0-9\-\—\"]+', '', text_line['text'])
                            verse_ending = f"{sub.strip()[-3:]} # " if self.prompt_ending else ""

                            body.append( num_str + verse_ending  + text_line['text'])
                            
                            i+=1
                            
                            if i in self.verse_len:
                                tokenized = self._tokenizer.encode(f"{rhyme}\n" +  "\n".join(body) + "\n\n", return_tensors="np", truncation=True)[0]
                                data.append({"input_ids" : tokenized,
                                     "rhyme":  [1 if rhyme == rhyme_schemes[i] or (rhyme_schemes[i] == None and rhyme not in rhyme_schemes )  else 0 for i in range(len(rhyme_schemes)) ]})
                                
                                if i == max(self.verse_len):
                                    body = []
                                    rhyme = ""
                                    rhyme_sequence = -1
                                    i=0
                        if len(body) > 0 and i not in self.verse_len:
                            tokenized = self._tokenizer.encode(f"{rhyme}\n" +  "\n".join(body) + "\n\n", return_tensors="np", truncation=True)[0]
                            data.append({"input_ids" : tokenized,
                                "rhyme": [1 if  rhyme == rhyme_schemes[i] or (rhyme_schemes[i] == None and rhyme not in rhyme_schemes )  else 0 for i in range(len(rhyme_schemes)) ]})
                                
                                                    
            return data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            return self.data[index]
    
    def load_json_filenames(self, prompt_length, prompt_ending, prompt_verse, verse_len=[4,6]):
        data_filenames = os.listdir(self.data_dir)
        data_by_files = []
        for filename in data_filenames:
            file_path = os.path.join(self.data_dir, filename)
            data_by_files.append(file_path)
        
        self.pytorch_dataset_body = CorpusDatasetPytorch.BodyDataset(data_by_files, self.tokenizer, prompt_ending=prompt_ending, 
                                                    prompt_length=prompt_length, prompt_verse=prompt_verse, verse_len=verse_len)
         
        
        self.pytorch_dataset_text = CorpusDatasetPytorch.TextDataset(data_by_files, self.tokenizer, prompt_ending=prompt_ending, 
                                                    prompt_length=prompt_length)
        
    @staticmethod
    def collate(batch):
        max_len = np.max([len(text['input_ids']) for text in batch])
        attention = np.zeros((len(batch), max_len), dtype=np.uint8)
        for pos, text in enumerate(batch):
            attention[pos,:len(text['input_ids'])] = 1
        padded_batch = np.asarray([np.append(text['input_ids'], [0] *(max_len - len(text['input_ids'])))  for text in batch], dtype=np.int32)
        padded_batch = torch.tensor(padded_batch,  dtype=torch.int32)
        
        nums = None
        if "num_vowels" in batch[0].keys():
            nums = torch.tensor(np.asarray([text['num_vowels'] for text in batch], dtype=np.int32), dtype=torch.int32)
            
        rhyme=None
        if "rhyme" in batch[0].keys():
            rhyme = torch.tensor(np.asarray([text["rhyme"] for text in batch], dtype=np.int32), dtype=torch.float32)
        
        verse_end = None
        if "verse_end" in batch[0].keys():
            verse_end = torch.tensor(np.asarray([text["verse_end"] for text in batch], dtype=np.int32), dtype=torch.float32)
        
        return {
            "input_ids": padded_batch,
            "labels": padded_batch.type(torch.LongTensor),
            "attention_mask": torch.tensor(attention, dtype=torch.bool),
            "nums" :  nums,
            "rhyme": rhyme,
            "verse_end" : verse_end
            }
    
    #TODO: Finish Rhyme Prompting
    def __init__(self,tokenizer,  data_dir = "PoetGen\corpusCzechVerse-master\ccv", cache_dir='./', prompt_length=True, prompt_ending=True, prompt_verse=True):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        if  os.path.isfile(os.path.join(cache_dir, "body_poet_data.json")) and os.path.isfile(os.path.join(cache_dir, "text_poet_data.json")):
            self.pytorch_dataset_body = pickle.load( open( os.path.join(cache_dir, "body_poet_data.json"), 'rb'))
            self.pytorch_dataset_text = pickle.load( open( os.path.join(cache_dir, "text_poet_data.json"), 'rb'))
        else:
            self.load_json_filenames(prompt_length, prompt_ending, prompt_verse)
            pickle.dump(self.pytorch_dataset_body, open( os.path.join(cache_dir, "body_poet_data.json"), 'wb+'))
            pickle.dump(self.pytorch_dataset_text, open( os.path.join(cache_dir, "text_poet_data.json"), 'wb+'))
        