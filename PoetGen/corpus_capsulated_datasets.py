import os
import json
import numpy as np
import torch
import re
import pickle

from poet_utils import RHYME_SCHEMES, VERSE_ENDS, POET_YEARS_BUCKETS, METER_TYPES
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
                 
        @staticmethod
        def _vowels_and_endings_vector(raw_text):
            vowels = len(re.findall("a|e|i|o|u|á|é|í|ú|ů|ó|ě|y|ý", raw_text.lower()))
            sub = re.sub(r'([^\w\s]+|[0-9]+)', '', raw_text)
            ending = sub.strip()[-2:].lower()
            verse_end_vector = np.zeros(len(VERSE_ENDS))
            if ending in VERSE_ENDS:
                verse_end_vector[VERSE_ENDS.index(ending)] = 1
            else:
                verse_end_vector[-1] = 1
            return vowels, verse_end_vector
        
        @staticmethod
        def _remove_most_nonchar(raw_text):
            text = re.sub(r'[–\„\“\’\;\:()\]\[\_\*\‘\”\'\-\—\"]+', "", raw_text)
            return text
                     
            
        def data_text_line_gen(self):
            data = []
            for step,file in enumerate(self.gen_files()):
                if step % 500 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                for data_line in datum:
                    for part_line in data_line['body']:
                        for text_line in part_line:
                            scanned_text = self._remove_most_nonchar(text_line['text'])
                            tokenized = self._tokenizer.encode(scanned_text, return_tensors="np", truncation=True)[0]
                            num_vowels, verse_end_vector = self._vowels_and_endings_vector(scanned_text)
                            data.append({
                                "input_ids" : tokenized,
                                "num_vowels": [num_vowels],
                                "verse_end": verse_end_vector
                                     })
            return data
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            return self.data[index]
        
    class BodyDataset(Dataset):
        def __init__(self, data_file_paths, tokenizer , shuffle_bool:bool= True, seed:int= 42, 
                     prompt_length=True, prompt_ending=True, prompt_verse=True, verse_len=[4,6],
                     context_size:int = 2048):
            self._data_file_paths = data_file_paths
            self._tokenizer = tokenizer
            self.prompt_length = prompt_length
            self.prompt_ending = prompt_ending
            self.prompt_verse = prompt_verse
            self.verse_len = verse_len
            self.context_size = context_size
            
            self.data = self.data_body_gen()       
        
        def gen_files(self):
            for filename in self._data_file_paths:
                 yield open(filename, 'r')
        
        @staticmethod
        def rhyme_sec(rhyme_ref, current_rhyme):
            rhyme_pos = ["A", "B", "C", "D", "E", "F", "G", "H"]
            return "X" if current_rhyme == None or current_rhyme < rhyme_ref or current_rhyme >= rhyme_ref + len(rhyme_pos) else rhyme_pos[current_rhyme - rhyme_ref]
        
        @staticmethod
        def _rhyme_string_and_vector(curr_rhyme_list):
            reference = None
            
            for num in curr_rhyme_list:
                if num != None:
                    reference = num
                    break
            rhyme_str = ""
            for num in curr_rhyme_list:
               rhyme_str += CorpusDatasetPytorch.BodyDataset.rhyme_sec(reference, num)
            rhyme_vector = np.zeros(len(RHYME_SCHEMES))
            if rhyme_str in RHYME_SCHEMES:
                rhyme_vector[RHYME_SCHEMES.index(rhyme_str)] = 1
            else:
                rhyme_vector[-1] = 1
            return rhyme_str, rhyme_vector
        
        @staticmethod
        def _publish_year_and_vector(year_string):
            publish_year = None if not year_string.isdigit() else int(year_string)
            publish_vector = np.zeros(len(POET_YEARS_BUCKETS))
            if publish_year == None:
                publish_vector[-1] = 1
            else:
                publish_vector[np.argmin( abs(np.asarray(POET_YEARS_BUCKETS[:-1]) - publish_year))] = 1
            return publish_year, publish_vector
        
        @staticmethod
        def _metre_and_vector(meter_string):
            meter_str = meter_string
            meter_vector = np.zeros(len(METER_TYPES))
            if meter_str in METER_TYPES:          
                meter_vector[METER_TYPES.index(meter_str)] = 1
            else:
                meter_vector[-1] = 1
            return meter_str, meter_vector
        
        def _construct_line(self, raw_text):
            num_str = f"{len(re.findall('a|e|i|o|u|á|é|í|ú|ů|ó|ě|y|ý', raw_text.lower()))} " if self.prompt_length else ""
            sub = re.sub(r'([^\w\s]+|[0-9]+)', '', raw_text)
            verse_end = f"{sub.strip()[-3:]} # " if self.prompt_ending else ""
            return num_str + verse_end + raw_text
        
        @staticmethod
        def _remove_most_nonchar(raw_text):
            text = re.sub(r'[–\„\“\’\;\:()\]\[\_\*\‘\”\'\-\—\"]+', "", raw_text)
            return text
            
                                                           
        def data_body_gen(self):
            data = []
            for step,file in enumerate(self.gen_files()):
                if step % 500 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                
                for data_line in datum:
                    publish_year, publish_vector = self._publish_year_and_vector(data_line["biblio"]["year"])
                    context = []

                    for part_line in data_line['body']:                                                        
                        body = []
                        rhyme= []
                        i = 0
                        for text_line in part_line:
                            
                            # In rare cases multiple, but from searching only 1 metre per line
                            metre, metre_vector = self._metre_and_vector(text_line["metre"][0]["type"])

                            rhyme.append(text_line["rhyme"])  
                            
                            scanned_text = self._remove_most_nonchar(text_line["text"])

                            body.append(self._construct_line(scanned_text))
                            
                            i+=1
                            
                            if i in self.verse_len:
                                rhyme_str, rhyme_vector = self._rhyme_string_and_vector(rhyme)
                                
                                tokenized = self._tokenizer.encode(f"{rhyme_str} ## {publish_year} ## {metre}\n" + "\n".join(body) + "\n" + self._tokenizer.eos_token , 
                                                                   return_tensors="np", truncation=True)[0]
                                context_tokenized = self._tokenizer.encode("\n".join(context) + self._tokenizer.eos_token, 
                                                                   return_tensors="np", truncation=False)[0][:self.context_size]
                                data.append({
                                    "input_ids" : tokenized,
                                    "context_ids" : context_tokenized,
                                    "year": publish_vector,
                                    "rhyme":  rhyme_vector,
                                    "metre" : metre_vector
                                     })
                                
                                if i == max(self.verse_len):
                                    context = body
                                    body = []
                                    rhyme = []
                                    i=0
                        if len(body) > 0 and i not in self.verse_len:
                            rhyme_str, rhyme_vector = self._rhyme_string_and_vector(rhyme)
                            
                            tokenized = self._tokenizer.encode(f"{rhyme_str} ## {publish_year} ## {metre}\n" + "\n".join(body) + "\n" + self._tokenizer.eos_token, 
                                                               return_tensors="np", truncation=True)[0]
                            context_tokenized = self._tokenizer.encode("\n".join(context) + self._tokenizer.eos_token, 
                                                                   return_tensors="np", truncation=False)[0][:self.context_size]
                            data.append({
                                "input_ids" : tokenized,
                                "context_ids" : context_tokenized,
                                "year": publish_vector,
                                "rhyme":  rhyme_vector,
                                "metre" : metre_vector
                                })
                                
                                                    
            return data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            return self.data[index]
    
    def load_json_filenames(self, prompt_length, prompt_ending, prompt_verse, verse_len=[4,6], context_len=2048):
        data_filenames = os.listdir(self.data_dir)
        data_by_files = []
        for filename in data_filenames:
            file_path = os.path.join(self.data_dir, filename)
            data_by_files.append(file_path)
        
        self.pytorch_dataset_body = CorpusDatasetPytorch.BodyDataset(data_by_files, self.tokenizer, prompt_ending=prompt_ending, 
                                                    prompt_length=prompt_length, prompt_verse=prompt_verse, verse_len=verse_len, context_size=context_len)
         
        
        self.pytorch_dataset_text = CorpusDatasetPytorch.TextDataset(data_by_files, self.tokenizer, prompt_ending=prompt_ending, 
                                                    prompt_length=prompt_length)
        
    @staticmethod
    def collate(batch, mask_rate = 0.0):
        max_len = np.max([len(text['input_ids']) for text in batch])
        attention = np.zeros((len(batch), max_len), dtype=np.uint8)
        for pos, text in enumerate(batch):
            attention[pos,:len(text['input_ids'])] = 1
        padded_batch = np.asarray([np.append(text['input_ids'], [0] *(max_len - len(text['input_ids'])))  for text in batch], dtype=np.int32)
        padded_batch = torch.tensor(padded_batch,  dtype=torch.int32)
        # Input Masking
        mask = torch.rand(padded_batch.shape) < 1 - mask_rate
        padded_batch = padded_batch * mask.int()
        
        nums = None
        if "num_vowels" in batch[0].keys():
            nums = torch.tensor(np.asarray([text['num_vowels'] for text in batch], dtype=np.int32), dtype=torch.int32)
            
        rhyme=None
        if "rhyme" in batch[0].keys():
            rhyme = torch.tensor(np.asarray([text["rhyme"] for text in batch], dtype=np.int32), dtype=torch.float32)
        
        verse_end = None
        if "verse_end" in batch[0].keys():
            verse_end = torch.tensor(np.asarray([text["verse_end"] for text in batch], dtype=np.int32), dtype=torch.float32)
        
        year = None
        if "year" in batch[0].keys():
            year = torch.tensor(np.asarray([text["year"] for text in batch], dtype=np.int32), dtype=torch.float32)
            
        metre = None
        if "metre" in batch[0].keys():
            metre = torch.tensor(np.asarray([text["metre"] for text in batch], dtype=np.int32), dtype=torch.float32)
        
        context_ids = None
        context_attention_mask = None
        if "context_ids" in batch[0].keys():
            max_len = np.max([len(text['context_ids']) for text in batch])
            context_attention_mask = np.zeros((len(batch), max_len), dtype=np.uint8)
            for pos, text in enumerate(batch):
                context_attention_mask[pos,:len(text['context_ids'])] = 1
            context_ids = np.asarray([np.append(text['context_ids'], [0] *(max_len - len(text['context_ids'])))  for text in batch], dtype=np.int32)
            context_ids = torch.tensor(context_ids,  dtype=torch.int32)
            context_attention_mask =  torch.tensor(context_attention_mask,dtype=torch.bool)          
        
        return {
            "input_ids": padded_batch,
            "labels": padded_batch.type(torch.LongTensor),
            "attention_mask": torch.tensor(attention, dtype=torch.bool),
            "context_ids" : context_ids,
            "context_attention_mask" : context_attention_mask,
            "nums" :  nums,
            "rhyme": rhyme,
            "verse_end" : verse_end,
            "year": year,
            "metre" : metre
            }
        
    def __init__(self,tokenizer,  data_dir = "PoetGen\corpusCzechVerse-master\ccv", cache_dir='./', prompt_length=True, prompt_ending=True, prompt_verse=True, verse_len=[4,6] ,context_len=2048):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        if  os.path.isfile(os.path.join(cache_dir, "body_poet_data.json")) and os.path.isfile(os.path.join(cache_dir, "text_poet_data.json")):
            self.pytorch_dataset_body = pickle.load( open( os.path.join(cache_dir, "body_poet_data.json"), 'rb'))
            self.pytorch_dataset_text = pickle.load( open( os.path.join(cache_dir, "text_poet_data.json"), 'rb'))
        else:
            self.load_json_filenames(prompt_length, prompt_ending, prompt_verse, verse_len=verse_len, context_len=context_len)
            pickle.dump(self.pytorch_dataset_body, open( os.path.join(cache_dir, "body_poet_data.json"), 'wb+'))
            pickle.dump(self.pytorch_dataset_text, open( os.path.join(cache_dir, "text_poet_data.json"), 'wb+'))
        