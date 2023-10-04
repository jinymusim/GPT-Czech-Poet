import os
import json
import numpy as np
import torch
import re

from utils.poet_utils import RHYME_SCHEMES, VERSE_ENDS, POET_YEARS_BUCKETS, METER_TYPES, SyllableMaker
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

class CorpusDatasetPytorch:
    
    class RawDataset:
        def __init__(self, data_file_paths):
            self._data_file_paths = data_file_paths
        
        def gen_files(self):
            for filename in self._data_file_paths:
                 yield open(filename, 'r') 
                 
        def get_text(self):
            for step,file in enumerate(self.gen_files()):
                if step % 500 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                for data_line in datum:
                    for part_line in data_line['body']:
                        for text_line in part_line:
                            yield text_line['text']
                            
        def get_part(self):
             for step,file in enumerate(self.gen_files()):
                if step % 500 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                for data_line in datum:
                    for part_line in data_line['body']:
                        part = []
                        for text_line in part_line:
                            part.append(text_line['text'])
                        yield "\n".join(part)
        
        def get_body(self):
             for step,file in enumerate(self.gen_files()):
                if step % 500 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                for data_line in datum:
                    body = []
                    for part_line in data_line['body']:
                        
                        for text_line in part_line:
                            body.append(text_line['text'])
                        body.append("\n")
                    yield "\n".join(body)
    
    class TextDataset(Dataset):
        
        def __init__(self, data_file_paths, prompt_length=True, prompt_ending=True):
            self._data_file_paths = data_file_paths
            self.prompt_length = prompt_length
            self.prompt_ending = prompt_ending
            
            self.data = []
         
         
        def gen_files(self):
            for filename in self._data_file_paths:
                 yield open(filename, 'r') 
                 
        @staticmethod
        def _vowels_and_endings(raw_text):
            vowels = len(SyllableMaker.syllabify(raw_text)) #INFO: Now counts the number of syllables
            sub = re.sub(r'([^\w\s]+|[0-9]+)', '', raw_text)
            ending = sub.strip()[-2:].lower()
            return vowels, ending
        
        @staticmethod
        def _ending_vector(end):
            verse_end_vector = np.zeros(len(VERSE_ENDS))
            if end in VERSE_ENDS:
                verse_end_vector[VERSE_ENDS.index(end)] = 1
            else:
                verse_end_vector[-1] = 1
            return verse_end_vector
        
        @staticmethod
        def _remove_most_nonchar(raw_text):
            text = re.sub(r'[–\„\“\’\;\:()\]\[\_\*\‘\”\'\-\—\"]+', "", raw_text)
            return text
                     
            
        def data_text_line_gen(self):
            for step,file in enumerate(self.gen_files()):
                if step % 500 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                for data_line in datum:
                    for part_line in data_line['body']:
                        for text_line in part_line:
                            scanned_text = self._remove_most_nonchar(text_line['text'])
                            
                            num_vowels, verse_end = self._vowels_and_endings(scanned_text)
                            self.data.append({
                                "input_ids" : scanned_text,
                                "nums": [num_vowels],
                                "verse_end": verse_end
                                     })
                            
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            return self.data[index]
        
    class BodyDataset(Dataset):
        def __init__(self, data_file_paths, end_token:str = "<|endoftext|>",
                     prompt_length=True, prompt_ending=True, prompt_verse=True, verse_len=[4,6]):
            self._data_file_paths = data_file_paths
            self.prompt_length = prompt_length
            self.prompt_ending = prompt_ending
            self.prompt_verse = prompt_verse
            self.verse_len = verse_len
            self.end_token = end_token
            
            self.data = []   
        
        def gen_files(self):
            for filename in self._data_file_paths:
                 yield open(filename, 'r')
        
        @staticmethod
        def rhyme_sec(rhyme_ref, current_rhyme):
            rhyme_pos = ["A", "B", "C", "D", "E", "F", "G", "H"]
            return "X" if current_rhyme == None or current_rhyme < rhyme_ref or current_rhyme >= rhyme_ref + len(rhyme_pos) else rhyme_pos[current_rhyme - rhyme_ref]
        
        @staticmethod
        def _rhyme_string(curr_rhyme_list):
            reference = None
            
            for num in curr_rhyme_list:
                if num != None:
                    reference = num
                    break
            rhyme_str = ""
            for num in curr_rhyme_list:
               rhyme_str += CorpusDatasetPytorch.BodyDataset.rhyme_sec(reference, num)
            
            return rhyme_str
        
        @staticmethod
        def _rhyme_vector(rhyme_str):
            rhyme_vector = np.zeros(len(RHYME_SCHEMES))
            if rhyme_str in RHYME_SCHEMES:
                rhyme_vector[RHYME_SCHEMES.index(rhyme_str)] = 1
            else:
                rhyme_vector[-1] = 1
            return rhyme_vector
            
        
        @staticmethod
        def _publish_year_vector(year_string):
            publish_year = None if not year_string.isdigit() else int(year_string)
            publish_vector = np.zeros(len(POET_YEARS_BUCKETS))
            if publish_year == None:
                publish_vector[-1] = 1
            else:
                publish_vector[np.argmin( abs(np.asarray(POET_YEARS_BUCKETS[:-1]) - publish_year))] = 1
            return publish_vector
        
        
        
        @staticmethod
        def _metre_vector(metre):
            meter_vector = np.zeros(len(METER_TYPES))
            if metre in METER_TYPES:          
                meter_vector[METER_TYPES.index(metre)] = 1
            else:
                meter_vector[-1] = 1
            return meter_vector
            
        
        def _construct_line(self, raw_text):
            num_str = f"{len(SyllableMaker.syllabify(raw_text))} " if self.prompt_length else ""
            sub = re.sub(r'([^\w\s]+|[0-9]+)', '', raw_text)
            verse_end = f"{sub.strip()[-3:]} # " if self.prompt_ending else ""
            return num_str + verse_end + raw_text
        
        @staticmethod
        def _remove_most_nonchar(raw_text):
            text = re.sub(r'[–\„\“\’\;\:()\]\[\_\*\‘\”\'\-\—\"]+', "", raw_text)
            return text
            
                                                           
        def data_body_gen(self):
            for step,file in enumerate(self.gen_files()):
                if step % 500 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                
                for data_line in datum:
                    publish_year = data_line["biblio"]["year"]
                    context = []

                    for part_line in data_line['body']:                                                        
                        body = []
                        rhyme= []
                        i = 0
                        for text_line in part_line:
                            
                            # In rare cases multiple, but from searching only 1 metre per line
                            metre = text_line["metre"][0]["type"]

                            rhyme.append(text_line["rhyme"])  
                            
                            scanned_text = self._remove_most_nonchar(text_line["text"])

                            body.append(self._construct_line(scanned_text))
                            
                            i+=1
                            
                            if i in self.verse_len:
                                rhyme_str = self._rhyme_string(rhyme)
                                
                                text = f"{rhyme_str} ## {publish_year} ## {metre}\n" + "\n".join(body) + "\n" + self.end_token
                                context_text= "\n".join(context) + self.end_token
                                self.data.append({
                                    "input_ids" : text,
                                    "context_ids" : context_text,
                                    "year": publish_year,
                                    "rhyme":  rhyme_str,
                                    "metre" : metre
                                     })
                                
                                if i == max(self.verse_len):
                                    context = body
                                    body = []
                                    rhyme = []
                                    i=0
                        if len(body) > 0 and i not in self.verse_len:
                            rhyme_str = self._rhyme_string(rhyme)
                            
                            text = f"{rhyme_str} ## {publish_year} ## {metre}\n" + "\n".join(body) + "\n" + self.end_token
                            context_text= "\n".join(context) + self.end_token
                            self.data.append({
                                    "input_ids" : text,
                                    "context_ids" : context_text,
                                    "year": publish_year,
                                    "rhyme":  rhyme_str,
                                    "metre" : metre
                                     })
                                
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            return self.data[index]
        
    def get_filenames(self):
        data_filenames = os.listdir(self.data_dir)
        data_by_files = []
        for filename in data_filenames:
            file_path = os.path.join(self.data_dir, filename)
            data_by_files.append(file_path)
        return data_by_files
        
    def load_raw_(self):
        filenames = self.get_filenames()
            
        self.raw_dataset = CorpusDatasetPytorch.RawDataset(filenames)
    
    def load_json_filenames(self, prompt_length, prompt_ending, prompt_verse, verse_len=[4,6]):
        filenames = self.get_filenames()
        
        self.pytorch_dataset_body = CorpusDatasetPytorch.BodyDataset(filenames, prompt_ending=prompt_ending, 
                                                    prompt_length=prompt_length, prompt_verse=prompt_verse, 
                                                    verse_len=verse_len)
        self.pytorch_dataset_body.data_body_gen()
         
        
        self.pytorch_dataset_text = CorpusDatasetPytorch.TextDataset(filenames, prompt_ending=prompt_ending, 
                                                    prompt_length=prompt_length)
        
        self.pytorch_dataset_text.data_text_line_gen()
        
    def create_empty(self):
        self.pytorch_dataset_body = CorpusDatasetPytorch.BodyDataset([])
        self.pytorch_dataset_text = CorpusDatasetPytorch.TextDataset([])
        
        
    @staticmethod
    def collate(batch, tokenizer: PreTrainedTokenizerBase ,max_len = 1024, max_context = 1024 ,mask_rate = 0.0):
        
        tokenizer.model_max_length = max_len
        tokenized = tokenizer([text['input_ids'] for text in batch],return_tensors='pt', truncation=True, padding=True)
        input_ids = tokenized['input_ids']
        attention = tokenized["attention_mask"]
        
        # Input Masking
        mask = torch.rand(input_ids.shape) < 1 - mask_rate
        input_ids = input_ids * mask.int()
        
        nums = None
        if "nums" in batch[0].keys():
            nums = torch.tensor(np.asarray([text['nums'] for text in batch], dtype=np.int32), dtype=torch.int32)
            
        rhyme=None
        if "rhyme" in batch[0].keys():
            rhyme = torch.tensor(np.asarray([CorpusDatasetPytorch.BodyDataset._rhyme_vector(text["rhyme"]) for text in batch], dtype=np.int32), dtype=torch.float32)
        
        verse_end = None
        if "verse_end" in batch[0].keys():       
            verse_end = torch.tensor(np.asarray([CorpusDatasetPytorch.TextDataset._ending_vector(text["verse_end"]) for text in batch], dtype=np.int32), dtype=torch.float32)
        
        year = None
        if "year" in batch[0].keys():      
            year = torch.tensor(np.asarray([CorpusDatasetPytorch.BodyDataset._publish_year_vector(text["year"]) for text in batch], dtype=np.int32), dtype=torch.float32)
            
        metre = None
        if "metre" in batch[0].keys():       
            metre = torch.tensor(np.asarray([CorpusDatasetPytorch.BodyDataset._metre_vector(text["metre"]) for text in batch], dtype=np.int32), dtype=torch.float32)
        
        context_ids = None
        context_attention_mask = None
        if "context_ids" in batch[0].keys():
            tokenizer.model_max_length = max_context
            tokenized_context = tokenizer([text['context_ids'] for text in batch],return_tensors='pt', truncation=True, padding=True)
            context_ids = tokenized_context['input_ids']
            context_attention_mask = tokenized_context['attention_mask'] 
        
        return {
            "input_ids": input_ids,
            "labels": input_ids.type(torch.LongTensor),
            "attention_mask": attention,
            "context_ids" : context_ids,
            "context_attention_mask" : context_attention_mask,
            "nums" :  nums,
            "rhyme": rhyme,
            "verse_end" : verse_end,
            "year": year,
            "metre" : metre}
        
    @staticmethod
    def collate_rhyme(batch, tokenizer: PreTrainedTokenizerBase, max_len:int = 32):
        tokenizer.model_max_length = max_len
        tokenized = tokenizer(
            ["\n".join(
                # Last line always contains just eof token, first line is always param line
                [CorpusDatasetPytorch.BodyDataset._remove_most_nonchar(line)[-6:] for line in text['input_ids'].splitlines()[1:-1]]
                ) for text in batch],
            return_tensors='pt', truncation=True, padding=True
            )
        
        input_ids = tokenized['input_ids']
        attention = tokenized["attention_mask"]
        
        rhyme=None
        if "rhyme" in batch[0].keys():
            rhyme = torch.tensor(np.asarray([CorpusDatasetPytorch.BodyDataset._rhyme_vector(text["rhyme"]) for text in batch], dtype=np.int32), dtype=torch.float32)
        
        return  {
            "input_ids": input_ids,
            "attention_mask": attention,
            "rhyme": rhyme,
            "metre": None}     
        
    def __init__(self, data_dir = "PoetGen\corpusCzechVerse-master\ccv", cache_dir='./', 
                 prompt_length=True, prompt_ending=True, prompt_verse=True, verse_len=[4,6]):
        self.data_dir = data_dir
        if  os.path.isfile(os.path.join(cache_dir, "body_poet_data.json")) and os.path.isfile(os.path.join(cache_dir, "text_poet_data.json")):
            self.create_empty()
            self.pytorch_dataset_body.data =list(json.load( open( os.path.join(cache_dir, "body_poet_data.json"), 'r')))
            self.pytorch_dataset_text.data =list(json.load( open( os.path.join(cache_dir, "text_poet_data.json"), 'r')))
        else:
            self.load_json_filenames(prompt_length, prompt_ending, prompt_verse, verse_len=verse_len)
            json.dump(self.pytorch_dataset_body.data, open( os.path.join(cache_dir, "body_poet_data.json"), 'w+'), indent = 6)
            json.dump(self.pytorch_dataset_text.data, open( os.path.join(cache_dir, "text_poet_data.json"), 'w+'), indent = 6)
            
        self.load_raw_()
        
        
        