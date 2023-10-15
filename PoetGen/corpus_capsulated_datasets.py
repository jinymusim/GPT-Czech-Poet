import os
import json
import numpy as np
import torch
import re

from utils.poet_utils import RHYME_SCHEMES, VERSE_ENDS, POET_YEARS_BUCKETS, METER_TYPES, VALID_CHARS, METER_TRANSLATE, SyllableMaker, TextAnalysis, TextManipulation
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
#TODO: Maybe replace year of book being written for year Author was born
class CorpusDatasetPytorch:
    
    class RawDataset:
        def __init__(self, data_file_paths, lower_case:bool = True):
            self._data_file_paths = data_file_paths
            self.lower_case = lower_case
        
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
                            yield text_line['text'].lower() if self.lower_case else text_line['text']
                            
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
                        yield "\n".join(part).lower() if self.lower_case else "\n".join(part)
        
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
                    yield "\n".join(body).lower() if self.lower_case else "\n".join(body)
    
    class TextDataset(Dataset):
        
        def __init__(self, data_file_paths, prompt_length=True, prompt_ending=True, lower_case=True, val_data_rate: float = 0.1):
            self._data_file_paths = data_file_paths
            self.prompt_length = prompt_length
            self.prompt_ending = prompt_ending
            self.lower_case = lower_case
            
            self.val_data_rate = val_data_rate
            
            self.data = []
            self.validation_data = []
         
         
        def gen_files(self):
            for filename in self._data_file_paths:
                 yield open(filename, 'r') 
                 
        @staticmethod
        def _vowels_and_endings(raw_text):
            syllabs = SyllableMaker.syllabify(raw_text)
            vowels = len(syllabs) #INFO: Now counts the number of syllables
            ending = syllabs[-1]
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
        def _syllable_line(raw_text):
            ending = raw_text[-1] if raw_text[-1] in [',','.','!','?'] else ''
            return " ".join(SyllableMaker.syllabify(raw_text)) + ending
                     
            
        def data_text_line_gen(self):
            for step,file in enumerate(self.gen_files()):
                if step % 500 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                for data_line in datum:
                    for part_line in data_line['body']:
                        for text_line in part_line:
                            scanned_text = TextManipulation._remove_most_nonchar(text_line['text'], self.lower_case)
                            
                            syllable_line = self._syllable_line(scanned_text)
                            num_vowels, verse_end = self._vowels_and_endings(scanned_text)
                            if np.random.rand() > self.val_data_rate: 
                                self.data.append({
                                "input_ids" : [scanned_text,syllable_line],
                                "nums": [num_vowels],
                                "verse_end": verse_end
                                     })
                            else:
                                self.validation_data.append({
                                "input_ids" : [scanned_text,syllable_line],
                                "nums": [num_vowels],
                                "verse_end": verse_end
                                     })
                            
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, index):
            return self.data[index]
        
    class BodyDataset(Dataset):
        def __init__(self, data_file_paths,
                     prompt_length=True, prompt_ending=True, prompt_verse=True, verse_len=[4,6], lower_case=True, val_data_rate: float = 0.1):
            self._data_file_paths = data_file_paths
            self.prompt_length = prompt_length
            self.prompt_ending = prompt_ending
            self.prompt_verse = prompt_verse
            self.verse_len = verse_len
            self.lower_case = lower_case
            self.val_data_rate = val_data_rate
            
            self.data = []
            self.validation_data = []
        
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
        def _publish_year_vector(year_string):
            publish_year = None if not year_string.isdigit() else int(year_string)
            publish_vector = np.zeros(len(POET_YEARS_BUCKETS))
            if publish_year == None:
                publish_vector[-1] = 1
            else:
                publish_vector[np.argmin( abs(np.asarray(POET_YEARS_BUCKETS[:-1]) - publish_year))] = 1
            return publish_vector     
                     
        
        def _construct_line(self, raw_text):
            num_str = f"{len(SyllableMaker.syllabify(raw_text))} " if self.prompt_length else ""
            sub = re.sub(r'([^\w\s]+|[0-9]+)', '', raw_text)
            verse_end = f"{sub.strip()[-3:]} # " if self.prompt_ending else ""
            return num_str + verse_end + raw_text
        
        def _construct_syllable_line(self, raw_text):
            ending = raw_text[-1] if raw_text[-1] in [',','.','!','?'] else ''
            syllables = SyllableMaker.syllabify(raw_text)
            num_str = f"{len(syllables)} " if self.prompt_length else ""
            verse_end = f"{syllables[-1]} # " if self.prompt_ending else ""
            return num_str + verse_end + " ".join(syllables) + ending
            
            
                                                           
        def data_body_gen(self):
            for step,file in enumerate(self.gen_files()):
                if step % 500 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                
                for data_line in datum:
                    publish_year = TextManipulation._year_bucketor(data_line["biblio"]["year"])
                    context = []

                    for part_line in data_line['body']:                                                        
                        body = []
                        body_syllabs = []
                        rhyme= []
                        i = 0
                        for text_line in part_line:
                            
                            # In rare cases multiple, but from searching only 1 metre per line
                            metre = METER_TRANSLATE.get(text_line["metre"][0]["type"], "N")

                            rhyme.append(text_line["rhyme"])  
                            
                            scanned_text = TextManipulation._remove_most_nonchar(text_line["text"], self.lower_case)

                            body.append(self._construct_line(scanned_text))
                            body_syllabs.append(self._construct_syllable_line(scanned_text))
                            
                            i+=1
                            
                            if i in self.verse_len:
                                rhyme_str = self._rhyme_string(rhyme)
                                
                                text = f"{rhyme_str} # {publish_year} # {metre}\n" + "\n".join(body) + "\n"
                                syllable_text = f"{rhyme_str} # {publish_year} # {metre}\n" + "\n".join(body_syllabs) + "\n"
                                context_text= "\n".join(context)
                                if np.random.rand() > self.val_data_rate:
                                    self.data.append({
                                    "input_ids" : [text,syllable_text],
                                    "context_ids" : context_text,
                                    "year": publish_year,
                                    "rhyme":  rhyme_str,
                                    "metre" : metre
                                     })
                                else:
                                    self.validation_data.append({
                                    "input_ids" : [text,syllable_text],
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
                        # The resulting schema throws the model often off
                        
                        # if len(body) > 0 and i not in self.verse_len:
                        #     rhyme_str = self._rhyme_string(rhyme)
                        #     
                        #     text = f"{rhyme_str} ## {publish_year} ## {metre}\n" + "\n".join(body) + "\n"
                        #     context_text= "\n".join(context) 
                        #     self.data.append({
                        #             "input_ids" : text,
                        #             "context_ids" : context_text,
                        #             "year": publish_year,
                        #             "rhyme":  rhyme_str,
                        #             "metre" : metre
                        #              })
                                
        
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
            
        self.raw_dataset = CorpusDatasetPytorch.RawDataset(filenames, self.lower_case)
    
    def load_json_filenames(self, prompt_length, prompt_ending, prompt_verse, verse_len=[4,6], val_data_rate=0.1):
        filenames = self.get_filenames()
        
        self.pytorch_dataset_body = CorpusDatasetPytorch.BodyDataset(filenames, prompt_ending=prompt_ending, 
                                                    prompt_length=prompt_length, prompt_verse=prompt_verse, 
                                                    verse_len=verse_len, lower_case=self.lower_case, 
                                                    val_data_rate=val_data_rate)
        self.pytorch_dataset_body.data_body_gen()
         
        
        self.pytorch_dataset_text = CorpusDatasetPytorch.TextDataset(filenames, prompt_ending=prompt_ending, 
                                                    prompt_length=prompt_length, lower_case=self.lower_case,
                                                    val_data_rate=val_data_rate)
        
        self.pytorch_dataset_text.data_text_line_gen()
        
    def create_empty(self):
        self.pytorch_dataset_body = CorpusDatasetPytorch.BodyDataset([])
        self.pytorch_dataset_text = CorpusDatasetPytorch.TextDataset([])
        
        
    @staticmethod
    def collate(batch, tokenizer: PreTrainedTokenizerBase ,max_len = 1024, max_context = 1024 ,mask_rate = 0.0, syllables: bool = False):
        index = 1 if syllables else 0
        
        tokenizer.model_max_length = max_len
        tokenized = tokenizer([text['input_ids'][index] + tokenizer.eos_token for text in batch],return_tensors='pt', truncation=True, padding=True)
        input_ids = tokenized['input_ids']
        attention = tokenized["attention_mask"]
        
        # Input Masking
        mask = torch.rand(input_ids.shape) < 1 - mask_rate
        input_ids = input_ids * mask.int()
        
        nums = None
        if "nums" in batch[0].keys():
            nums = torch.tensor(np.asarray([text['nums'] for text in batch], dtype=np.int32), dtype=torch.float32)
            
        rhyme=None
        if "rhyme" in batch[0].keys():
            rhyme = torch.tensor(np.asarray([TextAnalysis._rhyme_vector(text["rhyme"]) for text in batch], dtype=np.int32), dtype=torch.float32)
        
        verse_end = None
        if "verse_end" in batch[0].keys():       
            verse_end = torch.tensor(np.asarray([CorpusDatasetPytorch.TextDataset._ending_vector(text["verse_end"]) for text in batch], dtype=np.int32), dtype=torch.float32)
        
        year = None
        if "year" in batch[0].keys():      
            year = torch.tensor(np.asarray([CorpusDatasetPytorch.BodyDataset._publish_year_vector(text["year"]) for text in batch], dtype=np.int32), dtype=torch.float32)
            
        metre = None
        if "metre" in batch[0].keys():       
            metre = torch.tensor(np.asarray([TextAnalysis._metre_vector(text["metre"]) for text in batch], dtype=np.int32), dtype=torch.float32)
        
        context_ids = None
        context_attention_mask = None
        if "context_ids" in batch[0].keys():
            tokenizer.model_max_length = max_context
            tokenized_context = tokenizer([text['context_ids'] + tokenizer.eos_token  for text in batch],return_tensors='pt', truncation=True, padding=True)
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
    def collate_rhyme(batch, max_len:int = 36, max_verse_len:int = 6):
        chars_per_line = max_len//max_verse_len
        
        input_ids = torch.zeros((len(batch), max_len * len(VALID_CHARS)))
        for i, text in enumerate(batch):
            one_input = text['input_ids'][0]
            # First Line is parameter line
            lines = [re.sub(r'[^aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž]+', '', line.lower()) for line in one_input.splitlines()[1:max_verse_len + 1]]
            j = 0
            while j<len(lines) and j<max_verse_len:
                k = 0
                while k<len(lines[j]) and k<chars_per_line:    
                    input_ids[i,j * chars_per_line * len(VALID_CHARS) +  k * len(VALID_CHARS) + VALID_CHARS.index(lines[j][k])] = 1
                    k+=1
                while k<chars_per_line:
                    input_ids[i,j * chars_per_line * len(VALID_CHARS) +  k * len(VALID_CHARS)] = 1
                    k+=1
                j +=1
            while j<max_verse_len:
                input_ids[i,j*len(VALID_CHARS)] = 1
                k=0
                while k<chars_per_line:
                    input_ids[i,j * chars_per_line * len(VALID_CHARS) +  k * len(VALID_CHARS)] = 1
                    k+=1
                j +=1
                           
        attention = torch.ones_like(input_ids)
        
        rhyme=None
        if "rhyme" in batch[0].keys():
            rhyme = torch.tensor(np.asarray([TextAnalysis._rhyme_vector(text["rhyme"]) for text in batch], dtype=np.int32), dtype=torch.float32)
        
        return  {
            "input_ids": input_ids,
            "attention_mask": attention,
            "rhyme": rhyme,
            "metre": None}     
        
    def __init__(self, data_dir = "PoetGen\corpusCzechVerse-master\ccv", cache_dir='./', 
                 prompt_length=True, prompt_ending=True, prompt_verse=True, verse_len=[4,6], lower_case=True, val_data_rate=0.1):
        self.lower_case = lower_case
        self.data_dir = data_dir
        if  os.path.isfile(os.path.join(cache_dir, "body_poet_data.json")) and os.path.isfile(os.path.join(cache_dir, "text_poet_data.json")) \
            and os.path.isfile(os.path.join(cache_dir, "val_body_poet_data.json")) and os.path.isfile(os.path.join(cache_dir, "val_text_poet_data.json")):
            self.create_empty()
            self.pytorch_dataset_body.data =list(json.load( open( os.path.join(cache_dir, "body_poet_data.json"), 'r')))
            self.pytorch_dataset_text.data =list(json.load( open( os.path.join(cache_dir, "text_poet_data.json"), 'r')))
            self.pytorch_dataset_body.validation_data = list(json.load( open( os.path.join(cache_dir, "val_body_poet_data.json"), 'r')))
            self.pytorch_dataset_text.data = list(json.load( open( os.path.join(cache_dir, "val_text_poet_data.json"), 'r')))
        else:
            self.load_json_filenames(prompt_length, prompt_ending, prompt_verse, verse_len=verse_len, val_data_rate=val_data_rate)
            json.dump(self.pytorch_dataset_body.data, open( os.path.join(cache_dir, "body_poet_data.json"), 'w+'), indent = 6)
            json.dump(self.pytorch_dataset_text.data, open( os.path.join(cache_dir, "text_poet_data.json"), 'w+'), indent = 6)
            json.dump(self.pytorch_dataset_body.validation_data, open( os.path.join(cache_dir, "val_body_poet_data.json"), 'w+'), indent = 6)
            json.dump(self.pytorch_dataset_text.validation_data, open( os.path.join(cache_dir, "val_text_poet_data.json"), 'w+'), indent = 6)
            
        self.load_raw_()
        
        
        