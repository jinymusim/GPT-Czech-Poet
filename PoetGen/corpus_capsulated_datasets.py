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
    """Dataset class responsible for data loading.
    """
    
    class RawDataset:
        """Dataset distributing raw sting data with no preprocessing
        """
        def __init__(self, data_file_paths, lower_case:bool = True):
            """Construct the frame around Raw data generation

            Args:
                data_file_paths (_type_): list of paths to data files
                lower_case (bool, optional): if resulting data should be in lowercase. Defaults to True.
            """
            self._data_file_paths = data_file_paths
            self.lower_case = lower_case
        
        def gen_files(self):
            """Get individual opened files

            Yields:
                _type_: open file object
            """
            for filename in self._data_file_paths:
                 yield open(filename, 'r') 
                 
        def get_text(self):
            """Get lines of text of poetry

            Yields:
                str: individual verse line
            """
            for step,file in enumerate(self.gen_files()):
                if step % 500 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                for data_line in datum:
                    for part_line in data_line['body']:
                        for text_line in part_line:
                            yield text_line['text'].lower() if self.lower_case else text_line['text']
                            
        def get_part(self):
            """Get strophe of poetry

            Yields:
                str: 1 strophe of poetry
            """
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
            """Get whole poem

            Yields:
                str: 1 whole poem
            """
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
        """Dataset of preprocessed verse lines 

        Args:
            Dataset (_type_): Dataset is child of torch class for better integration with torch and huggingface
        """
        
        def __init__(self, data_file_paths, prompt_length=True, prompt_ending=True, lower_case=True, val_data_rate: float = 0.1):
            """Construct the class our given data files path and store variables

            Args:
                data_file_paths (_type_):  list of paths to data files
                prompt_length (bool, optional): If to prompt the syllable count. Defaults to True.
                prompt_ending (bool, optional): If to prompt verse ending. Defaults to True.
                lower_case (bool, optional): If the string should be in lowercase. Defaults to True.
                val_data_rate (float, optional): Amount of data to be left for validation. Defaults to 0.1.
            """
            self._data_file_paths = data_file_paths
            self.prompt_length = prompt_length
            self.prompt_ending = prompt_ending
            self.lower_case = lower_case
            
            self.val_data_rate = val_data_rate
            
            self.data = []
            self.validation_data = []
         
         
        def gen_files(self):
            """Get individual opened files

            Yields:
                _type_: open file object
            """
            for filename in self._data_file_paths:
                 yield open(filename, 'r') 
                 
        @staticmethod
        def _vowels_and_endings(raw_text):
            """Get the verse ending and number of syllables in verse

            Args:
                raw_text (str): raw verse to analyze

            Returns:
                tuple: number of syllables, ending syllable
            """
            syllabs = SyllableMaker.syllabify(raw_text)
            vowels = len(syllabs) #INFO: Now counts the number of syllables
            ending = syllabs[-1]
            return vowels, ending
        
        @staticmethod
        def _ending_vector(end):
            """Construct One-hot encoded vector for ending syllable

            Args:
                end (str): Ending syllable

            Returns:
                numpy.ndarray: One-hot encoded vector of ending syllable
            """
            verse_end_vector = np.zeros(len(VERSE_ENDS))
            if end in VERSE_ENDS:
                verse_end_vector[VERSE_ENDS.index(end)] = 1
            else:
                verse_end_vector[-1] = 1
            return verse_end_vector
        
        @staticmethod
        def _syllable_line(raw_text):
            """Construct verse as sequence of syllables

            Args:
                raw_text (str): raw verse line

            Returns:
                str: Verse line as sequence of syllables
            """
            ending = raw_text[-1] if raw_text[-1] in [',','.','!','?'] else ''
            return " ".join(SyllableMaker.syllabify(raw_text)) + ending
                     
            
        def data_text_line_gen(self):
            """Preprocess and process data for usage
            """
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
                            # Based on result of random chose proper set. Because data are large enough, will result in wanted split.
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
            """Return length of training data

            Returns:
                int: length of training data
            """
            return len(self.data)
        
        def __getitem__(self, index):
            """return indexed item

            Args:
                index (int): index from where to return

            Returns:
                dict: dict with indexed data
            """
            return self.data[index]
        
    class BodyDataset(Dataset):
        """Dataset of preprocessed strophe

        Args:
            Dataset (_type_): Dataset is child of torch class for better integration with torch and huggingface
        """
        def __init__(self, data_file_paths,
                     prompt_length=True, prompt_ending=True, prompt_verse=True, verse_len=[4,6], lower_case=True, val_data_rate: float = 0.1):
            """Construct the class our given data files path and store variables

            Args:
                data_file_paths (_type_): list of paths to data files
                prompt_length (bool, optional): If to prompt the syllable count. Defaults to True.
                prompt_ending (bool, optional): If to prompt verse ending. Defaults to True.
                prompt_verse (bool, optional): If to prompt rhyme schema . Defaults to True.
                verse_len (list, optional): Considered length of strophe. Defaults to [4,6].
                lower_case (bool, optional): If the string should be in lowercase. Defaults to True.
                val_data_rate (float, optional): Amount of data to be left for validation. Defaults to 0.1.
            """
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
            """Get individual opened files

            Yields:
                _type_: open file object
            """
            for filename in self._data_file_paths:
                 yield open(filename, 'r')
                 
        
   
                     
        
        def _construct_line(self, raw_text):
            """Construct individual content line

            Args:
                raw_text (str): raw verse line

            Returns:
                str: Processed verse line with line parameters
            """
            syllables = SyllableMaker.syllabify(raw_text)
            num_str = f"{len(syllables)} " if self.prompt_length else ""
            verse_end = f"{syllables[-1]} # " if self.prompt_ending else ""
            return num_str + verse_end + raw_text
        
        def _construct_syllable_line(self, raw_text):
            """Construct individual content line as sequence of syllables

            Args:
                raw_text (str): raw verse line

            Returns:
                str: Processed verse line as sequence of syllables with line parameters
            """
            ending = raw_text[-1] if raw_text[-1] in [',','.','!','?'] else ''
            syllables = SyllableMaker.syllabify(raw_text)
            num_str = f"{len(syllables)} " if self.prompt_length else ""
            verse_end = f"{syllables[-1]} # " if self.prompt_ending else ""
            return num_str + verse_end + " ".join(syllables) + ending
            
            
                                                           
        def data_body_gen(self):
            """Preprocess and process data for usage
            """
            for step,file in enumerate(self.gen_files()):
                if step % 500 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                
                for data_line in datum:
                    publish_year_text = TextManipulation._year_bucketor(data_line["biblio"]["year"])
                    publish_year_true = data_line["biblio"]["year"] if TextAnalysis._is_year(data_line["biblio"]["year"]) else 'NaN'
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
                                rhyme_str = TextManipulation._rhyme_string(rhyme)
                                
                                text = f"{rhyme_str} # {publish_year_text} # {metre}\n" + "\n".join(body) + "\n"
                                syllable_text = f"{rhyme_str} # {publish_year_text} # {metre}\n" + "\n".join(body_syllabs) + "\n"
                                context_text= "\n".join(context)
                                if np.random.rand() > self.val_data_rate:
                                    self.data.append({
                                    "input_ids" : [text,syllable_text],
                                    "context_ids" : context_text,
                                    "year": publish_year_true,
                                    "rhyme":  rhyme_str,
                                    "metre" : metre
                                     })
                                else:
                                    self.validation_data.append({
                                    "input_ids" : [text,syllable_text],
                                    "context_ids" : context_text,
                                    "year": publish_year_true,
                                    "rhyme":  rhyme_str,
                                    "metre" : metre
                                     })
                                
                                if i == max(self.verse_len):
                                    context = body
                                    body = []
                                    rhyme = []
                                    i=0
                                
        
        def __len__(self):
            """Return length of training data

            Returns:
                int: length of training data
            """
            return len(self.data)
        
        def __getitem__(self, index):
            """return indexed item

            Args:
                index (int): index from where to return

            Returns:
                dict: dict with indexed data
            """
            return self.data[index]
        
    def get_filenames(self):
        """Get paths of data files

        Returns:
            list: Paths of data files
        """
        data_filenames = os.listdir(self.data_dir)
        data_by_files = []
        for filename in data_filenames:
            file_path = os.path.join(self.data_dir, filename)
            data_by_files.append(file_path)
        return data_by_files
        
    def load_raw_(self):
        """Load Raw dataset with raw string data
        """
        filenames = self.get_filenames()
            
        self.raw_dataset = CorpusDatasetPytorch.RawDataset(filenames, self.lower_case)
    
    def load_json_filenames(self, prompt_length, prompt_ending, prompt_verse, verse_len=[4,6], val_data_rate=0.1):
        """Load Verse and Strophe datasets

        Args:
            prompt_length (bool, optional): If to prompt the syllable count. Defaults to True.
            prompt_ending (bool, optional): If to prompt verse ending. Defaults to True.
            prompt_verse (bool, optional): If to prompt rhyme schema . Defaults to True.
            verse_len (list, optional): Considered length of strophe. Defaults to [4,6].
            val_data_rate (float, optional): If the string should be in lowercase. Defaults to 0.1.
        """
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
        """Create empty holder for possible load of processed data from file
        """
        self.pytorch_dataset_body = CorpusDatasetPytorch.BodyDataset([])
        self.pytorch_dataset_text = CorpusDatasetPytorch.TextDataset([])
        
        
    @staticmethod
    def collate(batch, tokenizer: PreTrainedTokenizerBase ,max_len = 1024, max_context = 1024 ,mask_rate = 0.0, syllables: bool = False):
        """Process data for usage in LM

        Args:
            batch (_type_): Batch with selected data points
            tokenizer (PreTrainedTokenizerBase): tokenizer to tokenize input text
            max_len (int, optional): Maximum length of tokenization. Defaults to 1024.
            max_context (int, optional): Maximum length of tokenization of context. Defaults to 1024.
            mask_rate (float, optional): Rate in with to mask data. Defaults to 0.0.
            syllables (bool, optional): If to use sequence of syllables as input text. Defaults to False.

        Returns:
            dict: tokenized and processed to tensors data
        """
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
            year = torch.tensor(np.asarray([TextAnalysis._publish_year_vector(text["year"]) for text in batch], dtype=np.int32), dtype=torch.float32)
            
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
    def collate_validator(batch, tokenizer: PreTrainedTokenizerBase,syllables:bool, is_syllable:bool = False,max_len = 1024):
        """Process data for use in LM for metre,rhyme and year prediction

        Args:
            batch (_type_): Batch with selected data points
            tokenizer (PreTrainedTokenizerBase): tokenizer to tokenize input text   
            syllables (bool): If to use sequence of syllables as input text
            is_syllable (bool, optional): Signal if the preprocessed inputs contain syllable data. Defaults to False.
            max_len (int, optional): Maximum length of tokenization. Defaults to 1024.

        Returns:
            dict: tokenized and processed to tensors data
        """
        index = 1 if syllables and is_syllable else 0
        tokenizer.model_max_length = max_len
        data_ids = ["\n".join(
            [" ".join(
                    SyllableMaker.syllabify(line.split('#')[-1])
                ) if (syllables and not is_syllable) else line.split('#')[-1] for line in text['input_ids'][index].splitlines()[1:]] 
            ) for text in batch ]
        
        
        tokenized = tokenizer(data_ids, return_tensors='pt', truncation=True, padding=True)
        input_ids = tokenized['input_ids']
        attention = tokenized["attention_mask"]
        
        metre = None
        if "metre" in batch[0].keys():       
            metre = torch.tensor(np.asarray([TextAnalysis._metre_vector(text["metre"]) for text in batch], dtype=np.int32), dtype=torch.float32)
            
        rhyme=None
        if "rhyme" in batch[0].keys():
            rhyme = torch.tensor(np.asarray([TextAnalysis._rhyme_vector(text["rhyme"]) for text in batch], dtype=np.int32), dtype=torch.float32)
            
        year = None
        if "year" in batch[0].keys():      
            year = torch.tensor(np.asarray([TextAnalysis._publish_year_vector(text["year"]) for text in batch], dtype=np.int32), dtype=torch.float32)
        
        return  {
            "input_ids": input_ids,
            "attention_mask": attention,
            "rhyme": rhyme,
            "metre": metre,
            "year": year}
    
        
    def __init__(self, data_dir = "PoetGen\corpusCzechVerse-master\ccv", cache_dir='./', 
                 prompt_length=True, prompt_ending=True, prompt_verse=True, verse_len=[4,6], lower_case=True, val_data_rate=0.05):
        """Construct the Dataloader and create Datasets

        Args:
            data_dir (str, optional): Path to data. Defaults to "PoetGen\corpusCzechVerse-master\ccv".
            cache_dir (str, optional): Path where to store processed data. Defaults to './'.
            prompt_length (bool, optional): If to prompt the syllable count. Defaults to True.
            prompt_ending (bool, optional): If to prompt verse ending. Defaults to True.
            prompt_verse (bool, optional): If to prompt rhyme schema. Defaults to True.
            verse_len (list, optional): Considered length of strophe. Defaults to [4,6].
            lower_case (bool, optional): If the string should be in lowercase. Defaults to True.
            val_data_rate (float, optional): Amount of data to be left for validation. Defaults to 0.1.
        """
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
        
        
        
#if __name__ == "__main__":
# Line Count
#    print(len(list(CorpusDatasetPytorch(os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")) ).raw_dataset.get_text())))
# Strophe Count
#    print(len(list(CorpusDatasetPytorch(os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")) ).raw_dataset.get_part())))
# Poem Count
#    print(len(list(CorpusDatasetPytorch(os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")) ).raw_dataset.get_body())))