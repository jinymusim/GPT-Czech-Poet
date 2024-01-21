import os
import json
import numpy as np
import torch

from utils.poet_utils import StropheParams, SyllableMaker, TextAnalysis, TextManipulation
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, PreTrainedModel
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
        
        def __init__(self, data_file_paths, prompt_length=True, prompt_ending=True, lower_case=True, val_data_rate: float = 0.05, test_data_rate: float = 0.05):
            """Construct the class our given data files path and store variables

            Args:
                data_file_paths (_type_):  list of paths to data files
                prompt_length (bool, optional): If to prompt the syllable count. Defaults to True.
                prompt_ending (bool, optional): If to prompt verse ending. Defaults to True.
                lower_case (bool, optional): If the string should be in lowercase. Defaults to True.
                val_data_rate (float, optional): Amount of data to be left for validation. Defaults to 0.05.
                test_data_rate (float, optional): Amount of data to be left for validation. Defaults to 0.05.
            """
            self._data_file_paths = data_file_paths
            self.prompt_length = prompt_length
            self.prompt_ending = prompt_ending
            self.lower_case = lower_case
            
            self.val_data_rate = val_data_rate
            self.test_data_rate = test_data_rate
            
            self.data = []
            self.validation_data = []
            self.test_data = []
            
            self.custom_size = 1
         
         
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
            vowels = sum(map(len, syllabs)) #INFO: Now counts the number of syllables
            ending = syllabs[-1][-1]
            return vowels, ending
        
        @staticmethod
        def _ending_vector(end):
            """Construct One-hot encoded vector for ending syllable

            Args:
                end (str): Ending syllable

            Returns:
                numpy.ndarray: One-hot encoded vector of ending syllable
            """
            verse_end_vector = np.zeros(len(StropheParams.ENDS))
            if end in StropheParams.ENDS[:-1]:
                verse_end_vector[StropheParams.ENDS.index(end)] = 1
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
            return "  ".join([" ".join(syl) for syl in  SyllableMaker.syllabify(raw_text)]) + ending
        
        def _construct_line(self, raw_text, metre):
            """Construct individual content line

            Args:
                raw_text (str): raw verse line

            Returns:
                str: Processed verse line with line parameters
            """
            syllables = SyllableMaker.syllabify(raw_text)
            num_str = f"{sum(map(len, syllables))} # " if self.prompt_length else ""
            verse_end = f"{syllables[-1][-1]} # " if self.prompt_ending else ""
            metre_txt = f"{metre} # "
            return metre_txt + num_str + verse_end  + raw_text
        
        def _introduce_phonetics(self, raw_text:str, phonetics):
            phonetic_text = raw_text
            for word in phonetics['words']:
                phonetic_text = phonetic_text.replace(f'{word["token_lc"]}', f'{word["phoebe"]}') if self.lower_case else phonetic_text.replace(f'{word["token"]}', f'{word["phoebe"]}')
            return phonetic_text
        
        def _construct_syllable_line(self, raw_text, metre):
            """Construct individual content line as sequence of syllables

            Args:
                raw_text (str): raw verse line

            Returns:
                str: Processed verse line as sequence of syllables with line parameters
            """
            ending = raw_text[-1] if raw_text[-1] in [',','.','!','?'] else ''
            syllables = SyllableMaker.syllabify(raw_text)
            num_str = f"{sum(map(len, syllables))} # " if self.prompt_length else ""
            verse_end = f"{syllables[-1][-1]} # " if self.prompt_ending else ""
            metre_txt = f"{metre} # "
            return  metre_txt+ num_str + verse_end + "  ".join([" ".join(syl) for syl in syllables]) + ending
                     
            
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
                            metre = StropheParams.METER_TRANSLATE.get(text_line["metre"][0]["type"], "N")
                            
                            scanned_text = TextManipulation._remove_most_nonchar(text_line['text'], self.lower_case)
                            
                            text_line_scanned = self._construct_line(scanned_text, metre)
                            syllable_line = self._construct_syllable_line(scanned_text, metre)
                            #phonetic_text = self._introduce_phonetics(scanned_text, text_line)
                            
                            num_vowels, verse_end = self._vowels_and_endings(scanned_text)
                            
                            # Based on result of random chose proper set. Because data are large enough, will result in wanted split.
                            rand_split = np.random.rand()
                            if rand_split > self.val_data_rate + self.test_data_rate: 
                                self.data.append({
                                "input_ids" : [text_line_scanned,syllable_line],
                                "nums": [num_vowels],
                                "verse_end": verse_end,
                                "metre": metre
                                     })
                            elif rand_split < self.test_data_rate:
                                self.test_data.append({
                                "input_ids" : [text_line_scanned,syllable_line],
                                "nums": [num_vowels],
                                "verse_end": verse_end,
                                "metre": metre
                                     })
                            else:
                                self.validation_data.append({
                                "input_ids" : [text_line_scanned,syllable_line],
                                "nums": [num_vowels],
                                "verse_end": verse_end,
                                "metre": metre
                                     })
                            
            
        def __len__(self):
            """Return length of training data

            Returns:
                int: length of training data
            """
            return int(len(self.data) * self.custom_size)
        
        def __getitem__(self, index):
            """return indexed item

            Args:
                index (int): index from where to return

            Returns:
                dict: dict with indexed data
            """
            return self.data[index]
        
        def change_custom_size(self,float_size:float = 1):
            if float_size > 1:
                print("Improper size, revert to full size")
                self.custom_size = 1
            elif float_size <= 0:
                print("Size must be positive, revert to full size")
                self.custom_size = 1
            else:
                self.custom_size = float_size
        
    class BodyDataset(Dataset):
        """Dataset of preprocessed strophe

        Args:
            Dataset (_type_): Dataset is child of torch class for better integration with torch and huggingface
        """
        def __init__(self, data_file_paths,
                     prompt_length=True, prompt_ending=True, prompt_verse=True, verse_len=[4,6], lower_case=True, val_data_rate: float = 0.05, test_data_rate: float = 0.05):
            """Construct the class our given data files path and store variables

            Args:
                data_file_paths (_type_): list of paths to data files
                prompt_length (bool, optional): If to prompt the syllable count. Defaults to True.
                prompt_ending (bool, optional): If to prompt verse ending. Defaults to True.
                prompt_verse (bool, optional): If to prompt rhyme schema . Defaults to True.
                verse_len (list, optional): Considered length of strophe. Defaults to [4,6].
                lower_case (bool, optional): If the string should be in lowercase. Defaults to True.
                val_data_rate (float, optional): Amount of data to be left for validation. Defaults to 0.05.
                test_data_rate (float, optional): Amount of data to be left for validation. Defaults to 0.05.
            """
            self._data_file_paths = data_file_paths
            self.prompt_length = prompt_length
            self.prompt_ending = prompt_ending
            self.prompt_verse = prompt_verse
            self.verse_len = verse_len
            self.lower_case = lower_case
            
            self.val_data_rate = val_data_rate
            self.test_data_rate = test_data_rate
            
            self.data = []
            self.validation_data = []
            self.test_data = []
            
            self.custom_size = 1
        
        def gen_files(self):
            """Get individual opened files

            Yields:
                _type_: open file object
            """
            for filename in self._data_file_paths:
                 yield open(filename, 'r')
                 
                     
        
        def _construct_line(self, raw_text, metre):
            """Construct individual content line

            Args:
                raw_text (str): raw verse line

            Returns:
                str: Processed verse line with line parameters
            """
            syllables = SyllableMaker.syllabify(raw_text)
            num_str = f"{sum(map(len, syllables))} # " if self.prompt_length else ""
            verse_end = f"{syllables[-1][-1]} # " if self.prompt_ending else ""
            metre_txt = f"{metre} # "
            return  metre_txt + num_str + verse_end  + raw_text
        
        def _construct_syllable_line(self, raw_text, metre):
            """Construct individual content line as sequence of syllables

            Args:
                raw_text (str): raw verse line

            Returns:
                str: Processed verse line as sequence of syllables with line parameters
            """
            ending = raw_text[-1] if raw_text[-1] in [',','.','!','?'] else ''
            syllables = SyllableMaker.syllabify(raw_text)
            num_str = f"{sum(map(len, syllables))} # " if self.prompt_length else ""
            verse_end = f"{syllables[-1][-1]} # " if self.prompt_ending else ""
            metre_txt = f"{metre} # "
            return metre_txt + num_str + verse_end + "  ".join([" ".join(syl) for syl in syllables]) + ending
            
            
                                                           
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
                    context = ["NO CONTEXT"]

                    for part_line in data_line['body']:                                                        
                        body = []
                        body_syllabs = []
                        rhyme= []
                        metres = []
                        i = 0
                        for text_line in part_line:
                            
                            # In rare cases multiple, but from searching only 1 metre per line
                            metre = StropheParams.METER_TRANSLATE.get(text_line["metre"][0]["type"], "J")
                            metres +=  [metre]
                            
                            rhyme.append(text_line["rhyme"])  
                            
                            scanned_text = TextManipulation._remove_most_nonchar(text_line["text"], self.lower_case)

                            body.append(self._construct_line(scanned_text,metre))
                            body_syllabs.append(self._construct_syllable_line(scanned_text,metre))
                            
                            i+=1
                            
                            if i in self.verse_len:
                                
                                rhyme_str = TextManipulation._rhyme_string(rhyme)
                                
                                text = f"# {rhyme_str} # {publish_year_text}\n" + "\n".join(body) + "\n"
                                syllable_text = f"# {rhyme_str} # {publish_year_text}\n" + "\n".join(body_syllabs) + "\n"
                                context_text= "\n".join(context)
                                rand_split = np.random.rand()
                                if rand_split > self.val_data_rate + self.test_data_rate:
                                    self.data.append({
                                    "input_ids" : [text,syllable_text],
                                    "context_ids" : context_text,
                                    "year": publish_year_true,
                                    "rhyme":  rhyme_str,
                                    "metre_ids" : metres.copy()
                                     })
                                elif rand_split < self.test_data_rate:
                                    self.test_data.append({
                                    "input_ids" : [text,syllable_text],
                                    "context_ids" : context_text,
                                    "year": publish_year_true,
                                    "rhyme":  rhyme_str,
                                    "metre_ids" : metres.copy()
                                     })
                                else:
                                    self.validation_data.append({
                                    "input_ids" : [text,syllable_text],
                                    "context_ids" : context_text,
                                    "year": publish_year_true,
                                    "rhyme":  rhyme_str,
                                    "metre_ids" : metres.copy()
                                     })
                                
                                if i == max(self.verse_len):
                                    body = []
                                    body_syllabs = []
                                    rhyme = []
                                    metres = []
                                    i=0
                                
        
        def __len__(self):
            """Return length of training data

            Returns:
                int: length of training data
            """
            return int(len(self.data) * self.custom_size)
        
        def __getitem__(self, index):
            """return indexed item

            Args:
                index (int): index from where to return

            Returns:
                dict: dict with indexed data
            """
            return self.data[index]
        
        def change_custom_size(self,float_size:float = 1):
            if float_size > 1:
                print("Improper size, revert to full size")
                self.custom_size = 1
            elif float_size <= 0:
                print("Size must be positive, revert to full size")
                self.custom_size = 1
            else:
                self.custom_size = float_size
        
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
    
    def load_json_filenames(self, prompt_length, prompt_ending, prompt_verse, verse_len=[4,6], val_data_rate=0.05, test_data_rate=0.05):
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
                                                    val_data_rate=val_data_rate, test_data_rate=test_data_rate)
        self.pytorch_dataset_body.data_body_gen()
         
        
        self.pytorch_dataset_text = CorpusDatasetPytorch.TextDataset(filenames, prompt_ending=prompt_ending, 
                                                    prompt_length=prompt_length, lower_case=self.lower_case,
                                                    val_data_rate=val_data_rate, test_data_rate=test_data_rate)
        
        self.pytorch_dataset_text.data_text_line_gen()
        
        self.val_pytorch_dataset_body = CorpusDatasetPytorch.BodyDataset([])
        self.val_pytorch_dataset_text = CorpusDatasetPytorch.TextDataset([])
        
        self.val_pytorch_dataset_body.data = self.pytorch_dataset_body.validation_data
        self.val_pytorch_dataset_text.data = self.pytorch_dataset_text.validation_data
        
        self.pytorch_dataset_text.validation_data = []
        self.pytorch_dataset_body.validation_data = []
        
        self.test_pytorch_dataset_body = CorpusDatasetPytorch.BodyDataset([])
        self.test_pytorch_dataset_text = CorpusDatasetPytorch.TextDataset([])
        
        self.test_pytorch_dataset_body.data = self.pytorch_dataset_body.test_data
        self.test_pytorch_dataset_text.data = self.pytorch_dataset_text.test_data
        
        self.pytorch_dataset_text.test_data = []
        self.pytorch_dataset_body.test_data = []
        
    def create_empty(self):
        """Create empty holder for possible load of processed data from file
        """
        self.pytorch_dataset_body = CorpusDatasetPytorch.BodyDataset([])
        self.pytorch_dataset_text = CorpusDatasetPytorch.TextDataset([])
        self.val_pytorch_dataset_body = CorpusDatasetPytorch.BodyDataset([])
        self.val_pytorch_dataset_text = CorpusDatasetPytorch.TextDataset([])
        self.test_pytorch_dataset_body = CorpusDatasetPytorch.BodyDataset([])
        self.test_pytorch_dataset_text = CorpusDatasetPytorch.TextDataset([])
        
        
    @staticmethod
    def collate(batch, tokenizer: PreTrainedTokenizerBase ,max_len = 1024, max_context = 1024 ,mask_rate = 0.0, syllables: bool = False, format: str = 'METER_VERSE'):
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
        if batch[0]['input_ids'][0].startswith("#"):
            
            data = [text['input_ids'][index] for text in batch]
            if format == "BASIC":
                data =  ["\n".join
                         (
                        [line + f" # {datum.splitlines()[1].split()[0]}"
                         if i==0 else line.split('#')[-1] for i, line in enumerate(datum.splitlines())] 
                        ) + tokenizer.eos_token  for j, datum in enumerate(data) 
                         ]
            elif format == "VERSE_PAR":
                 data =  ["\n".join
                         (
                        [line + f" # {datum.splitlines()[1].split()[0]}"
                         if i==0 else "#".join(line.split('#')[1:]) for i, line in enumerate(datum.splitlines())] 
                        ) + tokenizer.eos_token for j, datum in enumerate(data) 
                         ]
            else:
                data = [text['input_ids'][index] + tokenizer.eos_token for text in batch]
                 
            tokenized = tokenizer(data,return_tensors='pt', truncation=True, padding=True)
            input_ids = tokenized['input_ids']
            attention = tokenized["attention_mask"]
        
        else:
            tokenized = tokenizer([text['input_ids'][index] + tokenizer.eos_token for text in batch],return_tensors='pt', truncation=True, padding=True)
            input_ids = tokenized['input_ids']
            attention = tokenized["attention_mask"]
    
        
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
    def collate_distil(batch, tokenizer: PreTrainedTokenizerBase ,surrogate_model: PreTrainedModel = None,surrogate_model_device=None ,max_len = 1024):
        tokenizer.model_max_length = max_len
        tokenized = tokenizer([text['input_ids'][0] + tokenizer.eos_token for text in batch], return_tensors='pt', truncation=True, padding=True)
        input_ids = tokenized['input_ids']
        attention = tokenized["attention_mask"]
        
        with torch.no_grad():
            # This is Tuple
            model_hidden_states = surrogate_model(input_ids=input_ids.to(surrogate_model_device), 
                                                  attention_mask=attention.to(surrogate_model_device), 
                                                  labels=input_ids.type(torch.LongTensor).to(surrogate_model_device))['hidden_states']
        model_hidden_states = [hidden.cpu().detach() for hidden in model_hidden_states]
        
        return {
            "input_ids": input_ids,
            "labels": input_ids.type(torch.LongTensor),
            "attention_mask": attention,
            "to_replicate_states": model_hidden_states
         }
        
    @staticmethod
    def collate_validator(batch, tokenizer: PreTrainedTokenizerBase,syllables:bool, is_syllable:bool = False,max_len = 512):
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
            ["  ".join(
                   [" ".join(syl) for syl in SyllableMaker.syllabify(line.split('#')[-1])]
                ) + (line[-1] if line[-1] in [',','.','!','?'] else '') if (syllables and not is_syllable and line) else line.split('#')[-1] for line in text['input_ids'][index].splitlines()[1:]] 
            ) for text in batch ]
        
        
        tokenized = tokenizer(data_ids, return_tensors='pt', truncation=True, padding=True)
        input_ids = tokenized['input_ids']
        attention = tokenized["attention_mask"]
            
        rhyme=None
        if "rhyme" in batch[0].keys():
            rhyme = torch.tensor(np.asarray([TextAnalysis._rhyme_vector(text["rhyme"]) for text in batch], dtype=np.int32), dtype=torch.float32)
            
        year_bucket = None
        year = None
        if "year" in batch[0].keys():      
            year_bucket = torch.tensor(np.asarray([TextAnalysis._publish_year_vector(text["year"]) for text in batch], dtype=np.int32), dtype=torch.float32)
            year = torch.tensor(np.asarray([ [int(text['year'])] if text['year'] != 'NaN' else [0] for text in batch], dtype=np.int32), dtype=torch.float32)
        
        return  {
            "input_ids": input_ids,
            "attention_mask": attention,
            "rhyme": rhyme,
            "metre_ids": None,
            "year_bucket": year_bucket,
            'year':year}
    
    @staticmethod
    def collate_meter(batch, tokenizer: PreTrainedTokenizerBase, syllables:bool, is_syllable:bool = False, max_len = 512):
        index = 1 if syllables and is_syllable else 0
        tokenizer.model_max_length = max_len
        data_ids = []
        metre = []
        for datum in batch:
            data_ids += [
                    "  ".join(
                    [" ".join(syl) for syl in SyllableMaker.syllabify(line.split('#')[-1])]
                ) + (line[-1] if line[-1] in [',','.','!','?'] else '') if (syllables and not is_syllable and line) else line.split('#')[-1] for line in datum['input_ids'][index].splitlines()[1:]
                ]
            if "metre_ids" in batch[0].keys():
                metre += [TextAnalysis._metre_vector(one_metre) for one_metre in datum['metre_ids']]
                
        tokenized = tokenizer(data_ids, return_tensors='pt', truncation=True, padding=True)
        input_ids = tokenized['input_ids']
        attention = tokenized["attention_mask"]
        
        metre_ids = None
        if len(metre) > 0:
            metre_ids = torch.tensor(np.asarray(metre, dtype=np.int32), dtype=torch.float32)
            
        return  {
            "input_ids": input_ids,
            "attention_mask": attention,
            "rhyme": None,
            "metre_ids": metre_ids,
            "year_bucket": None,
            "year": None}
        
    @staticmethod
    def collate_meter_context(batch, tokenizer: PreTrainedTokenizerBase, syllables:bool, is_syllable:bool = False, max_len = 512):
        index = 1 if syllables and is_syllable else 0
        tokenizer.model_max_length = max_len
        data_ids = []
        
        metre = []
        for datum in batch:
            base_datums = []
            base_datums += [
                    "  ".join(
                    [" ".join(syl) for syl in SyllableMaker.syllabify(line.split('#')[-1])]
                ) + (line[-1] if line[-1] in [',','.','!','?'] else '') if (syllables and not is_syllable and line) else line.split('#')[-1] for line in datum['input_ids'][index].splitlines()[1:]
                ]
            i = 0
            for i in range(len(base_datums)):
                data_ids.append(
                    "\n".join(base_datums[:i] + ['# ' + base_datums[i]] + base_datums[i+1:])
                    ) 
            if "metre_ids" in batch[0].keys():
                metre += [TextAnalysis._metre_vector(one_metre) for one_metre in datum['metre_ids']]
                
        tokenized = tokenizer(data_ids, return_tensors='pt', truncation=True, padding=True)
        input_ids = tokenized['input_ids']
        attention = tokenized["attention_mask"]
        
        metre_ids = None
        if len(metre) > 0:
            metre_ids = torch.tensor(np.asarray(metre, dtype=np.int32), dtype=torch.float32)
            
        return  {
            "input_ids": input_ids,
            "attention_mask": attention,
            "rhyme": None,
            "metre_ids": metre_ids,
            "year_bucket": None,
            "year": None}
        
        
    
        
    def __init__(self, data_dir = "PoetGen\corpusCzechVerse-master\ccv", cache_dir='./', 
                 prompt_length=True, prompt_ending=True, prompt_verse=True, verse_len=[4,6], lower_case=True, val_data_rate=0.05, test_data_rate=0.05):
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
            and os.path.isfile(os.path.join(cache_dir, "val_body_poet_data.json")) and os.path.isfile(os.path.join(cache_dir, "val_text_poet_data.json")) \
            and os.path.isfile(os.path.join(cache_dir, "test_body_poet_data.json")) and os.path.isfile(os.path.join(cache_dir, "test_text_poet_data.json")) :
            self.create_empty()
            self.pytorch_dataset_body.data =list(json.load( open( os.path.join(cache_dir, "body_poet_data.json"), 'r')))
            self.pytorch_dataset_text.data =list(json.load( open( os.path.join(cache_dir, "text_poet_data.json"), 'r')))
            self.val_pytorch_dataset_body.data = list(json.load( open( os.path.join(cache_dir, "val_body_poet_data.json"), 'r')))
            self.val_pytorch_dataset_text.data = list(json.load( open( os.path.join(cache_dir, "val_text_poet_data.json"), 'r')))
            self.test_pytorch_dataset_body.data = list(json.load( open( os.path.join(cache_dir, "test_body_poet_data.json"), 'r')))
            self.test_pytorch_dataset_text.data = list(json.load( open( os.path.join(cache_dir, "test_text_poet_data.json"), 'r')))
        else:
            self.load_json_filenames(prompt_length, prompt_ending, prompt_verse, verse_len=verse_len, val_data_rate=val_data_rate, test_data_rate=test_data_rate)
            json.dump(self.pytorch_dataset_body.data, open( os.path.join(cache_dir, "body_poet_data.json"), 'w+'), indent = 6)
            json.dump(self.pytorch_dataset_text.data, open( os.path.join(cache_dir, "text_poet_data.json"), 'w+'), indent = 6)
            json.dump(self.val_pytorch_dataset_body.data, open( os.path.join(cache_dir, "val_body_poet_data.json"), 'w+'), indent = 6)
            json.dump(self.val_pytorch_dataset_text.data, open( os.path.join(cache_dir, "val_text_poet_data.json"), 'w+'), indent = 6)
            json.dump(self.test_pytorch_dataset_body.data, open( os.path.join(cache_dir, "test_body_poet_data.json"), 'w+'), indent = 6)
            json.dump(self.test_pytorch_dataset_text.data, open( os.path.join(cache_dir, "test_text_poet_data.json"), 'w+'), indent = 6)
            
        self.load_raw_()
        
        
        
#if __name__ == "__main__":
# Line Count
#    print(len(list(CorpusDatasetPytorch(os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")) ).raw_dataset.get_text())))
# Strophe Count
#    print(len(list(CorpusDatasetPytorch(os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")) ).raw_dataset.get_part())))
# Poem Count
#    print(len(list(CorpusDatasetPytorch(os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")) ).raw_dataset.get_body())))