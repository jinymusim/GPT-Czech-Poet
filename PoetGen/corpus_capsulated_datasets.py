import os
import json
import numpy as np
import torch

from utils.poet_utils import StropheParams,  TextAnalysis, TextManipulation, SyllableMaker, VersologicalMaker
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
                if step % 100 == 0:
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
                if step % 100 == 0:
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
                if step % 100 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                for data_line in datum:
                    body = []
                    for part_line in data_line['body']:
                        
                        for text_line in part_line:
                            body.append(text_line['text'])
                        body.append("\n")
                    yield "\n".join(body).lower() if self.lower_case else "\n".join(body)
    
    class VersesDataset(Dataset):
        """Dataset of preprocessed verse lines 

        Args:
            Dataset (_type_): Dataset is child of torch class for better integration with torch and huggingface
        """
        
        def __init__(self, data_file_paths, dataset_parameters, segment_type, dataset_part):
            """Create Specific dataset part

            Args:
                data_file_paths (_type_): filenames
                dataset_parameters (_type_): dataset parames to use for structure
                segment_type (_type_): segmentation to do on the dataset
                dataset_part (_type_): part of dataset to be done ('train', 'val', 'test')
            """
            self._data_file_paths = data_file_paths
            self.params = dataset_parameters

            self.seg_type = segment_type

            self.dataset_part = dataset_part
            self.relevant_indexes = []
            if dataset_part == 'train':
                self.relevant_indexes = self.params['train_indexes']
            elif dataset_part == 'val':
                self.relevant_indexes = self.params['val_indexes']
            elif dataset_part == 'test':
                self.relevant_indexes = self.params['test_indexes']

            self.data = []

            self.line_constructor = None
            if segment_type == 'BASE':
                self.line_constructor = self._construct_line
            elif segment_type == 'SYLLABLE':
                self.line_constructor = self._construct_syllable_line
            elif segment_type == 'VERSEMARK':
                self.line_constructor = self._construct_verse_marks_line
        
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
            return "  ".join([" ".join(syl) for syl in  (SyllableMaker.syllabify(raw_text) )]) + ending
        
        @staticmethod
        def _verse_marks_line(raw_text):
            """Construct verse as sequence of syllables

            Args:
                raw_text (str): raw verse line

            Returns:
                str: Verse line as sequence of syllables
            """
            ending = raw_text[-1] if raw_text[-1] in [',','.','!','?'] else ''
            return "  ".join([" ".join(syl) for syl in  (VersologicalMaker.verse_segmnent(raw_text) )]) + ending
        
        
        def _construct_line(self, raw_text, metre):
            """Construct individual content line

            Args:
                raw_text (str): raw verse line

            Returns:
                str: Processed verse line with line parameters
            """
            syllables = SyllableMaker.syllabify(raw_text)
            verse_marks = VersologicalMaker.verse_segmnent(raw_text)
            num_str = f"{sum(map(len, syllables))} # " if self.params['prompt_len'] else ""
            verse_end = f"{ ''.join(verse_marks[-1][-2:])} # " if self.params['prompt_end'] else ""
            metre_txt = f"{metre} # "
            return metre_txt + num_str + verse_end  + raw_text
        
        def _construct_syllable_line(self, raw_text, metre):
            """Construct individual content line as sequence of syllables

            Args:
                raw_text (str): raw verse line

            Returns:
                str: Processed verse line as sequence of syllables with line parameters
            """
            ending = raw_text[-1] if raw_text[-1] in [',','.','!','?'] else ''
            syllables = SyllableMaker.syllabify(raw_text)
            #verse_marks = VersologicalMaker.verse_segmnent(raw_text)
            num_str = f"{sum(map(len, syllables))} # " if self.params['prompt_len'] else ""
            verse_end = f"{''.join(syllables[-1][-2:])} # " if self.params['prompt_end'] else ""
            metre_txt = f"{metre} # "
            return  metre_txt+ num_str + verse_end + "  ".join([" ".join(syl) for syl in syllables]) + ending
        
        def _construct_verse_marks_line(self, raw_text, metre):
            """Construct individual content line as sequence of syllables

            Args:
                raw_text (str): raw verse line

            Returns:
                str: Processed verse line as sequence of syllables with line parameters
            """
            ending = raw_text[-1] if raw_text[-1] in [',','.','!','?'] else ''
            #syllables = SyllableMaker.syllabify(raw_text)
            verse_marks = VersologicalMaker.verse_segmnent(raw_text)
            num_str = f"{sum(map(len, verse_marks))} # " if self.params['prompt_len'] else ""
            verse_end = f"{''.join(verse_marks[-1][-2:])} # " if self.params['prompt_end'] else ""
            metre_txt = f"{metre} # "
            return  metre_txt+ num_str + verse_end + "  ".join([" ".join(syl) for syl in verse_marks]) + ending
                     
            
        def data_text_line_gen(self):
            """Preprocess and process data for usage
            """
            i = 0
            for step,file in enumerate(self.gen_files()):
                if step % 100 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                for data_line in datum:
                    for part_line in data_line['body']:
                        if i in self.relevant_indexes:
                            for text_line in part_line:
                                metre = StropheParams.METER_TRANSLATE.get(text_line["metre"][0]["type"], "N")
                                scanned_text = TextManipulation._remove_most_nonchar(text_line['text'], self.params['lower_case'])
                            
                                constructed_line = self.line_constructor(scanned_text, metre)

                                num_vowels, verse_end = self._vowels_and_endings(scanned_text)
                        
                                self.data.append({
                                "input_ids" : constructed_line,
                                "nums": [num_vowels],
                                "verse_end": verse_end,
                                "metre": metre
                                     })
                        i+=1
                            
            
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
            """Customize the size of used dataset + shuffle it.

            Args:
                float_size (float, optional): Portion of new dataset. Defaults to 1.
            """
            if float_size > 1:
                print("Improper size, revert to full size")
                self.custom_size = 1
            elif float_size <= 0:
                print("Size must be positive, revert to full size")
                self.custom_size = 1
            else:
                self.custom_size = float_size
                
            np.random.shuffle(self.data)
        
    class StrophesDataset(Dataset):
        """Dataset of preprocessed strophe

        Args:
            Dataset (_type_): Dataset is child of torch class for better integration with torch and huggingface
        """
        def __init__(self, data_file_paths, dataset_parameters, segment_type, dataset_part):
            """Construct the class our given data files path and store variables

            Args:
                data_file_paths (_type_): list of paths to data files

            """
            self._data_file_paths = data_file_paths
            self.params = dataset_parameters

            self.seg_type = segment_type

            self.dataset_part = dataset_part
            self.relevant_indexes = []
            if dataset_part == 'train':
                self.relevant_indexes = self.params['train_indexes']
            elif dataset_part == 'val':
                self.relevant_indexes = self.params['val_indexes']
            elif dataset_part == 'test':
                self.relevant_indexes = self.params['test_indexes']

            self.data = []

            self.line_constructor = None
            if segment_type == 'BASE':
                self.line_constructor = self._construct_line
            elif segment_type == 'SYLLABLE':
                self.line_constructor = self._construct_syllable_line
            elif segment_type == 'VERSEMARK':
                self.line_constructor = self._construct_verse_marks_line
        
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
            verse_marks = VersologicalMaker.verse_segmnent(raw_text)
            num_str = f"{sum(map(len, syllables))} # " if self.params['prompt_len'] else ""
            verse_end = f"{''.join(verse_marks[-1][-2:])} # " if self.params['prompt_end'] else ""
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
            #verse_marks = VersologicalMaker.verse_segmnent(raw_text)
            num_str = f"{sum(map(len, syllables))} # " if self.params['prompt_len'] else ""
            verse_end = f"{''.join(syllables[-1][-2:])} # " if self.params['prompt_end'] else ""
            metre_txt = f"{metre} # "
            return metre_txt + num_str + verse_end + "  ".join([" ".join(syl) for syl in syllables]) + ending
        
        def _construct_verse_marks_line(self, raw_text, metre):
            """Construct individual content line as sequence of verse marks

            Args:
                raw_text (str): raw verse line

            Returns:
                str: Processed verse line as sequence of verse marks with line parameters
            """
            ending = raw_text[-1] if raw_text[-1] in [',','.','!','?'] else ''
            #syllables = SyllableMaker.syllabify(raw_text)
            verse_marks = VersologicalMaker.verse_segmnent(raw_text)
            num_str = f"{sum(map(len, verse_marks))} # " if self.params['prompt_len'] else ""
            verse_end = f"{''.join(verse_marks[-1][-2:])} # " if self.params['prompt_end'] else ""
            metre_txt = f"{metre} # "
            return metre_txt + num_str + verse_end + "  ".join([" ".join(syl) for syl in verse_marks]) + ending
            
            
                                                           
        def data_body_gen(self):
            """Preprocess and process data for usage
            """
            i=0
            for step,file in enumerate(self.gen_files()):
                if step % 100 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                
                for data_line in datum:

                    publish_year_text = TextManipulation._year_bucketor(data_line["biblio"]["year"])
                    publish_year_true = data_line["biblio"]["year"] if TextAnalysis._is_year(data_line["biblio"]["year"]) else 'NaN'

                    for part_line in data_line['body']:    
                        if i in self.relevant_indexes:                                                    
                            body = []
                            rhyme= []
                            metres = []
                            j = 0
                            for text_line in part_line:
                                # In rare cases multiple, but from searching only 1 metre per line
                                metre = StropheParams.METER_TRANSLATE.get(text_line["metre"][0]["type"], "J")
                                metres +=  [metre]
                                rhyme.append(text_line["rhyme"])  
                                scanned_text = TextManipulation._remove_most_nonchar(text_line["text"], self.params['lower_case'])
                                body.append(self.line_constructor(scanned_text,metre))
                                j+=1
                                if j in self.params['verse_lenght']:
                                    rhyme_str = TextManipulation._rhyme_string(rhyme)
                                    constructed_strophe = f"# {rhyme_str} # {publish_year_text}\n" + "\n".join(body) + "\n"
                                    self.data.append({
                                        "input_ids" : constructed_strophe,
                                        "context_ids" : "NO CONTEXT",
                                        "year": publish_year_true,
                                        "rhyme":  rhyme_str,
                                        "metre_ids" : metres.copy()
                                        })
                                    if j == max(self.params['verse_lenght']):
                                        body = []
                                        rhyme = []
                                        metres = []
                                        j=0
                        i +=1                     
        
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
            
            np.random.shuffle(self.data)
        
       
    @staticmethod
    def collate(batch, tokenizer: PreTrainedTokenizerBase ,max_len = 1024, max_context = 1024, format: str = 'METER_VERSE'):
        """Process data for usage in LM

        Args:
            batch (_type_): Batch with selected data points
            tokenizer (PreTrainedTokenizerBase): tokenizer to tokenize input text
            max_len (int, optional): Maximum length of tokenization. Defaults to 1024.
            max_context (int, optional): Maximum length of tokenization of context. Defaults to 1024.
            mask_rate (float, optional): Rate in with to mask data. Defaults to 0.0.

        Returns:
            dict: tokenized and processed to tensors data
        """
        
        tokenizer.model_max_length = max_len
        if batch[0]['input_ids'].startswith("#"):
            
            data = [text['input_ids'] for text in batch]
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
                data = [text['input_ids'] + tokenizer.eos_token for text in batch]
                 
            tokenized = tokenizer(data,return_tensors='pt', truncation=True, padding=True)
            input_ids = tokenized['input_ids']
            attention = tokenized["attention_mask"]
        
        else:
            tokenized = tokenizer([text['input_ids'] + tokenizer.eos_token for text in batch],return_tensors='pt', truncation=True, padding=True)
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
        """Process data for usage in distilled training

        Args:
            batch (_type_): Batch with selected data points
            tokenizer (PreTrainedTokenizerBase):  tokenizer to tokenize input text
            surrogate_model (PreTrainedModel, optional): Model to base the hidden layers data. Defaults to None.
            surrogate_model_device (_type_, optional): Device to compute the hidden layer data on. Defaults to None.
            max_len (int, optional): Maximum length of tokenization. Defaults to 1024.

        Returns:
            dict: tokenized and processed to tensors data
        """
        tokenizer.model_max_length = max_len
        tokenized = tokenizer([text['input_ids'] + tokenizer.eos_token for text in batch], return_tensors='pt', truncation=True, padding=True)
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
    def collate_validator(batch, tokenizer: PreTrainedTokenizerBase, make_syllables:bool = False, make_verse_marks:bool = False, max_len = 512):
        """Collate for Validator Training

        Args:
            batch (_type_): Batch with selected data points
            tokenizer (PreTrainedTokenizerBase): tokenizer to tokenize input text   
            make_syllables (bool): Make syllables to use in model (If data is not syllabic). Defaults to False
            make_verse_marks(bool): Make Verse Marks to use in model (If data is not verse marks). Defaults to False
            max_len (int, optional): Maximum length of tokenization. Defaults to 512.

        Returns:
            dict: tokenized and processed to tensors data
        """
        
        if make_syllables and make_verse_marks:
            raise RuntimeError("Can't both make syllables and make verse marks. Unset one to False")
        
        reform = None
        if make_syllables:
            reform = SyllableMaker.syllabify
        
        if make_verse_marks:
            reform = VersologicalMaker.verse_segmnent
        
        tokenizer.model_max_length = max_len
        data_ids = ["\n".join(
            ["  ".join(
                   [" ".join(syl) for syl in reform(line.split('#')[-1])]
                ) + (line[-1] if line[-1] in [',','.','!','?'] else '') if ( (make_syllables or make_verse_marks) and line) else line.split('#')[-1] for line in text['input_ids'].splitlines()[1:]] 
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
    def collate_meter(batch, tokenizer: PreTrainedTokenizerBase, make_syllables:bool = False, make_verse_marks:bool = False, max_len = 512):
        """Collate for Isolated Meter Training

        Args:
            batch (_type_): Batch with selected data points
            tokenizer (PreTrainedTokenizerBase): tokenizer to tokenize input text   
            make_syllables (bool): Make syllables to use in model (If data is not syllabic). Defaults to False
            make_verse_marks(bool): Make Verse Marks to use in model (If data is not verse marks). Defaults to False
            max_len (int, optional): Maximum length of tokenization. Defaults to 512.

        Returns:
            dict: tokenized and processed to tensors data
        """
        
        tokenizer.model_max_length = max_len
        data_ids = []
        metre = []
        
        if make_syllables and make_verse_marks:
            raise RuntimeError("Can't both make syllables and make verse marks. Unset one to False")
        
        reform = None
        if make_syllables:
            reform = SyllableMaker.syllabify
        
        if make_verse_marks:
            reform = VersologicalMaker.verse_segmnent
          
        for datum in batch:
            data_ids += [
                    "  ".join(
                    [" ".join(syl) for syl in reform(line.split('#')[-1])]
                ) + (line[-1] if line[-1] in [',','.','!','?'] else '') if ( (make_syllables or make_verse_marks) and line) else line.split('#')[-1] for line in datum['input_ids'].splitlines()[1:]
                ]
            if "metre_ids" in datum.keys():
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
    def collate_meter_context(batch, tokenizer: PreTrainedTokenizerBase, make_syllables:bool = False, make_verse_marks:bool = False, max_len = 512):
        """Collate for Context Meter Training

        Args:
            batch (_type_): Batch with selected data points
            tokenizer (PreTrainedTokenizerBase): tokenizer to tokenize input text   
            make_syllables (bool): Make syllables to use in model (If data is not syllabic). Defaults to False
            make_verse_marks(bool): Make Verse Marks to use in model (If data is not verse marks). Defaults to False
            max_len (int, optional): Maximum length of tokenization. Defaults to 512.

        Returns:
            dict: tokenized and processed to tensors data
        """
        
        tokenizer.model_max_length = max_len
        data_ids = []
        
        if make_syllables and make_verse_marks:
            raise RuntimeError("Can't both make syllables and make verse marks. Unset one to False")
        
        reform = None
        if make_syllables:
            reform = SyllableMaker.syllabify
        
        if make_verse_marks:
            reform = VersologicalMaker.verse_segmnent
        
        metre = []
        for datum in batch:
            base_datums = [
                    "  ".join(
                    [" ".join(syl) for syl in reform(line.split('#')[-1])]
                ) + (line[-1] if line[-1] in [',','.','!','?'] else '') if (( make_syllables or make_verse_marks ) and line) else line.split('#')[-1] for line in datum['input_ids'].splitlines()[1:]
                ]
            i = 0
            for i in range(len(base_datums)):
                data_ids.append(
                    "\n".join(base_datums[:i] + ['# ' + base_datums[i]] + base_datums[i+1:])
                    ) 
            if "metre_ids" in datum.keys():
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
            
        self.raw_dataset = CorpusDatasetPytorch.RawDataset(filenames, self.dataset_parameters['lower_case'])
    
    def load_json_filenames(self, dataset_parameters, segmentation_type):
        """Load and process datasets

        Args:
            dataset_parameters (_type_): Dataset parameters
            segmentation_type (_type_): Segementation to be done
        """
        
        filenames = self.get_filenames()

        # Parse Train Data
        
        self.train_strophes = CorpusDatasetPytorch.StrophesDataset(filenames,dataset_parameters, segment_type= segmentation_type, dataset_part='train')
        self.train_strophes.data_body_gen()
         
        self.train_verses = CorpusDatasetPytorch.VersesDataset(filenames,dataset_parameters, segment_type= segmentation_type, dataset_part='train')
        self.train_verses.data_text_line_gen()

        # Parse Val Data
        
        self.val_strophes = CorpusDatasetPytorch.StrophesDataset(filenames,dataset_parameters, segment_type= segmentation_type, dataset_part='val')
        self.val_strophes.data_body_gen()

        self.val_verses = CorpusDatasetPytorch.VersesDataset(filenames,dataset_parameters, segment_type= segmentation_type, dataset_part='val')
        self.val_verses.data_text_line_gen()
        
        # Parse Test Data
        
        self.test_strophes = CorpusDatasetPytorch.StrophesDataset(filenames,dataset_parameters, segment_type= segmentation_type, dataset_part='test')
        self.test_strophes.data_body_gen()

        self.test_verses = CorpusDatasetPytorch.VersesDataset(filenames,dataset_parameters, segment_type= segmentation_type, dataset_part='test')
        self.test_verses.data_text_line_gen()
        
        sub_dir = os.path.join(dataset_parameters['cache_dir'], segmentation_type)

        json.dump(self.train_strophes.data, open( os.path.join(sub_dir, "TRAIN_STOPHES.json"), 'w+'), indent = 6)
        json.dump(self.train_verses.data, open( os.path.join(sub_dir, "TRAIN_VERSES.json"), 'w+'), indent = 6)

        json.dump(self.val_strophes.data, open( os.path.join(sub_dir, "VAL_STOPHES.json"), 'w+'), indent = 6)
        json.dump(self.val_verses.data, open( os.path.join(sub_dir, "VAL_VERSES.json"), 'w+'), indent = 6)

        json.dump(self.test_strophes.data, open( os.path.join(sub_dir, "TEST_STOPHES.json"), 'w+'), indent = 6)
        json.dump(self.test_verses.data, open( os.path.join(sub_dir, "TEST_VERSES.json"), 'w+'), indent = 6)
              
        
    def create_config(self, cache_directory, 
                      prompt_length, prompt_ending ,prompt_verse,
                      verse_lenghts,lower_case,
                      validation_data_rate,test_data_rate):
        
        dataset_params = {'prompt_len' : prompt_length,
                          'prompt_end' : prompt_ending,
                          'prompt_verse' : prompt_verse,
                          'verse_lenght': [verse_lenghts] if not type(verse_lenghts) is list else verse_lenghts,
                          'lower_case': lower_case,
                          'train_indexes':[],
                          'val_indexes': [],
                          'test_indexes': [],
                          'cache_dir' : cache_directory}
        os.makedirs(cache_directory, exist_ok=True )
        os.makedirs(os.path.join(cache_directory, 'BASE'), exist_ok=True )
        os.makedirs(os.path.join(cache_directory, 'SYLLABLE'), exist_ok=True )
        os.makedirs(os.path.join(cache_directory, 'VERSEMARK'), exist_ok=True )

        i = 0
        for filename in self.get_filenames():
            file = open(filename, 'r')
            datum = json.load(file)
            for poem in datum:
                for strophe in poem['body']:
                    rand_split = np.random.rand()
                    if rand_split > validation_data_rate + test_data_rate:
                        dataset_params['train_indexes'].append(i)
                    elif rand_split < validation_data_rate:
                        dataset_params['val_indexes'].append(i)
                    else:
                        dataset_params['test_indexes'].append(i)
                    i+=1

        json.dump(dataset_params, open(os.path.join(os.path.dirname(__file__), 'config.json'), 'w+'), indent=6)

        return dataset_params
    
    def check_file_existence(self, dataset_parameters, segmentation_type):
        sub_dir = os.path.join(dataset_parameters['cache_dir'], segmentation_type)
        return  os.path.exists(os.path.join(sub_dir, 'TRAIN_STOPHES.json')) and os.path.exists(os.path.join(sub_dir, 'TRAIN_VERSES.json'))  and \
                os.path.exists(os.path.join(sub_dir, 'VAL_STOPHES.json')) and os.path.exists(os.path.join(sub_dir, 'VAL_VERSES.json')) and \
                os.path.exists(os.path.join(sub_dir, 'TEST_STOPHES.json')) and os.path.exists(os.path.join(sub_dir, 'TEST_VERSES.json'))
    
    def load_cached(self, dataset_parameters, segmentation_type):
        self.train_strophes = CorpusDatasetPytorch.StrophesDataset([], dataset_parameters =dataset_parameters, segment_type =segmentation_type, dataset_part='train' )
        self.train_verses = CorpusDatasetPytorch.VersesDataset([], dataset_parameters =dataset_parameters, segment_type =segmentation_type, dataset_part='train'  )

        self.val_strophes = CorpusDatasetPytorch.StrophesDataset([], dataset_parameters =dataset_parameters, segment_type =segmentation_type, dataset_part='val'  )
        self.val_verses = CorpusDatasetPytorch.VersesDataset([], dataset_parameters =dataset_parameters, segment_type =segmentation_type, dataset_part='val'  )

        self.test_strophes = CorpusDatasetPytorch.StrophesDataset([], dataset_parameters =dataset_parameters, segment_type =segmentation_type, dataset_part='test'  )
        self.test_verses = CorpusDatasetPytorch.VersesDataset([], dataset_parameters =dataset_parameters, segment_type =segmentation_type, dataset_part='test'  )

        sub_dir = os.path.join(dataset_parameters['cache_dir'], segmentation_type)

        self.train_strophes.data =json.load( open( os.path.join(sub_dir, "TRAIN_STOPHES.json"), 'r'))
        self.train_verses.data =json.load( open( os.path.join(sub_dir, "TRAIN_VERSES.json"), 'r'))

        self.val_strophes.data =json.load( open( os.path.join(sub_dir, "VAL_STOPHES.json"), 'r'))
        self.val_verses.data =json.load( open( os.path.join(sub_dir, "VAL_VERSES.json"), 'r'))

        self.test_strophes.data =json.load( open( os.path.join(sub_dir, "TEST_STOPHES.json"), 'r'))
        self.test_verses.data =json.load( open( os.path.join(sub_dir, "TEST_VERSES.json"), 'r'))

        
    def __init__(self, SEGMENT_TYPE, data_dir = "PoetGen\corpusCzechVerse-master\ccv", cache_dir=os.path.join(os.path.dirname(__file__), 'ProcessedData'), 
                 prompt_length=True, prompt_ending=True, prompt_verse=True, verse_len=[4,6], lower_case=True, val_data_rate=0.05, test_data_rate=0.05):
        """Create Dataset with specified segementatiom

        Args:
            SEGMENT_TYPE (_type_): Segmentation type. Choose from BASE, SYLLABLE, VERSEMARK
            data_dir (str, optional): path to uprocessed data. Defaults to "PoetGen\corpusCzechVerse-master\ccv".
            cache_dir (_type_, optional): where to store processed data. Defaults to os.path.join(os.path.dirname(__file__), 'ProcessedData').
            prompt_length (bool, optional): if to prompt length. Defaults to True.
            prompt_ending (bool, optional): if to prompt ending. Defaults to True.
            prompt_verse (bool, optional): if to prompt verses. Defaults to True.
            verse_len (list, optional): chosen strophe lengths. Defaults to [4,6].
            lower_case (bool, optional): if to use lowercase. Defaults to True.
            val_data_rate (float, optional): size of validation data. Defaults to 0.05.
            test_data_rate (float, optional): size of test data. Defaults to 0.05.
        """
        
        self.data_dir = data_dir
        self.SEGMENT_TYPE =SEGMENT_TYPE
        if os.path.isfile( os.path.join(os.path.dirname(__file__), 'config.json')):
            self.dataset_parameters =  json.load(open(os.path.join(os.path.dirname(__file__), 'config.json'), 'r'))
        else:
            self.dataset_parameters = self.create_config(cache_directory = cache_dir, 
                               prompt_length = prompt_length,
                               prompt_ending = prompt_ending,
                               prompt_verse = prompt_verse,
                               verse_lenghts = verse_len,
                               lower_case = lower_case,
                               validation_data_rate = val_data_rate,
                               test_data_rate = test_data_rate)

        if  self.check_file_existence(self.dataset_parameters, self.SEGMENT_TYPE):
            self.load_cached(self.dataset_parameters, self.SEGMENT_TYPE)
            
        else:
            self.load_json_filenames(self.dataset_parameters, self.SEGMENT_TYPE)
            
            
        self.load_raw_()
        
        
        
#if __name__ == "__main__":
# Line Count
#    print(len(list(CorpusDatasetPytorch(os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")) ).raw_dataset.get_text())))
# Strophe Count
#    print(len(list(CorpusDatasetPytorch(os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")) ).raw_dataset.get_part())))
# Poem Count
#    print(len(list(CorpusDatasetPytorch(os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")) ).raw_dataset.get_body())))