import os
import json
import numpy as np
import torch

from utils.poet_utils import StropheParams,  TextAnalysis, TextManipulation, SyllableMaker, VersologicalMaker, Tokens
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, PreTrainedModel
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
                 
        def get_verses(self):
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
                            
        def get_strophes(self):
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
        
        def get_poems(self):
            """Get whole poem

            Yields:
                str: 1 whole poem
            """
            for step,file in enumerate(self.gen_files()):
                if step % 100 == 0:
                    print(f"Processing file {step}")
                datum = json.load(file)
                for data_line in datum:
                    body = [ f"{data_line['biblio']['p_title']}:" ]
                    
                    for part_line in data_line['body']:
                        
                        for text_line in part_line:
                            body.append(text_line['text'])
                        body.append("\n")
                    yield "\n".join(body).lower() if self.lower_case else "\n".join(body)
    
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
                 
                     
        
        def _construct_line(self, raw_text):
            """Construct individual content line

            Args:
                raw_text (str): raw verse line

            Returns:
                str: Processed verse line with line parameters
            """
            syllables = SyllableMaker.syllabify(" ".join(raw_text.split()[-4:]))
            verse_marks = VersologicalMaker.verse_segmnent(''.join([''.join(word) for word in syllables[-2:]]))
            
            if len(syllables[-1]) == 1 and TextAnalysis.is_consonant(verse_marks[-1][-1][-1]) and (len(syllables) == 1 or len(syllables[-2]) > 1):
                verse_end = f"{ TextManipulation._shortify(''.join(verse_marks[-1][-1:])) } # "
            else:
                verse_end = f"{ TextManipulation._shortify(''.join(verse_marks[-1][-2:])) } # "
            return  verse_end  + raw_text
        
        def _construct_syllable_line(self, raw_text):
            """Construct individual content line as sequence of syllables

            Args:
                raw_text (str): raw verse line

            Returns:
                str: Processed verse line as sequence of syllables with line parameters
            """
            ending = raw_text[-1] if raw_text[-1] in [',','.','!','?'] else ''
            syllables = SyllableMaker.syllabify(raw_text)
            
            verse_end = f"{ TextManipulation._shortify(''.join(syllables[-1][-2:])) } # "
            return  verse_end + "  ".join([" ".join(syl) for syl in syllables]) + ending
        
        def _construct_verse_marks_line(self, raw_text):
            """Construct individual content line as sequence of verse marks

            Args:
                raw_text (str): raw verse line

            Returns:
                str: Processed verse line as sequence of verse marks with line parameters
            """
            ending = raw_text[-1] if raw_text[-1] in [',','.','!','?'] else ''
            #syllables = SyllableMaker.syllabify(raw_text)
            verse_marks = VersologicalMaker.verse_segmnent(raw_text)
            verse_end = f"{ TextManipulation._shortify(''.join(verse_marks[-1][-2:])) } # " 
            return  verse_end + "  ".join([" ".join(syl) for syl in verse_marks]) + ending
        
        def _create_header(self, author, title, year):
            """Create header for strophe

            Args:
                author (str): author of the strophe
                title (str): title of the strophe
                year (str): year of the strophe

            Returns:
                str: header of the strophe
            """
            return f"{Tokens.AUTHOR} {author}\n{Tokens.TITLE} {title}\n{Tokens.YEAR} {year}\n"
        
        def _format_strophe(self, meter:str, rhyme: str, verses: list):
            return f"{Tokens.STROPHE_START}\n{Tokens.METER} {meter}\n{Tokens.RHYME} {rhyme}\n" + "\n".join(verses) + f"\n{Tokens.STROPHE_END}\n"
                                                           
        def data_body_gen(self):
            """Preprocess and process data for usage
            """
            i=0
            for step,file in enumerate(self.gen_files()):
                if step % 100 == 0:
                    print(f"Processing file {step}")
                    
                datum = json.load(file)
                # Check if file in proper indexes
                if i in self.relevant_indexes:
                    for data_line in datum:

                        publish_year_text = TextManipulation._year_bucketor(data_line["biblio"]["year"])
                        publish_year_true = data_line["biblio"]["year"] if TextAnalysis._is_year(data_line["biblio"]["year"]) else 'NaN'
                        author = data_line["p_author"]["name"] if "p_author" in data_line.keys() else (data_line["b_author"]["name"] if "b_author" in data_line.keys() else "Unknown")

                        poem_header = self._create_header(author, data_line["biblio"]["p_title"], publish_year_text)
                        previous_strophe = ""

                        for part_line in data_line['body']:                                                     
                            body = []
                            rhyme= []
                            metres = []

                            for text_line in part_line:
                                # In rare cases multiple, but from searching only 1 metre per line
                                metres.append(StropheParams.METER_TRANSLATE.get(text_line["metre"][0]["type"], "J"))
                                rhyme.append(text_line["rhyme"])  
                                scanned_text = TextManipulation._remove_most_nonchar(text_line["text"], self.params['lower_case'])
                                body.append(self.line_constructor(scanned_text))

                            rhyme_str = TextManipulation._rhyme_string(rhyme)
                            meter = max(set(metres), key=metres.count)
                            current_strophe = self._format_strophe(meter, rhyme_str, body)
                            constructed_strophe = poem_header + previous_strophe + current_strophe  
                            self.data.append({
                                    "input_ids" : constructed_strophe,
                                    "context_ids" : "None",
                                    "year": publish_year_true,
                                    "rhyme":  rhyme_str,
                                    "metre_ids" : metres.copy()
                                    })     

                            previous_strophe = current_strophe
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
            verse_end = torch.tensor(np.asarray([CorpusDatasetPytorch.VersesDataset._ending_vector(text["verse_end"]) for text in batch], dtype=np.int32), dtype=torch.float32)
        
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
            "labels": input_ids.type(torch.LongTensor),
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
            "labels": input_ids.type(torch.LongTensor),
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
            "labels": input_ids.type(torch.LongTensor),
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
         

        # Parse Val Data
        
        self.val_strophes = CorpusDatasetPytorch.StrophesDataset(filenames,dataset_parameters, segment_type= segmentation_type, dataset_part='val')
        self.val_strophes.data_body_gen()

        
        # Parse Test Data
        
        self.test_strophes = CorpusDatasetPytorch.StrophesDataset(filenames,dataset_parameters, segment_type= segmentation_type, dataset_part='test')
        self.test_strophes.data_body_gen()

        
        sub_dir = os.path.join(dataset_parameters['cache_dir'], segmentation_type)

        json.dump(self.train_strophes.data, open( os.path.join(sub_dir, "TRAIN_STOPHES.json"), 'w+'), indent = 6)

        json.dump(self.val_strophes.data, open( os.path.join(sub_dir, "VAL_STOPHES.json"), 'w+'), indent = 6)

        json.dump(self.test_strophes.data, open( os.path.join(sub_dir, "TEST_STOPHES.json"), 'w+'), indent = 6)
              
        
    def create_config(self, cache_directory, lower_case,
                      validation_data_rate,test_data_rate):
        
        dataset_params = {'lower_case': lower_case,
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
        return  os.path.exists(os.path.join(sub_dir, 'TRAIN_STOPHES.json')) and  \
                os.path.exists(os.path.join(sub_dir, 'VAL_STOPHES.json')) and  \
                os.path.exists(os.path.join(sub_dir, 'TEST_STOPHES.json'))
    
    def load_cached(self, dataset_parameters, segmentation_type):
        self.train_strophes = CorpusDatasetPytorch.StrophesDataset([], dataset_parameters =dataset_parameters, segment_type =segmentation_type, dataset_part='train' )

        self.val_strophes = CorpusDatasetPytorch.StrophesDataset([], dataset_parameters =dataset_parameters, segment_type =segmentation_type, dataset_part='val'  )

        self.test_strophes = CorpusDatasetPytorch.StrophesDataset([], dataset_parameters =dataset_parameters, segment_type =segmentation_type, dataset_part='test'  )

        sub_dir = os.path.join(dataset_parameters['cache_dir'], segmentation_type)

        self.train_strophes.data =json.load( open( os.path.join(sub_dir, "TRAIN_STOPHES.json"), 'r'))

        self.val_strophes.data =json.load( open( os.path.join(sub_dir, "VAL_STOPHES.json"), 'r'))

        self.test_strophes.data =json.load( open( os.path.join(sub_dir, "TEST_STOPHES.json"), 'r'))

        
    def __init__(self, SEGMENT_TYPE, data_dir = "PoetGen\corpusCzechVerse-master\ccv", cache_dir=os.path.join(os.path.dirname(__file__), 'ProcessedData'), 
                  lower_case=True, val_data_rate=0.05, test_data_rate=0.05):
        """Create Dataset with specified segementatiom

        Args:
            SEGMENT_TYPE (_type_): Segmentation type. Choose from BASE, SYLLABLE, VERSEMARK
            data_dir (str, optional): path to uprocessed data. Defaults to "PoetGen\corpusCzechVerse-master\ccv".
            cache_dir (_type_, optional): where to store processed data. Defaults to os.path.join(os.path.dirname(__file__), 'ProcessedData').
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