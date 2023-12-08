import torch
import os
import random
import re
import argparse  
import numpy as np

from tqdm import tqdm
from transformers import  AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizerBase, AutoModelForCausalLM
from utils.poet_utils import RHYME_SCHEMES, TextAnalysis, TextManipulation, UNK, EOS, PAD, NORMAL_SCHEMES, SyllableMaker
from utils.poet_model_utils import PoetModelInterface
from utils.validators import ValidatorInterface

from poet_model_base_lm import PoetModelBase

from corpus_capsulated_datasets import CorpusDatasetPytorch

#TODO: Add Year Validator to the mix

class ModelValidator:
    """Class to Validate LMs using Validators and Analysis
    """
    def __init__(self, args: argparse.Namespace,
                 result_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__),"results"))) -> None:
        """Construct Validators using given arguments. Save the requested number of repeats

        Args:
            args (argparse.Namespace): Arguments of Validation
            result_dir (str, optional): Directory to store results. Defaults to os.path.abspath(os.path.join(os.path.dirname(__file__),"results")).
        """
        
        self.args = args
        
        self.model_name = args.model_path_full
        # Split Path to find only the LM name itself
        _ ,self.model_rel_name =  os.path.split(self.model_name)
        # Load Model as Pickle file or as stored LM 
        if "_LM" in self.model_rel_name:
            self.model_rel_name = re.sub("_LM", "", self.model_rel_name)
            self.model: PoetModelInterface = PoetModelBase(self.model_name)
        else:
            self.model: PoetModelInterface= (torch.load(self.model_name, map_location=torch.device('cpu')))
        self.model.eval()
        
        # Load validators 
        self.rhyme_model, self.meter_model, self.year_model = None, None, None
        self.rhyme_model_name, self.meter_model_name, self.year_model_name = "", "", ""
        if args.rhyme_model_path_full:
            self.rhyme_model: ValidatorInterface = (torch.load(args.rhyme_model_path_full, map_location=torch.device('cpu')))
            self.rhyme_model.eval()
            _,  self.rhyme_model_name = os.path.split(args.rhyme_model_path_full)
        
        if args.metre_model_path_full:
            self.meter_model: ValidatorInterface = (torch.load(args.metre_model_path_full, map_location=torch.device('cpu')))
            self.meter_model.eval()
            _, self.meter_model_name = os.path.split(args.metre_model_path_full)
            
        if args.year_model_path_full:
            self.year_model: ValidatorInterface = (torch.load(args.year_model_path_full, map_location=torch.device('cpu')))
            self.year_model.eval()
            _,  self.year_model_name = os.path.split(args.year_model_path_full)
            

            
        # Load Rhyme tokenizer
        self.validator_tokenizer_rhyme: PreTrainedTokenizerBase = None
        if args.validator_tokenizer_model_rhyme:
            try:
                self.validator_tokenizer_rhyme = AutoTokenizer.from_pretrained(args.validator_tokenizer_model_rhyme)
            except:
                self.validator_tokenizer_rhyme: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=args.validator_tokenizer_model_rhyme)
                self.validator_tokenizer_rhyme.eos_token = EOS
                self.validator_tokenizer_rhyme.eos_token_id = 0
                self.validator_tokenizer_rhyme.pad_token = PAD
                self.validator_tokenizer_rhyme.pad_token_id = 1
                self.validator_tokenizer_rhyme.unk_token = UNK
                self.validator_tokenizer_rhyme.unk_token_id = 2
                
        # Load Meter tokenizer
        self.validator_tokenizer_meter: PreTrainedTokenizerBase = None
        if args.validator_tokenizer_model_meter:
            try:
                self.validator_tokenizer_meter = AutoTokenizer.from_pretrained(args.validator_tokenizer_model_meter)
            except:
                self.validator_tokenizer_meter: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=args.validator_tokenizer_model_meter)
                self.validator_tokenizer_meter.eos_token = EOS
                self.validator_tokenizer_meter.eos_token_id = 0
                self.validator_tokenizer_meter.pad_token = PAD
                self.validator_tokenizer_meter.pad_token_id = 1
                self.validator_tokenizer_meter.unk_token = UNK
                self.validator_tokenizer_meter.unk_token_id = 2
                
        # Load Year tokenizer
        self.validator_tokenizer_year: PreTrainedTokenizerBase = None
        if args.validator_tokenizer_model_year:
            try:
                self.validator_tokenizer_year = AutoTokenizer.from_pretrained(args.validator_tokenizer_model_year)
            except:
                self.validator_tokenizer_year: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=args.validator_tokenizer_model_year)
                self.validator_tokenizer_year.eos_token = EOS
                self.validator_tokenizer_year.eos_token_id = 0
                self.validator_tokenizer_year.pad_token = PAD
                self.validator_tokenizer_year.pad_token_id = 1
                self.validator_tokenizer_year.unk_token = UNK
                self.validator_tokenizer_year.unk_token_id = 2
         
        # Load LM tokenizers       
        try:    
            self.tokenizer: PreTrainedTokenizerBase =  AutoTokenizer.from_pretrained(self.model_name)
        except:
            self.tokenizer: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=args.backup_tokenizer_model)
            self.tokenizer.eos_token = EOS
            self.tokenizer.eos_token_id = 0
            self.tokenizer.pad_token = PAD
            self.tokenizer.pad_token_id = 1
            self.tokenizer.unk_token = UNK
            self.tokenizer.unk_token_id = 2
            
        self.dataset = CorpusDatasetPytorch(data_dir=args.data_path_poet)
        self.validation_data = self.dataset.pytorch_dataset_body.validation_data
        
        # Store the Validation arguments  
        self.epochs = args.num_runs
        self.runs_per_epoch = args.num_samples
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            
            
    def decode_helper(self, type:str, index:int):
        """Wrapper around LM generation

        Args:
            type (str): Which type of generation to use ('BASIC', 'FORCED')

        Returns:
            str: Generated Strophe
        """
        
        
        
        
        if type  == "BASIC":
            # Up to first meter
            FORMAT = "METER_VERSE"
            if self.model_rel_name.startswith('CZ') or self.model_rel_name.startswith('ALT') or self.model_rel_name.startswith('EN') or self.model_rel_name.startswith('ENALT'):
                start = f"# {self.validation_data[index]['rhyme']} # {TextManipulation._year_bucketor(self.validation_data[index]['year'])}\n{self.validation_data[index]['metre_ids'][0]}"
                FORMAT = "METER_VERSE"
            elif self.model_rel_name.startswith('gpt'):
                start = f"# {self.validation_data[index]['rhyme']} # {TextManipulation._year_bucketor(self.validation_data[index]['year'])} # {self.validation_data[index]['metre_ids'][0]}"
                FORMAT = "BASIC"
            else:
                start = f"# {self.validation_data[index]['rhyme']} # {TextManipulation._year_bucketor(self.validation_data[index]['year'])} # {self.validation_data[index]['metre_ids'][0]}"
                FORMAT = "VERSE_PAR"
            tokenized_poet_start = self.tokenizer.encode(start, return_tensors='pt', truncation=True)
            if self.args.sample:
                out = self.model.model.generate(tokenized_poet_start, 
                                        max_length=256,
                                        do_sample=True,
                                        top_k=50,
                                        eos_token_id = self.tokenizer.eos_token_id,
                                        early_stopping=True,
                                        pad_token_id=self.tokenizer.pad_token_id)
                
            else:
                out = self.model.model.generate(tokenized_poet_start, 
                                        max_length=256,
                                        num_beams=8,
                                        no_repeat_ngram_size=2,
                                        eos_token_id = self.tokenizer.eos_token_id,
                                        early_stopping=True,
                                        pad_token_id=self.tokenizer.pad_token_id)
                
            return self.tokenizer.decode(out[0], skip_special_tokens=True)
        if type == "FORCED":
            
            if self.model_rel_name.startswith('CZ') or self.model_rel_name.startswith('ALT') or self.model_rel_name.startswith('EN') or self.model_rel_name.startswith('ENALT'):
                FORMAT = "METER_VERSE"
            elif self.model_rel_name.startswith('gpt'):
                FORMAT = "BASIC"
            else:
                FORMAT = "VERSE_PAR"
            
            start_forced = {'RHYME': self.validation_data[index]['rhyme'],
                            'YEAR': TextManipulation._year_bucketor(self.validation_data[index]['year'])}
            if self.model_rel_name.startswith('CZ') or self.model_rel_name.startswith('ALT') or self.model_rel_name.startswith('EN') or self.model_rel_name.startswith('ENALT'):
                for ch, id in zip(self.validation_data[index]['rhyme'], self.validation_data[index]['metre_ids']):
                    if ch == 'A':
                        start_forced['METER_0'] = id
                    elif ch == 'B':
                        start_forced['METER_1'] = id
                    elif ch == 'C':
                        start_forced['METER_2'] = id
            else:
                start_forced['STROPHE_METER'] = self.validation_data[index]['metre_ids'][0]
            
            return self.model.generate_forced(start_forced, self.tokenizer, verse_len= len(self.validation_data[index]['rhyme']), sample=self.args.sample, format=FORMAT)
            
            
            
    def validate_decoding(self, type:str):
        """Validate LM given generation type. Measure metrics (Rhyme acc, Metrum acc, End acc, Syllable count acc)

        Args:
            type (str): Type of generation to use
        """
        # Store of individual runs of evaluation
        end_accuracy, sylab_accuracy = [], []
        rhyme_accuracy, rhyme_top_k, rhyme_label_acc, levenshtein_dist = [], [], [], []
        metre_accuracy, metre_top_k, metre_label_acc  = [], [], []
        year_accuracy, year_top_k, year_label_acc  = [], [], []
        
        syllable_running_ration = []
        # Run the requested amount of evaluations
        for _ in tqdm(range(self.epochs), desc=f"Validation {type}"):
            # Store results of current evaluation
            end_all, sylab_all  = 0,0
            rhyme_all, rhyme_top_k_all, rhyme_label_all, lev_distance_all = 0,0,0,0
            metre_all, metre_top_k_all, metre_label_all = 0,0,0
            year_all, year_top_k_all, year_label_all = 0,0,0
            
            
            end_pos, sylab_pos = 0, 0
            rhyme_pos, rhyme_top_k_pos, rhyme_label_pos, lev_distance = 0,0,0,0
            metre_pos, metre_top_k_pos, metre_label_pos  = 0,0,0
            year_pos, year_top_k_pos, year_label_pos = 0,0,0
            
            
            samples = random.choices(list(range(len(self.validation_data))), k=self.runs_per_epoch)
            # Run the requested steps in evaluation
            for i in tqdm(range(self.runs_per_epoch), leave=False):
                # Get generated Strophe
                decoded_cont:str = self.decode_helper(type,samples[i])
                # Validate line by line
                STROPHE_METER = 'J'
                for line in decoded_cont.splitlines():
                    # Skip Empty lines
                    if not line.strip(): 
                        break
                    if not (TextManipulation._remove_most_nonchar(line)).strip():
                        break
                    # Validate for Strophe Parameters
                    if TextAnalysis._is_param_line(line):
                        values = TextAnalysis._first_line_analysis(line)
                        
                        rhyme_all +=1
                        rhyme_top_k_all +=1
                        rhyme_label_all +=1
                        
                        year_all +=1
                        year_top_k_all +=1
                        year_label_all +=1
                        
                        # Validate for Rhyme schema
                        if self.rhyme_model != None and "RHYME" in values.keys():
                            data = CorpusDatasetPytorch.collate_validator([{"input_ids" :[decoded_cont], 'rhyme' : values["RHYME"]}],tokenizer=self.validator_tokenizer_rhyme,
                                                                           is_syllable=False, syllables=self.args.val_syllables_rhyme,
                                                                           max_len=self.rhyme_model.model.config.max_position_embeddings)
                            res = self.rhyme_model.validate(input_ids=data['input_ids'],
                                                                   rhyme=data['rhyme'], k=self.args.top_k)
                            rhyme_pos += res['acc']
                            rhyme_top_k_pos += res['top_k']
                            rhyme_label_pos += res['predicted_label']
                            if res['acc'] < 0.5:
                                lev_distance_all += 1
                                lev_distance +=res['lev_distance']
                            
                        
                        #Validate for Year
                        if self.year_model != None and "YEAR" in values.keys():
                            data = CorpusDatasetPytorch.collate_validator([{"input_ids" :[decoded_cont], "year": values["YEAR"]}],tokenizer=self.validator_tokenizer_year,
                                                                           is_syllable=False, syllables=self.args.val_syllables_year,
                                                                           max_len=self.year_model.model.config.max_position_embeddings)
                            res = self.year_model.validate(input_ids=data['input_ids'],
                                                                   year_bucket=data['year_bucket'],k=self.args.top_k)
                            
                            year_pos += res['acc']
                            year_top_k_pos += res['top_k']
                            year_label_pos += res['predicted_label']
                            
                        if 'STROPHE_METER' in values.keys():
                            STROPHE_METER = values['STROPHE_METER']
                                
                        
                        # Measure Syllable uniqueness
                        all_sylabs = []
                        for line in decoded_cont.splitlines()[1:]:
                            all_sylabs += SyllableMaker.syllabify(line.split("#")[-1])
                        if len(all_sylabs)  != 0:
                            syllable_running_ration.append(len(set(all_sylabs))/len(all_sylabs))
                        
                        continue
                            
                    # Else validate for individual verse
                    line_analysis = TextAnalysis._continuos_line_analysis(line)
                    # Was Still empty in terms of any text
                    if len(line_analysis.keys()) == 0:
                        continue
                    
                    # Validate for Metrum
                    metre_all +=1
                    metre_top_k_all +=1
                    metre_label_all +=1
                    if self.meter_model != None and "METER" in line_analysis.keys():
                        data = CorpusDatasetPytorch.collate_meter([{"input_ids" :["FIRST LINE SKIP!\n" + line], "metre_ids": line_analysis["METER"]}],tokenizer=self.validator_tokenizer_meter,
                                                                       is_syllable=False, syllables=self.args.val_syllables_meter,
                                                                       max_len=self.meter_model.model.config.max_position_embeddings)
                        res = self.meter_model.validate(input_ids=data['input_ids'],
                                                            metre_ids=data['metre_ids'],k=self.args.top_k)
                        
                        metre_pos += res['acc']
                        metre_top_k_pos += res['top_k']
                        metre_label_pos += res['predicted_label']
                    elif self.meter_model != None:
                        data = CorpusDatasetPytorch.collate_meter([{"input_ids" :["FIRST LINE SKIP!\n" + line], "metre_ids": STROPHE_METER}],tokenizer=self.validator_tokenizer_meter,
                                                                       is_syllable=False, syllables=self.args.val_syllables_meter,
                                                                       max_len=self.meter_model.model.config.max_position_embeddings)
                        res = self.meter_model.validate(input_ids=data['input_ids'],
                                                            metre_ids=data['metre_ids'],k=self.args.top_k)
                        
                        metre_pos += res['acc']
                        metre_top_k_pos += res['top_k']
                        metre_label_pos += res['predicted_label']
                        
                    
                    end_all += 1
                    if "END" in line_analysis.keys() and "TRUE_END" in line_analysis.keys() and line_analysis["END"] == line_analysis["TRUE_END"]:
                        end_pos +=1
                    
                    sylab_all +=1
                    if "LENGTH" in line_analysis.keys() and "TRUE_LENGTH" in line_analysis.keys() and line_analysis["LENGTH"] == line_analysis["TRUE_LENGTH"]:
                        sylab_pos +=1
                    
                    
            # Store Results        
            end_accuracy.append(0 if end_all == 0 else end_pos/end_all)
            sylab_accuracy.append(0 if sylab_all==0 else sylab_pos/sylab_all)
            
            rhyme_accuracy.append(0 if rhyme_all==0 else rhyme_pos/rhyme_all)
            rhyme_top_k.append(0 if rhyme_top_k_all == 0 else rhyme_top_k_pos/rhyme_top_k_all)
            rhyme_label_acc.append(0 if rhyme_label_all==0 else rhyme_label_pos/rhyme_label_all)
            levenshtein_dist.append(0 if lev_distance_all == 0 else lev_distance/lev_distance_all)
            
            metre_accuracy.append(0 if metre_all==0 else metre_pos/metre_all)
            metre_top_k.append(0 if metre_top_k_all==0 else  metre_top_k_pos/metre_top_k_all)
            metre_label_acc.append(0 if metre_label_all==0 else metre_label_pos/metre_label_all)
            
            year_accuracy.append(0 if year_all==0 else year_pos/year_all)
            year_top_k.append(0 if year_top_k_all==0 else year_top_k_pos/year_top_k_all)
            year_label_acc.append(0 if year_label_all==0 else year_label_pos/year_label_all)
            
            
        # Log all results and configuration
        with open(os.path.abspath(os.path.join(self.result_dir, self.model_rel_name + ".txt")), 'a') as file:
            print(f" ===== {type} Decoding Validation: Epochs: {self.epochs}, Runs per epoch: {self.runs_per_epoch}, SAMPLING: {str(self.args.sample)} =====", file=file)
            # Line Metrics
            print(f"Num Sylabs Accuracy: {np.mean(sylab_accuracy):.4f} +- {np.std(sylab_accuracy, ddof=1):.4f}", file=file)
            print(f"Endings Accuracy: {np.mean(end_accuracy):.4f} +- {np.std(end_accuracy, ddof=1):.4f}", file=file)
            print(f"Unique Syllable Ratio: {np.mean(syllable_running_ration):.4f} +- {np.std(syllable_running_ration, ddof=1):.4f}\n", file=file)
            # Rhyme Related Metrics
            print(f"Rhyme model: {self.rhyme_model_name}, Syllable {str(self.args.val_syllables_rhyme)}", file=file)
            print(f"Rhyme Accuracy: {np.mean(rhyme_accuracy):.4f} +- {np.std(rhyme_accuracy, ddof=1):.4f}", file=file)
            print(f"Rhyme top {self.args.top_k} presence: {np.mean(rhyme_top_k):.4f} +- {np.std(rhyme_top_k, ddof=1):.4f}", file=file)
            print(f"Rhyme label presence: {np.mean(rhyme_label_acc):.4f} +- {np.std(rhyme_label_acc, ddof=1):.4f}", file=file)
            # Measures Levenshtein distance only on wrong examples!
            print(f"Rhyme Levenshtein distance: {np.mean(levenshtein_dist):.4f} +- {np.std(levenshtein_dist, ddof=1):.4f}\n", file=file)
            # Metre related metrics
            print(f"Metre model: {self.meter_model_name}, Syllable {str(self.args.val_syllables_meter)}", file=file)
            print(f"Metre Accuracy: {np.mean(metre_accuracy):.4f} +- {np.std(metre_accuracy, ddof=1):.4f}", file=file)
            print(f"Metre top {self.args.top_k} presence: {np.mean(metre_top_k):.4f} +- {np.std(metre_top_k, ddof=1):.4f}", file=file)
            print(f"Metre label presence: {np.mean(metre_label_acc):.4f} +- {np.std(metre_label_acc, ddof=1):.4f}\n", file=file)
            # Year related metrics
            print(f"Year model: {self.year_model_name}, Syllable {str(self.args.val_syllables_year)}", file=file)
            print(f"Year Accuracy: {np.mean(year_accuracy):.4f} +- {np.std(year_accuracy, ddof=1):.4f}", file=file)
            print(f"Year top {self.args.top_k} presence: {np.mean(year_top_k):.4f} +- {np.std(year_top_k, ddof=1):.4f}", file=file)
            print(f"Year label presence: {np.mean(year_label_acc):.4f} +- {np.std(year_label_acc, ddof=1):.4f}\n", file=file)
                    
            
    def full_validate(self):
        """Validate both generation types
        """
        self.args.sample = True
        self.validate_decoding("BASIC")
        self.validate_decoding("FORCED")
        self.args.sample = False
        self.validate_decoding("BASIC")
        self.validate_decoding("FORCED")
        
      
      

        
parser = argparse.ArgumentParser()

parser.add_argument("--backup_tokenizer_model", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "utils", "tokenizers", "BPE", "new_syllabs_processed_tokenizer.json")), type=str, help="Default Model from HF to use")
parser.add_argument("--data_path_poet",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")

parser.add_argument("--num_samples", default=10, type=int, help="Number of samples to test the tokenizer on")
parser.add_argument("--num_runs", default=2, type=int, help="Number of runs on datasets")

parser.add_argument("--model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'backup_LMS', "gpt-cz-poetry-basic-format-e0e4_LM")),  type=str, help="Path to Model")

parser.add_argument("--rhyme_model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'utils', 'validators', 'rhyme', 'distilroberta-base_syllable_BPE_validator_1701872157853')),  type=str, help="Path to Model")
parser.add_argument("--metre_model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'utils' ,"validators", 'meter', 'ufal-robeczech-base_syllable_BPE_validator_1701575698812')),  type=str, help="Path to Model")
parser.add_argument("--year_model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'utils' ,"validators", 'year', 'ufal-robeczech-base_BPE_validator_1701938156314')),  type=str, help="Path to Model")

parser.add_argument("--validator_tokenizer_model_rhyme", default='distilroberta-base', type=str, help="Validator tokenizer")
parser.add_argument("--validator_tokenizer_model_meter", default='ufal/robeczech-base', type=str, help="Validator tokenizer")
parser.add_argument("--validator_tokenizer_model_year", default='ufal/robeczech-base', type=str, help="Validator tokenizer")
parser.add_argument("--val_syllables_rhyme", default=True, type=bool, help="Does validator use syllables")
parser.add_argument("--val_syllables_meter", default=True, type=bool, help="Does validator use syllables")
parser.add_argument("--val_syllables_year", default=False, type=bool, help="Does validator use syllables")

parser.add_argument("--top_k", default=2, type=int, help="Top k number")

def main(args):
    val = ModelValidator(args)
    val.full_validate()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
    
