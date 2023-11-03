import torch
import os
import random
import re
import argparse  
import numpy as np

from tqdm import tqdm
from transformers import  AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizerBase, AutoModelForCausalLM
from utils.poet_utils import RHYME_SCHEMES, TextAnalysis, TextManipulation, UNK, EOS, PAD, NORMAL_SCHEMES
from utils.poet_model_utils import PoetModelInterface
from utils.validators import ValidatorInterface

from poet_model_base_lm import PoetModelBase

from corpus_capsulated_datasets import CorpusDatasetPytorch

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
        self.rhyme_model, self.meter_model = None, None
        self.rhyme_model_name, self.meter_model_name = "", ""
        if args.rhyme_model_path_full:
            self.rhyme_model: ValidatorInterface = (torch.load(args.rhyme_model_path_full, map_location=torch.device('cpu')))
            self.rhyme_model.eval()
            _,  self.rhyme_model_name = os.path.split(args.rhyme_model_path_full)
        
        if args.metre_model_path_full:
            self.meter_model: ValidatorInterface = (torch.load(args.metre_model_path_full, map_location=torch.device('cpu')))
            self.meter_model.eval()
            _, self.meter_model_name = os.path.split(args.metre_model_path_full)
            

            
        # Load validator tokenizer
        self.validator_tokenizer: PreTrainedTokenizerBase = None
        if args.validator_tokenizer_model:
            try:
                self.validator_tokenizer = AutoTokenizer.from_pretrained(args.validator_tokenizer_model)
            except:
                self.validator_tokenizer: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=args.validator_tokenizer_model)
                self.validator_tokenizer.eos_token = EOS
                self.validator_tokenizer.eos_token_id = 0
                self.validator_tokenizer.pad_token = PAD
                self.validator_tokenizer.pad_token_id = 1
                self.validator_tokenizer.unk_token = UNK
                self.validator_tokenizer.unk_token_id = 2
         
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
        start = f"{self.validation_data[index]['rhyme']} # {self.validation_data[index]['year']} # {self.validation_data[index]['metre']}" 
        
        if type  == "BASIC":
            tokenized_poet_start = self.tokenizer.encode(start, return_tensors='pt', truncation=True)
            if self.args.sample:
                out = self.model.model.generate(tokenized_poet_start, 
                                        max_length=256,
                                        num_beams=8,
                                        no_repeat_ngram_size=2,
                                        eos_token_id = self.tokenizer.eos_token_id,
                                        early_stopping=True,
                                        pad_token_id=self.tokenizer.pad_token_id)
            else:
                out = self.model.model.generate(tokenized_poet_start, 
                                        max_length=256,
                                        do_sample=True,
                                        top_k=50,
                                        eos_token_id = self.tokenizer.eos_token_id,
                                        early_stopping=True,
                                        pad_token_id=self.tokenizer.pad_token_id)
            return self.tokenizer.decode(out[0], skip_special_tokens=True)
        if type == "FORCED":
            return self.model.generate_forced(start, self.tokenizer, verse_len= len(self.validation_data[index]['rhyme']), sample=self.args.sample)
            
            
            
    def validate_decoding(self, type:str):
        """Validate LM given generation type. Measure metrics (Rhyme acc, Metrum acc, End acc, Syllable count acc)

        Args:
            type (str): Type of generation to use
        """
        # Store of individual runs of evaluation
        end_accuracy, sylab_accuracy, rhyme_accuracy, metre_accuracy, rhyme_top_k, metre_top_k, rhyme_label_acc, metre_label_acc, levenshtein_dist = [], [], [], [], [], [], [], [], []
        # Run the requested amount of evaluations
        for _ in tqdm(range(self.epochs), desc=f"Validation {type}"):
            # Store results of current evaluation
            end_all, sylab_all, rhyme_all, metre_all, rhyme_top_k_all,metre_top_k_all, rhyme_label_all, metre_label_all,lev_distance_all = 0,0,0,0,0,0,0,0,0
            
            end_pos, sylab_pos, rhyme_pos, metre_pos, rhyme_top_k_pos,metre_top_k_pos, rhyme_label_pos, metre_label_pos,lev_distance = 0,0,0,0,0,0,0,0,0
            
            
            samples = random.choices(list(range(len(self.validation_data))), k=self.runs_per_epoch)
            # Run the requested steps in evaluation
            for i in tqdm(range(self.runs_per_epoch), leave=False):
                # Get generated Strophe
                decoded_cont:str = self.decode_helper(type,samples[i])
                # Validate line by line
                for line in decoded_cont.splitlines():
                    # Skip Empty lines
                    if not line.strip(): 
                        break
                    if not (TextManipulation._remove_most_nonchar(line)).strip():
                        break
                    # Validate for Strophe Parameters
                    if TextAnalysis._is_param_line(line):
                        values = TextAnalysis._first_line_analysis(line)
                        metre_all +=1
                        metre_top_k_all +=1
                        metre_label_all +=1
                        
                        rhyme_all +=1
                        rhyme_top_k_all +=1
                        rhyme_label_all +=1
                        
                        lev_distance_all += 1
                        # Validate for Rhyme schema
                        if self.rhyme_model != None and "RHYME" in values.keys():
                            data = CorpusDatasetPytorch.collate_validator([{"input_ids" :[decoded_cont], 'rhyme' : values["RHYME"]}],tokenizer=self.validator_tokenizer,
                                                                           is_syllable=False, syllables=self.args.val_syllables_rhyme,
                                                                           max_len=self.rhyme_model.model.config.max_position_embeddings)
                            res = self.rhyme_model.validate(input_ids=data['input_ids'],
                                                                   rhyme=data['rhyme'], k=self.args.top_k)
                            rhyme_pos += res['acc']
                            rhyme_top_k_pos += res['top_k']
                            rhyme_label_pos += res['predicted_label']
                            lev_distance +=res['lev_distance']
                            
                        # Validate for Metrum
                        if self.meter_model != None and "METER" in values.keys():
                            data = CorpusDatasetPytorch.collate_validator([{"input_ids" :[decoded_cont], "metre": values["METER"]}],tokenizer=self.validator_tokenizer,
                                                                           is_syllable=False, syllables=self.args.val_syllables_meter,
                                                                           max_len=self.meter_model.model.config.max_position_embeddings)
                            res = self.meter_model.validate(input_ids=data['input_ids'],
                                                                   metre=data['metre'],k=self.args.top_k)
                            
                            metre_pos += res['acc']
                            metre_top_k_pos += res['top_k']
                            metre_label_pos += res['predicted_label']
                        continue
                            
                    # Else validate for individual verse
                    line_analysis = TextAnalysis._continuos_line_analysis(line)
                    # Was Still empty in terms of any text
                    if len(line_analysis.keys()) == 0:
                        continue
                    
                    end_all += 1
                    if "END" in line_analysis.keys() and "TRUE_END" in line_analysis.keys() and line_analysis["END"] == line_analysis["TRUE_END"]:
                        end_pos +=1
                    
                    sylab_all +=1
                    if "LENGTH" in line_analysis.keys() and "TRUE_LENGTH" in line_analysis.keys() and line_analysis["LENGTH"] == line_analysis["TRUE_LENGTH"]:
                        sylab_pos +=1
                    
                    
            # Store Results        
            end_accuracy.append(end_pos/end_all)
            sylab_accuracy.append(sylab_pos/sylab_all)
            rhyme_accuracy.append(rhyme_pos/rhyme_all)
            metre_accuracy.append(metre_pos/metre_all)
            rhyme_top_k.append(rhyme_top_k_pos/rhyme_top_k_all)
            metre_top_k.append(metre_top_k_pos/metre_top_k_all)
            rhyme_label_acc.append(rhyme_label_pos/rhyme_label_all)
            metre_label_acc.append(metre_label_pos/metre_label_all)
            levenshtein_dist.append(lev_distance/lev_distance_all)
        # Log all results and configuration
        with open(os.path.abspath(os.path.join(self.result_dir, self.model_rel_name + ".txt")), 'a') as file:
            print(f" ===== {type} Decoding Validation: Epochs: {self.epochs}, Runs per epoch: {self.runs_per_epoch}, SAMPLING: {str(self.args.sample)} =====", file=file)
            # Line Metrics
            print(f"Num Sylabs Accuracy: {np.mean(sylab_accuracy)} +- {np.std(sylab_accuracy, ddof=1)}", file=file)
            print(f"Endings Accuracy: {np.mean(end_accuracy)} +- {np.std(end_accuracy, ddof=1)}\n", file=file)
            # Rhyme Related Metrics
            print(f"Rhyme model: {self.rhyme_model_name}, Syllable {str(self.args.val_syllables_rhyme)}", file=file)
            print(f"Rhyme Accuracy: {np.mean(rhyme_accuracy)} +- {np.std(rhyme_accuracy, ddof=1)}", file=file)
            print(f"Rhyme top {self.args.top_k} presence: {np.mean(rhyme_top_k)} +- {np.std(rhyme_top_k, ddof=1)}", file=file)
            print(f"Rhyme label presence: {np.mean(rhyme_label_acc)} +- {np.std(rhyme_label_acc, ddof=1)}", file=file)
            print(f"Rhyme Levenshtein distance: {np.mean(levenshtein_dist)} +- {np.std(levenshtein_dist, ddof=1)}\n", file=file)
            # Metre related metrics
            print(f"Metre model: {self.meter_model_name}, Syllable {str(self.args.val_syllables_meter)}", file=file)
            print(f"Metre Accuracy: {np.mean(metre_accuracy)} +- {np.std(metre_accuracy, ddof=1)}", file=file)
            print(f"Metre top {self.args.top_k} presence: {np.mean(metre_top_k)} +- {np.std(metre_top_k, ddof=1)}", file=file)
            print(f"Metre label presence: {np.mean(metre_label_acc)} +- {np.std(metre_label_acc, ddof=1)}\n", file=file)
                    
            
    def full_validate(self):
        """Validate both generation types
        """
        self.validate_decoding("BASIC")
        self.validate_decoding("FORCED")
        
      
      

        
parser = argparse.ArgumentParser()

parser.add_argument("--backup_tokenizer_model", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "utils", "tokenizers", "BPE", "new_syllabs_processed_tokenizer.json")), type=str, help="Default Model from HF to use")
parser.add_argument("--data_path_poet",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")
parser.add_argument("--num_samples", default=50, type=int, help="Number of samples to test the tokenizer on")
parser.add_argument("--num_runs", default=2, type=int, help="Number of runs on datasets")
parser.add_argument("--model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'backup_LMS', "New-Syllable-BPE-NormalText-gpt-cz-poetry-base-e4e8_LM")),  type=str, help="Path to Model")
parser.add_argument("--rhyme_model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'utils', 'validators', 'rhyme', 'BPE_validator_1697993440889')),  type=str, help="Path to Model")
parser.add_argument("--metre_model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'utils' ,"validators", 'meter', 'BPE_validator_1697833311028')),  type=str, help="Path to Model")
parser.add_argument("--validator_tokenizer_model", default='roberta-base', type=str, help="Validator tokenizer")
parser.add_argument("--val_syllables_rhyme", default=True, type=bool, help="Does validator use syllables")
parser.add_argument("--val_syllables_meter", default=False, type=bool, help="Does validator use syllables")

parser.add_argument("--top_k", default=2, type=int, help="Top k number")
parser.add_argument("--sample", default=True, type=bool, help="Sample during generation")

def main(args):
    val = ModelValidator(args)
    val.full_validate()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
    
