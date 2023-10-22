import torch
import os
import random
import re
import argparse  
import numpy as np

from tqdm import tqdm
from transformers import  AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizerBase
from utils.poet_utils import RHYME_SCHEMES, TextAnalysis, TextManipulation, UNK, EOS, PAD
from utils.poet_model_utils import PoetModelInterface
from utils.validators import ValidatorInterface

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
        self.model: PoetModelInterface= (torch.load(self.model_name, map_location=torch.device('cpu')))
        
        # Load validators 
        self.rhyme_model, self.meter_model = None, None
        if args.rhyme_model_path_full:
            self.rhyme_model: ValidatorInterface = (torch.load(args.rhyme_model_path_full, map_location=torch.device('cpu')))    
        
        if args.metre_model_path_full:
            self.meter_model: ValidatorInterface = (torch.load(args.metre_model_path_full, map_location=torch.device('cpu')))
            
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
            self.tokenizer: PreTrainedTokenizerBase =  AutoTokenizer.from_pretrained(args.default_tokenizer_model)
        except:
            self.tokenizer: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=args.default_tokenizer_model)
            self.tokenizer.eos_token = EOS
            self.tokenizer.eos_token_id = 0
            self.tokenizer.pad_token = PAD
            self.tokenizer.pad_token_id = 1
            self.tokenizer.unk_token = UNK
            self.tokenizer.unk_token_id = 2
        
        # Store the Validation arguments    
        self.epochs = args.num_runs
        self.runs_per_epoch = args.num_samples
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            
            
    def decode_helper(self, type:str):
        """Wrapper around LM generation

        Args:
            type (str): Which type of generation to use ('basic', 'forced')

        Returns:
            str: Generated Strophe
        """
        if type  == "basic":
            tokenized_poet_start = self.tokenizer.encode(random.choice(RHYME_SCHEMES[:-1]), return_tensors='pt', truncation=True)
        
            out = self.model.model.generate(tokenized_poet_start, 
                                        max_length=192,
                                        num_beams=2,
                                        no_repeat_ngram_size=2,
                                        early_stopping=True,
                                        pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(out[0], skip_special_tokens=True)
        if type == "forced":
            rhyme = random.choice(RHYME_SCHEMES[:-1])
            return self.model.generate_forced(rhyme, self.tokenizer, verse_len= len(rhyme))
            
            
            
    def validate_decoding(self, type:str):
        """Validate LM given generation type. Measure metrics (Rhyme acc, Metrum acc, End acc, Syllable count acc)

        Args:
            type (str): Type of generation to use
        """
        # Store of individual runs of evaluation
        end_accuracy, sylab_accuracy, rhyme_accuracy, metre_accuracy = [], [], [],[]
        # Run the requested amount of evaluations
        for _ in tqdm(range(self.epochs), desc=f"Validation {type}"):
            # Store results of current evaluation
            end_all, sylab_all, rhyme_all, metre_all = 0,0,0,0
            end_pos, sylab_pos, rhyme_pos, metre_pos = 0,0,0,0
            # Run the requested steps in evaluation
            for _ in range(self.runs_per_epoch):
                # Get generated Strophe
                decoded_cont:str = self.decode_helper(type)
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
                        rhyme_all +=1
                        # Validate for Rhyme schema
                        if self.rhyme_model != None and "RHYME" in values.keys():
                            rhyme_vec = TextAnalysis._rhyme_vector(values["RHYME"])
                            input_ids = CorpusDatasetPytorch.collate_validator([{"input_ids" :[decoded_cont]}],tokenizer=self.validator_tokenizer,
                                                                           is_syllable=False, syllables=self.args.val_syllables,
                                                                           max_len=self.rhyme_model.model.config.max_position_embeddings)['input_ids']
                            rhyme_pos += self.rhyme_model.validate(input_ids=input_ids,
                                                                   rhyme=torch.tensor(rhyme_vec.reshape(1,-1)))
                        # Validate for Metrum
                        if self.meter_model != None and "METER" in values.keys():
                            metre_vec = TextAnalysis._metre_vector(values["METER"])
                            input_ids = CorpusDatasetPytorch.collate_validator([{"input_ids" :[decoded_cont]}],tokenizer=self.validator_tokenizer,
                                                                           is_syllable=False, syllables=self.args.val_syllables,
                                                                           max_len=self.meter_model.model.config.max_position_embeddings)['input_ids']
                            metre_pos += self.meter_model.validate(input_ids=input_ids,
                                                                   metre=torch.tensor(metre_vec.reshape(1,-1)))
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
        # Log all results and configuration
        with open(os.path.abspath(os.path.join(self.result_dir, self.model_rel_name)), 'a') as file:
             print(f"{type} Decoding Validation: Epochs: {self.epochs}, Runs per epoch: {self.runs_per_epoch}", file=file)
             print(f"Num Sylabs Accuracy: {np.mean(sylab_accuracy)} +- {np.std(sylab_accuracy, ddof=1)}", file=file)
             print(f"Endings Accuracy: {np.mean(end_accuracy)} +- {np.std(end_accuracy, ddof=1)}", file=file)
             print(f"Rhyme Accuracy: {np.mean(rhyme_accuracy)} +- {np.std(rhyme_accuracy, ddof=1)}", file=file)
             print(f"Metre Accuracy: {np.mean(metre_accuracy)} +- {np.std(metre_accuracy, ddof=1)}", file=file)
                    
            
    def full_validate(self):
        """Validate both generation types
        """
        self.validate_decoding("basic")
        self.validate_decoding("forced")
        
      
      

        
parser = argparse.ArgumentParser()

parser.add_argument("--default_tokenizer_model", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "utils", "tokenizers", "BPE", "tokenizer.json")), type=str, help="Default Model from HF to use")
parser.add_argument("--data_path_poet",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")
parser.add_argument("--num_samples", default=10, type=int, help="Number of samples to test the tokenizer on")
parser.add_argument("--num_runs", default=5, type=int, help="Number of runs on datasets")
parser.add_argument("--model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'backup_LMS', "gpt-cz-poetry-all-e32e32")),  type=str, help="Path to Model")
parser.add_argument("--rhyme_model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'utils', 'validators', 'rhyme', 'BPE_validator_1696540325706')),  type=str, help="Path to Model")
parser.add_argument("--metre_model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'utils' ,"validators", 'meter', 'BPE_validator_1696540325706')),  type=str, help="Path to Model")
parser.add_argument("--validator_tokenizer_model", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'utils', "tokenizers", "BPE", "processed_tokenizer.json")), type=str, help="Validator tokenizer")
parser.add_argument("--val_syllables", default=True, type=bool, help="Does validator use syllables")

def main(args):
    val = ModelValidator(args)
    val.full_validate()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
    
