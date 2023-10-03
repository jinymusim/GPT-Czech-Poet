import torch
import os
import random
import re
import numpy as np

from tqdm import tqdm
from transformers import  AutoTokenizer, PreTrainedTokenizerFast, PreTrainedTokenizerBase
from utils.poet_utils import RHYME_SCHEMES, TextAnalysis, TextManipulation, SyllableMaker
from utils.poet_model_utils import PoetModelInterface
from utils.validators import ValidatorInterface

from corpus_capsulated_datasets import CorpusDatasetPytorch

class ModelValidator:
    def __init__(self, model_name: str, tokenizer_name: str,
                 epochs:int = 20, runs_per_epoch: int = 10, 
                 result_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__),"results")),
                 rhyme_model_name: str = "", meter_model_name:str = "", validator_tokenizer_name: str = "",
                 rhyme_collate_fnc = None, meter_collate_fnc = None) -> None:
        self.model_name = model_name
        _ ,self.model_rel_name =  os.path.split(model_name)
        self.model: PoetModelInterface= (torch.load(model_name, map_location=torch.device('cpu')))
        
        self.rhyme_model, self.meter_model = None, None
        if rhyme_model_name:
            self.rhyme_model: ValidatorInterface = (torch.load(rhyme_model_name, map_location=torch.device('cpu')))    
        self.rhyme_collate = rhyme_collate_fnc
        
        if meter_model_name:
            self.meter_model: ValidatorInterface = (torch.load(meter_model_name, map_location=torch.device('cpu')))
        self.meter_collate = meter_collate_fnc
            
        self.validator_tokenizer: PreTrainedTokenizerBase = None
        if validator_tokenizer_name:
            try:
                self.validator_tokenizer = AutoTokenizer.from_pretrained(validator_tokenizer_name)
            except:
                self.validator_tokenizer: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=validator_tokenizer_name)
                self.validator_tokenizer.eos_token = "<|endoftext|>"
                self.validator_tokenizer.eos_token_id = 0
                self.validator_tokenizer.pad_token = '<|endoftext|>'
                self.validator_tokenizer.pad_token_id = 0
                self.validator_tokenizer.unk_token = '<|endoftext|>'
                self.validator_tokenizer.unk_token_id = 0
                self.validator_tokenizer.model_max_length = self.meter_model.model.config.n_positions
                
        try:    
            self.tokenizer: PreTrainedTokenizerBase =  AutoTokenizer.from_pretrained(tokenizer_name)
        except:
            self.tokenizer: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=tokenizer_name)
            self.tokenizer.eos_token = "<|endoftext|>"
            self.tokenizer.eos_token_id = 0
            self.tokenizer.pad_token = '<|endoftext|>'
            self.tokenizer.pad_token_id = 0
            self.tokenizer.unk_token = '<|endoftext|>'
            self.tokenizer.unk_token_id = 0
            self.tokenizer.model_max_length = self.model.model.config.n_positions
            
        self.epochs = epochs
        self.runs_per_epoch = runs_per_epoch
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            
            
    def decode_helper(self, type:str):
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
        end_accuracy, sylab_accuracy, rhyme_accuracy, metre_accuracy = [], [], [],[]
        for _ in tqdm(range(self.epochs), desc=f"Validation {type}"):
            end_all, sylab_all, rhyme_all, metre_all = 0,0,0,0
            end_pos, sylab_pos, rhyme_pos, metre_pos = 0,0,0,0
            for _ in range(self.runs_per_epoch):
                
                decoded_cont:str = self.decode_helper(type)
                
                for line in decoded_cont.splitlines():
                    if not line.strip(): 
                        break
                    if not (TextManipulation._remove_most_nonchar(line)).strip():
                        break
                    if TextAnalysis._is_param_line(line):
                        values = TextAnalysis._first_line_analysis(line)
                        metre_all +=1
                        rhyme_all +=1
                        if self.rhyme_model != None and "RHYME" in values.keys():
                            self.validator_tokenizer.model_max_length = self.rhyme_model.model.config.n_positions
                            rhyme_vec = TextAnalysis._rhyme_vector(values["RHYME"])
                            input_ids = self.rhyme_collate([{"input_ids" :decoded_cont}],
                                                          self.validator_tokenizer, 
                                                          max_len=self.rhyme_model.model.config.n_positions)['input_ids']
                            rhyme_pos += self.rhyme_model.validate(input_ids=input_ids,
                                                                   rhyme=torch.tensor(rhyme_vec.reshape(1,-1)))
                        if self.meter_model != None and "METER" in values.keys():
                            metre_vec = TextAnalysis._metre_vector(values["METER"])
                            
                            input_ids =self.meter_collate([{"input_ids" :decoded_cont}],
                                                          self.validator_tokenizer, 
                                                          max_len=self.meter_model.model.config.n_positions)['input_ids']
                            metre_pos += self.meter_model.validate(input_ids=input_ids,
                                                                   metre=torch.tensor(metre_vec.reshape(1,-1)))
                            
                    # Ended Verse

                    line_split = line.split()
                    # 0 = sylab count, 1 = ending, 2 = #, 3: line itself
                    # May struggle
                    try:
                        expected_sylab = int(line_split[0])
                        expected_end = line_split[1].strip()
                    except:
                        sylab_all += 1
                        end_all += 1
                        continue
                    
                    
                    raw_line = " ".join(line_split[3:])  
                               
                    sub = re.sub(r'([^\w\s]+|[0-9]+)', '', raw_line)
                    observed_end = sub.strip()[-3:]
                    end_all += 1 
                    if observed_end == expected_end:
                        end_pos += 1
                    
                    observed_sylab = len(SyllableMaker.syllabify(raw_line)) # INFO: Now properly counts syllables
                    sylab_all += 1
                    if observed_sylab == expected_sylab:
                        sylab_pos += 1
            end_accuracy.append(end_pos/end_all)
            sylab_accuracy.append(sylab_pos/sylab_all)
            rhyme_accuracy.append(rhyme_pos/rhyme_all)
            metre_accuracy.append(metre_pos/metre_all)
        with open(os.path.abspath(os.path.join(self.result_dir, self.model_rel_name)), 'a') as file:
             print(f"{type} Decoding Validation: Epochs: {self.epochs}, Runs per epoch: {self.runs_per_epoch}", file=file)
             print(f"Num Sylabs Accuracy: {np.mean(sylab_accuracy)} +- {np.std(sylab_accuracy, ddof=1)}", file=file)
             print(f"Endings Accuracy: {np.mean(end_accuracy)} +- {np.std(end_accuracy, ddof=1)}", file=file)
             print(f"Rhyme Accuracy: {np.mean(rhyme_accuracy)} +- {np.std(rhyme_accuracy, ddof=1)}", file=file)
             print(f"Metre Accuracy: {np.mean(metre_accuracy)} +- {np.std(metre_accuracy, ddof=1)}", file=file)
                    
            
    def full_validate(self):
        self.validate_decoding("basic")
        self.validate_decoding("forced")
        
      
      
import argparse  
        
parser = argparse.ArgumentParser()

parser.add_argument("--default_tokenizer_model", default="lchaloupsky/czech-gpt2-oscar", type=str, help="Default Model from HF to use")
parser.add_argument("--data_path_poet",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")
parser.add_argument("--num_samples", default=10, type=int, help="Number of samples to test the tokenizer on")
parser.add_argument("--num_runs", default=5, type=int, help="Number of runs on datasets")
parser.add_argument("--model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'backup_LMS', "gpt-cz-poetry-base")),  type=str, help="Path to Model")
parser.add_argument("--rhyme_model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'utils', 'validators', 'rhyme', 'BPE_validator')),  type=str, help="Path to Model")
parser.add_argument("--metre_model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'utils' ,"validators", 'meter', 'BPE_validator')),  type=str, help="Path to Model")
parser.add_argument("--validator_tokenizer_model", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'utils', "tokenizers", "BPE", "tokenizer.json")), type=str, help="Validator tokenizer")



def main(args):
    val = ModelValidator(model_name=args.model_path_full, 
                         tokenizer_name=args.default_tokenizer_model, 
                         epochs=args.num_runs, 
                         runs_per_epoch=args.num_samples,
                         rhyme_model_name= args.rhyme_model_path_full,
                         meter_model_name= args.metre_model_path_full,
                         validator_tokenizer_name=args.validator_tokenizer_model,
                         rhyme_collate_fnc=CorpusDatasetPytorch.collate_rhyme,
                         meter_collate_fnc=CorpusDatasetPytorch.collate)
    val.full_validate()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
    
