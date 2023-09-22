import torch
import os
import random
import re
import numpy as np

from tqdm import tqdm
from transformers import  AutoTokenizer
from poet_utils import RHYME_SCHEMES, TextAnalysis, TextManipulation
from poet_model_interface import PoetModelInterface

class ModelValidator:
    def __init__(self, model_name: str, tokenizer_name: str, epochs:int = 20, runs_per_epoch: int = 10, result_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "results")) ) -> None:
        self.model_name = model_name
        _ ,self.model_rel_name =  os.path.split(model_name)
        self.model: PoetModelInterface= (torch.load(model_name, map_location=torch.device('cpu')))
        self.tokenizer : AutoTokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.epochs = epochs
        self.runs_per_epoch = runs_per_epoch
        self.result_dir = result_dir
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            
            
    def decode_helper(self, type:str):
        if type  == "basic":
            tokenized_poet_start = self.tokenizer.encode(random.choice(RHYME_SCHEMES[:-1]), return_tensors='pt')
        
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
        end_accuracy = []
        sylab_accuracy = []
        for _ in tqdm(range(self.epochs), desc=f"Validation {type}"):
            end_pos = 0
            end_all = 0
            sylab_pos = 0
            sylab_all = 0
            for _ in range(self.runs_per_epoch):
                
                decoded_cont:str = self.decode_helper(type)
                
                for line in decoded_cont.splitlines():
                    if not line.strip(): 
                        break
                    if not (TextManipulation._remove_most_nonchar(line)).strip():
                        break
                    if TextAnalysis._is_param_line(line):
                        continue
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
                    
                    observed_sylab = len(re.findall('a|e|i|o|u|á|é|í|ú|ů|ó|ě|y|ý', raw_line))
                    sylab_all += 1
                    if observed_sylab == expected_sylab:
                        sylab_pos += 1
            end_accuracy.append(end_pos/end_all)
            sylab_accuracy.append(sylab_pos/sylab_all)
        with open(os.path.abspath(os.path.join(self.result_dir, self.model_rel_name)), 'a') as file:
             print(f"{type} Decoding Validation: Epochs: {self.epochs}, Runs per epoch: {self.runs_per_epoch}", file=file)
             print(f"Num Sylabs Accuracy: {np.mean(sylab_accuracy)} +- {np.std(sylab_accuracy, ddof=1)}", file=file)
             print(f"Endings Accuracy: {np.mean(end_accuracy)} +- {np.std(end_accuracy, ddof=1)}", file=file)
                    
            
    def full_validate(self):
        self.validate_decoding("basic")
        self.validate_decoding("forced")
        
      
      
import argparse  
        
parser = argparse.ArgumentParser()

parser.add_argument("--default_tokenizer_model", default="lchaloupsky/czech-gpt2-oscar", type=str, help="Default Model from HF to use")
parser.add_argument("--data_path_poet",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")
parser.add_argument("--num_samples", default=10, type=int, help="Number of samples to test the tokenizer on")
parser.add_argument("--num_runs", default=5, type=int, help="Number of runs on datasets")
parser.add_argument("--model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'backup_LMS', "gpt-cz-poetry-base_newline_e8_e32")),  type=str, help="Path to Model")

def main(args):
    val = ModelValidator(args.model_path_full, args.default_tokenizer_model, args.num_runs, args.num_samples)
    val.full_validate()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
    
