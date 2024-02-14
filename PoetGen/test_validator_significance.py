import os
import argparse
import torch
import random

from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from utils.validators import YearValidator, RhymeValidator, MeterValidator, ValidatorInterface
from utils.poet_utils import StropheParams, Tokens, TextManipulation, TextAnalysis, parse_boolean
from corpus_capsulated_datasets import CorpusDatasetPytorch

parser = argparse.ArgumentParser()

parser.add_argument("--base_val_model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils', 'validators', 'rhyme', 'roberta-base_BPE_validator_1704404730804')),  type=str, help="Path to Model")
parser.add_argument("--improved_val_model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'utils' ,"validators", 'rhyme', 'roberta-base_syllable_BPE_validator_1704572115049')),  type=str, help="Path to Model")

parser.add_argument("--validator_type", default='rhyme', type=str, choices=['rhyme', 'meter', 'year'], help='Type of validator that is tested')

parser.add_argument("--base_validator_tokenizer_model", default='roberta-base', type=str, help="Validator tokenizer")
parser.add_argument("--improved_validator_tokenizer_model", default='roberta-base', type=str, help="Validator tokenizer")

parser.add_argument("--base_val_syllables", default=False, type=parse_boolean, help="Does validator use syllables")
parser.add_argument("--improved_val_syllables", default=False, type=parse_boolean, help="Does validator use syllables")

parser.add_argument("--base_meter_context", default=False, type=parse_boolean, help="Does validator use Context input for meter")
parser.add_argument("--improved_meter_context", default=False, type=parse_boolean, help="Does validator use Context input for meter")

parser.add_argument("--num_repetitions", default=100, type=int, help="Number of repetitions of validator")
parser.add_argument("--per_repetitions", default=10_000, type=int, help="Number of repetitions of validator")
parser.add_argument("--result_file", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'results_new', "validator_significance_test.txt")), type=str, help="Where to store the result of significance test")

parser.add_argument("--data_path_poet",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    
    
def validate(base_model: ValidatorInterface, improved_model: ValidatorInterface , data, base_collate, improved_collate, device, val_str:str):
    req_val = 'metre_ids' if 'meter' in val_str else ('year' if 'year' in val_str else 'rhyme')
    
    
    result = 0
    count = 0
    for i in range(len(data)):
        
        base_datum = base_collate([data[i]][0])
        improved_datum = improved_collate([data[i]][1])
        if req_val == 'metre_ids':
            for j in range(base_datum['input_ids'].shape[0]):
                if random.random() < 0.5:
                    result += base_model.validate_model(input_ids=base_datum["input_ids"][j,:].reshape(1,-1).to(device),
                                    attention_mask=base_datum['attention_mask'][j,:].reshape(1,-1).to(device),
                                    rhyme=None, 
                                    metre_ids=base_datum["metre_ids"][j,:].reshape(1,-1),
                                    year_bucket=None)['acc']
                else:
                    result += improved_model.validate_model(input_ids=improved_datum["input_ids"][j,:].reshape(1,-1).to(device),
                                    attention_mask=improved_datum['attention_mask'][j,:].reshape(1,-1).to(device),
                                    rhyme=None, 
                                    metre_ids=improved_datum["metre_ids"][j,:].reshape(1,-1),
                                    year_bucket=None)['acc']
                
                count +=1
        else:      
            if random.random() < 0.5:
                result += base_model.validate_model(input_ids=base_datum["input_ids"].to(device),
                                    rhyme=base_datum["rhyme"], 
                                    metre_ids=None,
                                    year_bucket=base_datum['year_bucket'])['acc']
            else:
                result += improved_model.validate_model(input_ids=improved_datum["input_ids"].to(device),
                                    rhyme=improved_datum["rhyme"], 
                                    metre_ids=None,
                                    year_bucket=improved_datum['year_bucket'])['acc']
            count +=1
        
        
        
    print(f"Uniform Result: {result/count}")
    
    
    return result/count
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


base_model, improved_model = None, None
base_model_name, improved_model_name =  "", ""

base_model: ValidatorInterface = (torch.load(args.base_val_model_path_full, map_location=torch.device('cpu'))).to(device)
base_model.eval()
_,  base_model_name = os.path.split(args.base_val_model_path_full)


improved_model: ValidatorInterface = (torch.load(args.improved_val_model_path_full, map_location=torch.device('cpu'))).to(device)
improved_model.eval()
_, improved_model_name = os.path.split(args.improved_val_model_path_full)
    

base_validator_tokenizer: PreTrainedTokenizerBase = None
if args.base_val_model_path_full:
    try:
        base_validator_tokenizer = AutoTokenizer.from_pretrained(args.base_validator_tokenizer_model)
    except:
        base_validator_tokenizer: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=args.base_validator_tokenizer_model)
        base_validator_tokenizer.eos_token = Tokens.EOS
        base_validator_tokenizer.eos_token_id = Tokens.EOS_ID
        base_validator_tokenizer.pad_token = Tokens.PAD
        base_validator_tokenizer.pad_token_id = Tokens.PAD_ID
        base_validator_tokenizer.unk_token = Tokens.UNK
        base_validator_tokenizer.unk_token_id = Tokens.UNK_ID
        base_validator_tokenizer.cls_token = Tokens.CLS
        base_validator_tokenizer.cls_token_id = Tokens.CLS_ID
        base_validator_tokenizer.sep_token = Tokens.SEP
        base_validator_tokenizer.sep_token_id = Tokens.SEP_ID
        
# Load Meter tokenizer
improved_validator_tokenizer: PreTrainedTokenizerBase = None
if args.improved_validator_tokenizer_model:
    try:
        improved_validator_tokenizer = AutoTokenizer.from_pretrained(args.improved_validator_tokenizer_model)
    except:
        improved_validator_tokenizer: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=args.improved_validator_tokenizer_model)
        improved_validator_tokenizer.eos_token = Tokens.EOS
        improved_validator_tokenizer.eos_token_id = Tokens.EOS_ID
        improved_validator_tokenizer.pad_token = Tokens.PAD
        improved_validator_tokenizer.pad_token_id = Tokens.PAD_ID
        improved_validator_tokenizer.unk_token = Tokens.UNK
        improved_validator_tokenizer.unk_token_id = Tokens.UNK_ID
        improved_validator_tokenizer.cls_token = Tokens.CLS
        improved_validator_tokenizer.cls_token_id = Tokens.CLS_ID
        improved_validator_tokenizer.sep_token = Tokens.SEP
        improved_validator_tokenizer.sep_token_id = Tokens.SEP_ID
        
# Dataset to take the test data from
base_form = 'BASE'
if args.base_val_syllables:
    base_form='SYLLABLE'
    
improved_form = 'BASE'
if args.improved_val_syllables:
    improved_form='SYLLABLE'
        
base_dataset = CorpusDatasetPytorch(base_form, data_dir=args.data_path_poet)
improved_dataset = CorpusDatasetPytorch(improved_form, data_dir=args.data_path_poet)


base_collate, improved_collate = None, None
if args.validator_type in ['rhyme', 'year']:
    base_collate = partial(CorpusDatasetPytorch.collate_validator, tokenizer=base_validator_tokenizer, max_len=512) 
    improved_collate = partial(CorpusDatasetPytorch.collate_validator, tokenizer=improved_validator_tokenizer, max_len=512)
else:
    base_collate = partial(CorpusDatasetPytorch.collate_meter_context if args.base_meter_context else CorpusDatasetPytorch.collate_meter, tokenizer=base_validator_tokenizer, max_len=512)  
    improved_collate = partial(CorpusDatasetPytorch.collate_meter_context if args.improved_meter_context else CorpusDatasetPytorch.collate_meter, tokenizer=improved_validator_tokenizer, max_len=512) 
    
uniform_list = []
for _ in tqdm(range(args.num_repetitions), desc=f"Comparision"):
    uniform_list.append(
        validate(base_model, improved_model, random.sample(  list(zip(base_dataset.test_strophes.data, improved_dataset.test_strophes.data)), args.per_repetitions), base_collate, improved_collate, device, args.validator_type)
    )
    
import numpy as np

# Lows in percentile
lows = [1, 5, 95, 99]
lows_results = np.percentile(uniform_list, lows)
with open(args.result_file, 'a',  encoding="utf-8") as file:
    print("\n",file=file)
    print(f"### Comparision of BASE: {base_model_name}, IMPROVED: {improved_model_name}, Repetitions: {args.num_repetitions}, Type: {args.validator_type} ###", file=file)
    print(f"### Syllables: BASE: {str(args.base_val_syllables)} IMPROVED: {str(args.improved_val_syllables)} ###", file=file)
    print(f"Tested Lows: {lows}, Results: {lows_results}", file=file)
    print(f"RAW DATA", file=file)
    print(f'{uniform_list}\n', file=file)