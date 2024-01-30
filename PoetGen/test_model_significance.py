import os
import argparse
import torch
import random

from tqdm import tqdm
from functools import partial
from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

from utils.validators import YearValidator, RhymeValidator, MeterValidator, ValidatorInterface
from utils.poet_utils import StropheParams, Tokens, TextManipulation, TextAnalysis
from utils.base_poet_models import PoetModelBase
from corpus_capsulated_datasets import CorpusDatasetPytorch

parser = argparse.ArgumentParser()

parser.add_argument("--base_model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'backup_LMS', 'CZ-Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM' )),  type=str, help="Path to Model")
parser.add_argument("--improved_model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'backup_LMS', 'CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM')),  type=str, help="Path to Model")

parser.add_argument("--base_generate", default='BASIC', type=str, choices=['BASIC', 'FORCED'], help='Generation type done')
parser.add_argument("--improved_generate", default='BASIC', type=str, choices=['BASIC', 'FORCED'], help='Generation type done')

parser.add_argument("--base_input_type", default='METER_VERSE', type=str, choices=['BASIC', 'VERSE_PAR', 'METER_VERSE'], help='Input Format type ')
parser.add_argument("--improved_input_type", default='METER_VERSE', type=str, choices=['BASIC', 'VERSE_PAR', 'METER_VERSE'], help='Input Format type ')

parser.add_argument("--rhyme_model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'utils', 'validators', 'rhyme', 'distilroberta-base_BPE_validator_1704126399565')),  type=str, help="Path to Model")
parser.add_argument("--metre_model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'utils' ,"validators", 'meter', 'Context_ufal-robeczech-base_BPE_validator_1705689955968')),  type=str, help="Path to Model")
parser.add_argument("--year_model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'utils' ,"validators", 'year', 'ufal-robeczech-base_BPE_validator_1702393305267')),  type=str, help="Path to Model")

parser.add_argument("--validator_tokenizer_model_rhyme", default='distilroberta-base', type=str, help="Validator tokenizer")
parser.add_argument("--validator_tokenizer_model_meter", default='ufal/robeczech-base', type=str, help="Validator tokenizer")
parser.add_argument("--validator_tokenizer_model_year", default='ufal/robeczech-base', type=str, help="Validator tokenizer")
parser.add_argument("--val_syllables_rhyme", default=False, type=bool, help="Does validator use syllables")
parser.add_argument("--val_syllables_meter", default=False, type=bool, help="Does validator use syllables")
parser.add_argument("--val_syllables_year", default=False, type=bool, help="Does validator use syllables")

parser.add_argument("--meter_with_context", default=True, type=bool, help="Does Meter uses context")

parser.add_argument("--num_repetitions", default=100, type=int, help="Number of repetitions")
parser.add_argument("--per_repetitions", default=500, type=int, help="Number of samples")
parser.add_argument("--result_file", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'results', "model_significance_test.txt")), type=str, help="Where to store the result of significance test")

parser.add_argument("--data_path_poet",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_model = PoetModelBase(args.base_model_path_full).to(device)
base_model.eval()
_, base_model_name = os.path.split(args.base_model_path_full)

improved_model = PoetModelBase(args.improved_model_path_full).to(device)
improved_model.eval()
_, improved_model_name = os.path.split(args.improved_model_path_full)

rhyme_model, meter_model, year_model = None, None, None
rhyme_model_name, meter_model_name, year_model_name = "", "", ""
if args.rhyme_model_path_full:
    rhyme_model: ValidatorInterface = (torch.load(args.rhyme_model_path_full, map_location=torch.device('cpu'))).to(device)
    rhyme_model.eval()
    _,  rhyme_model_name = os.path.split(args.rhyme_model_path_full)

if args.metre_model_path_full:
    meter_model: ValidatorInterface = (torch.load(args.metre_model_path_full, map_location=torch.device('cpu'))).to(device)
    meter_model.eval()
    _, meter_model_name = os.path.split(args.metre_model_path_full)
    
if args.year_model_path_full:
    year_model: ValidatorInterface = (torch.load(args.year_model_path_full, map_location=torch.device('cpu'))).to(device)
    year_model.eval()
    _,  year_model_name = os.path.split(args.year_model_path_full)
# Load Rhyme tokenizer
validator_tokenizer_rhyme: PreTrainedTokenizerBase = None
if args.validator_tokenizer_model_rhyme:
    try:
        validator_tokenizer_rhyme = AutoTokenizer.from_pretrained(args.validator_tokenizer_model_rhyme)
    except:
        validator_tokenizer_rhyme: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=args.validator_tokenizer_model_rhyme)
        validator_tokenizer_rhyme.eos_token = Tokens.EOS
        validator_tokenizer_rhyme.eos_token_id = Tokens.EOS_ID
        validator_tokenizer_rhyme.pad_token = Tokens.PAD
        validator_tokenizer_rhyme.pad_token_id = Tokens.PAD_ID
        validator_tokenizer_rhyme.unk_token = Tokens.UNK
        validator_tokenizer_rhyme.unk_token_id = Tokens.UNK_ID
        validator_tokenizer_rhyme.cls_token = Tokens.CLS
        validator_tokenizer_rhyme.cls_token_id = Tokens.CLS_ID
        validator_tokenizer_rhyme.sep_token = Tokens.SEP
        validator_tokenizer_rhyme.sep_token_id = Tokens.SEP_ID
        
# Load Meter tokenizer
validator_tokenizer_meter: PreTrainedTokenizerBase = None
if args.validator_tokenizer_model_meter:
    try:
        validator_tokenizer_meter = AutoTokenizer.from_pretrained(args.validator_tokenizer_model_meter, revision='v1.0')
    except:
        validator_tokenizer_meter: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=args.validator_tokenizer_model_meter)
        validator_tokenizer_meter.eos_token = Tokens.EOS
        validator_tokenizer_meter.eos_token_id = Tokens.EOS_ID
        validator_tokenizer_meter.pad_token = Tokens.PAD
        validator_tokenizer_meter.pad_token_id = Tokens.PAD_ID
        validator_tokenizer_meter.unk_token = Tokens.UNK
        validator_tokenizer_meter.unk_token_id = Tokens.UNK_ID
        validator_tokenizer_meter.cls_token = Tokens.CLS
        validator_tokenizer_meter.cls_token_id = Tokens.CLS_ID
        validator_tokenizer_meter.sep_token = Tokens.SEP
        validator_tokenizer_meter.sep_token_id = Tokens.SEP_ID
        
# Load Year tokenizer
validator_tokenizer_year: PreTrainedTokenizerBase = None
if args.validator_tokenizer_model_year:
    try:
        validator_tokenizer_year = AutoTokenizer.from_pretrained(args.validator_tokenizer_model_year, revision='v1.0')
    except:
        validator_tokenizer_year: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=args.validator_tokenizer_model_year)
        validator_tokenizer_year.eos_token = Tokens.EOS
        validator_tokenizer_year.eos_token_id = Tokens.EOS_ID
        validator_tokenizer_year.pad_token = Tokens.PAD
        validator_tokenizer_year.pad_token_id = Tokens.PAD_ID
        validator_tokenizer_year.unk_token = Tokens.UNK
        validator_tokenizer_year.unk_token_id = Tokens.UNK_ID
        validator_tokenizer_year.cls_token = Tokens.CLS
        validator_tokenizer_year.cls_token_id = Tokens.CLS_ID
        validator_tokenizer_year.sep_token = Tokens.SEP
        validator_tokenizer_year.sep_token_id = Tokens.SEP_ID
 
# Load LM tokenizers       
base_tokenizer: PreTrainedTokenizerBase =  AutoTokenizer.from_pretrained(args.base_model_path_full)
improved_tokenizer: PreTrainedTokenizerBase =  AutoTokenizer.from_pretrained(args.improved_model_path_full)

dataset = CorpusDatasetPytorch(data_dir=args.data_path_poet)

def decoder_helper(type, rhyme, year, meter, tokenizer: PreTrainedTokenizerBase, model: PoetModelBase, input_type:str):
    if type == "BASIC":
        if input_type == 'METER_VERSE':
            start = f"# {rhyme} # {year}\n{meter}"
        else:
            start = f"# {rhyme} # {year} # {meter}"
        tokenized = tokenizer.encode(start, return_tensors='pt', truncation=True)
        out = model.model.generate(tokenized.to(device), 
                                        max_length=512,
                                        do_sample=True,
                                        top_k=50,
                                        eos_token_id = tokenizer.eos_token_id,
                                        early_stopping=True,
                                        pad_token_id= tokenizer.pad_token_id)
        return tokenizer.decode(out.cpu()[0], skip_special_tokens=True)
    if type=="FORCED":
        if input_type == 'METER_VERSE':
            start_forced = f"# {rhyme} # {year}\n{meter} #"
        else:
            start_forced = f"# {rhyme} # {year} # {meter}"
        return model.generate_forced(start_forced, tokenizer, verse_len=len(rhyme), sample=True, device=device, format=input_type)
    
def do_eval(generated_strophe):
    res_rhyme = 0
    res_meter = 0
    res_year = 0
    STROPHE_METER = 'J'
    PRESENT_METERS = []
    for line in generated_strophe.splitlines():
        # Skip Empty lines
        if not line.strip(): 
            break
        if not (TextManipulation._remove_most_nonchar(line)).strip():
            break
        # Validate for Strophe Parameters
        if TextAnalysis._is_param_line(line):
            values = TextAnalysis._first_line_analysis(line)
            
            
            # Validate for Rhyme schema
            if "RHYME" in values.keys():
                data = CorpusDatasetPytorch.collate_validator([{"input_ids" :[generated_strophe], 'rhyme' : values["RHYME"]}],tokenizer=validator_tokenizer_rhyme,
                                                               is_syllable=False, syllables=args.val_syllables_rhyme,
                                                               max_len=512)
                res_rhyme = rhyme_model.validate_model(input_ids=data['input_ids'].to(device),
                                                        rhyme=data['rhyme'], k=2)['acc']
                
            
            #Validate for Year
            if "YEAR" in values.keys():
                data = CorpusDatasetPytorch.collate_validator([{"input_ids" :[generated_strophe], "year": values["YEAR"]}],tokenizer=validator_tokenizer_year,
                                                               is_syllable=False, syllables=args.val_syllables_year,
                                                               max_len=512)
                res_year = year_model.validate_model(input_ids=data['input_ids'].to(device),
                                                       year_bucket=data['year_bucket'] ,k=2)['acc']
                
                
            if 'STROPHE_METER' in values.keys():
                STROPHE_METER = values['STROPHE_METER']
                    
                
        # Else validate for individual verse
        line_analysis = TextAnalysis._continuos_line_analysis(line)
        # Was Still empty in terms of any text
        if len(line_analysis.keys()) == 0:
            continue
        
        
        if "METER" in line_analysis.keys():
            PRESENT_METERS.append(line_analysis["METER"])
        else:
            PRESENT_METERS.append(STROPHE_METER)
                         
            
    # Validate for Metrum
    if args.meter_with_context:
        data = CorpusDatasetPytorch.collate_meter_context([{"input_ids" :[generated_strophe], "metre_ids": PRESENT_METERS}],tokenizer=validator_tokenizer_meter,
                                                       is_syllable=False, syllables=args.val_syllables_meter,
                                                       max_len=512)
    else:
        data = CorpusDatasetPytorch.collate_meter([{"input_ids" :[generated_strophe], "metre_ids": PRESENT_METERS}],tokenizer=validator_tokenizer_meter,
                                                       is_syllable=False, syllables=args.val_syllables_meter,
                                                       max_len=512)
    
    for j in range(min(data['input_ids'].shape[0], data['metre_ids'].shape[0])):
        res_meter += meter_model.validate_model(input_ids=data["input_ids"][j,:].reshape(1,-1).to(device),
                    attention_mask=data['attention_mask'][j,:].reshape(1,-1).to(device),
                    rhyme=None, 
                    metre_ids=data["metre_ids"][j,:].reshape(1,-1),
                    year_bucket=None)['acc']/min(data['input_ids'].shape[0], data['metre_ids'].shape[0])
        
    return res_rhyme, res_meter, res_year       
    
    
    
def do_epoch():
    samples = random.choices(list(range(len(dataset.test_pytorch_dataset_body.data))), k=args.per_repetitions)
    
    rhyme_res = 0
    meter_res = 0
    year_res = 0
    
    for i in range(args.per_repetitions):
        # Get generated Strophe
        rhyme = dataset.test_pytorch_dataset_body[samples[i]]['rhyme']
        meter = dataset.test_pytorch_dataset_body[samples[i]]['metre_ids'][0]
        year = dataset.test_pytorch_dataset_body[samples[i]]['year']
        base_decode:str = decoder_helper(args.base_generate, rhyme, year, meter, base_tokenizer, base_model, args.base_input_type)
        improved_decode:str = decoder_helper(args.improved_generate, rhyme, year, meter, improved_tokenizer, improved_model, args.improved_input_type) 
        
        if random.random() < 0.5:
            rhyme_one, meter_one, year_one = do_eval(base_decode)
        else:
            rhyme_one, meter_one, year_one = do_eval(improved_decode)
        
        rhyme_res += rhyme_one
        meter_res += meter_one
        year_res += year_one
        
        
        
    return rhyme_res/args.per_repetitions, meter_res/args.per_repetitions, year_res/args.per_repetitions


uniform_list_rhyme, uniform_list_meter, uniform_list_year = [], [], []
for i in tqdm(range(args.num_repetitions), desc=f"Comparision"):
    better_rhyme, better_meter, better_year = do_epoch()
    uniform_list_rhyme.append(better_rhyme)
    uniform_list_meter.append(better_meter)
    uniform_list_year.append(better_year)
     
import numpy as np

# Lows in percentile
lows = [1, 5, 95, 99]
lows_results_rhyme = np.percentile(uniform_list_rhyme, lows)
lows_results_meter = np.percentile(uniform_list_meter, lows)
lows_results_year = np.percentile(uniform_list_year, lows)
with open(args.result_file, 'a',  encoding="utf-8") as file:
    print("\n",file=file)
    print(f"### Comparision of BASE: {base_model_name}, IMPROVED: {improved_model_name}, Repetitions: {args.num_repetitions} ###", file=file)
    print(f"### Generation: BASE: {args.base_generate} IMPROVED: {args.improved_generate}, Input Type: BASE: {args.base_input_type} IMPROVED: {args.improved_input_type} ###", file=file)
    print(f"Tested Lows: {lows}, Results Rhyme: {lows_results_rhyme}, Results Meter: {lows_results_meter}, Results Year: {lows_results_year}", file=file)
    print(f"RAW DATA", file=file)
    print(f'{uniform_list_rhyme}', file=file)
    print(f'{uniform_list_meter}', file=file)
    print(f'{uniform_list_year}\n', file=file)
        
       