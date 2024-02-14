import argparse
import os
import torch
import numpy as np

from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast
from utils.poet_utils import StropheParams, Tokens, TextManipulation, TextAnalysis
from utils.base_poet_models import PoetModelBase
from utils.validators import ValidatorInterface

from corpus_capsulated_datasets import CorpusDatasetPytorch

parser = argparse.ArgumentParser()

#parser.add_argument("--model_path_full", default='jinymusim/gpt-czech-poet',  type=str, help="Path to Model")
parser.add_argument("--model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'backup_LMS', 'CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-all-e4e16_LM')),  type=str, help="Path to Model")
parser.add_argument("--result_file", default= os.path.abspath(os.path.join(os.path.dirname(__file__),'results', "unicode_generated_poems.txt")), type=str, help="Where to store the decoding efforts")
parser.add_argument("--sample", default=True, type=bool, help="If to sample during generation")

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

parser.add_argument("--runs_per_setting", default=1, type=int, help="Number of runs per setting")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    
_ ,model_rel_name =  os.path.split(args.model_path_full)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PoetModelBase(args.model_path_full).to(device)
model.eval()

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
        validator_tokenizer_meter = AutoTokenizer.from_pretrained(args.validator_tokenizer_model_meter)
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
        validator_tokenizer_year = AutoTokenizer.from_pretrained(args.validator_tokenizer_model_year)
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
tokenizer: PreTrainedTokenizerBase =  AutoTokenizer.from_pretrained(args.model_path_full)

def decoder_helper(type, rhyme, year, meter):
    if type == "BASIC":
        start = f"# {rhyme} # {year}\n{meter}"
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
        start_forced = f"# {rhyme} # {year}\n{meter} #"
        return model.generate_forced(start_forced, tokenizer, verse_len=len(rhyme), sample=True, device=device)
    
for rhyme in StropheParams.RHYME[:10]:
    for year in [1900, 1880, 1920]:
        for meter in ['J', 'T', 'A', 'D']:
            for type in ['BASIC', 'FORCED']:
                for _ in range(args.runs_per_setting):
                    generated_poem:str = decoder_helper(type, rhyme, year, meter)
                    meters = []
                    rhyme_pred = ''
                    year_pred = 0
                    for line in generated_poem.splitlines():
                        # Skip Empty lines
                        if not line.strip(): 
                            break
                        if not (TextManipulation._remove_most_nonchar(line)).strip():
                            break
                        # Validate for Strophe Parameters
                        if TextAnalysis._is_param_line(line):

                            data = CorpusDatasetPytorch.collate_validator([{"input_ids" :generated_poem}],tokenizer=validator_tokenizer_rhyme,
                                                                               make_syllables=args.val_syllables_rhyme,
                                                                               max_len=rhyme_model.model.config.max_position_embeddings - 2)
                            rhyme_pred =StropheParams.RHYME[np.argmax(rhyme_model.predict_state(input_ids=data['input_ids'].to(device)).detach().flatten().cpu().numpy())]

                            data = CorpusDatasetPytorch.collate_validator([{"input_ids" :generated_poem}],tokenizer=validator_tokenizer_year,
                                                                               make_syllables=args.val_syllables_year,
                                                                               max_len=year_model.model.config.max_position_embeddings - 2)
                            year_pred = round(year_model.predict_state(input_ids=data['input_ids'].to(device)).detach().flatten().cpu().numpy()[0])
                            continue
                        
                    if args.meter_with_context:
                        data = CorpusDatasetPytorch.collate_meter_context([{"input_ids" :generated_poem}],tokenizer=validator_tokenizer_meter,
                                                    make_syllables=args.val_syllables_meter,
                                                    max_len=meter_model.model.config.max_position_embeddings - 2)
                    else:
                        data = CorpusDatasetPytorch.collate_meter([{"input_ids" :generated_poem}],tokenizer=validator_tokenizer_meter,
                                                    make_syllables=args.val_syllables_meter,
                                                    max_len=meter_model.model.config.max_position_embeddings - 2)
                    for j in range(data['input_ids'].shape[0]):
                        meters.append(
                            StropheParams.METER[np.argmax(meter_model.predict_state(input_ids=data['input_ids'][j,:].reshape(1,-1).to(device)).detach().flatten().cpu().numpy())]
                        )

                    with open(args.result_file, 'a', encoding="utf-8") as file:
                        print(f"REQUESTED: {rhyme}, {year}, {meter}, GENERATED USING: {type}\n", file=file)
                        print(generated_poem, file=file)
                        print(f"PREDICTED: {rhyme_pred}, {year_pred}, {meters}\n\n", file=file)
                    print(f"REQUESTED: {rhyme}, {year}, {meter}, GENERATED USING: {type}\n")
                    print(generated_poem)
                    print(f"PREDICTED: {rhyme_pred}, {year_pred}, {meters}\n\n")