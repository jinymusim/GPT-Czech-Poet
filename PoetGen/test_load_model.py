import torch
import os
import argparse

from transformers import  AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast
from utils.base_poet_models import PoetModelBase
from utils.poet_model_utils import PoetModelInterface
from utils.poet_utils import Tokens

parser = argparse.ArgumentParser()

parser.add_argument("--model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'backup_LMS', "CZ-Base-Tokenizer-NormalText-TinyLama-cz-poetry-base-e16_LM")),  type=str, help="Path to Model")
# bigscience/bloom-560m
parser.add_argument("--backup_tokenizer_model", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "utils", "tokenizers", "BPE", "new_processed_tokenizer.json")), type=str, help="Default Model from HF to use")
parser.add_argument("--result_file", default= os.path.abspath(os.path.join(os.path.dirname(__file__),'results', "test_poet_model.txt")), type=str, help="Where to store the decoding efforts")
parser.add_argument("--sample", default=True, type=bool, help="If to sample during generation")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

# Load tokenizer
try:    
    tokenizer: PreTrainedTokenizerBase =  AutoTokenizer.from_pretrained(args.model_path_full)
except: 
    tokenizer: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=args.backup_tokenizer_model)
    tokenizer.eos_token = Tokens.EOS
    tokenizer.eos_token_id = Tokens.EOS_ID
    tokenizer.pad_token = Tokens.PAD
    tokenizer.pad_token_id = Tokens.PAD_ID
    tokenizer.unk_token = Tokens.UNK
    tokenizer.unk_token_id = Tokens.UNK_ID

# Load model
if "_LM" in args.model_path_full:
    model: PoetModelInterface= PoetModelBase(args.model_path_full)
else:
    model: PoetModelInterface= (torch.load(args.model_path_full, map_location=torch.device('cpu')))
# Free model generation
tokenized_poet_start = tokenizer.encode("<", return_tensors='pt')

if args.sample:
    out = model.model.generate(tokenized_poet_start, 
                                        max_length=256,
                                        do_sample=True,
                                        top_k=50,
                                        eos_token_id = tokenizer.eos_token_id,
                                        early_stopping=True,
                                        pad_token_id=tokenizer.pad_token_id)
                
else:
    out = model.model.generate(tokenized_poet_start, 
                                        max_length=256,
                                        num_beams=8,
                                        no_repeat_ngram_size=2,
                                        eos_token_id =tokenizer.eos_token_id,
                                        early_stopping=True,
                                        pad_token_id=tokenizer.pad_token_id)


decoded_cont = tokenizer.decode(out[0], skip_special_tokens=True)
# Print the result of generation
print("### Basic Decoding! ###\n", decoded_cont)
# Restricted generation 
FORMAT="METER_VERSE"
if os.path.split(args.model_path_full)[1].startswith('gpt'):
    FORMAT="BASIC"
if os.path.split(args.model_path_full)[1].startswith('NEW') or os.path.split(args.model_path_full)[1].startswith('BASE'):
    FORMAT='VERSE_PAR'
# Store both types of generation as well as the name of used LM
_, base_filename = os.path.split(args.model_path_full)
with open(args.result_file, 'a', encoding="utf-8") as file:
    print(f"--- Model {base_filename} ---", file=file)
    print("### Basic Decoding! ###\n", decoded_cont, file=file)