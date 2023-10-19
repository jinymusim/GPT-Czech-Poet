import torch
import os
import argparse

from transformers import  AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast
from utils.poet_model_utils import PoetModelInterface
from utils.poet_utils import UNK, PAD, EOS

parser = argparse.ArgumentParser()

parser.add_argument("--model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'backup_LMS', "Syllabs-BPE-NormalText-gpt-cz-poetry-base-e16e32")),  type=str, help="Path to Model")
# bigscience/bloom-560m
parser.add_argument("--default_tokenizer_model", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "utils", "tokenizers", "BPE", "syllabs_processed_tokenizer.json")), type=str, help="Default Model from HF to use")
parser.add_argument("--result_file", default= os.path.abspath(os.path.join(os.path.dirname(__file__),'results', "test_poet_model.txt")), type=str, help="Where to store the decoding efforts")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

try:    
    tokenizer: PreTrainedTokenizerBase =  AutoTokenizer.from_pretrained(args.default_tokenizer_model)
except: #TODO: Need model to update embedding matrix
    tokenizer: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=args.default_tokenizer_model)
    tokenizer.eos_token = EOS
    tokenizer.eos_token_id = 0
    tokenizer.pad_token = PAD
    tokenizer.pad_token_id = 1
    tokenizer.unk_token = UNK
    tokenizer.unk_token_id = 2
    
model: PoetModelInterface= (torch.load(args.model_path_full, map_location=torch.device('cpu')))

tokenized_poet_start = tokenizer.encode("A", return_tensors='pt')

out = model.model.generate(tokenized_poet_start, 
                                max_length=192,
                                num_beams=8,
                                no_repeat_ngram_size=2,
                                eos_token_id = tokenizer.eos_token_id,
                                early_stopping=True,
                                pad_token_id=tokenizer.pad_token_id)


decoded_cont = tokenizer.decode(out[0], skip_special_tokens=True)

print("### Basic Decoding! ###\n", decoded_cont)

out_forced = model.generate_forced("A", tokenizer, verse_len=4)

print("### Forced Decoding! ###\n", out_forced)

_, base_filename = os.path.split(args.model_path_full)
with open(args.result_file, 'a', encoding="utf-8") as file:
    print(f"--- Model {base_filename} ---", file=file)
    print("### Basic Decoding! ###\n", decoded_cont, file=file)
    print("### Forced Decoding! ###\n", out_forced, file=file)