import torch
import os
import argparse

from transformers import  AutoTokenizer
from utils.poet_model_utils import PoetModelInterface

parser = argparse.ArgumentParser()

parser.add_argument("--model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'backup_LMS', "gpt-cz-poetry-all_tasks_e8_e32")),  type=str, help="Path to Model")
# bigscience/bloom-560m
parser.add_argument("--default_hf_model", default="lchaloupsky/czech-gpt2-oscar", type=str, help="Default Model from HF to use")
parser.add_argument("--result_file", default= os.path.abspath(os.path.join(os.path.dirname(__file__),'results', "test_poet_model.txt")), type=str, help="Where to store the decoding efforts")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

tokenizer = AutoTokenizer.from_pretrained(args.default_hf_model)
model: PoetModelInterface= (torch.load(args.model_path_full, map_location=torch.device('cpu')))

tokenized_poet_start = tokenizer.encode("A", return_tensors='pt')

out = model.model.generate(tokenized_poet_start, 
                                max_length=192,
                                num_beams=2,
                                no_repeat_ngram_size=2,
                                early_stopping=True,
                                pad_token_id=tokenizer.eos_token_id)


decoded_cont = tokenizer.decode(out[0], skip_special_tokens=True)

print("### Basic Decoding! ###\n", decoded_cont)

out_forced = model.generate_forced("A", tokenizer, verse_len=4)

print("### Forced Decoding! ###\n", out_forced)

_, base_filename = os.path.split(args.model_path_full)
with open(args.result_file, 'a', encoding="utf-8") as file:
    print(f"--- Model {base_filename} ---", file=file)
    print("### Basic Decoding! ###\n", decoded_cont, file=file)
    print("### Forced Decoding! ###\n", out_forced, file=file)