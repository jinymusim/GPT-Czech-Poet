from poet_model import PoetModel
from transformers import  AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "gpt-cz-poetry")),  type=str, help="Path to Model")
parser.add_argument("--model_path_LM", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "gpt-cz-poetry_LM")),  type=str, help="Path to Model")
parser.add_argument("--default_hf_model", default="lchaloupsky/czech-gpt2-oscar", type=str, help="Default Model from HF to use")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

tokenizer = AutoTokenizer.from_pretrained(args.default_hf_model)
model = PoetModel(args.default_hf_model).load_state_dict(torch.load(args.model_path_full))
model_LM = AutoModelForCausalLM.from_pretrained(args.model_path_LM)

tokenized_poet_start = tokenizer.encode("ABBA\n", return_tensors='pt')

out = model_LM.generate(tokenized_poet_start, 
                                max_length=256,
                                num_beams=2,
                                no_repeat_ngram_size=2,
                                early_stopping=True,
                                pad_token_id=tokenizer.eos_token_id)


decoded_cont = tokenizer.decode(out[0], skip_special_tokens=True)

print(decoded_cont)