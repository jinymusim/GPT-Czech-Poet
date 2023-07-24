from transformers import AutoTokenizer
from datasets import load_dataset
import os
import argparse
import json
import random
import numpy


parser = argparse.ArgumentParser()

parser.add_argument("--default_tokenizer_model", default="lchaloupsky/czech-gpt2-oscar", type=str, help="Default Model from HF to use")
parser.add_argument("--data_path_poet",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")
parser.add_argument("--data_path_base", default="cs_restaurants", type=str, help="Base dataset")
parser.add_argument("--base_part", default="unshuffled_deduplicated_cs", type=str, help="Which part of base dataset to consider")
parser.add_argument("--num_samples", default=1000, type=int, help="Number of samples to test the tokenizer on")
parser.add_argument("--num_runs", default=100, type=int, help="Number of runs on datasets")
parser.add_argument("--result_file", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "tokenizer_analysis.txt")), type=str, help="Result of Analysis File")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

tokenizer = AutoTokenizer.from_pretrained(args.default_tokenizer_model)

def poet_samples(args, shuffle=True):
    data_filenames_poet = os.listdir(args.data_path_poet)
    if shuffle:
        random.shuffle(data_filenames_poet)
    text_lines_poet = []
    i=1
    for filename in data_filenames_poet:
        file_path_poet = os.path.join(args.data_path_poet, filename)
        with open(file_path_poet, 'r') as file:
            datum = json.load(file)
            for data_line in datum:
                for part_line in data_line['body']:
                    for text_line in part_line:
                        text_lines_poet.append(text_line['text'])
                        i+=1
                        if i >= args.num_samples:
                            return text_lines_poet
    return text_lines_poet

def base_samples(args, shuffle=True):
    dataset = load_dataset(args.data_path_base)
    if shuffle:
        dataset = dataset.shuffle()
    text_lines_base = []
    i=1
    for datum in dataset["train"]:
        text_lines_base.append(datum["text"])
        if i >= args.num_samples:
            return text_lines_base
    return text_lines_base
    
    
poet_runs = []
base_runs = []
for j in range(args.num_runs):
    poet = poet_samples(args)
    base = base_samples(args)
    i = 1
    poet_chars_per_token = []
    base_chars_per_token = []
    for sample_p, sample_b in zip(poet, base):
        poet_chars_per_token.append(len(sample_p) / len(tokenizer.encode(sample_p, return_tensors="np")[0]))
        base_chars_per_token.append(len(sample_b) / len(tokenizer.encode(sample_b, return_tensors="np")[0]))
        i+= 1

    print(f"Run {j}")
    print(f"Analyzed Samples: {i}")
    print(f"Chars per token Poet data: {sum(poet_chars_per_token)/len(poet_chars_per_token)}")
    print(f"Chars per token Base data: {sum(base_chars_per_token)/len(base_chars_per_token)}")
    print("##############################\n")
    poet_runs.append(sum(poet_chars_per_token)/len(poet_chars_per_token))
    base_runs.append(sum(base_chars_per_token)/len(base_chars_per_token))

# Write to STD   
print("All Runs Results")
print(f"Chars per token Poet data: {numpy.mean(poet_runs)} +- {numpy.std(poet_runs,ddof=1)}")
print(f"Chars per token Base data: {numpy.mean(base_runs)} +- {numpy.std(base_runs,ddof=1)}")
# Write to file
with open(args.result_file, 'a') as file:
    print("All Runs Results", file=file)
    print(f"Chars per token Poet data: {numpy.mean(poet_runs)} +- {numpy.std(poet_runs,ddof=1)}", file=file)
    print(f"Chars per token Base data: {numpy.mean(base_runs)} +- {numpy.std(base_runs,ddof=1)}", file=file)