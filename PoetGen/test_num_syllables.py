import os
import re
import argparse
import json
import matplotlib.pyplot as plt

from collections import Counter
from utils.poet_utils import SyllableMaker, TextManipulation


# r'[\,\.\?\!–\„\“\’\;\:()\]\[\_\*\‘\”\'0-9\-\—\"]+'
# r'[^a-zA-Z\s]+'
# r'[^\w\s]+'
# r'([^\w\s]+|[0-9]+)'

parser = argparse.ArgumentParser()

parser.add_argument("--data_path_poet",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")
parser.add_argument("--result_file", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'results_new', "syllable_aggreement.txt")), type=str, help="Result of Analysis File")
parser.add_argument("--sample_lines", default=2_000_000, type=int, help="Amount of lines to sample for syllable agreement")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

def poet_samples(args):
    """Collect samples of verse ending (only alpha part)

    Args:
        args (_type_): Regex to use for filtering

    Returns:
        list: All founds ends of verses
    """
    data_filenames_poet = os.listdir(args.data_path_poet)
    same_num_syllable = 0
    num_lines = 0
    errors = {}
    i = 0
    for filename in data_filenames_poet:
        file_path_poet = os.path.join(args.data_path_poet, filename)
        with open(file_path_poet, 'r', encoding="utf8") as file:
            datum = json.load(file)
            for data_line in datum:
                for part_line in data_line['body']:
                    for text_line in part_line:
                        syl_len = sum(map(len, (SyllableMaker.syllabify(text_line['text'])) ) ) 
                        syl_len_ref = len(text_line['stress'])
                        if syl_len == syl_len_ref:
                            same_num_syllable += 1
                        else:
                            errors[syl_len - syl_len_ref] = errors.get(syl_len - syl_len_ref, 0) + 1
                        num_lines += 1
        i += 1
        if i % 500 == 0:
            print(f"Processing file {i}")
        if num_lines > args.sample_lines:
            break
    return same_num_syllable / num_lines, errors


# Collect all ends
percentage_same, errors = poet_samples(args)
# Count ends by number of appearances
# Print and store requested amount of top appearing endings
with open(args.result_file, 'a', encoding="utf8") as file:
    print(f"--- LINE SAMPLE: {args.sample_lines}  ---", file=file)
    print(f"Agreement of {percentage_same * 100:.2f} %", file=file)
    print(f"Errors: {errors}", file=file)
    print(f"--- END ---", file=file)
    
plt.bar(errors.keys(), errors.values())
plt.show()