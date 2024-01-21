import os
import re
import argparse
import json

from collections import Counter
from utils.poet_utils import SyllableMaker, TextManipulation


# r'[\,\.\?\!–\„\“\’\;\:()\]\[\_\*\‘\”\'0-9\-\—\"]+'
# r'[^a-zA-Z\s]+'
# r'[^\w\s]+'
# r'([^\w\s]+|[0-9]+)'

parser = argparse.ArgumentParser()

parser.add_argument("--data_path_poet",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")
parser.add_argument("--result_file", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'results', "endings.txt")), type=str, help="Result of Analysis File")
parser.add_argument("--top", default=200, type=int, help="Amount of top endings (None for all)")
parser.add_argument("--regex", default=r'([^\w\s]+|[0-9]+)', type=str, help="Tested Regex")

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
    text_lines_poet = []
    i = 0
    for filename in data_filenames_poet:
        file_path_poet = os.path.join(args.data_path_poet, filename)
        with open(file_path_poet, 'r', encoding="utf8") as file:
            datum = json.load(file)
            for data_line in datum:
                for part_line in data_line['body']:
                    for text_line in part_line:
                        # r'[^a-zA-Z\s]+' Will also match í,ě and others, so not usable
                        # r'[^\w\s"]+' Doesn't work better than current regex
                        whitening = TextManipulation._remove_all_nonchar(TextManipulation._remove_most_nonchar(text_line['text'])).strip() 
                        text_lines_poet.append(SyllableMaker.syllabify( " ".join(whitening.split()[-2:]))[-1][-1])
        i += 1
        if i % 500 == 0:
            print(f"Processing file {i}")
    return text_lines_poet


# Collect all ends
poet_endings = poet_samples(args)
# Count ends by number of appearances
endings = Counter(poet_endings)
# Print and store requested amount of top appearing endings
print(endings.most_common(args.top))
with open(args.result_file, 'a', encoding="utf8") as file:
    print(f"--- Tested Regex {args.regex} --- Top {'All' if args.top == None else args.top}", file=file)
    print( [i[0] for i in endings.most_common(args.top)] , file=file)