
import os
import re
import argparse
import json
from collections import Counter
import random


parser = argparse.ArgumentParser()

parser.add_argument("--data_path_poet",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")
parser.add_argument("--result_file", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "endings.txt")), type=str, help="Result of Analysis File")
parser.add_argument("--top", default=200, type=int, help="Amount of top endings")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

def poet_samples(args):
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
                        sub = re.sub(r'[\,\.\?\!–\„\“\’\;\:()\]\[\_\*]+', '', text_line['text'])
                        text_lines_poet.append(sub.strip()[-2:].lower())
        i += 1
        if i % 500 == 0:
            print(f"Processing file {i}")
    return text_lines_poet



poet_endings = poet_samples(args)
endings = Counter(poet_endings)
print(endings.most_common(args.top))
with open(args.result_file, 'w+', encoding="utf8") as file:
    print( [i[0] for i in endings.most_common(args.top)] , file=file)