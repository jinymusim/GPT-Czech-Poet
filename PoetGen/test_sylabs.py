import os
import argparse

from transformers import AutoTokenizer
from utils.poet_utils import SyllableMaker
from corpus_capsulated_datasets import CorpusDatasetPytorch

parser = argparse.ArgumentParser()

parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")
parser.add_argument("--result_file", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'results', "sylabs.txt")), type=str, help="Result of Analysis File")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    

train_data = CorpusDatasetPytorch(AutoTokenizer.from_pretrained("gpt2"), data_dir=args.data_path)

sylables = set()
i = 0
for text in train_data.raw_dataset.get_text():
    for item in SyllableMaker.syllabify(text):
        sylables.add(item)
    i+=1
    if i > 500_000:
        break
        
with open(args.result_file, 'a', encoding="utf8") as file:
    print("--- Syllables ---", file=file)
    print( list(sylables), file=file)        