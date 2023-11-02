import json
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument("--data_files", default=[os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",  "body_poet_data.json")) , 
                                             os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..", "val_body_poet_data.json"))], type=list, help="Paths to data files")
parser.add_argument("--result_file", default=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","results",  "data_analysis.txt")), type=str, help="Path to result file")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    
all_data = []
for datafile in args.data_files:
    with open(datafile, "r") as file:
        all_data += json.load(file) 
rhymes = {}
metres = {}
years = {}
for line in all_data:
    rhymes[line['rhyme']] = rhymes.get(line['rhyme'], 0) + (1/len(all_data))
    metres[line['metre']] = metres.get(line['metre'],0) + (1/len(all_data))
    years[line['year']] = years.get(line['year'],0) + (1/len(all_data))
    
rhymes = sorted(rhymes.items(), key=lambda x: x[1], reverse=True)
metres = sorted(metres.items(), key=lambda x: x[1], reverse=True)
years = sorted(years.items(), key=lambda x: x[1], reverse=True)

with open(args.result_file, "w+") as file:
    print("=== RHYMES ===", file=file)
    for rhyme in rhymes[:100]:
        print(f"{rhyme[0]}, Presence: {rhyme[1] * 100:.2f} %", file=file)
    print("=== METRES ===", file=file)
    for metre in metres[:100]:
        print(f"{metre[0]}, Presence: {metre[1] * 100:.2f} %", file=file)
    print("=== YEARS ===", file=file)
    for years in years[:100]:
        print(f"{years[0]}, Presence: {years[1] * 100:.2f} %", file=file)