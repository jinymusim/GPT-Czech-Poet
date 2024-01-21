import json
import argparse
import os

from poet_utils import TextManipulation, SyllableMaker

parser = argparse.ArgumentParser()

parser.add_argument("--data_files", default=[os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..",  "body_poet_data.json")) , 
                                             os.path.abspath(os.path.join(os.path.dirname(__file__),"..","..", "val_body_poet_data.json"))], type=list, help="Paths to data files")
parser.add_argument("--result_file", default=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","results",  "data_analysis.txt")), type=str, help="Path to result file")
parser.add_argument("--raw_data_files", default=os.path.abspath(os.path.join(os.path.dirname(__file__),"..", "corpusCzechVerse", "ccv")), type=str, help="Path to raw data")
parser.add_argument("--raw_result_file", default=os.path.abspath(os.path.join(os.path.dirname(__file__),"..","results",  "raw_data_analysis.txt")), type=str, help="Path to result file")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    
all_data = []
for datafile in args.data_files:
    with open(datafile, "r") as file:
        all_data += json.load(file) 
rhymes = {}
metres = {}
years = {}
years_bucketed = {}
m_count = 0
for line in all_data:
    rhymes[line['rhyme']] = rhymes.get(line['rhyme'], 0) + (1/len(all_data))
    for meter in line['metre_ids']:
        metres[meter] = metres.get(meter,0) + 1
        m_count +=1
    years[line['year']] = years.get(line['year'],0) + (1/len(all_data))
    years_bucketed[TextManipulation._year_bucketor(line['year'])] = years_bucketed.get(TextManipulation._year_bucketor(line['year']),0) + (1/len(all_data))

for key, value in metres.items():
    metres[key] = value/m_count
    
rhymes = sorted(rhymes.items(), key=lambda x: x[1], reverse=True)
metres = sorted(metres.items(), key=lambda x: x[1], reverse=True)
years = sorted(years.items(), key=lambda x: x[1], reverse=True)
years_bucketed =sorted(years_bucketed.items(),key=lambda x: x[1], reverse=True)

with open(args.result_file, "w+") as file:
    print("=== RHYMES ===", file=file)
    for rhyme in rhymes:
        print(f"{rhyme[0]}, Presence: {rhyme[1] * 100:.2f} %", file=file)
    print("=== METRES ===", file=file)
    for metre in metres:
        print(f"{metre[0]}, Presence: {metre[1] * 100:.2f} %", file=file)
    print("=== YEARS ===", file=file)
    for years in years:
        print(f"{years[0]}, Presence: {years[1] * 100:.2f} %", file=file)
    print("=== BUCKETED YEARS ===", file=file)
    for years in years_bucketed:
        print(f"{years[0]}, Presence: {years[1] * 100:.2f} %", file=file)

writer_data = {}
data_filenames = os.listdir(args.raw_data_files)
syllable_uniqueness = []
for filename in data_filenames:
    file_path = os.path.join(args.raw_data_files, filename)
    
    with open(file_path, "r") as file:
        data = json.load(file)
    for book in data:
        # Real name is under 'identity'
        try:
            author = book['p_author']['identity']
        except:
            author = book['b_author']['identity']
        line_count = 0
        for body_part in book['body']:
            syllables = []
            for text_line in  body_part:
                line_count +=1
                syllables += [syl for syl_word in SyllableMaker.syllabify(text_line['text']) for syl in syl_word] 
            syllable_uniqueness += [len(set(syllables))/len(syllables)]
        writer_data[author] = writer_data.get(author, 0) + line_count
        
writers = sorted(writer_data.items(), key=lambda x: x[1], reverse=True)
with open(args.raw_result_file, "w+", encoding="utf-8") as file:
    print("=== WRITERS ===", file=file)
    for writer in writers:
        print(f"{writer[0]}, Number of Lines: {writer[1]}", file=file)
    print("=== SYLLAB UNIQUE ===", file=file)
    print(f"{sum(syllable_uniqueness)/len(syllable_uniqueness) * 100} % Syllable Uniqueness", file=file)
    