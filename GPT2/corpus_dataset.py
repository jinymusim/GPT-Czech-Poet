import os
import sys
import json
import numpy as np
from typing import Any
import tensorflow as tf

class CorpusDataLoad:
    
    
    class Dataset:
        
        def __init__(self, data_file_paths, shuffle_bool:bool= True, seed:int= 42):
            self._data_file_paths = data_file_paths
            self._size = len(self._data_file_paths)
            
        @property
        def size(self):
            return self._size
    
        @property
        def raw_data_gen(self):
            for filename in self._data_file_paths:
                with open(filename, 'r') as file:
                    datum = json.load(file)
                yield datum
        
        @property
        def data_text_line_gen(self):
            for filename in self._data_file_paths:
                with open(filename, 'r') as file:
                    datum = json.load(file)
                for data_line in datum:
                    for part_line in data_line['body']:
                        for text_line in part_line:
                            yield text_line['text']
        @property
        def data_part_gen(self):
            for filename in self._data_file_paths:
                with open(filename, 'r') as file:
                    datum = json.load(file)
                for data_line in datum:
                    part = []
                    for part_line in data_line['body']:
                        body = []
                        for text_line in part_line:
                            body.append(text_line['text'])
                        part.append("\n".join(body))
                    yield "\n".join(part)
        @property
        def data_body_gen(self):
            for filename in self._data_file_paths:
                with open(filename, 'r') as file:
                    datum = json.load(file)
                for data_line in datum:
                    for part_line in data_line['body']:
                        body = []
                        for text_line in part_line:
                            body.append(text_line['text'])
                        yield "\n".join(body)
            
            
    def load_json_filenames(self):
        data_filenames = os.listdir(self.data_dir)
        data_by_files = []
        for filename in data_filenames:
            file_path = os.path.join(self.data_dir, filename)
            data_by_files.append(file_path)
        self.dataset = CorpusDataLoad.Dataset(data_by_files)
    
    def __init__(self, data_dir = "GPT2\corpusCzechVerse-master\ccv"):
        self.data_dir = data_dir
        self.load_json_filenames()
        
        
        
if __name__ == "__main__":   
    train_dat = CorpusDataLoad()
    i = 0
    for datum in train_dat.dataset.data_body_gen:
        if i >= 100:
            break
        print("Body {} loaded".format(i))
        print()
        print("Body Text example: {}".format(datum))
        print()
        i+=1