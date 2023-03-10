import os
import sys
import json
import numpy as np
from typing import Any
import tensorflow as tf

class CorpusDataLoad:
    
    
    class Dataset:
        
        def __init__(self, data, shuffle_bool:bool= True, seed:int= 42):
            self._raw_data = data
            self._size = len(self._raw_data)
            
        @property
        def size(self):
            return self._size
    
        @property
        def raw_data(self):
            return self._raw_data
            
    def load_jsons(self):
        data_filenames = os.listdir(self.data_dir)
        data_by_files = []
        for filename in data_filenames:
            file_path = os.path.join(self.data_dir, filename)
            with open(file_path, 'r') as file:
               data_by_files.append(json.load(file))
        self.data = CorpusDataLoad.Dataset(data_by_files)
    
    def __init__(self, data_dir = "GPT2\corpusCzechVerse-master\ccv"):
        self.data_dir = data_dir
        self.load_jsons()
        
train_dat = CorpusDataLoad()
print(train_dat.data.size())