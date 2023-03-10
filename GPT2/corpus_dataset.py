import os
import sys
import numpy as np
import tensorflow as tf

class CorpusDataLoad:
    
    
    class Dataset:
        
        def __init__(self, data: np.ndarray, shuffle_bool:bool= True, seed:int= 42):
            self._data = data
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    