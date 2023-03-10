import os
import sys
import json
import numpy as np
from transformers import GPT2Tokenizer
import tensorflow as tf

class CorpusDataLoad:
     
    class Dataset:
        
        def __init__(self, data_file_paths, tokenizer ,shuffle_bool:bool= True, seed:int= 42):
            self._data_file_paths = data_file_paths
            self._tokenizer = tokenizer
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
        
        def data_text_line_gen(self):
            for filename in self._data_file_paths:
                with open(filename, 'r') as file:
                    datum = json.load(file)
                for data_line in datum:
                    for part_line in data_line['body']:
                        for text_line in part_line:
                            yield self._tokenizer.encode(text_line['text'], return_tensors="tf")
                            
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
                    yield self._tokenizer.encode("\n".join(part), return_tensors="tf")
                    
        def data_body_gen(self):
            for filename in self._data_file_paths:
                with open(filename, 'r') as file:
                    datum = json.load(file)
                for data_line in datum:
                    for part_line in data_line['body']:
                        body = []
                        for text_line in part_line:
                            body.append(text_line['text'])
                        yield self._tokenizer.encode("\n".join(body), return_tensors="tf")
        
        @property
        def dataset_text(self) -> tf.data.Dataset:
            return tf.data.Dataset.from_generator(self.data_text_line_gen, output_signature=tf.TensorSpec(shape=(1,None),dtype=tf.int32))
        
        @property
        def dataset_part(self) -> tf.data.Dataset:
            return tf.data.Dataset.from_generator(self.data_part_gen, output_signature=tf.TensorSpec(shape=(1,None), dtype=tf.int32))
        
        @property
        def dataset_body(self) -> tf.data.Dataset:
            return tf.data.Dataset.from_generator(self.data_body_gen, output_signature=tf.TensorSpec(shape=(1,None),dtype=tf.int32))
        

            
            
    def load_json_filenames(self):
        data_filenames = os.listdir(self.data_dir)
        data_by_files = []
        for filename in data_filenames:
            file_path = os.path.join(self.data_dir, filename)
            data_by_files.append(file_path)
        self.dataset = CorpusDataLoad.Dataset(data_by_files, self.tokenizer)
    
    def __init__(self,tokenizer,  data_dir = "GPT2\corpusCzechVerse-master\ccv"):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.load_json_filenames()
                      
if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("lchaloupsky/czech-gpt2-oscar")
    train_dat = CorpusDataLoad(tokenizer)
    
    tf_dataset = train_dat.dataset.dataset_body
    tf_dataset = tf_dataset.shuffle(128).padded_batch(8)
    i = 0
    for datum in tf_dataset.enumerate():
        if i >= 10:
            break
        print("Tensor type {}".format(type(datum)))
        print("\nBody {} loaded\n".format(i))
        print("\nBody Text example: {}\n".format(datum))
        i+=1
        
    i = 0
    for datum in train_dat.dataset.data_body_gen():
        if i >= 10:
            break
        print("\nBody {} loaded\n".format(i))
        print("\nBody Text example: {}\n".format(datum))
        i+=1
        
    