import argparse
import os
import torch
import json
import re
import random
from tqdm import tqdm

from poet_utils import StropheParams

from llama_cpp import Llama

parser = argparse.ArgumentParser()


#repo_id='TheBloke/Llama-2-70B-Chat-GGUF',
#filename="*Q5_K_M.gguf",

#repo_id='TheBloke/Llama-2-7B-Chat-GGUF',
#filename="*Q5_K_M.gguf",

#repo_id='TheBloke/Llama-2-7B-GGUF',
#filename="*Q5_K_M.gguf",

#repo_id='TheBloke/Mixtral-8x7B-v0.1-GGUF',
#filename="*Q5_K_M.gguf", 

#repo_id='TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF',
#filename="*Q5_K_M.gguf", 


#TheBloke/Yarn-Llama-2-13B-128K-GGUF
#TheBloke/Yarn-Mistral-7B-128k-GGUF

parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__),'..', "corpusCzechVerse", "ccv-new")), type=str, help="Path to Data")
parser.add_argument("--result_data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__),'..', "corpusCzechVerse", "ccv-new-summary-one-sentence")), type=str, help="Path to Data")

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)


os.makedirs(args.result_data_path, exist_ok=True)

model_name = 'TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF'

with torch.no_grad():
    
    model = Llama.from_pretrained(
        repo_id=model_name,
        filename="*.Q6_K.gguf",
        verbose=True,
        chat_format="llama-2",
        n_gpu_layers=-1,
        n_ctx=30000
    )
    
    dataset= os.listdir(args.data_path)
    random.shuffle(dataset)

    for poem_file in tqdm(dataset):
        if not os.path.isfile(os.path.join(args.data_path, poem_file)) or os.path.exists(os.path.join(args.result_data_path, poem_file)):
            continue
        
        file = json.load(open(os.path.join(args.data_path, poem_file) , 'r'))
        
         
        for i, poem_data in enumerate(file):
            try:
                poem_text = [] 
                autor = 'Unknown' 
                if poem_data['biblio']['p_title'] != None:
                    poem_text.append(poem_data['biblio']['p_title'])
                else:
                    poem_text.append("NO TITLE")
                    
                if  "p_author" in poem_data.keys() and 'identity' in poem_data['p_author'].keys():
                    autor = poem_data['p_author']['identity']
                for strophe in poem_data['body']:
                        for verse in strophe:
                            poem_text.append(verse['text'])
                        poem_text.append("\n")
                poem = "\n".join(poem_text)
                input_text = f"Author: {autor}\nPoem:\n{poem}\nCategories: {', '.join(StropheParams.POEM_TYPES)}. Best Category:"
                if 'Chat' in model_name or 'Instruct' in model_name:
                    out = model.create_chat_completion(
                        messages = [
                            {
                                "role": "assistant", 
                                "content": "You are a assistent that is proficient in poem categorization and picks the most representing category from list provided by the user and displays it in plain text."
                            },
                            {
                                "role": "user",
                                "content": input_text
                            }
                        ],
                        stop=["\n", '\t']
                        )
                    categories = out['choices'][0]['message']['content']
                else:
                    out = model(
                        f'{input_text}',
                        max_tokens=250
                        )
                    categories = out['choices'][0]['text']
                
            
            
                categories = list(map(str.strip, re.findall(r'\w+', categories)))
                categories = list(filter(lambda x: len(x) > 0, categories))
                categories = list(filter(lambda x: x in StropheParams.POEM_TYPES, categories))

                file[i]['categories'] = categories
            
                input_text = f"Author: {autor}\nPoem:\n{poem}\nSingle sentence poem analysis:"
                if 'Chat' in model_name or 'Instruct' in model_name in model_name:
                    out = model.create_chat_completion(
                        messages = [
                            {
                                "role": "assistant", 
                                "content": "You are a assistent that is proficient in poem analysis and provides it in ONE sentence in plain text."
                            },
                            {
                                "role": "user",
                                "content": input_text
                            }
                        ],
                        stop=["\n", '\t\t']
                        )
                    sumarization =  out['choices'][0]['message']['content']
                else:
                    out = model(
                        f'{input_text}',
                        max_tokens=250
                        )
                    sumarization =  out['choices'][0]['text']


                file[i]['summarization'] = sumarization
            except Exception as e:  
                print("Context too large: ", repr(e))
            
        
        json.dump(file, open(os.path.join(args.result_data_path, poem_file), 'w+'), indent=6)   