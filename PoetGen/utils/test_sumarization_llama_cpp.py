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

parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__),'..', "corpusCzechVerse", "ccv-new")), type=str, help="Path to Data")

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)


model_name = 'TheBloke/Mixtral-8x7B-v0.1-GGUF'

with torch.no_grad():
    
    model = Llama.from_pretrained(
        repo_id=model_name,
        filename="*Q5_K_M.gguf",
        verbose=True,
        chat_format="llama-2",
        n_gpu_layers=-1,
        n_ctx=30000
    )
    
    dataset= os.listdir(args.data_path)
    random.shuffle(dataset)

    for poem_file in tqdm(dataset):
        if not os.path.isfile(os.path.join(args.data_path, poem_file)):
            continue
        
        file = json.load(open(os.path.join(args.data_path, poem_file) , 'r'))
        
        if 'sumarization' in file[0].keys():
            continue
         
        for i, poem_data in enumerate(file):
            poem_text = []  
            if poem_data['biblio']['p_title'] != None:
                poem_text.append(poem_data['biblio']['p_title'])
            else:
                poem_text.append("Neznámý název")
            for strophe in poem_data['body']:
                    for verse in strophe:
                        poem_text.append(verse['text'])
                    poem_text.append("\n")
            poem = "\n".join(poem_text)
            input_text = f"Toto jsou kategorie: {', '.join(StropheParams.POEM_TYPES)}. \
Vyber z těchto kategorií ty, které nejlépe vystihují tuto báseň a vypiš pouze je: \
\n{poem} \nKategorie: "
            if 'Chat' in model_name:
                out = model.create_chat_completion(
                    messages = [
                        {
                            "role": "system", 
                            "content": "Jsi asistent který rozumí básním a umí je kategorizovat."
                        },
                        {
                            "role": "user",
                            "content": input_text
                        }
                    ],
                    response_format={
                        "type": "json_object",
                    },
                    )
                categories = out['choices'][0]['message']['content']
            else:
                out = model(
                    f'{input_text}',
                    )
                categories = out['choices'][0]['text']
                
            
            
            categories = list(map(str.strip, re.findall(r'\w+', categories)))
            categories = list(filter(lambda x: len(x) > 0, categories))
            categories = list(filter(lambda x: x in StropheParams.POEM_TYPES, categories))
        
            file[i]['categories'] = categories
            
            input_text = f"Toto je báseň: \
\n{poem}\n\nNapiš schrnutí předešlé básně: "
            if 'Chat' in model_name:
                out = model.create_chat_completion(
                    messages = [
                        {
                            "role": "system", 
                            "content": "Jsi asistent který rozumí básním a umí je sumarizovat."
                        },
                        {
                            "role": "user",
                            "content": input_text
                        }
                    ],
                    response_format={
                        "type": "json_object",
                    },
                    )
                sumarization =  out['choices'][0]['message']['content']
            else:
                out = model(
                    f'{input_text}',
                    )
                sumarization =  out['choices'][0]['text']
            
        
            file[i]['sumarization'] = sumarization
            
        
        json.dump(file, open(os.path.join(args.data_path, poem_file), 'w+'), indent=6)   