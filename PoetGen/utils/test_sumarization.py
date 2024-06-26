import argparse
import os
import torch
import json
import re
from tqdm import tqdm
import random

from poet_utils import StropheParams

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ['PYTORCH_CUDA_ALLOC_CONF']= 'expandable_segments:True'

parser = argparse.ArgumentParser()


# h2oai/h2ogpt-4096-llama2-7b-chat              Asi ne, vše je misticisum
# simecek/cswikimistral_0.1                     Neschrnuje, není použitelné    
# mistralai/Mistral-7B-Instruct-v0.2            Většina příroda, schrnutí takové meh
# NickyNicky/Mistral-7B-OpenOrca-oasst_top1_2023-08-25-v2       Použitelné, jak kategorie tak shrnutí
# HuggingFaceH4/zephyr-7b-beta          Ne uplně dobré
# Qwen/Qwen1.5-1.8B-Chat            Halucinace Kategorií
# Qwen/Qwen1.5-7B-Chat              Čeština není super
# Open-Orca/Mistral-7B-OpenOrca

# mistralai/Mixtral-8x7B-Instruct-v0.1
# jarradh/llama2_70b_chat_uncensored
# mistralai/Mixtral-8x7B-v0.1
parser.add_argument("--model", default='mistralai/Mixtral-8x7B-v0.1', type=str, help='Huggingface model id')
parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__),'..', "corpusCzechVerse", "ccv-new")), type=str, help="Path to Data")

if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)

tokenizer = AutoTokenizer.from_pretrained(args.model)

with torch.no_grad():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        four_bits =  BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_compute_dtype=torch.bfloat16)
    
        
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=four_bits)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model)
    
    model.eval()
    dataset= os.listdir(args.data_path)
    random.shuffle(dataset)

    for poem_file in tqdm(dataset):
        if not os.path.isfile(os.path.join(args.data_path, poem_file)):
            continue
        
        file = json.load(open(os.path.join(args.data_path, poem_file) , 'r'))
        if 'categories' in file[0].keys():
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
Vyber z těchto kategorií ty, které nejlépe vystihují tuto báseň: \
\n{poem}\n=========\nkategorie:"
            tokenized = tokenizer(input_text, return_tensors='pt').to(device)
            out = model.generate(**tokenized, 
                        do_sample=True,
                        max_new_tokens= 100,
                        top_k=50,
                        eos_token_id = tokenizer.eos_token_id,
                        pad_token_id = tokenizer.pad_token_id ).detach().cpu()[0]
            out_decoded = tokenizer.decode(out, skip_special_tokens=True)
            
            categories = list(map(str.strip, re.findall(r'\w+', out_decoded)))
            categories = list(filter(lambda x: len(x) > 0, categories))
            categories = list(filter(lambda x: x in StropheParams.POEM_TYPES, categories))
        
            file[i]['categories'] = categories
            
            input_text = f"Toto je báseň: \
\n{poem}\n=========\nToto je schrnutí předešlé básně:"
            tokenized = tokenizer(input_text, return_tensors='pt').to(device)
            out = model.generate(**tokenized, 
                        do_sample=True,
                        max_new_tokens= 500,
                        top_k=50,
                        eos_token_id = tokenizer.eos_token_id,
                        pad_token_id = tokenizer.pad_token_id ).detach().cpu()[0]
            out_decoded = tokenizer.decode(out, skip_special_tokens=True)
            
            sumarization =  out_decoded[len(input_text):].strip()
        
            file[i]['sumarization'] = sumarization
            
        
        json.dump(file, open(os.path.join(args.data_path, poem_file), 'w+'), indent=6)   