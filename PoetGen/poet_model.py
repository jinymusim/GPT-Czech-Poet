from transformers import  AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import os
import argparse
import constants
import re
import random


class PoetModel(torch.nn.Module):
    def __init__(self, pretrainedModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = AutoModelForCausalLM.from_pretrained(pretrainedModel, output_hidden_states=True)
        self.vowels_regressor = torch.nn.Linear(768,1) # Number of Emmbedings of gpt2 is 768, we want 1 num out
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, vowel_count=None, rhyme=None):
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        last_hidden = outputs['hidden_states'][-1]
        vowel_regression = self.vowels_regressor(last_hidden[:,0,:].view(-1, 768))
        
        vowel_loss = None
        if vowel_count is not None:
            loss_fct = torch.nn.MSELoss()
            vowel_loss = loss_fct(vowel_regression.view(-1, 1), vowel_count.view(-1, 1))
        
        return {"model_output" : outputs, 
                "vowel_regression_output": vowel_regression, "vowel_regression_loss": vowel_loss,}
    
    def save_LM(self, LM_path):
        self.model.save_pretrained(LM_path)
        
    def analyze_prompt(self, prompt:str):
        features_dict = {
            "rhyme_scheme" : "",
            "included_scheme" : False,
            "type_1_len" : 0,
            "included_1_len" : False,
            "type_2_len" : 0,
            "included_2_len" : False,
            "type_3_len" : 0,
            "included_3_len" : False,
                      
        }
        
        lines = prompt.splitlines()
        lines = list(map(str.strip, lines))
        i = 0
        while i < len(lines):
            if not lines[i]:
                lines.pop(i)
                i-=1
            i+=1
        if len(lines) == 0:
            raise Exception("Empty Prompt!")
        elif len(lines) == 1:
            if lines[0].lower() in constants.rhyme_schemes:
                features_dict["rhyme_scheme"] = lines[0].upper()
                features_dict["included_scheme"] = True
            elif lines[0][0].isdigit():
                features_dict["type_1_len"] = int(lines[0][0])
                features_dict["included_1_len"] = True
        elif len(lines) > 1:
            if lines[0].lower() in constants.rhyme_schemes:
                features_dict["rhyme_scheme"] = lines[0].upper()
                features_dict["included_scheme"] = True
            elif lines[0][0].isdigit():
                features_dict["type_1_len"] = int(lines[0][0])
                features_dict["included_1_len"] = True
            for i in range(1, min(lines, 3)):
                if lines[i][0].isdigit():
                    features_dict[f"type_{i}_len"] = int(lines[i][0])
                    features_dict[f"included_{i}_len"] = True
                else:
                    features_dict[f"type_{i}_len"] = len(re.findall("a|e|i|o|u|y", lines[i]))
        
        if features_dict["rhyme_scheme"] == "":
            features_dict["rhyme_scheme"] = random.choice(constants.rhyme_schemes)
        if features_dict["type_1_len"] == 0:
            features_dict["type_1_len"] = random.randint(6,12)
        if features_dict["type_2_len"] == 0:
            features_dict["type_2_len"] = random.randint(6,12)
        if features_dict["type_3_len"] == 0:
            features_dict["type_3_len"] = random.randint(6,12)
        return features_dict
                   
    
    def generate_forced(self, prompt:str, tokenizer: AutoTokenizer):
        
        features_dict = self.analyze_prompt(prompt)
        prompt_list = prompt.splitlines()
        if not features_dict["included_scheme"]:
            prompt_list.insert(0, features_dict["rhyme_scheme"])
        for i in range(1, len(prompt_list)):
            j = 1
            if features_dict["included_scheme"][(i - 1) % len(features_dict["included_scheme"])] == "B":
                j = 2
            elif features_dict["included_scheme"][(i - 1) % len(features_dict["included_scheme"])] == "C":
                j = 3
            if not features_dict[f'included_{j}_len']:  
                prompt_list[i] = features_dict[f"type_{j}_len"] + " " + prompt_list[i]
        # Generating 4 verse rhymes
        while len(prompt_list) < 5:
            j = 1
            if features_dict["included_scheme"][(len(prompt_list) - 1) % len(features_dict["included_scheme"])] == "B":
                j = 2
            elif features_dict["included_scheme"][(len(prompt_list) - 1) % len(features_dict["included_scheme"])] == "C":
                j = 3
            line_start = features_dict[f"type_{j}_len"]
            tokenized_poet_start = tokenizer.encode("\n".join(prompt_list) + "\n" + line_start, return_tensors='pt')
            out_line =  self.model.generate(tokenized_poet_start, 
                                max_length=1000,
                                max_new_tokens= 100,
                                num_beams=2,
                                no_repeat_ngram_size=2,
                                early_stopping=True,
                                pad_token_id=tokenizer.eos_token_id)
            decoded_line = tokenizer.decode(out_line[0], skip_special_tokens=True).splitlines()[len(prompt_list)]
            prompt_list.append(decoded_line)
        
        return "\n".join(prompt_list)
            