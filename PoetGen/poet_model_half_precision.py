from transformers import  AutoModelForCausalLM, AutoTokenizer
from poet_model_interface import PoetModelInterface
import torch
from poet_constants import rhyme_schemes
import re
import random


class PoetModelHalfBase(PoetModelInterface):
    def __init__(self, pretrainedModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        #if "llama" in pretrainedModel:
        #    self.model = AutoModelForCausalLM.from_pretrained(pretrainedModel, 
        #                                                      output_hidden_states=True, 
        #                                                      max_memory = {i: torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())}, 
        #                                                      torch_dtype=torch.float16)
        #else:
        #    self.model = AutoModelForCausalLM.from_pretrained(pretrainedModel, 
        #                                                      output_hidden_states=True,
        #                                                      max_memory = {i: torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())}, 
        #

        self.model = AutoModelForCausalLM.from_pretrained(pretrainedModel, 
                                                        output_hidden_states=True, torch_dtype=torch.float16)
            
        model_config = self.model.config
        self.model_size = 1
        # Check for Hidden layer size by Attribute Name
        if hasattr(model_config, "n_embd"):
            self.model_size = model_config.n_embd
        elif hasattr(model_config, "hidden_size"):
            self.model_size = model_config.hidden_size
        
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        
        return {"model_output" : outputs,
                "loss" : outputs.loss}
    
    def save_LM(self, LM_path):
        self.model.save_pretrained(LM_path)
        
    def analyze_prompt(self, prompt:str):
        features_dict = {
            "rhyme_scheme" : "",
            "included_scheme" : False,
            "type_1_len" : 0,
            "included_1_len" : False,
            "type_1_end" : "",
            "included_1_end" : False,
            "type_2_len" : 0,
            "included_2_len" : False,
            "type_2_end" : "",
            "included_2_end" : False,
            "type_3_len" : 0,
            "included_3_len" : False,
            "type_3_end" : "",
            "included_3_end" : False,
                      
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
            if (lines[0].upper() in rhyme_schemes) or PoetModelInterface.rhyme_like(lines[0]):
                features_dict["rhyme_scheme"] = lines[0].upper()
                features_dict["included_scheme"] = True
            elif lines[0].split()[0].isdigit():
                features_dict["type_1_len"] = int(lines[0].split()[0])
                features_dict["included_1_len"] = True
        elif len(lines) > 1:
            if (lines[0].upper() in rhyme_schemes) or PoetModelInterface.rhyme_like(lines[0]):
                features_dict["rhyme_scheme"] = lines[0].upper()
                features_dict["included_scheme"] = True
            elif lines[0].split()[0].isdigit():
                features_dict["type_1_len"] = int(lines[0].split()[0])
                features_dict["included_1_len"] = True
            for i in range(1, min(lines, 3)):
                if lines[i].split()[0].isdigit():
                    features_dict[f"type_{i}_len"] = int(lines[i].split()[0])
                    features_dict[f"included_{i}_len"] = True
                else:
                    features_dict[f"type_{i}_len"] = len(re.findall("a|e|i|o|u|y", lines[i]))
        
        if features_dict["rhyme_scheme"] == "":
            features_dict["rhyme_scheme"] = random.choice(rhyme_schemes)
        if features_dict["type_1_len"] == 0:
            features_dict["type_1_len"] = random.randint(6,14)
        if features_dict["type_2_len"] == 0:
            features_dict["type_2_len"] = random.randint(6,14)
        if features_dict["type_3_len"] == 0:
            features_dict["type_3_len"] = random.randint(6,14)
        return features_dict
                   
    
    def generate_forced(self, prompt:str, tokenizer: AutoTokenizer, verse_len:int = 4):
        
        features_dict = self.analyze_prompt(prompt)
        prompt_list = prompt.splitlines()
        if not features_dict["included_scheme"]:
            prompt_list.insert(0, features_dict["rhyme_scheme"])
        for i in range(1, len(prompt_list)):
            j = 1
            if features_dict["rhyme_scheme"][(i - 1) % len(features_dict["rhyme_scheme"])] == "B":
                j = 2
            elif features_dict["rhyme_scheme"][(i - 1) % len(features_dict["rhyme_scheme"])] == "C":
                j = 3
            if not features_dict[f'included_{j}_len']:  
                prompt_list[i] = str(features_dict[f"type_{j}_len"]) + " " + prompt_list[i]
        # Generating 4 verse rhymes
        while len(prompt_list) <= verse_len:
            j = 1
            if features_dict["rhyme_scheme"][(len(prompt_list) - 1) % len(features_dict["rhyme_scheme"])] == "B":
                j = 2
            elif features_dict["rhyme_scheme"][(len(prompt_list) - 1) % len(features_dict["rhyme_scheme"])] == "C":
                j = 3
            line_start = str(features_dict[f"type_{j}_len"]) + (f" {features_dict[f'type_{j}_end'] } #" if features_dict[f'type_{j}_end'] != "" else "")
            tokenized_poet_start = tokenizer.encode("\n".join(prompt_list) + "\n" + line_start, return_tensors='pt')
            out_line =  self.model.generate(tokenized_poet_start, 
                                max_new_tokens= 100,
                                num_beams=2,
                                no_repeat_ngram_size=2,
                                early_stopping=True,
                                pad_token_id=tokenizer.eos_token_id)
            decoded_line: str = tokenizer.decode(out_line[0], skip_special_tokens=True).splitlines()[len(prompt_list)]
            if features_dict[f'type_{j}_end'] == "":
                features_dict[f'type_{j}_end'] = decoded_line.split()[1]
            prompt_list.append(decoded_line)
        
        return "\n".join(prompt_list)
            