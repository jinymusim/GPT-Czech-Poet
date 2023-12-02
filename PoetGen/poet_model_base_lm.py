from transformers import  AutoModelForCausalLM, AutoTokenizer
from utils.poet_model_utils import PoetModelInterface
from utils.poet_utils import TextAnalysis, RHYME_SCHEMES

from transformers.utils import ModelOutput

import random

class PoetModelBase(PoetModelInterface):
    def __init__(self, pretrainedModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(pretrainedModel, 
                                                        output_hidden_states=True,
                                                        ignore_mismatched_sizes=True)
            
        model_config = self.model.config
        self.model_size = 1
        # Check for Hidden layer size by Attribute Name
        if hasattr(model_config, "n_embd"):
            self.model_size = model_config.n_embd
        elif hasattr(model_config, "hidden_size"):
            self.model_size = model_config.hidden_size
        
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        
        return ModelOutput(loss= outputs.loss, model_output=outputs) # {"model_output" : outputs,"loss" : outputs.loss}
    
    def save_LM(self, LM_path):
        self.model.save_pretrained(LM_path)
        
        
    def analyze_prompt(self, prompt):
        if isinstance(prompt, dict):
            return prompt 
        features_dict = {}  
        lines = prompt.splitlines()
        lines = list(map(str.strip, lines))
        i = 0
        while i < len(lines):
            if not lines[i]:
                lines.pop(i)
                i-=1
            i+=1
        cont_line = 0
        for line in lines:
            if TextAnalysis._is_param_line(line):
                for key, value in TextAnalysis._first_line_analysis(line).items():
                    features_dict[key] = value
            else:
                for key, value in TextAnalysis._continuos_line_analysis(line).items():
                    features_dict[f"{key}_{cont_line}"] = value
                    cont_line += 1
        return features_dict
                   
    
    def generate_forced(self, prompt, tokenizer: AutoTokenizer, verse_len:int = 4, sample: bool = False):
        
        features_dict_init = self.analyze_prompt(prompt)
        if isinstance(prompt, dict):
            prompt_list = []
        else:
            prompt_list = prompt.splitlines()
        # GENERATE FOR POSSIBLE MISSING POET PARAM
        token_gen_rhyme = tokenizer.encode("#", return_tensors='pt')
        if sample:
            rhyme_line = self.model.generate(token_gen_rhyme, 
                                max_new_tokens= 100,
                                do_sample=True,
                                top_k=50,
                                early_stopping=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
        else:
            rhyme_line = self.model.generate(token_gen_rhyme, 
                                max_new_tokens= 100,
                                num_beams=8,
                                no_repeat_ngram_size=2,
                                early_stopping=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
        rhyme_dec = tokenizer.decode(rhyme_line[0], skip_special_tokens=True).splitlines()[0]
        features_dict= TextAnalysis._first_line_analysis(rhyme_dec)
        for key, value in features_dict_init.items():
            features_dict[key] = value
        # CONSTRUCT BEST INPUT LINE
        # BACKUP RHYME
        if "RHYME" not in features_dict.keys():
            features_dict["RHYME"] = random.choice(RHYME_SCHEMES[:-1])
        poet_param_str = "# "
        if "RHYME" in features_dict.keys():
            poet_param_str += features_dict["RHYME"]
        if "YEAR" in features_dict.keys():
            poet_param_str += f" # {features_dict['YEAR']}"
        # REPLACE OR INSERT BASED ON PRESENCE
        if len(features_dict_init.keys()) == 0: # Wierd Input
            prompt_list = [poet_param_str]
        elif len(prompt_list) == 0: # Inputed as Dict
            prompt_list.append(poet_param_str)
        elif "RHYME" not in features_dict_init.keys():
            if "YEAR" in features_dict_init.keys(): # Replace the Uncomplete first line 
                prompt_list[0] = poet_param_str
            else:
                prompt_list.insert(0, poet_param_str)
        else:
            prompt_list[0] = poet_param_str
            
        verse_len = len(features_dict["RHYME"])
        
        
        # Generating 4 verse rhymes
        has_rep= False
        has_rep_again = False
        while len(prompt_list) <= verse_len:
            j = 0
            if features_dict["RHYME"][(len(prompt_list) - 1) % len(features_dict["RHYME"])] == "B":
                j = 1
            elif features_dict["RHYME"][(len(prompt_list) - 1) % len(features_dict["RHYME"])] == "C":
                j = 2
            elif features_dict["RHYME"][(len(prompt_list) - 1) % len(features_dict["RHYME"])] == "D":
                j = 3
            elif features_dict["RHYME"][(len(prompt_list) - 1) % len(features_dict["RHYME"])] == "X":
                j=-1
            line_start = (f"{features_dict[f'METER_{j}'] } #" if f"METER_{j}" in features_dict.keys() else "") + \
                (f" {features_dict[f'LENGTH_{j}']} #" if f"LENGTH_{j}" in features_dict.keys() else "" ) + \
                (f" {features_dict[f'END_{j}'] } #" if  f"END_{j}" in features_dict.keys() else "") 
            tokenized_poet_start = tokenizer.encode("\n".join(prompt_list) + "\n" + line_start,  return_tensors='pt')
            if sample:
                out_line =  self.model.generate(tokenized_poet_start, 
                                max_new_tokens= 100,
                                do_sample=True,
                                top_k=50,
                                early_stopping=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
            else:
                out_line =  self.model.generate(tokenized_poet_start, 
                                max_new_tokens= 100,
                                num_beams=2,
                                no_repeat_ngram_size=2,
                                early_stopping=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
            decoded_lines = tokenizer.decode(out_line[0], skip_special_tokens=True).splitlines()
            # Repetition catcher
           
            # Possible 
            if len(decoded_lines) <= len(prompt_list) and not(has_rep_again and has_rep):
                if has_rep:
                    prompt_list.pop()
                    has_rep= False
                    has_rep_again = True
                else:
                    has_rep = True
                continue
            if has_rep_again and has_rep:
                decoded_line: str = decoded_lines[-1]
            else:
                decoded_line: str = decoded_lines[len(prompt_list)]
            if  f"END_{j}" not in features_dict.keys() and len(decoded_line.split()) > 4 and j>=0:
                features_dict[f'METER_{j}'] = decoded_line.split()[0]
                features_dict[f'LENGTH_{j}'] = decoded_line.split()[2]
                features_dict[f'END_{j}'] = decoded_line.split()[4]
                
            prompt_list.append(decoded_line)
        
        return "\n".join(prompt_list)
            