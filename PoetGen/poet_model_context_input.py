import torch
import random

from transformers import  AutoModelForCausalLM, AutoTokenizer
from utils.poet_model_utils import PoetModelInterface, ContextModule
from utils.poet_utils import RHYME_SCHEMES, TextAnalysis

class PoetModelContextInput(PoetModelInterface):
    def __init__(self, pretrainedModel, context_input_size:int = 2048, block_count:int=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = AutoModelForCausalLM.from_pretrained(pretrainedModel, 
                                                            output_hidden_states=True)
            
        
        model_config = self.model.config
        self.model_size = -1
        # Check for Hidden layer size by Attribute Name
        if hasattr(model_config, "n_embd"):
            self.model_size = model_config.n_embd
        elif hasattr(model_config, "hidden_size"):
            self.model_size = model_config.hidden_size  # Number of Emmbedings taken from config
        self.context_size = context_input_size
            
            
        self.model.base_model.h.insert(3, ContextModule(block_count, context_input_size, self.model_size, self.model_size))
        # Because of Inserted Layer, Head Masks don't match => Add 1 more
        self.model.base_model.config.n_layer += 1 
        
        self.rhyme_regressor = torch.nn.Linear(self.model_size, len(RHYME_SCHEMES)) # Rhyme Type
        
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, rhyme=None, context_ids=None, context_attention_mask=None,*args, **kwargs):
        # Inject Context to bypass GPT2Blocks (Can't Forward it)
        self.model.base_model.h[3].context_ids = context_ids
        self.model.base_model.h[3].context_attention_mask = context_attention_mask
        outputs = self.model(input_ids=input_ids, 
                             labels=labels, 
                             attention_mask=attention_mask)
        
        last_hidden = outputs['hidden_states'][-1]
        
        rhyme_regression = self.rhyme_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
        
        full_loss = outputs.loss
             
        rhyme_loss = None
        if rhyme is not None:
            softmaxed = torch.softmax(rhyme_regression, dim=1)
            loss_fct = torch.nn.CrossEntropyLoss()
            rhyme_loss = loss_fct(softmaxed, rhyme)
            full_loss = full_loss + rhyme_loss
        # Delete the Injection to prevent Dataloss
        self.model.base_model.h[3].context_ids = None
        self.model.base_model.h[3].context_attention_mask = None
        
        return {"model_output" : outputs,
                "rhyme_regression_output": rhyme_regression,
                "rhyme_regression_loss": rhyme_loss,
                "loss": full_loss}
    
    def save_LM(self, LM_path):
        self.model.save_pretrained(LM_path)
        

        
    def analyze_prompt(self, prompt:str):
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
                   
    
    def generate_forced(self, prompt:str, tokenizer: AutoTokenizer, verse_len:int = 4, sample: bool = False):
        
        features_dict_init = self.analyze_prompt(prompt)
        prompt_list = prompt.splitlines()
        # GENERATE FOR POSSIBLE MISSING POET PARAM
        token_gen_rhyme = tokenizer.encode("A", return_tensors='pt')
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
        poet_param_str = ""
        if "RHYME" in features_dict.keys():
            poet_param_str += features_dict["RHYME"]
        if "YEAR" in features_dict.keys():
            poet_param_str += f" # {features_dict['YEAR']}"
        # REPLACE OR INSERT BASED ON PRESENCE
        if len(features_dict_init.keys()) == 0: # Wierd Input
            prompt_list = [poet_param_str]
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
            line_start =  (features_dict[f"LENGTH_{j}"] if f"LENGTH_{j} # " in features_dict.keys() else "" )  + \
                (f"{features_dict[f'END_{j}'] } # " if  f"END_{j}" in features_dict.keys() else "") + \
                (f"{features_dict[f'METER_{j}'] } # " if f"METER_{j}" in features_dict.keys() else "")
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
            if  f"METER_{j}" not in features_dict.keys() and len(decoded_line.split()) > 4 and j>=0:
                features_dict[f'LENGTH_{j}'] = decoded_line.split()[0]
                features_dict[f'END_{j}'] = decoded_line.split()[2]
                features_dict[f'METER_{j}'] = decoded_line.split()[4]
            prompt_list.append(decoded_line)
        
        return "\n".join(prompt_list)
            