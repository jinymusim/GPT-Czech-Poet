import torch

from transformers import  AutoModelForCausalLM, AutoTokenizer
from poet_model_interface import PoetModelInterface
from poet_utils import RHYME_SCHEMES,TextAnalysis

class PoetModelSecondaryTasks(PoetModelInterface):
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
                                                            output_hidden_states=True)
            
        model_config = self.model.config
        self.model_size = -1
        # Check for Hidden layer size by Attribute Name
        if hasattr(model_config, "n_embd"):
            self.model_size = model_config.n_embd
        elif hasattr(model_config, "hidden_size"):
            self.model_size = model_config.hidden_size  # Number of Emmbedings taken from config
        self.vowels_regressor = torch.nn.Linear(self.model_size,1) # Vowel count
        self.rhyme_regressor = torch.nn.Linear(self.model_size, len(RHYME_SCHEMES)) # Rhyme Type
        
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, vowel_count=None, rhyme=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        last_hidden = outputs['hidden_states'][-1]
        vowel_regression = self.vowels_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
        
        rhyme_regression = self.rhyme_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
        
        full_loss = outputs.loss
        
        vowel_loss = None
        if vowel_count is not None:
            loss_fct = torch.nn.MSELoss()
            vowel_loss = loss_fct(vowel_regression.view(-1, 1), vowel_count.view(-1, 1))
            full_loss = full_loss + vowel_loss
            
        rhyme_loss = None
        if rhyme is not None:
            softmaxed = torch.softmax(rhyme_regression, dim=1)
            loss_fct = torch.nn.CrossEntropyLoss()
            rhyme_loss = loss_fct(softmaxed, rhyme)
            full_loss = full_loss + rhyme_loss
            
        
        return {"model_output" : outputs,
                "vowel_regression_output": vowel_regression, 
                "vowel_regression_loss": vowel_loss,
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
                    features_dict[f"{key}_{cont_line}"]
                    cont_line += 1
        return features_dict
                   
    
    def generate_forced(self, prompt:str, tokenizer: AutoTokenizer, verse_len:int = 4):
        
        features_dict_init = self.analyze_prompt(prompt)
        prompt_list = prompt.splitlines()
        # GENERATE FOR POSSIBLE MISSING POET PARAM
        token_gen_rhyme = tokenizer.encode("A", return_tensors='pt')
        rhyme_line = self.model.generate(token_gen_rhyme, 
                                max_new_tokens= 100,
                                num_beams=2,
                                no_repeat_ngram_size=2,
                                early_stopping=True,
                                pad_token_id=tokenizer.eos_token_id)
        rhyme_dec = tokenizer.decode(rhyme_line[0], skip_special_tokens=True).splitlines()[0]
        features_dict= self.analyze_prompt(rhyme_dec)
        for key, value in features_dict_init.items():
            features_dict[key] = value
        # CONSTRUCT BEST INPUT LINE
        poet_param_str = ""
        if "RHYME" in features_dict.keys():
            poet_param_str += features_dict["RHYME"]
        if "YEAR" in features_dict.keys():
            poet_param_str += f" ## {features_dict['YEAR']}"
        if "METER" in features_dict.keys():
            poet_param_str += f" ## {features_dict['METER']}"
        # REPLACE OR INSERT BASED ON PRESENCE
        if "RHYME" not in features_dict_init.keys():
            prompt_list.insert(0, poet_param_str)
        else:
            prompt_list[0] = poet_param_str
        
        
        # Generating 4 verse rhymes
        while len(prompt_list) <= verse_len:
            j = 0
            if features_dict["RHYME"][(len(prompt_list) - 1) % len(features_dict["RHYME"])] == "B":
                j = 1
            elif features_dict["RHYME"][(len(prompt_list) - 1) % len(features_dict["RHYME"])] == "C":
                j = 2
            elif features_dict["RHYME"][(len(prompt_list) - 1) % len(features_dict["RHYME"])] == "D":
                j = 3
            line_start =  (features_dict[f"LENGTH_{j}"] if f"LENGTH_{j}" in features_dict.keys() else "" )  + \
                (f" {features_dict[f'END_{j}'] } #" if  f"END_{j}" in features_dict.keys() else "")
            tokenized_poet_start = tokenizer.encode("\n".join(prompt_list) + "\n" + line_start,  return_tensors='pt')
            out_line =  self.model.generate(tokenized_poet_start, 
                                max_new_tokens= 100,
                                num_beams=2,
                                no_repeat_ngram_size=2,
                                early_stopping=True,
                                pad_token_id=tokenizer.eos_token_id)
            decoded_line: str = tokenizer.decode(out_line[0], skip_special_tokens=True).splitlines()[len(prompt_list)]
            if  f"LENGTH_{j}" not in features_dict.keys() and len(decoded_line.split()) > 1:
                features_dict[f'LENGTH_{j}'] = decoded_line.split()[0]
                features_dict[f'END_{j}'] = decoded_line.split()[1]
            prompt_list.append(decoded_line)
        
        return "\n".join(prompt_list)
            