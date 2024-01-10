from .poet_model_utils import PoetModelInterface
from .poet_utils import TextAnalysis, StropheParams

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import ModelOutput
import random
import torch

class PoetModelFunctionalInterface(PoetModelInterface):
    """Poet Model Functional Interface. Abstract class with implementation of 

    Args:
        PoetModelInterface (_type_): Is child of PoetModelInterface for carrying core methods 
    """
    def __init__(self, *args, **kwargs) -> None:
        """ Constructor. As child Class needs to construct Parent
        """
        super().__init__(*args, **kwargs)
        
    def analyze_prompt(self, prompt) -> dict:
        """Analysis of users prompt

        Args:
            prompt (_type_): dict or string, carrying users intent

        Returns:
            dict: Analysis with users intended input
        """
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
                   
    def generate_forced(self, prompt, tokenizer: AutoTokenizer, sample: bool = True, format: str = 'METER_VERSE', device= torch.device('cpu'), *args, **kwargs) -> str:
        """Generate Strophe using the FORCED generation

        Args:
            prompt (_type_): dict or string of users intended parameters of strophe start
            tokenizer (AutoTokenizer): tokenizer to be used during generation. Should be model specific.
            sample (bool, optional): If to sample. Defaults to False.
            format (str, optional): Format of generation to be used. Should be same as trained on. possible formats: BASIC, VERSE_PAR, METER_VERSE, OLD (DEPRECATED! For old models compatibility only). Defaults to 'METER_VERSE'.
            device (_type_, optional): Device to generate on. CPU as default. Defaults to torch.device('cpu').

        Returns:
            str: Generated Strophe
        """     
        features_dict_init = self.analyze_prompt(prompt)
        # If user parameters as dict, list is initialized to carry future verses.
        if isinstance(prompt, dict):
            prompt_list = []
        else:
            prompt_list = prompt.splitlines()
        # GENERATE FOR POSSIBLE MISSING POET PARAM
        token_gen_rhyme = tokenizer.encode("#", return_tensors='pt')
        if sample:
            rhyme_line = self.model.generate(token_gen_rhyme.to(device), 
                                max_new_tokens= 100,
                                do_sample=True,
                                top_k=50,
                                early_stopping=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
        else:
            rhyme_line = self.model.generate(token_gen_rhyme.to(device), 
                                max_new_tokens= 100,
                                num_beams=8,
                                no_repeat_ngram_size=2,
                                early_stopping=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
        rhyme_dec = tokenizer.decode(rhyme_line.cpu()[0], skip_special_tokens=True).splitlines()[0]
        features_dict= TextAnalysis._first_line_analysis(rhyme_dec)
        for key, value in features_dict_init.items():
            features_dict[key] = value
        # CONSTRUCT BEST INPUT LINE
        # BACKUP RHYME
        if "RHYME" not in features_dict.keys():
            features_dict["RHYME"] = random.choice(StropheParams.RHYME[:-1])
        #OLD
        if format == 'OLD':
            poet_param_str = ""
            if "RHYME" in features_dict.keys():
                poet_param_str += features_dict["RHYME"]
            if "YEAR" in features_dict.keys():
                poet_param_str += f" # {features_dict['YEAR']}"
            if 'STROPHE_METER' in features_dict.keys():
                poet_param_str += f" # {features_dict['STROPHE_METER']}"
            
        elif format != 'METER_VERSE':
            poet_param_str = "# "
            if "RHYME" in features_dict.keys():
                poet_param_str += features_dict["RHYME"]
            if "YEAR" in features_dict.keys():
                poet_param_str += f" # {features_dict['YEAR']}"
            if 'STROPHE_METER' in features_dict.keys():
                poet_param_str += f" # {features_dict['STROPHE_METER']}"
        # NEW
        else:
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
            if "YEAR" in features_dict_init.keys() or 'STROPHE_METER' in features_dict_init.keys(): # Replace the Uncomplete first line 
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
            #OLD
            if format == 'BASIC':
                line_start = ""
            elif format == 'OLD':
                line_start = (f"{features_dict[f'LENGTH_{j}']} " if f"LENGTH_{j}" in features_dict.keys() else "" ) + \
                        (f" {features_dict[f'END_{j}'] } #" if  f"END_{j}" in features_dict.keys() else "") 
            elif format == 'VERSE_PAR':
                line_start = (f"{features_dict[f'LENGTH_{j}']} #" if f"LENGTH_{j}" in features_dict.keys() else "" ) + \
                        (f" {features_dict[f'END_{j}'] } #" if  f"END_{j}" in features_dict.keys() else "") 
            else:
                line_start = (f"{features_dict[f'METER_{j}'] } #" if f"METER_{j}" in features_dict.keys() else "") + \
                (f" {features_dict[f'LENGTH_{j}']} #" if f"LENGTH_{j}" in features_dict.keys() else "" ) + \
                (f" {features_dict[f'END_{j}'] } #" if  f"END_{j}" in features_dict.keys() else "") 
            tokenized_poet_start = tokenizer.encode("\n".join(prompt_list) + "\n" + line_start,  return_tensors='pt')
            if sample:
                out_line =  self.model.generate(tokenized_poet_start.to(device), 
                                max_new_tokens= 100,
                                do_sample=True,
                                top_k=50,
                                early_stopping=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
            else:
                out_line =  self.model.generate(tokenized_poet_start.to(device), 
                                max_new_tokens= 100,
                                num_beams=2,
                                no_repeat_ngram_size=2,
                                early_stopping=True,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id)
            decoded_lines = tokenizer.decode(out_line.cpu()[0], skip_special_tokens=True).splitlines()
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
            #OLD
            if format == 'VERSE_PAR' or format == 'OLD':
                if  f"END_{j}" not in features_dict.keys() and len(decoded_line.split()) > 1 and j>=0 and decoded_line.count("#") <=1:
                    features_dict[f'LENGTH_{j}'] = decoded_line.split()[0]
                    features_dict[f'END_{j}'] = decoded_line.split()[1]
                elif f"END_{j}" not in features_dict.keys() and len(decoded_line.split()) > 2 and j>=0:
                    features_dict[f'LENGTH_{j}'] = decoded_line.split()[0]
                    features_dict[f'END_{j}'] = decoded_line.split()[2]            
            # NEW
            elif format == 'METER_VERSE':    
                if  f"END_{j}" not in features_dict.keys() and len(decoded_line.split()) > 4 and j>=0:
                    features_dict[f'METER_{j}'] = decoded_line.split()[0]
                    features_dict[f'LENGTH_{j}'] = decoded_line.split()[2]
                    features_dict[f'END_{j}'] = decoded_line.split()[4]
                
            prompt_list.append(decoded_line)
        
        return "\n".join(prompt_list)

    
class PoetModelBase(PoetModelFunctionalInterface):
    def __init__(self, pretrainedModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(pretrainedModel, output_hidden_states=True)
            
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
        self.model.save_pretrained(LM_path, safe_serialization=False)
        
        
class PoetModelAllTasks(PoetModelFunctionalInterface):
    def __init__(self, pretrainedModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(pretrainedModel, output_hidden_states=True)
            
        model_config = self.model.config
        self.model_size = 1
        # Check for Hidden layer size by Attribute Name
        if hasattr(model_config, "n_embd"):
            self.model_size = model_config.n_embd
        elif hasattr(model_config, "hidden_size"):
            self.model_size = model_config.hidden_size
            
        self.vowels_regressor = torch.nn.Linear(self.model_size,1) # Vowel Count
        self.rhyme_regressor = torch.nn.Linear(self.model_size, len(StropheParams.RHYME)) # Rhyme Type
        self.verse_endings = torch.nn.Linear(self.model_size, len(StropheParams.ENDS)) # Verse End Syllable
        self.metre_regressor = torch.nn.Linear(self.model_size,len(StropheParams.METER)) # Meter Type
        self.year_regressor = torch.nn.Linear(self.model_size,len(StropheParams.YEAR)) # Year Bucket
        
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, nums=None, rhyme=None, verse_end=None, year=None, metre=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        last_hidden = outputs['hidden_states'][-1]
        vowel_regression = self.vowels_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
        rhyme_regression = self.rhyme_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
        verse_end_reg = self.verse_endings((last_hidden[:,0,:].view(-1, self.model_size)))
        metre_regression = self.metre_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
        year_regression = self.year_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
        full_loss = outputs.loss
        
        vowel_loss = None
        if nums is not None:
            loss_fct = torch.nn.MSELoss()
            vowel_loss = loss_fct(vowel_regression.view(-1, 1), nums.view(-1, 1))
            full_loss = full_loss + 0.1*vowel_loss
            
        rhyme_loss = None
        if rhyme is not None:
            softmaxed = torch.softmax(rhyme_regression, dim=1)
            loss_fct = torch.nn.CrossEntropyLoss()
            rhyme_loss = loss_fct(softmaxed, rhyme)
            full_loss = full_loss + 0.1*rhyme_loss
            
        verse_loss = None
        if verse_end is not None:
            softmaxed = torch.softmax(verse_end_reg, dim=1)
            loss_fct = torch.nn.CrossEntropyLoss()
            verse_loss = loss_fct(softmaxed, verse_end)
            full_loss = full_loss + 0.1*verse_loss
            
        metre_loss = None
        if metre is not None:
            softmaxed = torch.softmax(metre_regression, dim=1)
            loss_fct = torch.nn.CrossEntropyLoss()
            metre_loss = loss_fct(softmaxed, metre)
            full_loss = full_loss + 0.1*metre_loss
        
        year_loss = None
        if year is not None:
            softmaxed = torch.softmax(year_regression, dim=1)
            loss_fct = torch.nn.CrossEntropyLoss()
            year_loss = loss_fct(softmaxed, year)
            full_loss = full_loss + 0.1*year_loss
            
        
        return {"model_output" : outputs,
                "vowel_regression_output": vowel_regression, 
                "vowel_regression_loss": vowel_loss,
                "rhyme_regression_output": rhyme_regression,
                "rhyme_regression_loss": rhyme_loss,
                "verse_end_regression_output" : verse_end_reg,
                "verse_end_regression_loss" : verse_loss,
                "metre_regression_output" : metre_regression,
                "metre_regression_loss" : metre_loss,
                "year_regression_output" : year_regression,
                "year_regression_loss" : year_loss,
                "loss": full_loss}
    
    def save_LM(self, LM_path):
        self.model.save_pretrained(LM_path, safe_serialization=False)

from .poet_model_utils import ContextModule
        
class PoetModelContextInput(PoetModelFunctionalInterface):
    def __init__(self, pretrainedModel, context_input_size:int = 2048, block_count:int=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = AutoModelForCausalLM.from_pretrained(pretrainedModel,output_hidden_states=True)
             
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
        
        self.rhyme_regressor = torch.nn.Linear(self.model_size, len(StropheParams.RHYME)) # Rhyme Type
        
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, rhyme=None, context_ids=None, context_attention_mask=None,*args, **kwargs):
        # Inject Context to bypass GPT2Blocks (Can't Forward it)
        self.model.base_model.h[3].context_ids = context_ids
        self.model.base_model.h[3].context_attention_mask = context_attention_mask
        
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)   
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

from .poet_model_utils import PoetTypeModule  
        
class PoetModelContextYear(PoetModelFunctionalInterface):
    def __init__(self, pretrainedModel, context_input_size:int = 2048, block_count:int=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = AutoModelForCausalLM.from_pretrained(pretrainedModel, output_hidden_states=True)        
        
        model_config = self.model.config
        self.model_size = -1
        # Check for Hidden layer size by Attribute Name
        if hasattr(model_config, "n_embd"):
            self.model_size = model_config.n_embd
        elif hasattr(model_config, "hidden_size"):
            self.model_size = model_config.hidden_size  # Number of Emmbedings taken from config
        self.context_size = context_input_size
            
            
        self.model.base_model.h.insert(3, ContextModule(block_count, context_input_size, self.model_size, self.model_size))
        self.model.base_model.h.insert(3, PoetTypeModule(block_count, context_input_size, self.model_size, self.model_size))
        # Because of Inserted Layer, Head Masks don't match => Add 1 more
        self.model.base_model.config.n_layer += 2
        
        self.rhyme_regressor = torch.nn.Linear(self.model_size, len(StropheParams.RHYME)) # Rhyme Type
        self.year_regressor = torch.nn.Linear(self.model_size, len(StropheParams.YEAR)) # Year Bucket
        
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, rhyme=None, context_ids=None, context_attention_mask=None, year=None,*args, **kwargs):
        # Inject Context to bypass GPT2Blocks (Can't Forward it)
        self.model.base_model.h[3].context_ids = context_ids
        self.model.base_model.h[3].context_attention_mask = context_attention_mask
        self.model.base_model.h[3].type_labels = year
        
        self.model.base_model.h[4].context_ids = context_ids
        self.model.base_model.h[4].context_attention_mask = context_attention_mask
        
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        last_hidden = outputs['hidden_states'][-1]
        rhyme_regression = self.rhyme_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
        full_loss = outputs.loss
             
        rhyme_loss = None
        if rhyme is not None:
            softmaxed = torch.softmax(rhyme_regression, dim=1)
            loss_fct = torch.nn.CrossEntropyLoss()
            rhyme_loss = loss_fct(softmaxed, rhyme)
            full_loss = full_loss + rhyme_loss
            
        
        year_regression = self.year_regressor((last_hidden[:,0,:].view(-1, self.model_size)))    
        
        year_loss = None
        if year is not None:
            softmaxed = torch.softmax(year_regression, dim=1)
            loss_fct = torch.nn.CrossEntropyLoss()
            year_loss = loss_fct(softmaxed, year)
            full_loss = full_loss + year_loss +  self.model.base_model.h[3].indiv_loss
        
        # Delete the Injection to prevent Dataloss
        self.model.base_model.h[3].context_ids = None
        self.model.base_model.h[3].context_attention_mask = None
        self.model.base_model.h[3].type_labels = None
        # Delete Loss 
        self.model.base_model.h[3].indiv_loss = None
        
        self.model.base_model.h[4].context_ids = None
        self.model.base_model.h[4].context_attention_mask = None
        
        return {"model_output" : outputs,
                "rhyme_regression_output": rhyme_regression,
                "rhyme_regression_loss": rhyme_loss,
                "year_regression_output" : year_regression,
                "year_loss" : year_loss,
                "loss": full_loss}
    
    def save_LM(self, LM_path):
        self.model.save_pretrained(LM_path)
        
        
class DistilModel(PoetModelFunctionalInterface):
    
    def __init__(self, pretrainedModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(pretrainedModel, output_hidden_states=True)
            
        model_config = self.model.config
        self.model_size = 1
        # Check for Hidden layer size by Attribute Name
        if hasattr(model_config, "n_embd"):
            self.model_size = model_config.n_embd
        elif hasattr(model_config, "hidden_size"):
            self.model_size = model_config.hidden_size
        
        self.kept_states = [1, 3, 5, 7, 9, 11]
            
        for pop_index in sorted(list(set(range(len(self.model.base_model.h))) - set(self.kept_states)), reverse=True):
            
            self.model.base_model.h.pop(pop_index)
        # Because of Inserted Layer, Head Masks don't match => Add 1 more
        self.model.base_model.config.n_layer = len(self.kept_states)
        
        self.loss_fnc = torch.nn.MSELoss()
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, to_replicate_states= None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        # The 6 layers + embeddings (add + 1 to shift the original_index)
        for distil_index, original_index in enumerate([-1] + self.kept_states):
            loss += self.loss_fnc(outputs['hidden_states'][distil_index], to_replicate_states[original_index + 1])
        
        return {"model_output" : outputs,
                "loss": loss}
    
    def save_LM(self, LM_path):
        self.model.save_pretrained(LM_path, safe_serialization=False)
        
    def generate_forced(self, *args, **kwargs):
        raise NotImplementedError("Currently without")
    
class PoetModelHalfBase(PoetModelFunctionalInterface):
    def __init__(self, pretrainedModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
        self.model = AutoModelForCausalLM.from_pretrained(pretrainedModel, output_hidden_states=True, torch_dtype=torch.float16)
            
        model_config = self.model.config
        self.model_size = -1
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
        
        
class PoetModelSecondaryTasks(PoetModelFunctionalInterface):
    def __init__(self, pretrainedModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = AutoModelForCausalLM.from_pretrained(pretrainedModel, output_hidden_states=True)
            
        model_config = self.model.config
        self.model_size = -1
        # Check for Hidden layer size by Attribute Name
        if hasattr(model_config, "n_embd"):
            self.model_size = model_config.n_embd
        elif hasattr(model_config, "hidden_size"):
            self.model_size = model_config.hidden_size  # Number of Emmbedings taken from config
        self.vowels_regressor = torch.nn.Linear(self.model_size,1) # Vowel count
        self.rhyme_regressor = torch.nn.Linear(self.model_size, len(StropheParams.RHYME)) # Rhyme Type
        
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, nums=None, rhyme=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        last_hidden = outputs['hidden_states'][-1]
        vowel_regression = self.vowels_regressor((last_hidden[:,0,:].view(-1, self.model_size))) 
        rhyme_regression = self.rhyme_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
        full_loss = outputs.loss
        
        vowel_loss = None
        if nums is not None:
            loss_fct = torch.nn.MSELoss()
            vowel_loss = loss_fct(vowel_regression.view(-1, 1), nums.view(-1, 1))
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
        
        
class PoetModelVerseEnd(PoetModelFunctionalInterface):
    def __init__(self, pretrainedModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.model = AutoModelForCausalLM.from_pretrained(pretrainedModel, output_hidden_states=True)
            
        model_config = self.model.config
        self.model_size = -1
        # Check for Hidden layer size by Attribute Name
        if hasattr(model_config, "n_embd"):
            self.model_size = model_config.n_embd
        elif hasattr(model_config, "hidden_size"):
            self.model_size = model_config.hidden_size  # Number of Emmbedings taken from config
        self.vowels_regressor = torch.nn.Linear(self.model_size,1) # Vowel count
        self.rhyme_regressor = torch.nn.Linear(self.model_size, len(StropheParams.RHYME)) # Rhyme Type
        self.verse_endings = torch.nn.Linear(self.model_size, len(StropheParams.ENDS)) # Verse End Syllable
        
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, nums=None, rhyme=None, verse_end = None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        last_hidden = outputs['hidden_states'][-1]
        vowel_regression = self.vowels_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
        rhyme_regression = self.rhyme_regressor((last_hidden[:,0,:].view(-1, self.model_size)))
        verse_end_reg = self.verse_endings((last_hidden[:,0,:].view(-1, self.model_size)))
        full_loss = outputs.loss
        
        vowel_loss = None
        if nums is not None:
            loss_fct = torch.nn.MSELoss()
            vowel_loss = loss_fct(vowel_regression.view(-1, 1), nums.view(-1, 1))
            full_loss = full_loss + vowel_loss
            
        rhyme_loss = None
        if rhyme is not None:
            softmaxed = torch.softmax(rhyme_regression, dim=1)
            loss_fct = torch.nn.CrossEntropyLoss()
            rhyme_loss = loss_fct(softmaxed, rhyme)
            full_loss = full_loss + rhyme_loss
            
        verse_loss = None
        if verse_end is not None:
            softmaxed = torch.softmax(verse_end_reg, dim=1)
            loss_fct = torch.nn.CrossEntropyLoss()
            verse_loss = loss_fct(softmaxed, verse_end)
            full_loss = full_loss + verse_loss
            
        
        return {"model_output" : outputs,
                "vowel_regression_output": vowel_regression, 
                "vowel_regression_loss": vowel_loss,
                "rhyme_regression_output": rhyme_regression,
                "rhyme_regression_loss": rhyme_loss,
                "verse_end_regression_output" : verse_end_reg,
                "verse_end_regression_loss" : verse_loss,
                "loss": full_loss}
    
    def save_LM(self, LM_path):
        self.model.save_pretrained(LM_path)