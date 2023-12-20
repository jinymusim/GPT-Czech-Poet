import torch
from transformers import  AutoModelForCausalLM, AutoTokenizer
from utils.poet_model_utils import PoetModelInterface
from utils.poet_utils import TextAnalysis, RHYME_SCHEMES

class DistilModel(PoetModelInterface):
    
    def __init__(self, pretrainedModel, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(pretrainedModel, 
                                                        output_hidden_states=True)
            
        model_config = self.model.config
        self.model_size = 1
        # Check for Hidden layer size by Attribute Name
        if hasattr(model_config, "n_embd"):
            self.model_size = model_config.n_embd
        elif hasattr(model_config, "hidden_size"):
            self.model_size = model_config.hidden_size
        
        self.kept_states = [3, 7, 11]
            
        for pop_index in sorted(list(set(range(len(self.model.base_model.h))) - set(self.kept_states)), reverse=True):
            
            self.model.base_model.h.pop(pop_index)
        # Because of Inserted Layer, Head Masks don't match => Add 1 more
        self.model.base_model.config.n_layer = len(self.kept_states)
        
        self.loss_fnc = torch.nn.MSELoss()
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, to_replicate_states= None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        # The 3 layers + embeddings (add + 1 to shift the original_index)
        for distil_index, original_index in enumerate([-1] + self.kept_states):
            loss += self.loss_fnc(outputs['hidden_states'][distil_index], to_replicate_states[original_index + 1])
        
        return {"model_output" : outputs,
                "loss": loss}
    
    def save_LM(self, LM_path):
        self.model.save_pretrained(LM_path, safe_serialization=False)
        
    def generate_forced(self, *args, **kwargs):
        raise NotImplementedError("Currently without")