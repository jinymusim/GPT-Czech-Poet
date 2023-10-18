import torch

class PoetModelInterface(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, *args, **kwargs):
        raise NotImplementedError()
    
    def generate_forced(self,  *args, **kwargs):
        raise NotImplementedError()

    @staticmethod
    def rhyme_like(rhyme:str):
        return rhyme.isupper() and len(rhyme) in [4,6]
    
    def save_LM(self, LM_path):
        raise NotImplementedError()
    

from transformers import GPT2Config, GPT2Model
from .poet_utils import POET_YEARS_BUCKETS

class ContextModule(torch.nn.Module):
    
    def __init__(self, block_count, input_size, n_embd ,output_size,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = GPT2Config(n_positions=input_size, n_head=(n_embd//(768//12)),n_embd=n_embd, 
                                 n_layer=block_count, output_hidden_states=True,  output_attentions =True)
        self.context_model = GPT2Model(self.config)
        self.linear_downscale = torch.nn.Linear(n_embd, output_size)
        self.input_size = input_size
        self.n_embd = n_embd
        self.output_size = output_size
        self.context_ids = None
        self.context_attention_mask = None
    
    # Context is getting injected from Top
    def forward(self, hidden_states,layer_past=None,*args, **kwargs):
        down = torch.zeros_like(hidden_states)
        model_output = None
        if self.context_ids != None:
            model_output = self.context_model.forward(input_ids=self.context_ids, attention_mask=self.context_attention_mask)
            down = self.linear_downscale.forward(model_output["hidden_states"][-1][:,0,:].view(-1, self.n_embd))[:, None, :]
        # torch.zeros( base n_head ,  ,base n_embd // base n_head))
        return  (hidden_states + down,
                 down[None, :, :, :],
                 (None if model_output == None else model_output["attentions"], 
                None))
        
class PoetTypeModule(torch.nn.Module):
    
    def __init__(self, block_count, input_size, n_embd,output_size,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = GPT2Config(n_positions=input_size, n_head=(n_embd//(768//12)),n_embd=n_embd, 
                                 n_layer=block_count, output_hidden_states=True,  output_attentions =True)
        self.type_model = GPT2Model(self.config)
        self.type_predict = torch.nn.Linear(n_embd, len(POET_YEARS_BUCKETS))
        self.softmax = torch.nn.Softmax()
        self.linear_scale = torch.nn.Linear(len(POET_YEARS_BUCKETS), output_size)
        self.input_size = input_size
        self.n_embd = n_embd
        self.output_size = output_size
        self.context_ids = None
        self.context_attention_mask = None
        self.type_labels=None
        # Store for loss for model itself
        self.indiv_loss=None
    
    # Context And type labels are to be injected to bypass GPT2Blocks 
    def forward(self, hidden_states,layer_past=None,*args, **kwargs):
        type_prob = torch.zeros((hidden_states.shape[0], len(POET_YEARS_BUCKETS))).to("cuda" if torch.cuda.is_available() else "cpu")
        model_output = None
        if self.context_ids != None:
            model_output = self.type_model.forward(input_ids=self.context_ids, attention_mask=self.context_attention_mask)
            poet_type = self.type_predict.forward(model_output["hidden_states"][-1][:,0,:].view(-1, self.n_embd))
            type_prob = self.softmax.forward(poet_type) 
        if self.type_labels != None:
            loss_fct = torch.nn.CrossEntropyLoss()
            self.indiv_loss = loss_fct(type_prob, self.type_labels)
            type_prob = (self.type_labels.type(torch.FloatTensor)).to("cuda" if torch.cuda.is_available() else "cpu")
        linear_up = self.linear_scale.forward(type_prob)
        return (hidden_states + linear_up[:, None, :],
                linear_up[None, :, None, :], 
                (None if model_output == None else model_output["attentions"], 
                None))
            
from transformers import PreTrainedTokenizerBase

class ModelManipulation:
    
    # Code Inspired by article: Fine-tuning the English GPT-2 in any language with Hugging Face
    # Link: https://github.com/piegu/fastai-projects/blob/master/finetuning-English-GPT2-any-language-Portuguese-HuggingFace-fastaiv2.ipynb
     
    @staticmethod
    def exchange_embedding(poet_model: PoetModelInterface, new_tokenizer: PreTrainedTokenizerBase, old_tokenizer: PreTrainedTokenizerBase):
        old_embed = poet_model.model.get_input_embeddings().weight.clone().detach()
        old_mean = old_embed.mean(0)
        
        new_embd = old_embed.new_zeros(new_tokenizer.vocab_size, old_embed.size(1))
        old_vocab = old_tokenizer.get_vocab()
        
        vocab_hit = 0
        
        for w, idx_new in new_tokenizer.get_vocab().items():
            idx_old = old_vocab.get(w, -1)
            if idx_old >= 0:
                new_embd[idx_new] = old_embed[idx_old]
                vocab_hit +=1
            else:
                new_embd[idx_new] = old_mean
                
        print(f"Vocab hit rate: {vocab_hit}/{old_tokenizer.vocab_size}")
        
        new_embd_layer = torch.nn.Embedding(new_tokenizer.vocab_size, old_embed.size(1))
        new_embd_layer.weight.data = new_embd
        poet_model.model.transformer.set_input_embeddings(new_embd_layer)
        new_decoder = torch.nn.Linear( old_embed.size(1), new_tokenizer.vocab_size, bias=False)
        new_decoder.weight = poet_model.model.transformer.wte.weight
        poet_model.model.lm_head = new_decoder
        
        poet_model.model.config.vocab_size = new_tokenizer.vocab_size
        
        
    @staticmethod
    def exchange_embedding_roberta(metre_model, new_tokenizer: PreTrainedTokenizerBase, old_tokenizer: PreTrainedTokenizerBase):
        old_embed = metre_model.model.get_input_embeddings().weight.clone().detach()
        old_mean = old_embed.mean(0)
        
        new_embd = old_embed.new_zeros(new_tokenizer.vocab_size, old_embed.size(1))
        old_vocab = old_tokenizer.get_vocab()
        
        vocab_hit = 0
        
        for w, idx_new in new_tokenizer.get_vocab().items():
            idx_old = old_vocab.get(w, -1)
            if idx_old >= 0:
                new_embd[idx_new] = old_embed[idx_old]
                vocab_hit +=1
            else:
                new_embd[idx_new] = old_mean
                
        print(f"Vocab hit rate: {vocab_hit}/{old_tokenizer.vocab_size}")
        new_embd_layer = torch.nn.Embedding(new_tokenizer.vocab_size, old_embed.size(1))
        new_embd_layer.weight.data = new_embd
        metre_model.model.set_input_embeddings(new_embd_layer)
        new_decoder = torch.nn.Linear( old_embed.size(1), new_tokenizer.vocab_size)
        new_decoder.weight = metre_model.model.roberta.embeddings.word_embeddings.weight
        metre_model.model.lm_head.decoder = new_decoder
        
        metre_model.model.config.vocab_size = new_tokenizer.vocab_size
        