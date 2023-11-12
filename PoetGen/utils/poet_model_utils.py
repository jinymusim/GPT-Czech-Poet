import torch

class PoetModelInterface(torch.nn.Module):
    """Pytorch Model Interface. Abstract class for all Poet model types

    Args:
        torch (_type_): Is child of torch.nn.Module for integration with torch and huggingface 
    """
    def __init__(self, *args, **kwargs) -> None:
        """ Constructor. As child Class needs to construct Parent
        """
        super().__init__(*args, **kwargs)
        
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, *args, **kwargs):
        """Compute model output and model loss

        Args:
            input_ids (_type_, optional): Model inputs. Defaults to None.
            labels (_type_, optional): Language Model labels. Defaults to None.
            attention_mask (_type_, optional): Attention mask where padding starts. Defaults to None.

        Raises:
            NotImplementedError: Abstract class
        """
        raise NotImplementedError()
    
    def generate_forced(self,  *args, **kwargs):
        """Generates model output with restriction on inputs and past generation

        Raises:
            NotImplementedError: Abstract class
        """
        raise NotImplementedError()

    @staticmethod
    def rhyme_like(rhyme:str):
        """DEPRECATED: Check string in rhyme format 

        Args:
            rhyme (str): String with possible rhyme

        Returns:
            bool: Boolean if string like rhyme
        """
        return rhyme.isupper() and len(rhyme) in [4,6]
    
    def save_LM(self, LM_path):
        """Save raw LM

        Args:
            LM_path (str): Where to store the LM

        Raises:
            NotImplementedError: Abstract class
        """
        raise NotImplementedError()
    

from transformers import GPT2Config, GPT2Model
from .poet_utils import POET_YEARS_BUCKETS

class ContextModule(torch.nn.Module):
    """Module for understanding poet context

    Args:
        torch (_type_): Is child of torch.nn.Module for integration with torch and huggingface 
    """
    def __init__(self, block_count, input_size, n_embd ,output_size,*args, **kwargs) -> None:
        """Construct the underlying small LM for context

        Args:
            block_count (_type_): LM number of blocks of GPT2Block
            input_size (_type_): LM size of input
            n_embd (_type_): LM size of hidden layers
            output_size (_type_): LM size of output
        """
        super().__init__(*args, **kwargs)
        self.config = GPT2Config(n_positions=input_size, n_head=(n_embd//(768//12)),n_embd=n_embd, 
                                 n_layer=block_count, output_hidden_states=True,  output_attentions =True)
        self.context_model = GPT2Model(self.config)
        self.linear_downscale = torch.nn.Linear(n_embd, output_size)
        self.input_size = input_size
        self.n_embd = n_embd
        self.output_size = output_size
        # Context is getting injected from Outside
        self.context_ids = None
        self.context_attention_mask = None
    
    
    def forward(self, hidden_states,layer_past=None,*args, **kwargs):
        """Compute Context LM output, Data are injected from outside

        Args:
            hidden_states (_type_): Current hidden states
            layer_past (_type_, optional): Past layer outputs. Defaults to None.

        Returns:
            _type_: GPT2Block structured output (hidden states, layer past, attention, keys)
        """
        down = torch.zeros_like(hidden_states)
        model_output = None
        # Sometimes there might be no context
        if self.context_ids != None:
            model_output = self.context_model.forward(input_ids=self.context_ids, attention_mask=self.context_attention_mask)
            # Take only the Class token as 
            down = self.linear_downscale.forward(model_output["hidden_states"][-1][:,0,:].view(-1, self.n_embd))[:, None, :]
        return  (hidden_states + down,
                 down[None, :, :, :],
                 (None if model_output == None else model_output["attentions"], 
                None))
        
class PoetTypeModule(torch.nn.Module):
    """Module to classify poet type

    Args:
        torch (_type_): Is child of torch.nn.Module for integration with torch and huggingface 
    """
    
    def __init__(self, block_count, input_size, n_embd,output_size,*args, **kwargs) -> None:
        """Construct LM for poet classification from inputs

        Args:
            block_count (_type_): LM number of blocks of GPT2Block
            input_size (_type_): LM size of input
            n_embd (_type_): LM size of hidden layers
            output_size (_type_): LM size of output
        """
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
        # Context and labels are getting injected from Outside
        self.context_ids = None
        self.context_attention_mask = None
        self.type_labels=None
        # Store for loss for model itself
        self.indiv_loss=None
    
    def forward(self, hidden_states,layer_past=None,*args, **kwargs):
        """Compute Classification LM output and loss 

        Args:
            hidden_states (_type_): Current hidden states
            layer_past (_type_, optional): Past layer outputs. Defaults to None.

        Returns:
            _type_: GPT2Block structured output (hidden states, layer past, attention, keys)
        """
        type_prob = torch.zeros((hidden_states.shape[0], len(POET_YEARS_BUCKETS))).to("cuda" if torch.cuda.is_available() else "cpu")
        model_output = None
        # Sometimes there might be no context
        if self.context_ids != None:
            model_output = self.type_model.forward(input_ids=self.context_ids, attention_mask=self.context_attention_mask)
            # Only Class token is taken
            poet_type = self.type_predict.forward(model_output["hidden_states"][-1][:,0,:].view(-1, self.n_embd))
            type_prob = self.softmax.forward(poet_type) 
        # If type labels are present, inject the true labels to future blocks
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
    """Static Class incorporating methods for Manipulation with LMs
    Code Inspired by article: Fine-tuning the English GPT-2 in any language with Hugging Face
    Link: https://github.com/piegu/fastai-projects/blob/master/finetuning-English-GPT2-any-language-Portuguese-HuggingFace-fastaiv2.ipynb
    """
     
    @staticmethod
    def exchange_embedding(poet_model: PoetModelInterface, new_tokenizer: PreTrainedTokenizerBase, old_tokenizer: PreTrainedTokenizerBase):
        """Exchange embedding matrixes for GPT2 Models

        Args:
            poet_model (PoetModelInterface): Model to manipulate with
            new_tokenizer (PreTrainedTokenizerBase): New tokenization
            old_tokenizer (PreTrainedTokenizerBase): Old tokenization
        """
        # Get old Embeddings
        old_embed_in = poet_model.model.get_input_embeddings().weight.clone().detach()
        old_mean_in = old_embed_in.mean(0)
        old_embed_out = poet_model.model.get_output_embeddings().weight.clone().detach()
        old_mean_out = old_embed_out.mean(0)
        # Generate new Embedding based on new tokenization
        new_embd_in = old_embed_in.new_zeros(new_tokenizer.vocab_size, old_embed_in.size(1))
        new_embd_out = old_mean_out.new_zeros(new_tokenizer.vocab_size, old_embed_out.size(1))
        old_vocab = old_tokenizer.get_vocab()
        
        vocab_hit = 0
        # Keep as much from old Embeddings as possible
        for w, idx_new in new_tokenizer.get_vocab().items():
            idx_old = old_vocab.get(w, -1)
            if idx_old >= 0:
                new_embd_in[idx_new] = old_embed_in[idx_old]
                new_embd_out[idx_new] = old_embed_out[idx_old]
                vocab_hit +=1
            else:
                new_embd_in[idx_new] = old_mean_in
                new_embd_out[idx_new] = old_mean_out
                
        print(f"Vocab hit rate: {vocab_hit}/{old_tokenizer.vocab_size}")
        #Exchange Embeddings and Decoding
        new_embd_layer_in = torch.nn.Embedding(new_tokenizer.vocab_size, old_embed_in.size(1))
        new_embd_layer_in.weight.data = new_embd_in
        new_embd_layer_out = torch.nn.Linear(old_embed_out.size(1), new_tokenizer.vocab_size, bias=False)
        new_embd_layer_out.weight.data = new_embd_out
        
        poet_model.model.set_input_embeddings(new_embd_layer_in)
        poet_model.model.set_output_embeddings(new_embd_layer_out)
        
        # Update LM config to reflect possible change in vocab size
        poet_model.model.config.vocab_size = new_tokenizer.vocab_size
        
        
    @staticmethod
    def exchange_embedding_roberta(metre_model, new_tokenizer: PreTrainedTokenizerBase, old_tokenizer: PreTrainedTokenizerBase):
        """Exchange embedding matrixes for Roberta Models

        Args:
            poet_model (PoetModelInterface): Model to manipulate with
            new_tokenizer (PreTrainedTokenizerBase): New tokenization
            old_tokenizer (PreTrainedTokenizerBase): Old tokenization
        """
        # Get old Embeddings
        old_embed = metre_model.model.get_input_embeddings().weight.clone().detach()
        old_mean = old_embed.mean(0)
        # Generate new Embedding based on new tokenization
        new_embd = old_embed.new_zeros(new_tokenizer.vocab_size, old_embed.size(1))
        old_vocab = old_tokenizer.get_vocab()
        
        vocab_hit = 0
        # Keep as much from old Embeddings as possible
        for w, idx_new in new_tokenizer.get_vocab().items():
            idx_old = old_vocab.get(w, -1)
            if idx_old >= 0:
                new_embd[idx_new] = old_embed[idx_old]
                vocab_hit +=1
            else:
                new_embd[idx_new] = old_mean
                
        print(f"Vocab hit rate: {vocab_hit}/{old_tokenizer.vocab_size}")
        #Exchange Embeddings and Decoding
        new_embd_layer = torch.nn.Embedding(new_tokenizer.vocab_size, old_embed.size(1))
        new_embd_layer.weight.data = new_embd
        metre_model.model.set_input_embeddings(new_embd_layer)
        new_decoder = torch.nn.Linear( old_embed.size(1), new_tokenizer.vocab_size)
        new_decoder.weight = metre_model.model.roberta.embeddings.word_embeddings.weight
        metre_model.model.lm_head.decoder = new_decoder
        # Update LM config to reflect possible change in vocab size
        metre_model.model.config.vocab_size = new_tokenizer.vocab_size
        