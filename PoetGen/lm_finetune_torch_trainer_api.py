# Outide Packages
import torch
import os
import argparse
import torch.distributed as dist


from accelerate import Accelerator
from transformers import  AutoTokenizer, TrainingArguments, Trainer, PreTrainedTokenizerFast, PreTrainedTokenizerBase, AutoModelForCausalLM, EarlyStoppingCallback, IntervalStrategy
from functools import partial

# Project Packages
from utils.base_poet_models import PoetModelBase, PoetModelSecondaryTasks, PoetModelHalfBase, PoetModelVerseEnd, PoetModelContextInput, PoetModelContextYear, PoetModelAllTasks, DistilModel, PoetModelSmall


from corpus_capsulated_datasets import CorpusDatasetPytorch
from utils.poet_model_utils import ModelManipulation, PoetModelInterface

from utils.poet_utils import Tokens, parse_boolean


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size_poet", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs_poet", default=16, type=int, help="Number of epochs for poet gen")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning Rate for Finetuning")
parser.add_argument("--train_masked", default=False, type=bool, help="Train for consistency secondary training")
parser.add_argument("--input_mask_rate", default=0.0, type=float, help="Rate of input masking")

parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv-new-summary-one-sentence")), type=str, help="Path to Data")

#TODO: Join syllabification by better symbol (maybe extra space arround) DONE
#TODO: Make meter validator with context
#   NO_MARK verse       
#   MARK    verse     <- Predicting this METER
#   NO_MARK verse
#   NO_MARK verse
# DONE

# huggyllama/llama-7b 4096
# bigscience/bloom-560m 2048
# TheBloke/Llama-2-7B-fp16 4096

# lchaloupsky/czech-gpt2-oscar 1024 Czech Model
# spital/gpt2-small-czech-cs 1024 Alt Czech Model
# distilgpt2 1024 Alt En Model
# gpt2 1024 EN Model
# stabilityai/StableBeluga-7B 4096 Large
# RWKV/rwkv-4-169m-pile 1024 RNN
# unsloth/Llama-3.2-1B-Instruct 4096

# Introduce Layered Model, Best done by modifiing 
# self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])

# This gives Model only 5 blocks
# model.base_model.h = torch.nn.ModuleList([transformers.models.gpt2.modeling_gpt2.GPT2Block(model.base_model.config) for _ in range(5)])

# Adding Custom Modules to model is possible
# model.base_model.h = torch.nn.ModuleList(
    # [transformers.models.gpt2.modeling_gpt2.GPT2Block(model.base_model.config) for _ in range(5)] + \
    # [torch.nn.Linear(model.base_model.config.hidden_size, model.base_model.config.hidden_size) for _ in range(2)]
    # )

# Extending Appending and Inserting Modules Also Possible
# model.base_model.h.extend([torch.nn.Linear(768,1)])
# model.base_model.h.append(torch.nn.Linear(1,768))
# model.base_model.h.insert(7,torch.nn.Linear(768,768))

parser.add_argument("--default_hf_model", default='unsloth/Llama-3.2-1B', type=str, help="Default Model from HF to use")
parser.add_argument("--use_default_model",  default=True, type=bool, help="Use Default Model")
#parser.add_argument("--tokenizer", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "utils", "tokenizers", "Unicode", "unicode_tokenizer.json")), type=str, help="Tokenizer to use")
parser.add_argument("--tokenizer", default='unsloth/Llama-3.2-1B', type=str, help="Tokenizer to use")
#parser.add_argument("--tokenizer", default=os.path.join(os.path.dirname(__file__), 'backup_LMS','CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM' ), type=str, help="Tokenizer to use")
parser.add_argument("--model_type",  default="base", type=str, choices=["base", "secondary_tasks", "half", "verse", "context", "year", "all", 'distil', 'small'], help="What type of Model is to be constructed")
parser.add_argument("--model_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "Test-Model")),  type=str, help="Path to Model")
parser.add_argument("--max_len", default=4096, type=int, help="Max length for tokenizer")
parser.add_argument("--context_max_len", default=1, type=int, help="Max length of context for tokenizer")

parser.add_argument("--syllables", default=False, type=bool, help="If inputs should be parsed by syllables")
parser.add_argument("--lower_case", default=True, type=bool, help="If to lower case data")

parser.add_argument("--mirror_imbed", default=True, type=bool, help="If to mirror input embedding to output ones")

parser.add_argument("--val_data_rate", default=0.05, type=float, help="Rate of validation data")
parser.add_argument("--test_data_rate", default=0.05, type=float, help="Rate of test data")

parser.add_argument("--size_test", default=False, type=parse_boolean, help='If to conduct size test on data')
parser.add_argument("--sizes_to_test", default=1, type=float, help='Size to test on')

def train_model(model: PoetModelInterface, tokenizer: PreTrainedTokenizerBase ,dataset: CorpusDatasetPytorch, collate_fnc, args: argparse.Namespace):
    # Verse Training
    if args.epochs_poet !=0:
            
        training_args = TrainingArguments(
                                  output_dir=args.model_path + "TEMP",
                                  overwrite_output_dir= True,
                                  save_strategy  = IntervalStrategy.EPOCH,
                                  save_safetensors=False,
                                  save_total_limit=1,
                                  warmup_steps = len(dataset.train_strophes)//args.batch_size_poet,
                                  auto_find_batch_size = True,
                                  #do_eval = True,
                                  #evaluation_strategy =IntervalStrategy.EPOCH,
                                  logging_steps = 500,
                                  num_train_epochs = args.epochs_poet,
                                  learning_rate = args.learning_rate,
                                  fp16 = True if torch.cuda.is_available() else False,
                                  #fp16_full_eval  = True if torch.cuda.is_available() else False,
                                  optim='adamw_torch',
                                  lr_scheduler_type="constant_with_warmup",
                                  warmup_ratio=0.1,
                                  disable_tqdm=True,
                                  logging_dir = './logs',
                                  #metric_for_best_model='eval_loss',
                                  #load_best_model_at_end=True,
                                  #greater_is_better=False
                                )
    
        trainer = Trainer(model = model,
                           args = training_args,
                           train_dataset= dataset.train_strophes,
                           #eval_dataset= dataset.val_strophes,
                           #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
                           data_collator=collate_fnc).train()


def create_model_and_tokenizer(args: argparse.Namespace):
    """Create Model and Tokenizer, put model on best device

    Args:
        args (argparse.Namespace): Arguments of the model and tokenizer

    Raises:
        TypeError: Model type not recognized

    Returns:
        tuple: tuple of model and tokenizer
    """
    if args.use_default_model:
        if args.model_type == "base":         
            model = PoetModelBase(args.default_hf_model)
        elif args.model_type == "secondary_tasks":
            model = PoetModelSecondaryTasks(args.default_hf_model)
        elif args.model_type == "half":
            model = PoetModelHalfBase(args.default_hf_model)
        elif args.model_type == "verse":
            model =  PoetModelVerseEnd(args.default_hf_model)
        elif args.model_type == "context":
            model = PoetModelContextInput(args.default_hf_model, args.context_max_len)
        elif args.model_type == "year":
            model = PoetModelContextYear(args.default_hf_model, args.context_max_len)
        elif args.model_type == "all":
            model = PoetModelAllTasks(args.default_hf_model)
        elif args.model_type == 'distil':
            model = DistilModel(args.default_hf_model)
        elif args.model_type == 'small':
            model = PoetModelSmall()
        else:
            raise TypeError("Given model type doesn't exists")
        
        try:    
            tokenizer: PreTrainedTokenizerBase =  AutoTokenizer.from_pretrained(args.tokenizer)
            if tokenizer.pad_token == None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            if args.model_type == 'small':
                ModelManipulation.exchange_embedding(model, tokenizer, AutoTokenizer.from_pretrained(args.default_hf_model), args.mirror_imbed)
                
        except: #TODO: Need model to update embedding matrix
            tokenizer: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
            tokenizer.eos_token = Tokens.EOS
            tokenizer.eos_token_id = Tokens.EOS_ID
            tokenizer.pad_token = Tokens.PAD
            tokenizer.pad_token_id = Tokens.PAD_ID
            tokenizer.unk_token = Tokens.UNK
            tokenizer.unk_token_id = Tokens.UNK_ID
            
            ModelManipulation.exchange_embedding(model, tokenizer, AutoTokenizer.from_pretrained(args.default_hf_model), args.mirror_imbed)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = torch.load(args.model_path, map_location=torch.device('cpu'))
    
    # Parallel Plugin
    from accelerate import FullyShardedDataParallelPlugin
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
            )
    
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    model = accelerator.prepare(model)
    
    return model, tokenizer

def main(args: argparse.Namespace):
    
    model, tokenizer = create_model_and_tokenizer(args)
    
    
    # Partial Function to use as data collection with input masking
    if args.model_type == 'distil':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        collate = partial(CorpusDatasetPytorch.collate_distil, tokenizer=tokenizer,
                          surrogate_model=AutoModelForCausalLM.from_pretrained(args.default_hf_model,output_hidden_states=True).to(device), surrogate_model_device=device, max_len=args.max_len)
    else:
        collate = partial(CorpusDatasetPytorch.collate, tokenizer=tokenizer,max_len=args.max_len, 
                      max_context=args.context_max_len)
    

    train_data = CorpusDatasetPytorch(SEGMENT_TYPE='BASE', data_dir=args.data_path, 
                                    lower_case=args.lower_case,
                                    val_data_rate=args.val_data_rate, test_data_rate=args.test_data_rate)
    
    
    if not args.size_test:
        train_model(model, tokenizer, train_data, collate, args)
        
        torch.save(model.cpu(), args.model_path + ".model")
        model.save_LM(f"{args.model_path}_LM")
        tokenizer.save_pretrained(f"{args.model_path}_LM")
        
    else:
        train_data.train_strophes.change_custom_size(args.sizes_to_test)
        
        # Size compensation
        args.epochs_poet =  int(args.epochs_poet/args.sizes_to_test)
        train_model(model, tokenizer, train_data, collate, args)
        
        torch.save(model.cpu(), args.model_path + f"_data_size={args.sizes_to_test}.model")
        model.save_LM(f"{args.model_path}_data_size={args.sizes_to_test}_LM")
        tokenizer.save_pretrained(f"{args.model_path}_data_size={args.sizes_to_test}_LM")
      


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    print("Cuda is available: ", torch.cuda.is_available())
    main(args)
