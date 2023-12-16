# Outide Packages
import torch
import os
import argparse


from accelerate import Accelerator
from transformers import  AutoTokenizer, TrainingArguments, Trainer, PreTrainedTokenizerFast, PreTrainedTokenizerBase, AutoModelForCausalLM
from functools import partial

# Project Packages
from poet_model_base_lm import PoetModelBase
from poet_model_secondary_tasks import PoetModelSecondaryTasks
from poet_model_half_precision import PoetModelHalfBase
from poet_model_verse_end import PoetModelVerseEnd
from poet_model_context_input import PoetModelContextInput
from poet_model_context_year import PoetModelContextYear
from poet_model_all_tasks import PoetModelAllTasks
from poet_model_distil import DistilModel

from corpus_capsulated_datasets import CorpusDatasetPytorch
from utils.poet_model_utils import ModelManipulation

from utils.poet_utils import EOS, PAD, UNK, parse_boolean


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size_LM", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs_LM", default=0, type=int, help="Number of epochs to run.")
parser.add_argument("--batch_size_poet", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs_poet", default=0, type=int, help="Number of epochs for poet gen")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="Learning Rate for Finetuning")
parser.add_argument("--train_masked", default=False, type=bool, help="Train for consistency secondary training")
parser.add_argument("--input_mask_rate", default=0.00, type=float, help="Rate of input masking")

parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")

# huggyllama/llama-7b 4096
# bigscience/bloom-560m 2048
# TheBloke/Llama-2-7B-fp16 4096

# lchaloupsky/czech-gpt2-oscar 1024 Czech Model
# spital/gpt2-small-czech-cs 1024 Alt Czech Model
# distilgpt2 1024 Alt En Model
# gpt2 1024 EN Model
# stabilityai/StableBeluga-7B 4096 Large
# RWKV/rwkv-4-169m-pile 1024 RNN

#TODO: Introduce Layered Model, Best done by modifiing 
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

#TODO: BaseTokenize, all e4 e8, base e4 e8
#TODO: UnicodeTokenizer, all e4 e8, base e4 e8
#TODO: SyllableTokenizer, all e4 e8, base e4 e8
#TODO: ProcessedTokenizer, all e4 e8, base e4 e8

#parser.add_argument("--default_hf_model", default="lchaloupsky/czech-gpt2-oscar", type=str, help="Default Model from HF to use")
parser.add_argument("--default_hf_model", default=os.path.join(os.path.dirname(__file__), 'backup_LMS','CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM' ), type=str, help="Default Model from HF to use")
parser.add_argument("--use_default_model",  default=True, type=bool, help="Use Default Model")
#parser.add_argument("--tokenizer", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "utils", "tokenizers", "Unicode", "unicode_tokenizer.json")), type=str, help="Tokenizer to use")
#parser.add_argument("--tokenizer", default='lchaloupsky/czech-gpt2-oscar', type=str, help="Tokenizer to use")
parser.add_argument("--tokenizer", default=os.path.join(os.path.dirname(__file__), 'backup_LMS','CZ-Unicode-Tokenizer-NormalText-gpt-cz-poetry-base-e4e16_LM' ), type=str, help="Tokenizer to use")
parser.add_argument("--model_type",  default="distil", type=str, choices=["base", "secondary_tasks", "half", "verse", "context", "year", "all", 'distil'], help="What type of Model is to be constructed")
parser.add_argument("--model_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "Unicode-Distil")),  type=str, help="Path to Model")
parser.add_argument("--max_len", default=1024, type=int, help="Max length for tokenizer")
parser.add_argument("--context_max_len", default=1, type=int, help="Max length of context for tokenizer")
parser.add_argument("--verse_len", default=[4,6], type=list, help="Lengths of verses")


parser.add_argument("--prompt_rhyme", default=True, type=bool, help="Rhyme is prompted into training data")
parser.add_argument("--prompt_length", default=True, type=bool, help="Verse length is prompted into training data")
parser.add_argument("--prompt_ending", default=True, type=bool, help="Ending of Verse is prompted into training data")

parser.add_argument("--syllables", default=False, type=bool, help="If inputs should be parsed by syllables")
parser.add_argument("--lower_case", default=True, type=bool, help="If to lower case data")

parser.add_argument("--mirror_imbed", default=True, type=bool, help="If to mirror input embedding to output ones")

parser.add_argument("--val_data_rate", default=0.05, type=float, help="Rate of validation data")

parser.add_argument("--model_input_format",  default="METER_VERSE", type=str, choices=["BASIC", "VERSE_PAR", 'METER_VERSE'], help="Input format to use for model")


def main(args: argparse.Namespace):
    

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
        else:
            raise TypeError("Given model type doesn't exists")
        
        try:    
            tokenizer: PreTrainedTokenizerBase =  AutoTokenizer.from_pretrained(args.tokenizer)
            if tokenizer.pad_token == None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                
        except: #TODO: Need model to update embedding matrix
            tokenizer: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
            tokenizer.eos_token = EOS
            tokenizer.eos_token_id = 0
            tokenizer.pad_token = PAD
            tokenizer.pad_token_id = 1
            tokenizer.unk_token = UNK
            tokenizer.unk_token_id = 2
            
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
    
    # Partial Function to use as data collection with input masking
    if args.model_type == 'distil':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        collate = partial(CorpusDatasetPytorch.collate_distil, tokenizer=tokenizer,
                          surrogate_model=AutoModelForCausalLM.from_pretrained(args.default_hf_model,output_hidden_states=True).to(device), surrogate_model_device=device, max_len=args.max_len)
    else:
        collate = partial(CorpusDatasetPytorch.collate, tokenizer=tokenizer,max_len=args.max_len, 
                      max_context=args.context_max_len, mask_rate=args.input_mask_rate, syllables=args.syllables, format=args.model_input_format)
    

    train_data = CorpusDatasetPytorch(data_dir=args.data_path, prompt_ending=args.prompt_ending, 
                                      prompt_length=args.prompt_length, prompt_verse=args.prompt_rhyme,
                                      verse_len=args.verse_len, lower_case=args.lower_case, val_data_rate=args.val_data_rate)
    
    # Text Line Training
    if args.epochs_LM !=0:
        training_args = TrainingArguments(
                                  save_strategy  = "no",
                                  warmup_steps = len(train_data.pytorch_dataset_text)//args.batch_size_LM,
                                  logging_steps = 500,
                                  weight_decay = 0.0,
                                  num_train_epochs = args.epochs_LM,
                                  learning_rate = args.learning_rate,
                                  fp16 = True if torch.cuda.is_available() else False,
                                  ddp_backend = "nccl",
                                  lr_scheduler_type="cosine",
                                  logging_dir = './logs',
                                  output_dir = './results',
                                  per_device_train_batch_size = args.batch_size_LM)
    
    
        trainer = Trainer(model = model,
                           args = training_args,
                           train_dataset= train_data.pytorch_dataset_text,
                           data_collator=collate).train()
    
    # Verse Training
    if args.epochs_poet !=0:
        training_args = TrainingArguments(
                                  save_strategy  = "no",
                                  warmup_steps = len(train_data.pytorch_dataset_body)//args.batch_size_poet,
                                  logging_steps = 500,
                                  weight_decay = 0.0,
                                  num_train_epochs = args.epochs_poet,
                                  learning_rate = args.learning_rate,
                                  fp16 =  True if torch.cuda.is_available() else False,
                                  ddp_backend = "nccl",
                                  lr_scheduler_type="cosine",
                                  logging_dir = './logs',
                                  output_dir = './results',
                                  per_device_train_batch_size = args.batch_size_poet)
    
    
        trainer = Trainer(model = model,
                           args = training_args,
                           train_dataset= train_data.pytorch_dataset_body,
                           data_collator=collate).train()
    
    
    
    torch.save(model, args.model_path)
    model.save_LM(f"{args.model_path}_LM")
    tokenizer.save_pretrained(f"{args.model_path}_LM")
      


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
