# Outide Packages
import torch
import os
import argparse


from accelerate import Accelerator
from transformers import  AutoTokenizer, TrainingArguments, Trainer
from functools import partial

# Project Packages
from poet_model_base_lm import PoetModelBase
from poet_model_secondary_tasks import PoetModelSecondaryTasks
from poet_model_half_precision import PoetModelHalfBase
from poet_model_verse_end import PoetModelVerseEnd
from poet_model_context_input import PoetModelContextInput
from poet_model_context_year import PoetModelContextYear
from poet_model_all_tasks import PoetModelAllTasks

from corpus_capsulated_datasets import CorpusDatasetPytorch


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size_LM", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs_LM", default=32, type=int, help="Number of epochs to run.")
parser.add_argument("--batch_size_poet", default=16, type=int, help="Batch size.")
parser.add_argument("--epochs_poet", default=32, type=int, help="Number of epochs for poet gen")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="Learning Rate for Finetuning")
parser.add_argument("--use_gpu_if_available", default=True, type=bool, help="If GPU should be used")
parser.add_argument("--use_multiple_gpu_if_available", default=True, type=bool, help="If to use multiple gpus")
parser.add_argument("--train_masked", default=True, type=bool, help="Train for consistency secondary training")
parser.add_argument("--input_mask_rate", default=0.05, type=float, help="Rate of input masking")

parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")

# huggyllama/llama-7b 4096
# lchaloupsky/czech-gpt2-oscar 1024
# bigscience/bloom-560m 2048
# TheBloke/Llama-2-7B-fp16 4096

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

parser.add_argument("--default_hf_model", default="gpt2-xl", type=str, help="Default Model from HF to use")
parser.add_argument("--use_default_model",  default=True, type=bool, help="Use Default Model")
parser.add_argument("--model_type",  default="all", type=str, choices=["base", "secondary_tasks", "half", "verse", "context", "year", "all"], help="What type of Model is to be constructed")
parser.add_argument("--model_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "gpt-xl-poetry-all_tasks_e32_e32")),  type=str, help="Path to Model")
parser.add_argument("--max_len", default=1024, type=int, help="Max length for tokenizer")
parser.add_argument("--context_max_len", default=1024, type=int, help="Max length of context for tokenizer")
parser.add_argument("--verse_len", default=[3,4,5,6,7,8], type=list, help="Lengths of verses")


parser.add_argument("--prompt_rhyme", default=True, type=bool, help="Rhyme is prompted into training data")
parser.add_argument("--prompt_length", default=True, type=bool, help="Verse length is prompted into training data")
parser.add_argument("--prompt_ending", default=True, type=bool, help="Ending of Verse is prompted into training data")



def main(args: argparse.Namespace):
    # Base Device is CPU
    device = torch.device('cpu')
    # If Wanted and GPU is available, use it
    if args.use_gpu_if_available:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.use_default_model:
        tokenizer = AutoTokenizer.from_pretrained(args.default_hf_model)
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
        else:
            model = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.default_hf_model)
        model = torch.load(args.model_path_full, map_location=torch.device('cpu'))
    
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
    collate = partial(CorpusDatasetPytorch.collate, mask_rate=args.input_mask_rate)
    
    # Data Loading
    tokenizer.model_max_length = args.max_len
    train_data = CorpusDatasetPytorch(tokenizer, data_dir=args.data_path, 
                                      prompt_ending=args.prompt_ending, prompt_length=args.prompt_length, prompt_verse=args.prompt_rhyme,
                                      verse_len=args.verse_len, context_len=args.context_max_len)
    
    # Text Line Training
    training_args = TrainingArguments(
                                  save_strategy  = "no",
                                  warmup_steps = len(train_data.pytorch_dataset_text)//args.batch_size_LM,
                                  logging_steps = 500,
                                  weight_decay = 0.0,
                                  num_train_epochs = args.epochs_LM,
                                  learning_rate = args.learning_rate,
                                  fp16 = True if torch.cuda.is_available() else False,
                                  ddp_backend = "nccl",
                                  lr_scheduler_type="constant",
                                  logging_dir = './logs',
                                  output_dir = './results',
                                  per_device_train_batch_size = args.batch_size_LM)
    
    
    trainer = Trainer(model = model,
                           args = training_args,
                           train_dataset= train_data.pytorch_dataset_text,
                           data_collator=collate).train()
    
    # Verse Training
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
