# Outide Packages
import transformers
import torch
import os
import argparse

from transformers import  AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import DataLoader
#from torch.distributed.tensor.parallel import parallelize_module, PairwiseParallel

# Project Packages
from poet_model_base_lm import PoetModelBase
from poet_model_secondary_tasks import PoetModelSecondaryTasks
from poet_model_half_precision import PoetModelHalfBase


from corpus_capsulated_datasets import CorpusDatasetPytorch



parser = argparse.ArgumentParser()

parser.add_argument("--batch_size_LM", default=1, type=int, help="Batch size.")
parser.add_argument("--epochs_LM", default=2, type=int, help="Number of epochs to run.")
parser.add_argument("--batch_size_poet", default=1, type=int, help="Batch size.")
parser.add_argument("--epochs_poet", default=2, type=int, help="Number of epochs for poet gen")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="Learning Rate for Finetuning")
parser.add_argument("--use_gpu_if_available", default=True, type=bool, help="If GPU should be used")
parser.add_argument("--use_multiple_gpu_if_available", default=True, type=bool, help="If to use multiple gpus")
parser.add_argument("--train_masked", default=True, type=bool, help="Train for consistency secondary training")
parser.add_argument("--input_mask_rate", default=0.05, type=float, help="Rate of input masking")

parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")

# huggyllama/llama-7b 4096
# lchaloupsky/czech-gpt2-oscar 1024

parser.add_argument("--default_hf_model", default="huggyllama/llama-7b", type=str, help="Default Model from HF to use")
parser.add_argument("--use_default_model",  default=True, type=bool, help="Use Default Model")
parser.add_argument("--model_type",  default="half", type=str, choices=["base", "secondary_tasks", "half"], help="What type of Model is to be constructed")
parser.add_argument("--model_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "llama-cz-poetry-base")),  type=str, help="Path to Model")
parser.add_argument("--max_len", default=4096, type=int, help="Max length for tokenizer")


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
        else:
            model = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.default_hf_model)
        model = torch.load(args.model_path_full, map_location=torch.device('cpu'))
    
    model = model.to(device)
    
    
    # Data Loading
    tokenizer.model_max_length = args.max_len
    train_data = CorpusDatasetPytorch(tokenizer, data_dir=args.data_path, 
                                      prompt_ending=args.prompt_ending, prompt_length=args.prompt_length, prompt_verse=args.prompt_rhyme)
    
    # Text Line Training
    training_args = TrainingArguments(
                                  save_strategy  = "no",
                                  warmup_steps = len(train_data.pytorch_dataset_text)//args.batch_size_LM,
                                  logging_steps = 500,
                                  weight_decay = 0.05,
                                  num_train_epochs = args.epochs_LM,
                                  learning_rate = args.learning_rate,
                                  lr_scheduler_type="cosine",
                                  logging_dir = './logs',
                                  output_dir = './results',
                                  per_device_train_batch_size = args.batch_size_LM)
    
    
    trainer = Trainer(model = model,
                           args = training_args,
                           train_dataset= train_data.pytorch_dataset_text,
                           data_collator= CorpusDatasetPytorch.collate).train()
    
    # Verse Training
    training_args = TrainingArguments(
                                  save_strategy  = "no",
                                  warmup_steps = len(train_data.pytorch_dataset_body)//args.batch_size_poet,
                                  logging_steps = 500,
                                  weight_decay = 0.05,
                                  num_train_epochs = args.epochs_poet,
                                  learning_rate = args.learning_rate,
                                  lr_scheduler_type="constant_with_warmup",
                                  logging_dir = './logs',
                                  output_dir = './results',
                                  per_device_train_batch_size = args.batch_size_poet)
    
    
    trainer = Trainer(model = model,
                           args = training_args,
                           train_dataset= train_data.pytorch_dataset_body,
                           data_collator= CorpusDatasetPytorch.collate).train()
    
    
    
    
    model.save_LM(f"{args.model_path}_LM")
    tokenizer.save_pretrained(f"{args.model_path}_LM")
    torch.save(model, args.model_path)
      


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
