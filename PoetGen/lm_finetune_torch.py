# Outide Packages
import transformers
import torch
import os
import argparse

from transformers import  AutoTokenizer
from torch.utils.data import DataLoader
#from torch.distributed.tensor.parallel import parallelize_module, PairwiseParallel

# Project Packages
from poet_model_base_lm import PoetModelBase
from poet_model_secondary_tasks import PoetModelSecondaryTasks

from trainer_torch import Trainer

from corpus_dataset_torch import CorpusDatasetPytorch


parser = argparse.ArgumentParser()

parser.add_argument("--batch_size_LM", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs_LM", default=2, type=int, help="Number of epochs to run.")
parser.add_argument("--batch_size_poet", default=4, type=int, help="Batch size.")
parser.add_argument("--epochs_poet", default=2, type=int, help="Number of epochs for poet gen")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning Rate for Finetuning")
parser.add_argument("--use_gpu_if_available", default=True, type=bool, help="If GPU should be used")
parser.add_argument("--use_multiple_gpu_if_available", default=True, type=bool, help="If to use multiple gpus")
parser.add_argument("--train_masked", default=False, type=bool, help="Train for consistency secondary training")
parser.add_argument("--input_mask_rate", default=0.05, type=float, help="Rate of input masking")

parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")

# huggyllama/llama-7b 4096
# lchaloupsky/czech-gpt2-oscar 1024

parser.add_argument("--default_hf_model", default="lchaloupsky/czech-gpt2-oscar", type=str, help="Default Model from HF to use")
parser.add_argument("--use_default_model",  default=True, type=bool, help="Use Default Model")
parser.add_argument("--model_type",  default="base", type=str, choices=["base", "secondary_tasks"], help="What type of Model is to be constructed")
parser.add_argument("--model_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "gpt-cz-poetry-base")),  type=str, help="Path to Model")
parser.add_argument("--max_len", default=1024, type=int, help="Max length for tokenizer")


parser.add_argument("--prompt_rhyme", default=True, type=bool, help="Rhyme is prompted into training data")
parser.add_argument("--prompt_length", default=True, type=bool, help="Verse length is prompted into training data")
parser.add_argument("--prompt_ending", default=True, type=bool, help="Ending of Verse is prompted into training data")



def main(args: argparse.Namespace):
    # Base Device is CPU
    device = torch.device('cpu')
    multi_gpu = False
    # If Wanted and GPU is available, use it
    if args.use_gpu_if_available:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if args.use_default_model:
        tokenizer = AutoTokenizer.from_pretrained(args.default_hf_model)
        if args.model_type == "base":         
            model = PoetModelBase(args.default_hf_model)
        elif args.model_type == "secondary_tasks":
            model = PoetModelSecondaryTasks(args.default_hf_model)
        else:
            model = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.default_hf_model)
        model = torch.load(args.model_path_full, map_location=torch.device('cpu'))
    
    #if args.use_gpu_if_available and torch.cuda.is_available():
    #    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    #else:
    #if args.use_multiple_gpu_if_available and torch.cuda.device_count() > 1:
    #    model = parallelize_module(model, PairwiseParallel())
    #else:
    model = model.to(device)
    
    tokenizer.model_max_length = args.max_len
    train_data = CorpusDatasetPytorch(tokenizer, data_dir=args.data_path, prompt_ending=args.prompt_ending, prompt_length=args.prompt_length)
    
    ### Basic Text Learning
    dataloader_text = DataLoader(train_data.pytorch_dataset_text , batch_size=args.batch_size_LM, collate_fn=CorpusDatasetPytorch.collate, shuffle=True)
    optimizer_text = torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    scheduler_text = transformers.get_cosine_schedule_with_warmup(optimizer_text, 
                                                         len(dataloader_text)//args.batch_size_LM,
                                                         len(dataloader_text)//args.batch_size_LM *args.epochs_LM)
    trainer_text = Trainer(model=model, device=device ,epochs=args.epochs_LM, optimizer=optimizer_text, 
                           scheduler=scheduler_text, dataloader=dataloader_text, train_masked=args.train_masked, 
                           masking_rate=args.input_mask_rate, multi_gpu=multi_gpu)
    trainer_text.train()
    
    
    ### Part based learning
    dataloader_body = DataLoader(train_data.pytorch_dataset_body , batch_size=args.batch_size_poet, collate_fn=CorpusDatasetPytorch.collate, shuffle=True)
    optimizer_body= torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    ### To learn the structure => Constant scheduler
    scheduler_body= transformers.get_constant_schedule_with_warmup(optimizer_body, 
                                                         len(dataloader_body)//args.batch_size_poet)
    
    trainer_body = Trainer(model=model,device=device,epochs=args.epochs_poet,optimizer=optimizer_body, 
                           scheduler=scheduler_body, dataloader=dataloader_body, train_masked=args.train_masked, 
                           masking_rate=args.input_mask_rate, multi_gpu=multi_gpu)
    trainer_body.train()
    
    
    model.save_LM(f"{args.model_path}_LM")
    tokenizer.save_pretrained(f"{args.model_path}_LM")
    torch.save(model, args.model_path)
      


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
