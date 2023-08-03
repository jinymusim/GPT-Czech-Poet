from transformers import  AutoTokenizer
from poet_model import PoetModel
from torch.utils.data import DataLoader
from corpus_dataset_torch import CorpusDatasetPytorch
from trainer_torch import Trainer
import transformers
import torch
import os
import argparse

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size_LM", default=4, type=int, help="Batch size.")
parser.add_argument("--epochs_LM", default=1, type=int, help="Number of epochs to run.")
parser.add_argument("--batch_size_poet", default=4, type=int, help="Batch size.")
parser.add_argument("--epochs_poet", default=1, type=int, help="Number of epochs for poet gen")

parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning Rate for Finetuning")
parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")
parser.add_argument("--model_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "m-gpt-cz-poetry")),  type=str, help="Path to Model")
parser.add_argument("--use_default_model",  default=True, type=bool, help="Use Default Model")
# huggyllama/llama-7b 4096
# lchaloupsky/czech-gpt2-oscar 1024
parser.add_argument("--default_hf_model", default="lchaloupsky/czech-gpt2-oscar", type=str, help="Default Model from HF to use")
parser.add_argument("--max_len", default=1024, type=int, help="Max length for tokenizer")
parser.add_argument("--use_gpu_if_available", default=True, type=bool, help="If GPU should be used")
parser.add_argument("--train_for_consistency", default=True, type=bool, help="Train for consistency secondary training")
parser.add_argument("--input_mask_rate", default=0.05, type=float, help="Rate of input masking")

parser.add_argument("--prompt_rhyme", default=True, type=bool, help="Rhyme is prompted into training data")
parser.add_argument("--prompt_length", default=True, type=bool, help="Verse length is prompted into training data")
parser.add_argument("--prompt_ending", default=True, type=bool, help="Ending of Verse is prompted into training data")
#TODO: Rhyme Prompting
#TODO: Tokenizer Analysis
#TODO: 


def ddp_setup(rank: int, world_size: int):
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   init_process_group(backend="nccl", rank=rank, world_size=world_size)
   torch.cuda.set_device(rank)
   
   
def main_(rank, world_size, args: argparse.Namespace):
    
    # Base Device is CPU
    ddp_setup(rank, world_size)
    multi_gpu = True
    
    if args.use_default_model:
        tokenizer = AutoTokenizer.from_pretrained(args.default_hf_model)
        model = PoetModel(args.default_hf_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.default_hf_model)
        model = PoetModel(args.default_hf_model).load_state_dict(torch.load(args.model_path))
    
    model = DistributedDataParallel(model, device_ids=[rank])
    
    tokenizer.model_max_length = args.max_len
    train_data = CorpusDatasetPytorch(tokenizer, data_dir=args.data_path, prompt_ending=args.prompt_ending, prompt_length=args.prompt_length)
    
    ### Basic Text Learning
    dataloader_text = DataLoader(train_data.pytorch_dataset_text , batch_size=args.batch_size_LM, 
                                     collate_fn=CorpusDatasetPytorch.collate, shuffle=False, sampler=DistributedSampler(train_data.pytorch_dataset_text))
    optimizer_text = torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    scheduler_text = transformers.get_cosine_schedule_with_warmup(optimizer_text, 
                                                         len(dataloader_text)//args.batch_size_LM,
                                                         len(dataloader_text)//args.batch_size_LM *args.epochs_LM)
    trainer_text = Trainer(model, rank ,args.epochs_LM, optimizer_text, scheduler_text, dataloader_text, args.train_for_consistency, args.input_mask_rate, multi_gpu)
    trainer_text.train()
    
    
    ### Part based learning
    dataloader_body = DataLoader(train_data.pytorch_dataset_body , batch_size=args.batch_size_poet, 
                                     collate_fn=CorpusDatasetPytorch.collate, shuffle=False, sampler=DistributedSampler(train_data.pytorch_dataset_body))
    optimizer_body= torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    ### To learn the structure => Constant scheduler
    scheduler_body= transformers.get_constant_schedule_with_warmup(optimizer_body, 
                                                         len(dataloader_body)//args.batch_size_poet)
    
    trainer_body = Trainer(model, rank ,args.epochs_poet, optimizer_body, scheduler_body, dataloader_body, args.train_for_consistency, args.input_mask_rate, multi_gpu)
    trainer_body.train()
    
    
    model.save_LM(f"{args.model_path}_LM")
    tokenizer.save_pretrained(f"{args.model_path}_LM")
    torch.save(model, args.model_path)
    destroy_process_group()

def main(args: argparse.Namespace):
    # Base Device is CPU
    device = torch.device('cpu')
    multi_gpu = False
    # If Wanted and GPU is available, use it
    if args.use_gpu_if_available:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.use_default_model:
        tokenizer = AutoTokenizer.from_pretrained(args.default_hf_model)
        model = PoetModel(args.default_hf_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.default_hf_model)
        model = PoetModel(args.default_hf_model).load_state_dict(torch.load(args.model_path))
    
    model = model.to(device) 
    
    tokenizer.model_max_length = args.max_len
    train_data = CorpusDatasetPytorch(tokenizer, data_dir=args.data_path, prompt_ending=args.prompt_ending, prompt_length=args.prompt_length)
    
    ### Basic Text Learning
    dataloader_text = DataLoader(train_data.pytorch_dataset_text , batch_size=args.batch_size_LM, collate_fn=CorpusDatasetPytorch.collate, shuffle=True)
    optimizer_text = torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    scheduler_text = transformers.get_cosine_schedule_with_warmup(optimizer_text, 
                                                         len(dataloader_text)//args.batch_size_LM,
                                                         len(dataloader_text)//args.batch_size_LM *args.epochs_LM)
    trainer_text = Trainer(model, device ,args.epochs_LM, optimizer_text, scheduler_text, dataloader_text, args.train_for_consistency, args.input_mask_rate, multi_gpu)
    trainer_text.train()
    
    
    ### Part based learning
    dataloader_body = DataLoader(train_data.pytorch_dataset_body , batch_size=args.batch_size_poet, collate_fn=CorpusDatasetPytorch.collate, shuffle=True)
    optimizer_body= torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    ### To learn the structure => Constant scheduler
    scheduler_body= transformers.get_constant_schedule_with_warmup(optimizer_body, 
                                                         len(dataloader_body)//args.batch_size_poet)
    
    trainer_body = Trainer(model, device ,args.epochs_poet, optimizer_body, scheduler_body, dataloader_body, args.train_for_consistency, args.input_mask_rate, multi_gpu)
    trainer_body.train()
    
    
    model.save_LM(f"{args.model_path}_LM")
    tokenizer.save_pretrained(f"{args.model_path}_LM")
    torch.save(model, args.model_path)
      


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(main_, args=(world_size, args), nprocs=world_size)
    else:
        main(args)
