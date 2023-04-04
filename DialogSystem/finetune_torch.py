import transformers
from transformers import  AutoTokenizer, AutoModelForCausalLM
from finetune_dataloader import DialogDataset
from trainer_torch import Trainer
from torch.utils.data import DataLoader
import torch
import os
import argparse
from special_tokens import SPECIAL_TOKENS

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="Seed to set for Torch")
parser.add_argument("--batch_size", default=4,  type=int, help="Batch size")
parser.add_argument("--epochs", default=4, type=int, help="Number of Epochs to finetune")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning rate for finetune")
parser.add_argument("--max_token_len", default=1024, type=int, help="Max length for tokenizer")
parser.add_argument("--use_default_model", default=True, type=bool, help="Bool if default huggingface model used")
parser.add_argument("--default_hf_model", default="microsoft/DialoGPT-small", type=str, help="Default huggingface model path")
parser.add_argument("--model_path", default=os.path.abspath(os.path.join(os.path.dirname("__file__"), "dialogmodel")), type=str, help="Model path")
parser.add_argument("--use_gpu_if_available", default=True, type=bool, help="If GPU should be used")
parser.add_argument("--dataset", default="daily_dialog", type=str, help="Dialog Dataset to use for finetune")

def main(args: argparse.Namespace):
    # Base Device is CPU
    device = torch.device('cpu')
    # If Wanted and GPU is available, use it
    if args.use_gpu_if_available:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use default HF Model if requested
    if args.use_default_model:
        tokenizer = AutoTokenizer.from_pretrained(args.default_hf_model)
        model = AutoModelForCausalLM.from_pretrained(args.default_hf_model)
    else: # Use Own Model if default not used
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
    
    tokenizer.model_max_length = args.max_token_len
    # Embedding Size + Tokens
    model.resize_token_embeddings(50257 + len(SPECIAL_TOKENS))
    # Move Model to desired Device
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    train_dat, validation_dat = DialogDataset(args.dataset, 'train', tokenizer), DialogDataset(args.dataset, 'validation', tokenizer)
    
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, len(train_dat.data)//args.batch_size, len(train_dat.data) * args.epochs // args.batch_size)
    
    
    train_loader = DataLoader(train_dat.data,batch_size=args.batch_size, collate_fn=DialogDataset.collate)
    val_loader = DataLoader(validation_dat.data,batch_size=args.batch_size, collate_fn=DialogDataset.collate)
    
    trainer = Trainer(model, args.epochs, optimizer, scheduler, train_loader)
    trainer.train()
    
    # Save Model
    model.save_pretrained(args.model_path)
              
        
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
