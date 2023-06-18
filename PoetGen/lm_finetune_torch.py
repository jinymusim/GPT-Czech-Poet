from transformers import  AutoTokenizer
from poet_model import PoetModel
from torch.utils.data import DataLoader
from corpus_dataset_torch import CorpusDatasetPytorch
from trainer_torch import Trainer
import transformers
import torch
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs to run.")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning Rate for Finetuning")
parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")
parser.add_argument("--model_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "gpt-cz-poetry")),  type=str, help="Path to Model")
parser.add_argument("--use_default_model",  default=True, type=bool, help="Use Default Model")
parser.add_argument("--default_hf_model", default="jinymusim/gpt-czech-poet", type=str, help="Default Model from HF to use")
parser.add_argument("--max_len", default=1024, type=int, help="Max length for tokenizer")
parser.add_argument("--use_gpu_if_available", default=True, type=bool, help="If GPU should be used")
parser.add_argument("--train_for_consistency", default=True, type=bool, help="Train for consistency secondary training")
parser.add_argument("--input_mask_rate", default=0.05, type=float, help="Rate of input masking")

def main(args: argparse.Namespace):
    # Base Device is CPU
    device = torch.device('cpu')
    # If Wanted and GPU is available, use it
    if args.use_gpu_if_available:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.use_default_model:
        tokenizer = AutoTokenizer.from_pretrained(args.default_hf_model)
        model = PoetModel(args.default_hf_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.default_hf_model)
        model = PoetModel(args.default_hf_model).load_state_dict(torch.load(args.model_path))
    
    tokenizer.model_max_length = args.max_len
        
    train_data = CorpusDatasetPytorch(tokenizer, data_dir=args.data_path)
    complete_list = train_data.pytorch_dataset_body  + train_data.pytorch_dataset_text  # + train_data.pytorch_dataset_part 
    dataloader = DataLoader(complete_list , batch_size=args.batch_size, collate_fn=CorpusDatasetPytorch.collate)
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 
                                                         len(complete_list)//args.batch_size,
                                                         len(complete_list)//args.batch_size *args.epochs)
    
    trainer = Trainer(model, device ,args.epochs, optimizer, scheduler, dataloader, args.train_for_consistency, args.input_mask_rate)
    trainer.train()
    
    model.save_LM(f"{args.model_path}_LM")
    torch.save(model.state_dict(), args.model_path)
      


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
