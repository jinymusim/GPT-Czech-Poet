from transformers import  AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from corpus_dataset_torch import CorpusDatasetPytorch
from trainer_torch import Trainer
import transformers
import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs to run.")
parser.add_argument("--learning_rate", default=1e-5, type=float, help="Learning Rate for Finetuning")
parser.add_argument("--seed", default=99, type=int, help="Random seed")
parser.add_argument("--data_path",  default="GPT2/corpusCzechVerse-master/ccv", type=str, help="Path to Data")
parser.add_argument("--model_path", default="./gpt2-cz-poetry",  type=str, help="Path to Model")
parser.add_argument("--use_default_model",  default=True, type=bool, help="Use Default Model")
parser.add_argument("--default_hf_model", default="lchaloupsky/czech-gpt2-oscar", type=str, help="Default Model from HF to use")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args: argparse.Namespace):
    
    
    if args.use_default_model:
        tokenizer = AutoTokenizer.from_pretrained(args.default_hf_model)
        model = AutoModelForCausalLM.from_pretrained(args.default_hf_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path)

        
    train_data = CorpusDatasetPytorch(tokenizer, data_dir=args.data_path)
    dataloader = DataLoader(train_data.pytorch_dataset_body, batch_size=args.batch_size, collate_fn=CorpusDatasetPytorch.collate)
    
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 
                                                         train_data.dataset.size//args.batch_size,
                                                         train_data.dataset.size//args.batch_size *args.epochs)
    
    trainer = Trainer(model, args.epochs, optimizer, scheduler, dataloader)
    trainer.train()
    
    model.save_pretrained(args.model_path)
    


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)