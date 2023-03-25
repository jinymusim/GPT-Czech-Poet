from transformers import PreTrainedModel, PreTrainedTokenizer
from corpus_dataset_pytorch import CorpusDatasetPytorch
import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs to run.")
parser.add_argument("--lr", default=1e-5, type=float, help="Learning Rate for Finetuning")
parser.add_argument("--seed", default=99, type=int, help="Random seed")
parser.add_argument("--data_path",  default="GPT2/corpusCzechVerse-master/ccv", type=str, help="Path to Data")
parser.add_argument("--model_path", default="./gpt2-cz-poetry",  type=str, help="Path to Model")
parser.add_argument("--use_default_model",  default=True, type=bool, help="Use Default Model")
parser.add_argument("--default_hf_model", default="lchaloupsky/czech-gpt2-oscar", type=str, help="Default Model from HF to use")


def main(args: argparse.Namespace):
    
    
    if args.use_default_model:
        tokenizer = PreTrainedTokenizer.from_pretrained(args.default_hf_model)
        model = PreTrainedModel.from_pretrained(args.default_hf_model)
    else:
        tokenizer = PreTrainedTokenizer.from_pretrained(args.model_path)
        model = PreTrainedModel.from_pretrained(args.model_path)
        
    train_data = CorpusDatasetPytorch(tokenizer, data_dir=args.data_path)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)