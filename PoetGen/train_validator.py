
import torch
import os
import argparse

from transformers import  AutoTokenizer, TrainingArguments, Trainer, PreTrainedTokenizerFast, PreTrainedTokenizerBase
from functools import partial


from corpus_capsulated_datasets import CorpusDatasetPytorch
from utils.validators import MeterValidator, RhymeValidator, ValidatorInterface

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--epochs", default=128, type=int, help="Number of epochs to run.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning Rate for Finetuning")
parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")

parser.add_argument("--tokenizer", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "utils", "tokenizers", "BPE", "tokenizer.json")), type=str, help="Default Model from HF to use")
parser.add_argument("--model_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "utils", "validators")),  type=str, help="Path to Model")
parser.add_argument("--max_len", default=1024, type=int, help="Max length for tokenizer")
parser.add_argument("--context_max_len", default=1024, type=int, help="Max length of context for tokenizer")
parser.add_argument("--verse_len", default=[4,6], type=list, help="Lengths of verses")

parser.add_argument("--prompt_rhyme", default=True, type=bool, help="Rhyme is prompted into training data")
parser.add_argument("--prompt_length", default=True, type=bool, help="Verse length is prompted into training data")
parser.add_argument("--prompt_ending", default=True, type=bool, help="Ending of Verse is prompted into training data")

parser.add_argument("--block_count", default=3, type=int, help="Max length for tokenizer")
parser.add_argument("--n_embd", default=512, type=int, help="Max length for tokenizer")

def validate(model: ValidatorInterface, data, times: int = 1000):
    true_hits = 0
    for i in range(times):
        true_hits += model.validate(input_ids=torch.tensor(data[i]["input_ids"].reshape(1,-1)),
                                    rhyme=torch.tensor(data[i]["rhyme"].reshape(1,-1)), 
                                    metre=torch.tensor(data[i]["metre"].reshape(1,-1)))
    print(f"Validation acc: {true_hits/times}")


def main(args):
    
    if not os.path.exists(os.path.abspath(os.path.join(args.model_path, "rhyme"))):
        os.makedirs(os.path.abspath(os.path.join(args.model_path, "rhyme")))
        
    if not os.path.exists(os.path.abspath(os.path.join(args.model_path, "meter"))):
        os.makedirs(os.path.abspath(os.path.join(args.model_path, "meter")))
        
    rhyme_model = RhymeValidator(block_count=args.block_count, n_embd=args.n_embd, input_size=args.max_len)
    meter_model = MeterValidator(block_count=args.block_count, n_embd=args.n_embd, input_size=args.max_len)
    
    try:    
        tokenizer: PreTrainedTokenizerBase =  AutoTokenizer.from_pretrained(args.tokenizer)
    except:
        tokenizer: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.eos_token_id = 0
        tokenizer.pad_token = '<|endoftext|>'
        tokenizer.pad_token_id = 0
        tokenizer.unk_token = "<|endoftext|>"
        tokenizer.unk_token_id = 0
        
    collate = partial(CorpusDatasetPytorch.collate, mask_rate=0.0)
    
    tokenizer.model_max_length = args.max_len
    train_data = CorpusDatasetPytorch(tokenizer, data_dir=args.data_path, 
                                      prompt_ending=args.prompt_ending, prompt_length=args.prompt_length, prompt_verse=args.prompt_rhyme,
                                      verse_len=args.verse_len, context_len=args.context_max_len)
    
    training_args = TrainingArguments(
                                  save_strategy  = "no",
                                  warmup_steps = len(train_data.pytorch_dataset_body)//args.batch_size,
                                  logging_steps = 500,
                                  weight_decay = 0.0,
                                  num_train_epochs = args.epochs,
                                  learning_rate = args.learning_rate,
                                  fp16 = True if torch.cuda.is_available() else False,
                                  ddp_backend = "nccl",
                                  lr_scheduler_type="constant",
                                  logging_dir = './logs',
                                  output_dir = './results',
                                  per_device_train_batch_size = args.batch_size)
    
    
    trainer = Trainer(model = rhyme_model,
                           args = training_args,
                           train_dataset= train_data.pytorch_dataset_body,
                           data_collator=collate).train()
    
    validate(rhyme_model.cpu(), train_data.pytorch_dataset_body)
    
    torch.save(rhyme_model, os.path.abspath(os.path.join(args.model_path, "rhyme", f"{type(tokenizer.backend_tokenizer.model).__name__}_validator")) )
    
    
    training_args = TrainingArguments(
                                  save_strategy  = "no",
                                  warmup_steps = len(train_data.pytorch_dataset_body)//args.batch_size,
                                  logging_steps = 500,
                                  weight_decay = 0.0,
                                  num_train_epochs = args.epochs,
                                  learning_rate = args.learning_rate,
                                  fp16 = True if torch.cuda.is_available() else False,
                                  ddp_backend = "nccl",
                                  lr_scheduler_type="constant",
                                  logging_dir = './logs',
                                  output_dir = './results',
                                  per_device_train_batch_size = args.batch_size)
    
    
    trainer = Trainer(model = meter_model,
                           args = training_args,
                           train_dataset= train_data.pytorch_dataset_body,
                           data_collator=collate).train()
    
    validate(meter_model.cpu(), train_data.pytorch_dataset_body)
    
    torch.save(meter_model, os.path.abspath(os.path.join(args.model_path, "meter", f"{type(tokenizer.backend_tokenizer.model).__name__}_validator")) )
    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
    