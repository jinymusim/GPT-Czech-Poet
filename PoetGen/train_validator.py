
import torch
import os
import argparse

from accelerate import Accelerator
from transformers import  AutoTokenizer, TrainingArguments, Trainer, PreTrainedTokenizerFast, PreTrainedTokenizerBase
from functools import partial


from corpus_capsulated_datasets import CorpusDatasetPytorch
from utils.validators import MeterValidator, RhymeValidator, ValidatorInterface

from utils.poet_utils import VALID_CHARS

parser = argparse.ArgumentParser()


parser.add_argument("--epochs", default=32, type=int, help="Number of epochs to run.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning Rate for Finetuning")
parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")

parser.add_argument("--tokenizer", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "utils", "tokenizers", "BPE", "tokenizer.json")), type=str, help="Default Model from HF to use")
parser.add_argument("--model_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "utils", "validators")),  type=str, help="Path to Model")
parser.add_argument("--max_len_rhyme", default=24, type=int, help="Max length for tokenizer")
parser.add_argument("--max_len_metre", default=1024, type=int, help="Max length for tokenizer")
parser.add_argument("--verse_len", default=[4,6], type=list, help="Lengths of verses")

parser.add_argument("--prompt_rhyme", default=True, type=bool, help="Rhyme is prompted into training data")
parser.add_argument("--prompt_length", default=True, type=bool, help="Verse length is prompted into training data")
parser.add_argument("--prompt_ending", default=True, type=bool, help="Ending of Verse is prompted into training data")

parser.add_argument("--block_count", default=3, type=int, help="Max length for tokenizer")
parser.add_argument("--n_embd_metre", default=512, type=int, help="Max length for tokenizer")
parser.add_argument("--batch_size_metre", default=32, type=int, help="Batch size.")

parser.add_argument("--hidden_layers", default=3, type=int, help="Max length for tokenizer")
parser.add_argument("--hidden_layer_rhyme", default=512, type=int, help="Max length for tokenizer")
parser.add_argument("--batch_size_rhyme", default=128, type=int, help="Batch size.")

def validate(model: ValidatorInterface, data, collate_fnc,times: int = 1000):
    true_hits = 0
    for i in range(times):
        datum = collate_fnc([data[i]])
        true_hits += model.validate(input_ids=datum["input_ids"],
                                    rhyme=datum["rhyme"], 
                                    metre=datum["metre"])
    print(f"Validation acc: {true_hits/times}")


def main(args):
    
    if not os.path.exists(os.path.abspath(os.path.join(args.model_path, "rhyme"))):
        os.makedirs(os.path.abspath(os.path.join(args.model_path, "rhyme")))
        
    if not os.path.exists(os.path.abspath(os.path.join(args.model_path, "meter"))):
        os.makedirs(os.path.abspath(os.path.join(args.model_path, "meter")))
        
    rhyme_model = RhymeValidator(hidden_layers=args.hidden_layers, hidden_size=args.hidden_layer_rhyme, 
                                 input_size=args.max_len_rhyme * len(VALID_CHARS), raw_size=args.max_len_rhyme)
    meter_model = MeterValidator(block_count=args.block_count, n_embd=args.n_embd_metre, input_size=args.max_len_metre)
    
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
        
    collate_rhyme = partial(CorpusDatasetPytorch.collate_rhyme, tokenizer=tokenizer,max_len=args.max_len_rhyme, max_verse_len= max(args.verse_len))
    
    train_data = CorpusDatasetPytorch(data_dir=args.data_path, prompt_ending=args.prompt_ending, 
                                      prompt_length=args.prompt_length, prompt_verse=args.prompt_rhyme,
                                      verse_len=args.verse_len)
    
    # Parallel Plugin
    
    training_args = TrainingArguments(
                                  save_strategy  = "no",
                                  warmup_steps = len(train_data.pytorch_dataset_body)//args.batch_size_rhyme,
                                  logging_steps = 500,
                                  weight_decay = 0.0,
                                  num_train_epochs = args.epochs,
                                  learning_rate = args.learning_rate,
                                  fp16 = True if torch.cuda.is_available() else False,
                                  ddp_backend = "nccl",
                                  lr_scheduler_type="constant",
                                  logging_dir = './logs',
                                  output_dir = './results',
                                  per_device_train_batch_size = args.batch_size_rhyme)
    
    
    trainer = Trainer(model = rhyme_model,
                           args = training_args,
                           train_dataset= train_data.pytorch_dataset_body,
                           data_collator=collate_rhyme).train()
    
    validate(rhyme_model.cpu(), train_data.pytorch_dataset_body,collate_rhyme)
    
    torch.save(rhyme_model, os.path.abspath(os.path.join(args.model_path, "rhyme", f"{type(tokenizer.backend_tokenizer.model).__name__}_validator")) )
    
    
    collate  = partial(CorpusDatasetPytorch.collate, tokenizer=tokenizer, max_len=args.max_len_metre)
    
    training_args = TrainingArguments(
                                  save_strategy  = "no",
                                  warmup_steps = len(train_data.pytorch_dataset_body)//args.batch_size_metre,
                                  logging_steps = 500,
                                  weight_decay = 0.0,
                                  num_train_epochs = args.epochs,
                                  learning_rate = args.learning_rate,
                                  fp16 = True if torch.cuda.is_available() else False,
                                  ddp_backend = "nccl",
                                  lr_scheduler_type="constant",
                                  logging_dir = './logs',
                                  output_dir = './results',
                                  per_device_train_batch_size = args.batch_size_metre)
    
    
    trainer = Trainer(model = meter_model,
                           args = training_args,
                           train_dataset= train_data.pytorch_dataset_body,
                           data_collator=collate).train()
    
    validate(meter_model.cpu(), train_data.pytorch_dataset_body, collate)
    
    torch.save(meter_model, os.path.abspath(os.path.join(args.model_path, "meter", f"{type(tokenizer.backend_tokenizer.model).__name__}_validator")) )
    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
    