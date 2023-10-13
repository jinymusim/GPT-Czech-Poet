
import torch
import os
import argparse
import time

from accelerate import Accelerator
from transformers import  AutoTokenizer, TrainingArguments, Trainer, PreTrainedTokenizerFast, PreTrainedTokenizerBase
from functools import partial


from corpus_capsulated_datasets import CorpusDatasetPytorch
from utils.validators import MeterValidator, RhymeValidator, ValidatorInterface

from utils.poet_utils import VALID_CHARS, UNK, PAD, EOS

parser = argparse.ArgumentParser()



parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning Rate for Finetuning")
parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")

parser.add_argument("--tokenizer", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "utils", "tokenizers", "BPE", "syllabs_processed_tokenizer.json")), type=str, help="Default Model from HF to use")
parser.add_argument("--model_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "utils", "validators")),  type=str, help="Path to Model")
parser.add_argument("--max_len_rhyme", default=48, type=int, help="Max length for tokenizer")
parser.add_argument("--max_len_metre", default=1024, type=int, help="Max length for tokenizer")
parser.add_argument("--verse_len", default=[4,6], type=list, help="Lengths of verses")

parser.add_argument("--prompt_rhyme", default=True, type=bool, help="Rhyme is prompted into training data")
parser.add_argument("--prompt_length", default=True, type=bool, help="Verse length is prompted into training data")
parser.add_argument("--prompt_ending", default=True, type=bool, help="Ending of Verse is prompted into training data")

parser.add_argument("--syllables", default=False, type=bool, help="If to use syllable data")

parser.add_argument("--block_count", default=3, type=int, help="Max length for tokenizer")
parser.add_argument("--n_embd_metre", default=512, type=int, help="Max length for tokenizer")
parser.add_argument("--batch_size_metre", default=256, type=int, help="Batch size.")
parser.add_argument("--epochs_metre", default=128, type=int, help="Number of epochs to run.")

parser.add_argument("--hidden_layers", default=2, type=int, help="Max length for tokenizer")
parser.add_argument("--hidden_layer_rhyme", default=2048, type=int, help="Max length for tokenizer")
parser.add_argument("--batch_size_rhyme", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs_rhyme", default=1024, type=int, help="Number of epochs to run.")

parser.add_argument("--lower_case", default=True, type=bool, help="If to lower case data")
parser.add_argument("--val_data_rate", default=0.1, type=float, help="Rate of validation data")

parser.add_argument("--result_file", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'results', "validators_acc.txt")), type=str, help="Result of Analysis File")

def validate(model: ValidatorInterface, data, collate_fnc):
    model.eval()
    
    true_hits = 0
    for i in range(len(data)):
        datum = collate_fnc([data[i]])
        true_hits += model.validate(input_ids=datum["input_ids"],
                                    rhyme=datum["rhyme"], 
                                    metre=datum["metre"])
    print(f"Validation acc: {true_hits/len(data)}")
    
    model.train()
    
    return true_hits/len(data)


def main(args):
    # Time stamp for the validators
    time_stamp = int(round(time.time() * 1000))
    
    if not os.path.exists(os.path.abspath(os.path.join(args.model_path, "rhyme"))):
        os.makedirs(os.path.abspath(os.path.join(args.model_path, "rhyme")))
        
    if not os.path.exists(os.path.abspath(os.path.join(args.model_path, "meter"))):
        os.makedirs(os.path.abspath(os.path.join(args.model_path, "meter")))
        
    try:    
        tokenizer: PreTrainedTokenizerBase =  AutoTokenizer.from_pretrained(args.tokenizer)
    except:
        tokenizer: PreTrainedTokenizerBase = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer)
        tokenizer.eos_token = EOS
        tokenizer.eos_token_id = 0
        tokenizer.pad_token = PAD
        tokenizer.pad_token_id = 1
        tokenizer.unk_token = UNK
        tokenizer.unk_token_id = 2
        
    rhyme_model = RhymeValidator(hidden_layers=args.hidden_layers, hidden_size=args.hidden_layer_rhyme, 
                                 input_size=args.max_len_rhyme * len(VALID_CHARS), raw_size=args.max_len_rhyme)
    meter_model = MeterValidator(block_count=args.block_count, n_embd=args.n_embd_metre, input_size=args.max_len_metre, vocab_size=tokenizer.vocab_size)
        
    collate_rhyme = partial(CorpusDatasetPytorch.collate_rhyme,max_len=args.max_len_rhyme, max_verse_len= max(args.verse_len))
    
    train_data = CorpusDatasetPytorch(data_dir=args.data_path, prompt_ending=args.prompt_ending, 
                                      prompt_length=args.prompt_length, prompt_verse=args.prompt_rhyme,
                                      verse_len=args.verse_len, lower_case=args.lower_case, val_data_rate=args.val_data_rate)
    
    # Parallel Plugin
    
    training_args = TrainingArguments(
                                  save_strategy  = "no",
                                  logging_steps = 500,
                                  weight_decay = 0.0,
                                  num_train_epochs = args.epochs_rhyme,
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
    
    rhyme_acc =  validate(rhyme_model.cpu(), train_data.pytorch_dataset_body.validation_data,collate_rhyme)

    
    torch.save(rhyme_model, os.path.abspath(os.path.join(args.model_path, "rhyme", f"{'syllable_' if args.syllables else ''}{type(tokenizer.backend_tokenizer.model).__name__}_validator_{time_stamp}")) )
    
    
    collate  = partial(CorpusDatasetPytorch.collate, tokenizer=tokenizer, max_len=args.max_len_metre, syllables=args.syllables)
    
    training_args = TrainingArguments(
                                  save_strategy  = "no",
                                  warmup_steps = 0,
                                  logging_steps = 500,
                                  weight_decay = 0.0,
                                  num_train_epochs = args.epochs_metre,
                                  learning_rate = args.learning_rate,
                                  fp16 = True if torch.cuda.is_available() else False,
                                  ddp_backend = "nccl",
                                  lr_scheduler_type="cosine_with_restarts",
                                  logging_dir = './logs',
                                  output_dir = './results',
                                  per_device_train_batch_size = args.batch_size_metre)
    
    
    trainer = Trainer(model = meter_model,
                           args = training_args,
                           train_dataset= train_data.pytorch_dataset_body,
                           data_collator=collate).train()
    
    metre_acc = validate(meter_model.cpu(), train_data.pytorch_dataset_body.validation_data, collate)
    
    with open(args.result_file, 'a') as file:
        print(f"### {args.tokenizer} ### {time_stamp}", file=file)
        print(f"Rhyme Validator: MLP {args.hidden_layers},{args.hidden_layer_rhyme} Epochs: {args.epochs_rhyme} Accuracy: {rhyme_acc}", file=file)
        print(f"Metre Validator: GPT {args.block_count},{args.n_embd_metre},{args.max_len_metre} Epochs: {args.epochs_metre} Accuracy: {metre_acc}", file=file)
    
    torch.save(meter_model, os.path.abspath(os.path.join(args.model_path, "meter", f"{type(tokenizer.backend_tokenizer.model).__name__}_validator_{time_stamp}")) )
    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
    