
import torch
import os
import argparse
import time

from transformers import  AutoTokenizer, TrainingArguments, Trainer, PreTrainedTokenizerFast, PreTrainedTokenizerBase
from functools import partial


from corpus_capsulated_datasets import CorpusDatasetPytorch
from utils.validators import MeterValidator, RhymeValidator, YearValidator,ValidatorInterface, ValidatorTrainer

from utils.poet_utils import VALID_CHARS, UNK, PAD, EOS
from utils.poet_model_utils import ModelManipulation

parser = argparse.ArgumentParser()



parser.add_argument("--learning_rate_rhyme", default=5e-5, type=float, help="Learning Rate for Finetuning")
parser.add_argument("--learning_rate_metre", default=5e-5, type=float, help="Learning Rate for Finetuning")
parser.add_argument("--learning_rate_year", default=5e-5, type=float, help="Learning Rate for Finetuning")

parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")
#parser.add_argument("--tokenizer", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "utils", "tokenizers", "BPE", "syllabs_processed_tokenizer.json")), type=str, help="Tokenizer to use")
parser.add_argument("--tokenizer", default="roberta-base", type=str, help="Tokenizer to use")
parser.add_argument("--model_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "utils", "validators")),  type=str, help="Path to Model")
parser.add_argument("--max_len", default=512, type=int, help="Max length for tokenizer")
parser.add_argument("--verse_len", default=[4,6], type=list, help="Lengths of verses")

parser.add_argument("--prompt_rhyme", default=True, type=bool, help="Rhyme is prompted into training data")
parser.add_argument("--prompt_length", default=True, type=bool, help="Verse length is prompted into training data")
parser.add_argument("--prompt_ending", default=True, type=bool, help="Ending of Verse is prompted into training data")

parser.add_argument("--syllables", default=True, type=bool, help="If to use syllable data")

parser.add_argument("--pretrained_model", default="roberta-base", type=str, help="Roberta Model")

parser.add_argument("--batch_size_metre", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs_metre", default=0, type=int, help="Number of epochs to run.")

parser.add_argument("--batch_size_rhyme", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs_rhyme", default=4, type=int, help="Number of epochs to run.")

parser.add_argument("--batch_size_year", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs_year", default=4, type=int, help="Number of epochs to run.")

parser.add_argument("--lower_case", default=True, type=bool, help="If to lower case data")
parser.add_argument("--val_data_rate", default=0.05, type=float, help="Rate of validation data")

parser.add_argument("--result_file", default=os.path.abspath(os.path.join(os.path.dirname(__file__),'results', "validators_acc.txt")), type=str, help="Result of Analysis File")

def validate(model: ValidatorInterface, data, collate_fnc):
    """Validate model for accuracy on trained task

    Args:
        model (ValidatorInterface): Model to validate
        data (_type_): Validation data
        collate_fnc (_type_): Function to transform data for model

    Returns:
        float: Accuracy of model
    """
    model.eval()
    
    true_hits = 0
    for i in range(len(data)):
        datum = collate_fnc([data[i]])
        true_hits += model.validate(input_ids=datum["input_ids"],
                                    rhyme=datum["rhyme"], 
                                    metre=datum["metre"],
                                    year=datum['year'])['acc']
    print(f"Validation acc: {true_hits/len(data)}")
    
    model.train()
    
    return true_hits/len(data)


def main(args):
    # Time stamp for the validators
    time_stamp = int(round(time.time() * 1000))
    
    # Create directory for Validators to store to
    if not os.path.exists(os.path.abspath(os.path.join(args.model_path, "rhyme"))):
        os.makedirs(os.path.abspath(os.path.join(args.model_path, "rhyme")))
        
    if not os.path.exists(os.path.abspath(os.path.join(args.model_path, "meter"))):
        os.makedirs(os.path.abspath(os.path.join(args.model_path, "meter")))
        
    if not os.path.exists(os.path.abspath(os.path.join(args.model_path, "year"))):
        os.makedirs(os.path.abspath(os.path.join(args.model_path, "year")))
        
    # Create Validators    
    rhyme_model = RhymeValidator(pretrained_model=args.pretrained_model)
    meter_model = MeterValidator(pretrained_model=args.pretrained_model)
    year_model = YearValidator(pretrained_model=args.pretrained_model)
        
    # Load tokenizer for validators
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
        
        ModelManipulation.exchange_embedding_roberta(meter_model, new_tokenizer=tokenizer, old_tokenizer=AutoTokenizer.from_pretrained(args.pretrained_model))
        ModelManipulation.exchange_embedding_roberta(rhyme_acc, new_tokenizer=tokenizer, old_tokenizer=AutoTokenizer.from_pretrained(args.pretrained_model))
        ModelManipulation.exchange_embedding_roberta(year_model, new_tokenizer=tokenizer, old_tokenizer=AutoTokenizer.from_pretrained(args.pretrained_model))
        
      
    collate  = partial(CorpusDatasetPytorch.collate_validator, tokenizer=tokenizer, max_len=args.max_len, syllables=args.syllables, is_syllable=True)  
   
    # Train Rhyme Validator 

    
    train_data = CorpusDatasetPytorch(data_dir=args.data_path, prompt_ending=args.prompt_ending, 
                                      prompt_length=args.prompt_length, prompt_verse=args.prompt_rhyme,
                                      verse_len=args.verse_len, lower_case=args.lower_case, val_data_rate=args.val_data_rate)
    
    if torch.cuda.device_count() > -1:
        if args.epochs_rhyme > 0:
            training_args = TrainingArguments(
                                      save_strategy  = "no",
                                      logging_steps = 500,
                                      warmup_steps = len(train_data.pytorch_dataset_body)//args.batch_size_rhyme,
                                      weight_decay = 0.0,
                                      num_train_epochs = args.epochs_rhyme,
                                      learning_rate = args.learning_rate_rhyme,
                                      fp16 = True if torch.cuda.is_available() else False,
                                      ddp_backend = "nccl",
                                      lr_scheduler_type="cosine",
                                      logging_dir = './logs',
                                      output_dir = './results',
                                      per_device_train_batch_size = args.batch_size_rhyme)

            trainer = Trainer(model = rhyme_model,
                               args = training_args,
                               train_dataset= train_data.pytorch_dataset_body,
                               data_collator=collate).train()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        training_args = {"lr" : args.learning_rate_rhyme,
                         "epochs" : args.epochs_rhyme,
                         "batch_size" : args.batch_size_rhyme}
        
        rhyme_model = rhyme_model.to(device)
        
        trainer = ValidatorTrainer(model=rhyme_model, 
                                   args=training_args, 
                                   train_dataset=train_data.pytorch_dataset_body, 
                                   data_collator=collate,
                                   device=device).train()
        
        
    # Validate rhyme Validator on validation data
    rhyme_acc = 0
    if args.epochs_rhyme > 0:
        rhyme_acc =  validate(rhyme_model.cpu(), train_data.pytorch_dataset_body.validation_data, collate)
    
        torch.save(rhyme_model, os.path.abspath(os.path.join(args.model_path, "rhyme", f"{args.pretrained_model.replace('/', '-')}_{'syllable_' if args.syllables else ''}{type(tokenizer.backend_tokenizer.model).__name__}_validator_{time_stamp}")) )
    
    # Train Metrum Validator
    
    if torch.cuda.device_count() > -1:
        if args.epochs_metre > 0:
            training_args = TrainingArguments(
                                      save_strategy  = "no",
                                      warmup_steps = len(train_data.pytorch_dataset_body)//args.batch_size_metre,
                                      logging_steps = 500,
                                      weight_decay = 0.0,
                                      num_train_epochs = args.epochs_metre,
                                      learning_rate = args.learning_rate_metre,
                                      fp16 = True if torch.cuda.is_available() else False,
                                      ddp_backend = "nccl",
                                      lr_scheduler_type="cosine",
                                      logging_dir = './logs',
                                      output_dir = './results',
                                      per_device_train_batch_size = args.batch_size_metre)


            trainer = Trainer(model = meter_model,
                               args = training_args,
                               train_dataset= train_data.pytorch_dataset_body,
                               data_collator=collate).train()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        training_args = {"lr" : args.learning_rate_metre,
                         "epochs" : args.epochs_metre,
                         "batch_size" : args.batch_size_metre}
        
        meter_model = meter_model.to(device)
        
        trainer = ValidatorTrainer(model=meter_model, 
                                   args=training_args, 
                                   train_dataset=train_data.pytorch_dataset_body, 
                                   data_collator=collate,
                                   device=device).train()
    # Validate Metrum validator on validation data
    metre_acc = 0
    if args.epochs_metre > 0:
        metre_acc = validate(meter_model.cpu(), train_data.pytorch_dataset_body.validation_data, collate)
    
        torch.save(meter_model, os.path.abspath(os.path.join(args.model_path, "meter", f"{args.pretrained_model.replace('/', '-')}_{'syllable_' if args.syllables else ''}{type(tokenizer.backend_tokenizer.model).__name__}_validator_{time_stamp}")) )
    
    # Train Year Validator
    
    if torch.cuda.device_count() > -1:
        if args.epochs_year > 0:
            training_args = TrainingArguments(
                                      save_strategy  = "no",
                                      warmup_steps = len(train_data.pytorch_dataset_body)//args.batch_size_year,
                                      logging_steps = 500,
                                      weight_decay = 0.0,
                                      num_train_epochs = args.epochs_year,
                                      learning_rate = args.learning_rate_year,
                                      fp16 = True if torch.cuda.is_available() else False,
                                      ddp_backend = "nccl",
                                      lr_scheduler_type="cosine",
                                      logging_dir = './logs',
                                      output_dir = './results',
                                      per_device_train_batch_size = args.batch_size_year)


            trainer = Trainer(model = year_model,
                               args = training_args,
                               train_dataset= train_data.pytorch_dataset_body,
                               data_collator=collate).train()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        training_args = {"lr" : args.learning_rate_metre,
                         "epochs" : args.epochs_metre,
                         "batch_size" : args.batch_size_metre}
        
        year_model = year_model.to(device)
        
        trainer = ValidatorTrainer(model=year_model, 
                                   args=training_args, 
                                   train_dataset=train_data.pytorch_dataset_body, 
                                   data_collator=collate,
                                   device=device).train()
    
    year_acc = 0
    if args.epochs_year > 0:
        year_acc = validate(year_model.cpu(), train_data.pytorch_dataset_body.validation_data, collate)
    
        torch.save(year_model, os.path.abspath(os.path.join(args.model_path, "year", f"{args.pretrained_model.replace('/', '-')}_{'syllable_' if args.syllables else ''}{type(tokenizer.backend_tokenizer.model).__name__}_validator_{time_stamp}")) )
    
    
    # Store result and model
    with open(args.result_file, 'a') as file:
        print(f"### {type(tokenizer.backend_tokenizer.model).__name__} ### {time_stamp} ### Syllable: {str(args.syllables)} ", file=file)
        print(f"Rhyme Validator: {args.pretrained_model}, Epochs: {args.epochs_rhyme} Accuracy: {rhyme_acc}", file=file)
        print(f"Metre Validator: {args.pretrained_model}, Epochs: {args.epochs_metre} Accuracy: {metre_acc}", file=file)
        print(f"Year Validator: {args.pretrained_model}, Epochs: {args.epochs_year} Accuracy: {year_acc}", file=file)
    
    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
    