# GPT Czech Poet
Learning models for Czech Poet Generation

## Getting training data
Training data needs to be separately downloaded from github project `https://github.com/versotym/corpusCzechVerse`  
This project by Institute of Czech Literature, Czech Academy of Sciences, incorporates 1~305 books of poetry.  
For use, all modeling scripts automatically converts data to needed format.  

## GPT Models
For learning Czech GPT Models use script `PoetGen\lm_finetune_torch_trainer_api.py`.  
The script allows multiple different types of models and tokenizers.  
For example, if custom tokenizer is used, model will update its embeddings to match new vocabulary.

```
parser.add_argument("--default_hf_model", default="lchaloupsky/czech-gpt2-oscar", type=str, help="Default Model from HF to use")
parser.add_argument("--use_default_model",  default=True, type=bool, help="Use Default Model")
parser.add_argument("--tokenizer", default='lchaloupsky/czech-gpt2-oscar', type=str, help="Tokenizer to use")
parser.add_argument("--model_type",  default="base", type=str, choices=["base", "secondary_tasks", "half", "verse", "context", "year", "all"], help="What type of Model is to be constructed")
parser.add_argument("--model_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "Base-Tokenizer-NormalText-gpt-cz-poetry-all-e4e8")),  type=str, help="Path to Model")
parser.add_argument("--max_len", default=1024, type=int, help="Max length for tokenizer")
parser.add_argument("--context_max_len", default=8, type=int, help="Max length of context for tokenizer")
parser.add_argument("--verse_len", default=[4,6], type=list, help="Lengths of verses")


parser.add_argument("--prompt_rhyme", default=True, type=bool, help="Rhyme is prompted into training data")
parser.add_argument("--prompt_length", default=True, type=bool, help="Verse length is prompted into training data")
parser.add_argument("--prompt_ending", default=True, type=bool, help="Ending of Verse is prompted into training data")

parser.add_argument("--syllables", default=False, type=bool, help="If inputs should be parsed by syllables")
parser.add_argument("--lower_case", default=True, type=bool, help="If to lower case data")

parser.add_argument("--mirror_imbed", default=True, type=bool, help="If to mirror input embedding to output ones")

parser.add_argument("--val_data_rate", default=0.05, type=float, help="Rate of validation data")
```

### Generation
Each model can be generated for in standard way by calling `model.model.generate(tokenized_start, **kwarg)`  
or by using the Forced Generation function `model.generate_forced(poem_parameters)` that iteratively generates to match rhyme schema.  

### Testing Model
For model testing use script `test_load_model.py` that load the model and tests it on empty inputs.
Script will generate strophe using both type of model generation and output it to specified file

## Validators
For multiple parameters, it's possible to train validators to check if they are followed.  
For this use `train_validator.py` script to train validators for rhyme schemes, meters or years of publishing.  

### Model Validation
To validate model using validators, use script `model_validator.py` with your model and validators.  
Do not forget to specify if validators used syllables in training.

# License
CC-BY-SA
