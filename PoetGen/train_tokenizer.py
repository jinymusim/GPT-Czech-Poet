import os
import argparse
import time


from transformers import AutoTokenizer
from tokenizers import Tokenizer

from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer

from tokenizers.pre_tokenizers import ByteLevel as BytePre, Whitespace
from tokenizers.processors import ByteLevel as BytePost, RobertaProcessing
from tokenizers.decoders import ByteLevel as ByteDec, WordPiece as WordDec

from tokenizers.normalizers import NFD

from utils.poet_utils import StropheParams, Tokens
from corpus_capsulated_datasets import CorpusDatasetPytorch

parser = argparse.ArgumentParser()

parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")
# huggyllama/llama-7b 4096
# lchaloupsky/czech-gpt2-oscar 1024
# bigscience/bloom-560m 2048
# TheBloke/Llama-2-7B-fp16 4096
# spital/gpt2-small-czech-cs 1024
parser.add_argument("--default_tokenizer", default="BUT-FIT/CSTinyLlama-1.2B", type=str, help="Default Model from HF to use")
parser.add_argument("--tokenizer_type", default="Unicode", type=str, choices=["BPE", "Unigram", "WordLevel", "WordPiece", "Unicode", "VerseMarks"], help="What type of tokenize to train")

parser.add_argument("--tokenizer_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__),"utils","tokenizers")),  type=str, help="Path to Model")
parser.add_argument("--raw_data", default=False,  type=bool, help="If to use raw data")
parser.add_argument("--syllables", default=False,  type=bool, help="If to use syllables")

parser.add_argument("--lower_case", default=True, type=bool, help="If to lower case data")
parser.add_argument("--class_token", default=True, type=bool, help="If to add class token")


def main(args):
    special_token_map = [Tokens.EOS, Tokens.PAD, Tokens.UNK]
    if args.class_token:
        special_token_map += [Tokens.CLS]
        
    
    
    # Create tokenizer based on arguments. Keep the structural parameters (vocabulary size) from default tokenizer
    tok = AutoTokenizer.from_pretrained(args.default_tokenizer)
    
    true_vocab_size  = tok.vocab_size // 3 if args.syllables else (tok.vocab_size + 2 if args.class_token else tok.vocab_size)
    if args.tokenizer_type == "BPE" or args.tokenizer_type == 'VerseMarks':
        tokenizer = Tokenizer(BPE())
        trainer = BpeTrainer(special_tokens=special_token_map, vocab_size = true_vocab_size, min_frequency=2,
                             initial_alphabet= ["#"] + StropheParams.METER_TYPES[:-1] + StropheParams.RHYME_SCHEMES[:-1])
        
        tokenizer.pre_tokenizer = BytePre(add_prefix_space=False)
        tokenizer.decoder = ByteDec()
        if args.class_token:
            tokenizer.post_processor = RobertaProcessing((Tokens.EOS, 0), (Tokens.CLS, 3), trim_offsets=False, add_prefix_space=False)
        else:
            tokenizer.post_processor = BytePost(trim_offsets=False)
    elif args.tokenizer_type == "Unigram":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(unk_token=Tokens.UNK,special_tokens=special_token_map, vocab_size = true_vocab_size,
                                 initial_alphabet= ["#"] + StropheParams.METER_TYPES[:-1] + StropheParams.RHYME_SCHEMES[:-1])
        
        tokenizer.pre_tokenizer = BytePre(add_prefix_space=False)
        tokenizer.decoder = ByteDec()
        if args.class_token:
            tokenizer.post_processor = RobertaProcessing((Tokens.EOS, 0), (Tokens.CLS, 3), trim_offsets=False, add_prefix_space=False)
        else:
            tokenizer.post_processor = BytePost(trim_offsets=False)
    elif args.tokenizer_type == "Unicode":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(unk_token=Tokens.UNK,special_tokens=special_token_map, vocab_size = true_vocab_size,
                                 initial_alphabet= ["#"] + StropheParams.METER_TYPES[:-1] + StropheParams.RHYME_SCHEMES[:-1], max_piece_length=1)
        
        tokenizer.pre_tokenizer = BytePre(add_prefix_space=False)
        tokenizer.decoder = ByteDec()
        if args.class_token:
            tokenizer.post_processor = RobertaProcessing((Tokens.EOS, 0), (Tokens.CLS, 3), trim_offsets=False, add_prefix_space=False)
        else:
            tokenizer.post_processor = BytePost(trim_offsets=False)
    
    elif args.tokenizer_type == "WordLevel":
        tokenizer = Tokenizer(WordLevel(unk_token=Tokens.UNK))
        trainer = WordLevelTrainer(special_tokens=special_token_map, vocab_size = true_vocab_size, min_frequency=2)
        
        tokenizer.normalizer = NFD()
        tokenizer.pre_tokenizer = Whitespace()
    elif args.tokenizer_type == "WordPiece":
        tokenizer = Tokenizer(WordPiece(unk_token=Tokens.UNK))
        trainer = WordPieceTrainer(special_tokens=special_token_map , vocab_size = true_vocab_size, min_frequency=2, 
                                   initial_alphabet= ["#"] + StropheParams.METER_TYPES[:-1] + StropheParams.RHYME_SCHEMES[:-1])
        tokenizer.normalizer = NFD()
        tokenizer.decoder = WordDec()
    else:
        raise ValueError("Unknown tokenize type")
    

    SEG_TYPE = 'BASE'
    if args.syllables:
        SEG_TYPE = "SYLLABLE"
    
    if args.tokenizer_type == 'VerseMarks':
        SEG_TYPE ="VERSEMARK"

    # Create or load data
    train_data = CorpusDatasetPytorch(SEGMENT_TYPE= SEG_TYPE ,data_dir=args.data_path, lower_case=args.lower_case)
    # Train on raw or processed data
    if args.raw_data:
        tokenizer.train_from_iterator(train_data.raw_dataset.get_poems(),trainer=trainer)
    else:
        # Train on syllable or normal processed data
        tokenizer.train_from_iterator([text['input_ids'] for text in train_data.train_strophes.data]  \
                                        + [text['input_ids'] for text in train_data.val_strophes.data] \
                                        + [text['input_ids'] for text in train_data.test_strophes.data]
                                        ,trainer=trainer)

        
    # Store tokenizer        
    if not os.path.exists(os.path.join(args.tokenizer_path ,args.tokenizer_type)):
        os.makedirs(os.path.join(args.tokenizer_path, args.tokenizer_type))
    if args.tokenizer_type not in ["Unicode", 'VerseMarks']:
        tokenizer.save(os.path.join(args.tokenizer_path, args.tokenizer_type, f"new_{'class_' if args.class_token else ''}{'syllabs_' if args.syllables else ''}{'raw' if args.raw_data else 'processed'}_tokenizer.json"))
    elif args.tokenizer_type == "Unicode":
        tokenizer.save(os.path.join(args.tokenizer_path, args.tokenizer_type, f"{'class_' if args.class_token else ''}unicode_tokenizer.json"))
    elif args.tokenizer_type == "VerseMarks":
        tokenizer.save(os.path.join(args.tokenizer_path, args.tokenizer_type, f"{'class_' if args.class_token else ''}versemarks_tokenizer.json"))
        
    # Simple tokenizer test With some basic needs for Strophe generation
    print(train_data.train_strophes.data[0]['input_ids'])
    print(tokenizer.encode(train_data.train_strophes.data[0]['input_ids']).ids)
    print(tokenizer.decode(tokenizer.encode(train_data.train_strophes.data[0]['input_ids']).ids))
    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

    