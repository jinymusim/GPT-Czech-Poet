import os
import argparse


from transformers import AutoTokenizer
from tokenizers import Tokenizer

from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer

from tokenizers.pre_tokenizers import ByteLevel as BytePre, Whitespace
from tokenizers.processors import ByteLevel as BytePost
from tokenizers.decoders import ByteLevel as ByteDec, WordPiece as WordDec

from tokenizers.normalizers import NFD

from corpus_capsulated_datasets import CorpusDatasetPytorch

parser = argparse.ArgumentParser()

parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")
# huggyllama/llama-7b 4096
# lchaloupsky/czech-gpt2-oscar 1024
# bigscience/bloom-560m 2048
# TheBloke/Llama-2-7B-fp16 4096
# spital/gpt2-small-czech-cs 1024
parser.add_argument("--default_tokenizer", default="lchaloupsky/czech-gpt2-oscar", type=str, help="Default Model from HF to use")
parser.add_argument("--tokenizer_type", default="WordLevel", type=str, choices=["BPE", "Unigram", "WordLevel", "WordPiece"], help="What type of tokenize to train")
parser.add_argument("--tokenizer_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__),"utils","tokenizers")),  type=str, help="Path to Model")


def main(args):
    tok = AutoTokenizer.from_pretrained(args.default_tokenizer)
    if args.tokenizer_type == "BPE":
        tokenizer = Tokenizer(BPE())
        trainer = BpeTrainer(special_tokens=tok.all_special_tokens, vocab_size = tok.vocab_size, min_frequency=2)
        
        tokenizer.pre_tokenizer = BytePre(add_prefix_space=False)
        tokenizer.decoder = ByteDec()
        tokenizer.post_processor = BytePost(trim_offsets=False)
    elif args.tokenizer_type == "Unigram":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(special_tokens=tok.all_special_tokens, vocab_size = tok.vocab_size)
        
        tokenizer.pre_tokenizer = BytePre(add_prefix_space=False)
        tokenizer.decoder = ByteDec()
        tokenizer.post_processor = BytePost(trim_offsets=False)
    elif args.tokenizer_type == "WordLevel":
        tokenizer = Tokenizer(WordLevel(unk_token=tok.all_special_tokens[0]))
        trainer = WordLevelTrainer(special_tokens=tok.all_special_tokens, vocab_size = tok.vocab_size, min_frequency=2)
        
        tokenizer.normalizer = NFD()
        tokenizer.pre_tokenizer = Whitespace()
    elif args.tokenizer_type == "WordPiece":
        tokenizer = Tokenizer(WordPiece(unk_token=tok.all_special_tokens[0]))
        trainer = WordPieceTrainer(special_tokens=tok.all_special_tokens , vocab_size = tok.vocab_size, min_frequency=2)
        tokenizer.normalizer = NFD()
        tokenizer.decoder = WordDec()
    else:
        raise ValueError("Unknown tokenize type")
    

    
    train_data = CorpusDatasetPytorch(tok, data_dir=args.data_path)
    #tokenizer.train_from_iterator(train_data.raw_dataset.get_text(),trainer=trainer)
    #tokenizer.train_from_iterator(train_data.raw_dataset.get_part(),trainer=trainer)
    tokenizer.train_from_iterator(train_data.raw_dataset.get_body(),trainer=trainer)

    if not os.path.exists(os.path.join(args.tokenizer_path ,args.tokenizer_type)):
        os.makedirs(os.path.join(args.tokenizer_path, args.tokenizer_type))
    tokenizer.save(os.path.join(args.tokenizer_path, args.tokenizer_type, "tokenizer.json"))
    
    
    print("Strc prist zkrz krk\n Hola hej")
    print(tokenizer.encode("Strc prist zkrz krk\n Hola hej").ids)
    print(tokenizer.decode(tokenizer.encode("Strc prist zkrz krk\n Hola hej").ids))
    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

    