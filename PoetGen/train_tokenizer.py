import os
import argparse
import time


from transformers import AutoTokenizer
from tokenizers import Tokenizer

from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordLevelTrainer, WordPieceTrainer

from tokenizers.pre_tokenizers import ByteLevel as BytePre, Whitespace
from tokenizers.processors import ByteLevel as BytePost
from tokenizers.decoders import ByteLevel as ByteDec, WordPiece as WordDec

from tokenizers.normalizers import NFD

from utils.poet_utils import METER_TYPES, RHYME_SCHEMES, EOS, PAD, UNK
from corpus_capsulated_datasets import CorpusDatasetPytorch

parser = argparse.ArgumentParser()

parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")
# huggyllama/llama-7b 4096
# lchaloupsky/czech-gpt2-oscar 1024
# bigscience/bloom-560m 2048
# TheBloke/Llama-2-7B-fp16 4096
# spital/gpt2-small-czech-cs 1024
parser.add_argument("--default_tokenizer", default="lchaloupsky/czech-gpt2-oscar", type=str, help="Default Model from HF to use")
parser.add_argument("--tokenizer_type", default="BPE", type=str, choices=["BPE", "Unigram", "WordLevel", "WordPiece"], help="What type of tokenize to train")
parser.add_argument("--tokenizer_path", default=os.path.abspath(os.path.join(os.path.dirname(__file__),"utils","tokenizers")),  type=str, help="Path to Model")
parser.add_argument("--raw_data", default=False,  type=bool, help="If to use raw data")
parser.add_argument("--syllables", default=False,  type=bool, help="If to use syllables")


def main(args):
    
    tok = AutoTokenizer.from_pretrained(args.default_tokenizer)
    if args.tokenizer_type == "BPE":
        tokenizer = Tokenizer(BPE())
        trainer = BpeTrainer(special_tokens=[EOS, PAD, UNK], vocab_size = tok.vocab_size, min_frequency=2,
                             initial_alphabet= ["#"] + METER_TYPES[:-1] + RHYME_SCHEMES[:-1])
        
        tokenizer.pre_tokenizer = BytePre(add_prefix_space=False)
        tokenizer.decoder = ByteDec()
        tokenizer.post_processor = BytePost(trim_offsets=False)
    elif args.tokenizer_type == "Unigram":
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(unk_token=UNK,special_tokens=[EOS, PAD, UNK], vocab_size = tok.vocab_size,
                                 initial_alphabet= ["#"] + METER_TYPES[:-1] + RHYME_SCHEMES[:-1])
        
        tokenizer.pre_tokenizer = BytePre(add_prefix_space=False)
        tokenizer.decoder = ByteDec()
        tokenizer.post_processor = BytePost(trim_offsets=False)
    elif args.tokenizer_type == "WordLevel":
        tokenizer = Tokenizer(WordLevel(unk_token=UNK))
        trainer = WordLevelTrainer(special_tokens=[EOS, PAD, UNK], vocab_size = tok.vocab_size, min_frequency=2)
        
        tokenizer.normalizer = NFD()
        tokenizer.pre_tokenizer = Whitespace()
    elif args.tokenizer_type == "WordPiece":
        tokenizer = Tokenizer(WordPiece(unk_token=UNK))
        trainer = WordPieceTrainer(special_tokens=[EOS, PAD, UNK] , vocab_size = tok.vocab_size, min_frequency=2, 
                                   initial_alphabet= ["#"] + METER_TYPES[:-1] + RHYME_SCHEMES[:-1])
        tokenizer.normalizer = NFD()
        tokenizer.decoder = WordDec()
    else:
        raise ValueError("Unknown tokenize type")
    

    
    train_data = CorpusDatasetPytorch(data_dir=args.data_path)
    if args.raw_data:
        tokenizer.train_from_iterator(train_data.raw_dataset.get_body(),trainer=trainer)
    else:
        if args.syllables:
            tokenizer.train_from_iterator([text['input_ids'][1] for text in train_data.pytorch_dataset_body.data], trainer=trainer)
        else:      
            tokenizer.train_from_iterator([text['input_ids'][0] for text in train_data.pytorch_dataset_body.data], trainer=trainer)
                
    if not os.path.exists(os.path.join(args.tokenizer_path ,args.tokenizer_type)):
        os.makedirs(os.path.join(args.tokenizer_path, args.tokenizer_type))
    tokenizer.save(os.path.join(args.tokenizer_path, args.tokenizer_type, f"{'syllabs_' if args.syllables else ''}{'raw' if args.raw_data else 'processed'}_tokenizer.json"))
    
    
    print("AABB # J # 1899\n strc prist # zkrz krk\n Hola hej")
    print(tokenizer.encode("AABB # J # 1899\n strc prist # zkrz krk\n Hola hej").ids)
    print(tokenizer.decode(tokenizer.encode("AABB # J # 1899\n strc prist # zkrz krk\n Hola hej").ids))
    
if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)

    