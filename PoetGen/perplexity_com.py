import argparse
import os
import torch
import evaluate
import random

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from utils.base_poet_models import PoetModelBase

from corpus_capsulated_datasets import CorpusDatasetPytorch


parser = argparse.ArgumentParser()


parser.add_argument("--model_path_full", default=os.path.abspath(os.path.join(os.path.dirname(__file__), 'backup_LMS', 'CZ-New-Syllable-BPE-NormalText-gpt-cz-poetry-all-e4e16_LM')),  type=str, help="Path to Model")
parser.add_argument("--sample", default=True, type=bool, help="If to sample during generation")
parser.add_argument("--data_path",  default=os.path.abspath(os.path.join(os.path.dirname(__file__), "corpusCzechVerse", "ccv")), type=str, help="Path to Data")
parser.add_argument("--reference_model", default='lchaloupsky/czech-gpt2-oscar', type=str, help="Model to measure perplexity")
#parser.add_argument("--reference_model", default='BUT-FIT/csmpt7b', type=str, help="Model to measure perplexity")
#parser.add_argument("--reference_model", default='simecek/cswikimistral_0.1', type=str, help="Model to measure perplexity")

parser.add_argument("--result_file", default= os.path.abspath(os.path.join(os.path.dirname(__file__),'results_new', "model_perplexity.txt")), type=str, help="Where to store the decoding efforts")

parser.add_argument("--num_runs", default=10_000, type=int, help="Number of runs per setting")

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    
_ ,model_rel_name =  os.path.split(args.model_path_full)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with torch.no_grad():
    model = PoetModelBase(args.model_path_full).to(device)
    model.eval()
    
    # Load LM tokenizers       
    tokenizer: PreTrainedTokenizerBase =  AutoTokenizer.from_pretrained(args.model_path_full)

    def remove_params(gen_poem:str):
        lines = gen_poem.splitlines()[1:]
        lines = list(filter(lambda x: x, lines))
        lines = [line.split('#')[-1].strip() for line in lines ]
        return "\n".join(lines)


    def decoder_helper(type, rhyme, year, meter):
        if type == "BASIC":
            start = f"# {rhyme} # {year}\n{meter}"
            tokenized = tokenizer.encode(start, return_tensors='pt', truncation=True)
            out = model.model.generate(tokenized.to(device), 
                                            max_length=512,
                                            do_sample=True,
                                            top_k=50,
                                            eos_token_id = tokenizer.eos_token_id,
                                            early_stopping=True,
                                            pad_token_id= tokenizer.pad_token_id)
            return tokenizer.decode(out.cpu()[0], skip_special_tokens=True)
        if type=="FORCED":
            start_forced = f"# {rhyme} # {year}\n{meter} #"
            return model.generate_forced(start_forced, tokenizer, verse_len=len(rhyme), sample=True, device=device)


    dataset = CorpusDatasetPytorch('BASE', data_dir=args.data_path)

    sampled_data = random.choices(dataset.test_strophes, k=args.num_runs)
    ground = list(filter(lambda x: x, [remove_params(datum['input_ids']) for datum in sampled_data])) 
    generated_basic =  list(filter(lambda x: x, [remove_params(decoder_helper("BASIC", datum['rhyme'], datum['year'], datum['metre_ids'][0])) for datum in sampled_data]))
    generated_forced =  list(filter(lambda x: x, [remove_params(decoder_helper("FORCED", datum['rhyme'], datum['year'], datum['metre_ids'][0])) for datum in sampled_data]))

    perplexity = evaluate.load("perplexity", module_type="metric")


    results_ground = perplexity.compute(model_id=args.reference_model,
                                 add_start_token=False,
                                 predictions=ground)

    results_basic = perplexity.compute(model_id=args.reference_model,
                                 add_start_token=False,
                                 predictions=generated_basic)

    results_forced = perplexity.compute(model_id=args.reference_model,
                                 add_start_token=False,
                                 predictions=generated_forced)

with open(args.result_file, 'a', encoding="utf-8") as file:
    print(f"==== MODEL: {model_rel_name} ==== REFERENCE MODEL: {args.reference_model} ==== SAMPLES: {args.num_runs} ====", file=file)
    print(f"GROUND PERPLEXITY: {results_ground['mean_perplexity']}, BASIC GEN PER: {results_basic['mean_perplexity']}, FORCED GEN PER: {results_forced['mean_perplexity']}\n", file=file)
