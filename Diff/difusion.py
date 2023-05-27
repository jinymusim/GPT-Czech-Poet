from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer
from corpus_dataset_torch import CorpusDatasetPytorch
import torch

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

tokenizer = AutoTokenizer.from_pretrained("lchaloupsky/czech-gpt2-oscar")
train_data = CorpusDatasetPytorch(tokenizer)

for datum in train_data.dataset.raw_data_gen:
    for data_line in datum:
        for part_line in data_line['body']:
            for text_line in part_line:
                prompt = text_line['text']
                image = pipe(prompt).images[0]  
    
                image.save(prompt + ".png")
