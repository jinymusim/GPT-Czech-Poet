from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf

tokenizer = GPT2Tokenizer.from_pretrained("lchaloupsky/czech-gpt2-oscar")
model = TFGPT2LMHeadModel.from_pretrained("lchaloupsky/czech-gpt2-oscar")

# Get sequence length max of 1024
tokenizer.model_max_length=1024
# For older versions of the 'transformers' library use this
# tokenizer.max_len=1024

#model.eval()  # disable dropout (or leave in train mode to finetune)

text = "Polámal se mraveneček,\nví to celá obora,\no půlnoci zavolali\nmravenčího doktora.\n\nDoktor klepe na srdíčko,\npotom píše recepis,\ntřikrát denně prášek cukru,\nbude chlapík jako rys."
input_ids = tokenizer.encode(text, return_tensors="tf")

outputs = model.generate(input_ids, eos_token_id=50256, pad_token_id=50256,
                         do_sample=True, max_length=150, top_k=50)
print(tokenizer.decode(outputs[0]))

#Doesn't generate poem 