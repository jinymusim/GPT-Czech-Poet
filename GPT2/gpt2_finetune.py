from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from corpus_dataset import CorpusDataLoad
import tensorflow as tf

tokenizer = GPT2Tokenizer.from_pretrained("lchaloupsky/czech-gpt2-oscar")
model = TFGPT2LMHeadModel.from_pretrained("lchaloupsky/czech-gpt2-oscar")

train_data = CorpusDataLoad(tokenizer)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE),
    metrics=['accuracy']
)

model.fit(train_data.dataset.dataset_body, epochs=1)