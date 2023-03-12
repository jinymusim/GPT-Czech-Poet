from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from corpus_dataset import CorpusDataLoad
import tensorflow as tf

MODEL_PATH = "./gpt2-cz-poetry" 

tokenizer = GPT2Tokenizer.from_pretrained("lchaloupsky/czech-gpt2-oscar")
tokenizer.model_max_length=1024
model = TFGPT2LMHeadModel.from_pretrained("lchaloupsky/czech-gpt2-oscar")

train_data = CorpusDataLoad(tokenizer)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE),
    metrics=['accuracy']
)
tf_dataset_text = train_data.dataset.dataset_text.shuffle(128).padded_batch(4, padded_shapes=([None],[None]))

model.fit(tf_dataset_text, epochs=1)

tf_dataset_body = train_data.dataset.dataset_body.shuffle(128).padded_batch(4, padded_shapes=([None],[None]))

model.fit(tf_dataset_body, epochs=1)

model.save_model(MODEL_PATH)