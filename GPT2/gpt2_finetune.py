from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
from corpus_dataset import CorpusDataLoad
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs to run.")
parser.add_argument("--lr", default=1e-5, type=float, help="Learning Rate for Finetuning")
parser.add_argument("--seed", default=99, type=int, help="Random seed")
parser.add_argument("--threads", default=2, type=int, help="Maximum number of threads to use.")
parser.add_argument("--data_path",  default="GPT2/corpusCzechVerse-master/ccv", type=str, help="Path to Data")
parser.add_argument("--model_path", default="./gpt2-cz-poetry",  type=str, help="Path to Model")
parser.add_argument("--use_default_model",  default=True, type=bool, help="Use Default Model")
parser.add_argument("--default_hf_model", default="lchaloupsky/czech-gpt2-oscar", type=str, help="Default Model from HF to use")


def main(args: argparse.Namespace):
    
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    
    if args.use_default_model:
        tokenizer = GPT2Tokenizer.from_pretrained(args.default_hf_model)
        model = TFGPT2LMHeadModel.from_pretrained(args.default_hf_model)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
        model = TFGPT2LMHeadModel.from_pretrained(args.model_path)
        
    train_data = CorpusDataLoad(tokenizer, data_dir=args.data_path)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE),
        metrics=['accuracy']
    )
    
    tf_dataset_text = train_data.dataset.dataset_text.shuffle(128).padded_batch(args.batch_size, padded_shapes=([None],[None]))

    model.fit(tf_dataset_text, epochs=args.epochs)

    tf_dataset_body = train_data.dataset.dataset_body.shuffle(128).padded_batch(args.batch_size, padded_shapes=([None],[None]))

    model.fit(tf_dataset_body, epochs=args.epochs)

    model.save_model(args.model_path)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)