import os
import re
from typing import Dict
import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


import tensorflow as tf
from mnist import MNIST
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs to run.")
parser.add_argument("--seed", default=99, type=int, help="Random seed")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--num_filters", default=32, type=int, help="Number of filters in convolution layer")
parser.add_argument("--dense_layer", default=128, type=int, help="Size of the dense layer")

class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:

        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        convolution = tf.keras.layers.Conv2D(args.num_filters, 3, padding='valid', use_bias=False)(inputs)
        batch_norm = tf.keras.layers.BatchNormalization()(convolution)
        relu = tf.keras.layers.ReLU()(batch_norm)

        flatten = tf.keras.layers.Flatten()(relu)
        dense = tf.keras.layers.Dense(args.dense_layer, activation="relu")(flatten)

        outputs = tf.keras.layers.Dense(MNIST.LABELS, activation=tf.nn.softmax)(dense)

        super().__init__(inputs=inputs, outputs=outputs)


        self.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")]
        )
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, histogram_freq=1)

def main(args: argparse.Namespace) -> Dict[str, float]:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%Sd_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the model and train it
    model = Model(args)

    train_logs = model.fit(
        mnist.train.data["x"], mnist.train.data["y"],
        batch_size=args.batch_size, epochs=args.epochs,
        callbacks=[model.tb_callback]
    )
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(
        mnist.test.data["x"], mnist.test.data["y"],
        batch_size=args.batch_size,
        callbacks=[model.tb_callback]    
    )

    # Return Logs on Test Data
    return {metric: values[-1] for metric, values in train_logs.history.items()}, {'test_accuracy': test_accuracy, 'test_loss': test_loss}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
