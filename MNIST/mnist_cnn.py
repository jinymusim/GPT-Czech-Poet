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

class Model(tf.keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:

        inputs = tf.keras.layers.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        convolution = tf.keras.layers.Conv2D(32, 3, padding='valid', use_bias=False)(input)
        batch_norm = tf.keras.layers.BatchNormalization()(convolution)
        relu = tf.keras.layers.ReLU()(batch_norm)

        flatten = tf.keras.layers.Flatten()(relu)
        dense = tf.keras.layers.Dense(200, activation="relu")(flatten)

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
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data
    mnist = MNIST()

    # Create the model and train it
    model = Model(args)

    logs = model.fit(
        mnist.train.data["images"], mnist.train.data["labels"],
        batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
        callbacks=[model.tb_callback],
    )

    # Return development metrics for ReCodEx to validate
    return {metric: values[-1] for metric, values in logs.history.items() if metric.startswith("val_")}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
