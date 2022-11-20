import os
import sys
import urllib.request
from typing import Iterator, Optional, Dict

import tensorflow as tf
import numpy as np

class MNIST:
    H: int=28
    W: int=28
    C: int=1
    LABELS: int=10

    _URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets"

    class Dataset:
        def __init__(self, data: dict[str, np.ndarray], shuffle_batches: bool, seed:int = 99) -> None:
            self._data = data
            self._data["x"] = self._data["x"].astype(np.float32) /255
            self._size = len(self._data["x"])

        @property
        def data(self) -> dict[str, np.ndarray]:
            return self._data

        @property
        def size(self) -> int:
            return self._size

        def batches(self, size:Optional[int] = None) -> Iterator[dict[str, np.ndarray]]:
            permutation = self._shuffler.permutation(self._size) if self._shuffler else np.arange(self._size)
            while len(permutation):
                batch_size = min(size or np.inf, len(permutation))
                batch_perm = permutation[:batch_size]
                permutation = permutation[batch_size:]

                batch = {}
                for key in self._data:
                    batch[key] = self._data[key][batch_perm]
                yield batch



        @property
        def dataset(self) -> tf.data.Dataset:
            return tf.data.Dataset.from_tensor_slices(self._data)

    def __init__(self, dataset:str = "mnist", size: Dict[str, int] = {}) -> None:
        path = "{}.npz".format(dataset)
        if not os.path.exists(path):
            print("Downloading Dataset... {}".format(dataset))
            urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename=path)

        mnist = np.load(path)
        for dataset in ["train", "test"]:
            data = dict((key[:-(len(dataset) + 1)], mnist[key][:size.get(dataset, None)]) for key in mnist if key.endswith(dataset))
            setattr(self, dataset, self.Dataset(data, shuffle_batches=dataset == "train"))
