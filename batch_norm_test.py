#!/usr/bin/env python3

import os
import ssl
from argparse import ArgumentParser
from copy import copy
from functools import reduce
from operator import add
from time import time
from typing import Any, Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as kr
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.keras.utils import to_categorical


class ModelBuilder:
    def __init__(self, model):
        self.model = model

    def _add(self, layer):
        self.model.add(layer)

    def add(self, layer):
        self._add(layer)
        return self

    def add_multiple(self, layers):
        for layer in layers:
            self._add(layer)
        return self

    def build(self):
        return self.model


class BatchNormalizationLessModelBuilder(ModelBuilder):
    def _add(self, layer):
        if not isinstance(layer, BatchNormalization):
            super()._add(layer)


class MNISTEvaluator:
    @staticmethod
    def fetch_mnist_data():
        # Grab MNIST dataset
        mnist = kr.datasets.mnist

        # HACK: Disble verification of SSL certificate,
        # or an error prevent you from donwloading dataset
        ssl._create_default_https_context = ssl._create_unverified_context
        data = mnist.load_data()
        ssl._create_default_https_context = ssl._create_default_https_context
        return data

    @staticmethod
    def normalize_x(x: np.ndarray) -> np.ndarray:
        # Normalize to [0, 1] & Reshape to 4 dimensions
        return (x.astype(np.float32) / 255.).reshape(x.shape + (1,))

    @staticmethod
    def normalize_y(y: np.ndarray) -> np.ndarray:
        # Convert labels to one-hot
        return to_categorical(y, 10)

    @staticmethod
    def halve_array(array: np.ndarray) -> np.ndarray:
        return array[: array.shape[0] // 2]

    def __init__(self, optimizer: Optimizer, halve_data: bool) -> None:
        (x_train, y_train), (
            x_test,
            y_test,
        ) = MNISTEvaluator.fetch_mnist_data()

        print(f"Data: {x_train.shape[0]} (Training) / {x_test} (Test)")

        # Verify data
        self.per_data_dimension = x_train.shape[1:]
        if self.per_data_dimension != (28, 28):
            raise RuntimeError("Images must be 28x28.")

        # Preprocess data for NN

        self.x_train = MNISTEvaluator.normalize_x(
            MNISTEvaluator.halve_array(x_train) if halve_data else x_train
        )
        self.x_test = MNISTEvaluator.normalize_x(
            MNISTEvaluator.halve_array(x_test) if halve_data else x_test
        )

        self.y_train = MNISTEvaluator.normalize_y(
            MNISTEvaluator.halve_array(y_train) if halve_data else y_train
        )
        self.y_test = MNISTEvaluator.normalize_y(
            MNISTEvaluator.halve_array(y_test) if halve_data else y_test
        )

        self.optimizer = optimizer

    @staticmethod
    def plot_losses(history, title: str, out=None) -> None:
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        ax1.plot(history["loss"], label="Training")
        ax2.plot(history["acc"], label="Training")
        ax1.plot(history["val_loss"], label="Test")
        ax2.plot(history["val_acc"], label="Test")
        ax1.set_ylabel("Loss")
        ax2.set_ylabel("Accuracy")
        ax1.set_xlabel("epochs - 1")
        ax2.set_xlabel("epochs - 1")
        ax1.set_ylim((0., 0.15))
        ax2.set_ylim(((0.95, 1.)))
        fig.suptitle(title)
        ax1.legend()
        ax2.legend()
        plt.savefig(out, transparent=True)
        plt.close(fig)

    def train_and_evaluate(self, model_label, model) -> None:
        print(f"** {model_label} **")
        model.compile(
            loss="categorical_crossentropy",
            optimizer=self.optimizer,
            metrics=["accuracy"],
        )
        start_time = time()
        history = model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_test, self.y_test),
            epochs=6,
        )
        elapsed_seconds = time() - start_time
        print(f"Learning time: {elapsed_seconds:.03f} sec")

        loss, accuracy = model.evaluate(self.x_test, self.y_test)
        print(f"Loss: {loss} / Accuracy: {accuracy}")

        MNISTEvaluator.plot_losses(
            history.history,
            f"{model_label} (Time: {elapsed_seconds:.03f} s)",
            f"{model_label}.pdf",
        )

    def make_cnn_model(self, builder: ModelBuilder):
        builder.add(
            Conv2D(
                16,
                (3, 3),
                padding="same",
                input_shape=self.per_data_dimension + (1,),
            )
        )
        builder.add(BatchNormalization())
        builder.add(Activation("relu"))
        builder.add(Conv2D(8, (3, 3), padding="same"))
        builder.add(BatchNormalization())
        builder.add(Activation("relu"))
        builder.add(MaxPooling2D(2))
        builder.add(Conv2D(16, (3, 3), padding="same"))
        builder.add(BatchNormalization())
        builder.add(Activation("relu"))
        builder.add(Conv2D(16, (3, 3), padding="same"))
        builder.add(BatchNormalization())
        builder.add(Activation("relu"))
        builder.add(MaxPooling2D(2))
        builder.add(Flatten())
        builder.add(Dense(64))
        builder.add(BatchNormalization())
        builder.add(Activation("relu"))
        builder.add(Dense(10, activation="softmax"))
        return builder.build()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--halve-data", action="store_true", help="Halve data")
    args = parser.parse_args()

    if os.name != "nt" and "DISPLAY" not in os.environ:
        plt.switch_backend("Agg")

    evaluator = MNISTEvaluator(Adam(), args.halve_data)

    # Make CNN models
    model_with_bn = evaluator.make_cnn_model(ModelBuilder(Sequential()))
    model_without_bn = evaluator.make_cnn_model(
        BatchNormalizationLessModelBuilder(Sequential())
    )

    # Train & evaluate
    evaluator.train_and_evaluate("with Batch Normalization", model_with_bn)
    evaluator.train_and_evaluate(
        "without Batch Normalization", model_without_bn
    )
