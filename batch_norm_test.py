#!/usr/bin/env python3

import os
import ssl
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
from tensorflow.keras.optimizers import Adam
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

    def __init__(self, optimizer):
        (x_train, y_train), (
            x_test,
            y_test,
        ) = MNISTEvaluator.fetch_mnist_data()

        # Verify data
        self.per_data_dimension = x_train.shape[1:]
        if self.per_data_dimension != (28, 28):
            raise RuntimeError("Images must be 28x28.")

        # Preprocess data for NN

        # Normalize to [0, 1] & Reshape to 4 dimensions
        self.x_train = (x_train.astype(np.float32) / 255.).reshape(
            x_train.shape + (1,)
        )
        self.x_test = (x_test.astype(np.float32) / 255.).reshape(
            (x_test.shape + (1,))
        )

        # Convert labels to one-hot
        self.y_train = to_categorical(y_train, 10)
        self.y_test = to_categorical(y_test, 10)

        self.optimizer = optimizer

    @staticmethod
    def plot_losses(history, title, out=None):
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


    def train_and_evaluate(self, model_label, model):

        print(f"** {model_label} **")
        model.compile(
            loss="categorical_crossentropy",
            optimizer=self.optimizer,
            metrics=["accuracy"],
        )
        start_time = time()
        history = model.fit(
            self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=6
        )
        elapsed_seconds = time() - start_time
        print("Learning time: {:.03f} sec".format(elapsed_seconds))

        loss, accuracy = model.evaluate(self.x_test, self.y_test)
        print("Loss: {} / Accuracy: {}".format(loss, accuracy))

        MNISTEvaluator.plot_losses(history.history, model_label + "(Time: {:.03f} s)".format(elapsed_seconds), model_label + ".pdf")

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

if os.name != "nt" and "DISPLAY" not in os.environ:
    plt.switch_backend("Agg")

evaluator = MNISTEvaluator(Adam())

# Make CNN models
model_with_bn = evaluator.make_cnn_model(ModelBuilder(Sequential()))
model_without_bn = evaluator.make_cnn_model(
    BatchNormalizationLessModelBuilder(Sequential())
)

# Train & evaluate
evaluator.train_and_evaluate("with Batch Normalization", model_with_bn)
evaluator.train_and_evaluate("without Batch Normalization", model_without_bn)
