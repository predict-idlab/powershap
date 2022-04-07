__author__ = "Jeroen Van Der Donckt"

from powershap import PowerShap
from .conftest import dummy_classification, dummy_regression

import tensorflow as tf


def test_deep_learning_class_powershap(dummy_classification):
    X, y = dummy_classification
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(5, input_shape=(X.shape[1] + 1,), activation="relu")
    )
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    selector = PowerShap(
        model=model,
        power_iterations=5,
        automatic=False,
    )

    selector.fit(
        X, y, loss="binary_crossentropy", optimizer="adam", batch_size=16, epochs=5
    )
    _ = selector.transform(X)


def test_deep_learning_regr_powershap(dummy_regression):
    X, y = dummy_regression
    n_informative = sum([c.startswith("informative") for c in X.columns])
    assert n_informative > 0, "No informative columns in the dummy data!"

    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(5, input_shape=(X.shape[1] + 1,), activation="linear")
    )
    model.add(tf.keras.layers.Dense(1, activation="linear"))

    selector = PowerShap(
        model=model,
        power_iterations=5,
        automatic=False,
    )

    selector.fit(X, y, loss="mse", optimizer="adam", batch_size=16, epochs=5)
    _ = selector.transform(X)
