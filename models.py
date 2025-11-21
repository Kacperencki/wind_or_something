# models.py

import numpy as np
from typing import Tuple

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks


def evaluate_regression(y_true, y_pred) -> Tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


# ---------- Baseline (persistence) ----------

def persistence_baseline(X_test, y_test):
    """
    Assume last feature is the target itself (if you include it),
    or adapt to use y_ts directly.
    Here: use last step's target approximated by last feature.
    """
    y_pred = X_test[:, -1, -1]
    return evaluate_regression(y_test, y_pred)


# ---------- Ridge regression on flattened window ----------

def train_ridge(X_train, y_train, X_val, y_val, alpha=1.0):
    s_train, w, f = X_train.shape
    s_val = X_val.shape[0]

    Xtr = X_train.reshape(s_train, w * f)
    Xval = X_val.reshape(s_val, w * f)

    model = Ridge(alpha=alpha)
    model.fit(Xtr, y_train)

    y_val_pred = model.predict(Xval)
    val_mae, val_rmse = evaluate_regression(y_val, y_val_pred)
    return model, val_mae, val_rmse


def test_ridge(model, X_test, y_test):
    s_test, w, f = X_test.shape
    Xte = X_test.reshape(s_test, w * f)
    y_pred = model.predict(Xte)
    return evaluate_regression(y_test, y_pred)


# ---------- MLP on flattened window ----------

def train_mlp(X_train, y_train, X_val, y_val,
              hidden=(128, 64), max_iter=300):
    s_train, w, f = X_train.shape
    s_val = X_val.shape[0]

    Xtr = X_train.reshape(s_train, w * f)
    Xval = X_val.reshape(s_val, w * f)

    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=10
    )
    model.fit(Xtr, y_train)

    y_val_pred = model.predict(Xval)
    val_mae, val_rmse = evaluate_regression(y_val, y_val_pred)
    return model, val_mae, val_rmse


def test_mlp(model, X_test, y_test):
    s_test, w, f = X_test.shape
    Xte = X_test.reshape(s_test, w * f)
    y_pred = model.predict(Xte)
    return evaluate_regression(y_test, y_pred)


# ---------- LSTM (improved) ----------

def build_lstm(input_shape: Tuple[int, int]):
    """
    input_shape = (window, n_features)
    """
    inp = layers.Input(shape=input_shape)

    # Slightly larger, regularized LSTM stack
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32, return_sequences=False)(x)

    # MLP head
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)

    model = models.Model(inp, out)
    model.compile(
        loss="mse",
        optimizer=optimizers.Adam(learning_rate=1e-4),
        metrics=["mae"],
    )
    return model


def train_lstm(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs: int = 100,
    batch_size: int = 32,
):
    window = X_train.shape[1]
    n_features = X_train.shape[2]

    model = build_lstm((window, n_features))

    early = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,           # let it go a bit longer than before
        min_delta=1e-3,
        restore_best_weights=True,
    )

    lr_sched = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=6,
        min_lr=1e-6,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early, lr_sched],
        verbose=1,
        shuffle=False,         # keep temporal order
    )
    return model, history


def test_lstm(model, X_test, y_test):
    y_pred = model.predict(X_test, verbose=0).ravel()
    return evaluate_regression(y_test, y_pred)

# ---------- 1D CNN ----------

def build_cnn(window, n_features):
    inp = layers.Input(shape=(window, n_features))
    x = layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")(inp)
    x = layers.Conv1D(32, kernel_size=3, padding="causal", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)

    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def train_cnn(X_train, y_train, X_val, y_val,
              epochs: int = 50, batch_size: int = 64):
    window = X_train.shape[1]
    n_features = X_train.shape[2]

    model = build_cnn(window, n_features)

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early],
        verbose=1
    )
    return model, history


def test_cnn(model, X_test, y_test):
    y_pred = model.predict(X_test, verbose=0).ravel()
    return evaluate_regression(y_test, y_pred)
