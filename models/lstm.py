import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pandas as pd
import keras
import keras_tuner as kt
from sklearn.preprocessing import MinMaxScaler

from utils.metrics import evaluate  # ajuste para o nome real do seu módulo


def setup_device():
    """Informa onde o treino vai rodar (no backend torch, a GPU é usada automaticamente)."""
    backend = keras.backend.backend()
    if backend == "torch":
        import torch
        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (CUDA indisponível)"
    else:
        dev = backend
    print(f"Keras backend: {backend} | dispositivo: {dev}")


def _windows(x, y, janela, start, end):
    """Janelas para os alvos i em [start, end), olhando só os 'janela' dias anteriores a i."""
    X = np.stack([x[i - janela:i] for i in range(start, end)]).astype("float32")
    return X, y[start:end].astype("float32")


def run_lstm(
    train_dataset,
    test_dataset,
    stock_name,
    target="close",
    janela=20,
    val_frac=0.15,
    max_epochs=50,
    batch_size=64,
    seed=42,
    directory="lstm_tuning",
    verbose=True,
):
    col = f"{stock_name}_{target}"
    for nome, df in [("treino", train_dataset), ("teste", test_dataset)]:
        if col not in df.columns:
            raise KeyError(f"Coluna '{col}' não existe no dataset de {nome}.")

    keras.utils.set_random_seed(seed)
    setup_device()

    feature_cols = train_dataset.select_dtypes(include=[np.number]).columns.tolist()

    # 1. Split: validação = últimos val_frac do TREINO; teste = o mesmo dos outros modelos
    n_train = len(train_dataset)
    n_fit = int(n_train * (1 - val_frac))
    full = pd.concat([train_dataset, test_dataset])
    n_all = len(full)

    # 2. Normalização com fit só na parte de ajuste do treino
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    scaler_x.fit(full[feature_cols].iloc[:n_fit])
    scaler_y.fit(full[[col]].iloc[:n_fit])
    x_all = scaler_x.transform(full[feature_cols])
    y_all = scaler_y.transform(full[[col]])

    # 3. Janelas com contexto
    X_train, y_train = _windows(x_all, y_all, janela, janela, n_fit)
    X_val, y_val = _windows(x_all, y_all, janela, n_fit, n_train)
    X_test, _ = _windows(x_all, y_all, janela, n_train, n_all)

    if verbose:
        print(f"--- LSTM {stock_name.upper()}: treino {X_train.shape}, "
              f"val {X_val.shape}, teste {X_test.shape} ---")

    n_features = X_train.shape[2]

    # 4. Modelo parametrizado para o Keras Tuner
    def build_model(hp):
        model = keras.Sequential()
        model.add(keras.Input(shape=(janela, n_features)))
        n_layers = hp.Int("n_layers", 1, 3)
        for i in range(n_layers):
            units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=32)
            model.add(keras.layers.LSTM(units, return_sequences=i < n_layers - 1))
            model.add(keras.layers.Dropout(hp.Float(f"dropout_{i}", 0.0, 0.5, step=0.1)))
        model.add(keras.layers.Dense(1))
        lr = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                      loss="mse", metrics=["mae"])
        return model

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    tuner = kt.Hyperband(
        build_model,
        objective="val_loss",
        max_epochs=max_epochs,
        factor=3,
        seed=seed,
        directory=directory,
        project_name=f"{stock_name}_lstm",
    )
    tuner.search(X_train, y_train, validation_data=(X_val, y_val),
                 epochs=max_epochs, batch_size=batch_size,
                 callbacks=[early_stop], verbose=1 if verbose else 0)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    if verbose:
        print("Melhores hiperparâmetros:", best_hps.values)

    # 5. Treino final com os melhores hiperparâmetros
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=100, batch_size=batch_size,
                        callbacks=[early_stop], verbose=1 if verbose else 0)

    # 6. Previsão no teste, de volta à escala de preço
    pred_scaled = model.predict(X_test, batch_size=batch_size, verbose=0)
    preds = pd.Series(scaler_y.inverse_transform(pred_scaled).ravel(),
                      index=test_dataset.index)

    y_real = test_dataset[col].astype(float)
    ontem = full[col].astype(float).shift(1).iloc[n_train:]
    ontem.index = test_dataset.index

    resultado = evaluate(f"LSTM {stock_name}", y_real, ontem, preds)

    return {
        "stock": stock_name,
        "order": "LSTM",
        "model": model,
        "best_params": best_hps.values,
        "history": history.history,
        "predictions": preds,
        "real": y_real,
        "ontem": ontem,
        **resultado,
    }