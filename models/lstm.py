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

def run_lstm_30(
    train_dataset,
    test_dataset,
    stock_name,
    params,
    target="close",
    janela=20,
    horizon=30,
    val_frac=0.15,
    max_epochs=100,
    batch_size=64,
    seed=42,
    verbose=True,
    origens=None,
):
    """
    Treina uma LSTM que prevê os próximos 'horizon' pregões de uma vez e devolve um dict com:
      "backtest": {data da origem: DataFrame(previsao, real, naive)}, previsões feitas a partir de
                  datas do TESTE, onde existe preço real para comparar
      "metricas": RMSE / MAE / MAPE de cada origem, junto com o RMSE do naive (último fechamento)
      "futuro":   previsão para depois do fim da base (não tem preço real para comparar)
      "model":    o modelo treinado

    params:  res_lstm["best_params"].
    origens: datas do último dia conhecido (ex.: ["2025-03-31", "2025-09-30"]). Cada uma precisa estar
             no teste e ter 'horizon' pregões reais depois dela. Se None, usa janelas consecutivas de
             'horizon' pregões cobrindo o teste inteiro.
    """
    col = f"{stock_name}_{target}"
    for nome, df in [("treino", train_dataset), ("teste", test_dataset)]:
        if col not in df.columns:
            raise KeyError(f"Coluna '{col}' não existe no dataset de {nome}. Colunas: {list(df.columns)}")
    keras.utils.set_random_seed(seed)

    feature_cols = train_dataset.select_dtypes(include=[np.number]).columns.tolist()
    n_train = len(train_dataset)
    n_fit = int(n_train * (1 - val_frac))
    full = pd.concat([train_dataset, test_dataset])
    n_all = len(full)
    datas = pd.to_datetime(full["date"]).reset_index(drop=True)
    precos = full[col].astype(float).to_numpy()

    # Mesma normalização do run_lstm (fit só na parte de ajuste do treino)
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    scaler_x.fit(full[feature_cols].iloc[:n_fit])
    scaler_y.fit(full[[col]].iloc[:n_fit])
    x_all = scaler_x.transform(full[feature_cols])
    y_all = scaler_y.transform(full[[col]])

    # Janelas: entrada = 'janela' dias anteriores à origem s; alvo = os 'horizon' dias a partir de s
    def windows(first, limit):
        origins = range(first, limit - horizon + 1)
        X = np.stack([x_all[s - janela:s] for s in origins]).astype("float32")
        Y = np.stack([y_all[s:s + horizon, 0] for s in origins]).astype("float32")
        return X, Y

    X_train, Y_train = windows(janela, n_fit)
    X_val, Y_val = windows(n_fit, n_train)

    # Mesma arquitetura do run_lstm, mas com saída de 'horizon' valores
    model = keras.Sequential([keras.Input(shape=(janela, X_train.shape[2]))])
    n_layers = params["n_layers"]
    for i in range(n_layers):
        model.add(keras.layers.LSTM(params[f"units_{i}"], return_sequences=i < n_layers - 1))
        model.add(keras.layers.Dropout(params[f"dropout_{i}"]))
    model.add(keras.layers.Dense(horizon))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]), loss="mse")

    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=max_epochs,
              batch_size=batch_size, callbacks=[early_stop], verbose=1 if verbose else 0)

    def prever(s):
        """Preços previstos (R$) para os 'horizon' dias a partir da posição s, vendo os 'janela' dias antes."""
        pred = model.predict(x_all[s - janela:s][None].astype("float32"), verbose=0).reshape(-1, 1)
        return scaler_y.inverse_transform(pred).ravel()

    # --- Backtest: origens dentro do teste, com preço real para comparar ---
    if origens is None:
        posicoes = list(range(n_train, n_all - horizon + 1, horizon))
    else:
        posicoes = [int(datas.searchsorted(pd.Timestamp(o), side="right")) for o in origens]

    backtest, linhas = {}, []
    for s in posicoes:
        if s < n_train or s + horizon > n_all:
            raise ValueError(
                f"Origem inválida. O último dia conhecido deve ficar entre "
                f"{datas[n_train - 1]:%d/%m/%Y} e {datas[n_all - horizon - 1]:%d/%m/%Y}."
            )
        df = pd.DataFrame({"previsao": prever(s), "real": precos[s:s + horizon], "naive": precos[s - 1]},
                          index=pd.DatetimeIndex(datas.iloc[s:s + horizon]))
        erro = df["real"] - df["previsao"]
        linhas.append({
            "origem": datas[s - 1],
            "rmse": np.sqrt((erro ** 2).mean()),
            "rmse_naive": np.sqrt(((df["real"] - df["naive"]) ** 2).mean()),
            "mae": erro.abs().mean(),
            "mape": (erro.abs() / df["real"]).mean(),
        })
        backtest[datas[s - 1]] = df
    metricas = pd.DataFrame(linhas)

    if verbose:
        print(f"\nBacktest em {len(metricas)} origens (média):")
        print(metricas[["rmse", "rmse_naive", "mae", "mape"]].mean().round(4).to_string())

    # --- Futuro: depois do fim da base (sem preço real) ---
    datas_futuras = pd.bdate_range(datas.iloc[-1] + pd.Timedelta(days=1), periods=horizon)
    futuro = pd.DataFrame({"previsao": prever(n_all), "real": np.nan}, index=datas_futuras)

    return {"backtest": backtest, "metricas": metricas, "futuro": futuro, "model": model}


def plot_previsao(res, train_dataset, test_dataset, stock_name, target="close",
                  origem=None, zoom=60, titulo=None, save_path=None):
    """
    Plota uma janela de previsão do run_lstm_30.
      origem=None            -> previsão futura (depois do fim da base, sem preço real)
      origem="2025-03-31"    -> backtest a partir da origem mais próxima dessa data, com o preço real
    Precisa de: import matplotlib.pyplot as plt (no topo do lstm.py).
    """
    full = pd.concat([train_dataset, test_dataset])
    datas = pd.to_datetime(full["date"]).reset_index(drop=True)
    precos = full[f"{stock_name}_{target}"].astype(float).reset_index(drop=True)

    if origem is None:
        df, ref = res["futuro"], datas.iloc[-1]
    else:
        ref = min(res["backtest"], key=lambda d: abs(d - pd.Timestamp(origem)))
        df = res["backtest"][ref]
    fim = int(datas.searchsorted(ref, side="right"))     # 1ª posição depois da origem
    ultimo = precos.iloc[fim - 1]
    n = len(df)
    tem_real = df["real"].notna().any()

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.plot(datas.iloc[max(0, fim - zoom):fim], precos.iloc[max(0, fim - zoom):fim],
            color="#1f77b4", linewidth=1.8, label="Histórico (conhecido)")
    if tem_real:
        ax.plot([ref, *df.index], [ultimo, *df["real"]], color="#2ca02c", linewidth=2,
                label=f"Real (próximos {n} dias)")
    ax.plot([ref, *df.index], [ultimo, *df["previsao"]], color="#ff7f0e", linestyle="--",
            linewidth=1.8, label=f"Previsão LSTM ({n} dias)")
    ax.axhline(ultimo, color="#7f7f7f", linestyle=":", linewidth=1.3,
               label="Baseline Naïve (último fechamento)")
    ax.axvline(ref, color="k", linestyle=":", alpha=0.4)

    if titulo is None:
        if tem_real:
            rmse = np.sqrt(((df["real"] - df["previsao"]) ** 2).mean())
            rmse_naive = np.sqrt(((df["real"] - ultimo) ** 2).mean())
            titulo = (f"Backtest de {n} dias a partir de {ref:%d/%m/%Y} | "
                      f"RMSE LSTM {rmse:.2f} vs. Naïve {rmse_naive:.2f}")
        else:
            titulo = f"Previsão futura de {n} dias — {stock_name.upper()}"
    ax.set_title(titulo, fontsize=14, fontweight="bold")
    ax.set_xlabel("Data", fontsize=12)
    ax.set_ylabel("Preço de Fechamento (R$)", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.setp(ax.get_xticklabels(), rotation=30)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig