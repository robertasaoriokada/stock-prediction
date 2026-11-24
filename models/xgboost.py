import pandas as pd
import xgboost as xgb
from scipy.stats import binomtest
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from utils.library import np
from utils.metrics import evaluate  # ajuste para o nome real do seu módulo

DEFAULT_PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
}


def _make_lags(df, columns, n_lags):
    """Cria <col>_lag_k para cada coluna e k = 1..n_lags (de uma vez, sem fragmentar o DataFrame)."""
    lagged = {f"{c}_lag_{k}": df[c].shift(k) for c in columns for k in range(1, n_lags + 1)}
    return pd.concat([df, pd.DataFrame(lagged, index=df.index)], axis=1)


def run_xgboost(
    train_dataset,
    test_dataset,
    stock_name,
    target="close",
    n_lags=3,
    param_grid=None,
    n_splits=5,
    verbose=True,
):
    """
    XGBoost com lags, tuning temporal e avaliação one-step-ahead, para qualquer ação.

    stock_name: prefixo do load_stock (ex.: "vale" -> "vale_close", "vale_open").
    Só o <prefixo>_open entra sem lag (é conhecido antes do fechamento);
    todas as outras colunas entram defasadas.
    """
    col = f"{stock_name}_{target}"
    open_col = f"{stock_name}_open"
    for nome, df in [("treino", train_dataset), ("teste", test_dataset)]:
        if col not in df.columns:
            raise KeyError(f"Coluna '{col}' não existe no dataset de {nome}. Colunas: {list(df.columns)}")

    n_train = len(train_dataset)
    test_index = test_dataset.index
    full = pd.concat([train_dataset, test_dataset]).reset_index(drop=True)

    # 1. Features: open sem lag; todas as outras colunas numéricas (inclusive o alvo) com lag
    safe = [open_col] if open_col in full.columns else []
    to_lag = [c for c in full.select_dtypes("number").columns if c not in safe]
    full = _make_lags(full, to_lag, n_lags)
    lag_cols = [c for c in full.columns if "_lag_" in c]
    features = safe + lag_cols

    # 2. Separa de novo nos mesmos pontos de corte (teste idêntico ao do ARIMA)
    train_df = full.iloc[:n_train].dropna(subset=lag_cols)
    test_df = full.iloc[n_train:].copy()
    test_df.index = test_index

    X_train, y_train = train_df[features], train_df[col]
    X_test, y_test = test_df[features], test_df[col]

    if verbose:
        print(f"--- XGBoost {stock_name.upper()}: {len(features)} features, "
              f"{len(X_train)} treino / {len(X_test)} teste ---")

    # 3. Grid search com validação temporal (sem embaralhar)
    grid = GridSearchCV(
        estimator=xgb.XGBRegressor(random_state=42),
        param_grid=param_grid or DEFAULT_PARAM_GRID,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1 if verbose else 0,
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    if verbose:
        print(f"Melhores hiperparâmetros: {grid.best_params_}")
        print(f"Melhor RMSE médio (validação): {-grid.best_score_:.4f}")

    # 4. Previsão no teste
    preds = pd.Series(model.predict(X_test), index=test_index)
    ontem = test_df[f"{col}_lag_1"]

    # 5. Métricas (mesmas dos outros modelos) + baseline naive
    resultado = evaluate(f"XGBoost {stock_name}", y_test, ontem, preds)
    resultado_naive = evaluate(f"Naive {stock_name}", y_test, ontem, ontem)

    # 6. Teste binomial da acurácia acerto direcional (H0: 50%)
    acertos = np.sign(y_test.values - ontem.values) == np.sign(preds.values - ontem.values)
    p_valor = binomtest(int(acertos.sum()), len(acertos), p=0.5).pvalue
    if verbose:
        print(f"p-valor (teste binomial, acurácia direcional): {p_valor:.4f}")

    return {
        "stock": stock_name,
        "order": "XGB",
        "model": model,
        "best_params": grid.best_params_,
        "features": features,
        "X_test": X_test,            # usado depois pelo módulo de SHAP
        "predictions": preds,
        "real": y_test,
        "ontem": ontem,
        "dir_pvalue": p_valor,
        "naive_rmse": resultado_naive["rmse"],
        "naive_mape": resultado_naive["mape"],
        **resultado,                 # rmse, mae, mape, dir_acc
    }



def run_xgboost_30(
    train_dataset,
    test_dataset,
    stock_name,
    target="close",
    horizon=30,
    n_lags=3,
    param_grid=None,
    n_splits=5,
    verbose=True,
):
    """
    XGBoost multi-horizonte (estratégia direta), no mesmo formato do run_arima_30.
    - Um modelo por passo k = 1..horizon, cada um prevendo (preço[t+k] - preço[t]).
    - Hiperparâmetros ajustados uma vez, no passo final (k = horizon), e reaproveitados.
    - Features: colunas numéricas no dia t + lags 1..n_lags. Nada do futuro entra.
    - 'backtest': {data_da_origem: DataFrame(real, previsao)} para cada origem do teste
    - 'futuro':   DataFrame(real=NaN, previsao) depois do fim da base
    """
    col = f"{stock_name}_{target}"
    for nome, df in [("treino", train_dataset), ("teste", test_dataset)]:
        if col not in df.columns:
            raise KeyError(f"Coluna '{col}' não existe no dataset de {nome}. Colunas: {list(df.columns)}")

    n_train, n_test = len(train_dataset), len(test_dataset)
    if n_test < horizon:
        raise ValueError(f"Teste tem {n_test} pontos, menor que o horizonte {horizon}.")

    raw = pd.concat([train_dataset, test_dataset])
    datas = (pd.to_datetime(raw["date"]).reset_index(drop=True)
             if "date" in raw.columns
             else pd.Series(pd.to_datetime(raw.index)))
    full = raw.reset_index(drop=True)

    # 1. Features: todas as colunas numéricas em t + lags 1..n_lags
    num_cols = list(full.select_dtypes("number").columns)
    full = _make_lags(full, num_cols, n_lags)
    lag_cols = [c for c in full.columns if "_lag_" in c]
    features = num_cols + lag_cols
    price = full[col].astype(float)

    # 2. Alvo do passo k: variação do preço k dias à frente, calculada SÓ dentro do treino
    train_price = price.iloc[:n_train]
    train_ok = full.iloc[:n_train][lag_cols].notna().all(axis=1)

    def _train_xy(k):
        y_k = train_price.shift(-k) - train_price        # NaN nos últimos k dias do treino
        m = train_ok & y_k.notna()
        return full.iloc[:n_train].loc[m, features], y_k[m]

    # 3. Tuning no passo final, com 'gap' para os alvos de 30 dias não vazarem entre folds
    X_h, y_h = _train_xy(horizon)
    if verbose:
        print(f"--- XGBoost {stock_name.upper()} h={horizon}: {len(features)} features, "
              f"{len(X_h)} amostras de treino ---")
    grid = GridSearchCV(
        estimator=xgb.XGBRegressor(random_state=42),
        param_grid=param_grid or DEFAULT_PARAM_GRID,
        cv=TimeSeriesSplit(n_splits=n_splits, gap=horizon),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1 if verbose else 0,
    )
    grid.fit(X_h, y_h)
    if verbose:
        print(f"Melhores hiperparâmetros: {grid.best_params_}")
        print(f"Melhor RMSE médio (validação): {-grid.best_score_:.4f}")

    # 4. Um modelo por passo, com os mesmos hiperparâmetros
    models = {}
    for k in range(1, horizon + 1):
        if k == horizon:
            models[k] = grid.best_estimator_
        else:
            Xk, yk = _train_xy(k)
            models[k] = xgb.XGBRegressor(**grid.best_params_, random_state=42).fit(Xk, yk)

    # 5. Backtest: origem = último dia conhecido (a 1ª é o último dia do treino)
    origins = np.arange(n_train - 1, len(full) - horizon)
    X_o = full.loc[origins, features]
    X_o.index = datas.iloc[origins].values
    last = price.values[origins]
    P = np.column_stack([models[k].predict(X_o) for k in range(1, horizon + 1)]) + last[:, None]

    backtest = {}
    for j, pos in enumerate(origins):
        idx = pd.DatetimeIndex(datas.iloc[pos + 1: pos + 1 + horizon])
        real = price.iloc[pos + 1: pos + 1 + horizon].values
        backtest[datas.iloc[pos]] = pd.DataFrame({"real": real, "previsao": P[j]}, index=idx)

    # 6. Futuro: depois do fim da base, sem preço real (dias úteis; feriados ignorados)
    X_f = full.iloc[[-1]][features]
    fut = np.array([models[k].predict(X_f)[0] for k in range(1, horizon + 1)]) + price.iloc[-1]
    idx_fut = pd.bdate_range(datas.iloc[-1] + pd.Timedelta(days=1), periods=horizon)
    futuro = pd.DataFrame({"real": np.nan, "previsao": fut}, index=idx_fut)

    # 7. Métricas agregadas em t+h, em todas as origens, contra o Naïve (último preço)
    idx_alvo = datas.iloc[origins + horizon].values
    preds = pd.Series(P[:, -1], index=idx_alvo)
    reals = pd.Series(price.values[origins + horizon], index=idx_alvo)
    base = pd.Series(last, index=idx_alvo)

    resultado = evaluate(f"XGBoost {stock_name} h={horizon}", reals, base, preds)
    resultado_naive = evaluate(f"Naive {stock_name} h={horizon}", reals, base, base)

    acertos = np.sign(reals.values - base.values) == np.sign(preds.values - base.values)
    p_valor = binomtest(int(acertos.sum()), len(acertos), p=0.5).pvalue
    if verbose:
        print(f"p-valor (teste binomial, acurácia direcional): {p_valor:.4f}")

    return {
        "stock": stock_name, "order": "XGB", "horizon": horizon,
        "model": models[horizon], "models": models,
        "best_params": grid.best_params_, "features": features,
        "X_test": X_o,                      # features nas origens, para o SHAP
        "backtest": backtest, "futuro": futuro,
        "predictions": preds, "real": reals, "baseline": base,
        "dir_pvalue": p_valor,
        "naive_rmse": resultado_naive["rmse"], "naive_mape": resultado_naive["mape"],
        **resultado,
    }