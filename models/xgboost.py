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


def run_xgbooster(
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