import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from utils.metrics import evaluate  # ajuste para o nome real do seu módulo


def run_arima(train_dataset, test_dataset, stock_name, target="close", verbose=True):
    """
    ARIMA básico com avaliação one-step-ahead, para qualquer ação.

    stock_name: prefixo usado no load_stock (ex.: "vale" -> coluna "vale_close").
    target: sufixo da coluna alvo (padrão "close").
    """
    col = f"{stock_name}_{target}"
    for nome, df in [("treino", train_dataset), ("teste", test_dataset)]:
        if col not in df.columns:
            raise KeyError(f"Coluna '{col}' não existe no dataset de {nome}. Colunas: {list(df.columns)}")

    y_train = train_dataset[col].astype(float)
    y_test = test_dataset[col].astype(float)

    # 1. auto_arima escolhe (p, d, q) usando só o TREINO
    if verbose:
        print(f"--- Identificando ordem do ARIMA para {stock_name.upper()} ---")
    modelo_auto = auto_arima(
        y_train,
        seasonal=False,
        trace=verbose,
        suppress_warnings=True,
        error_action="ignore",
    )
    ordem = modelo_auto.order
    if verbose:
        print(f"Ordem ARIMA selecionada: {ordem}")

    # 2. Treina no TREINO
    trained_model = ARIMA(y_train, order=ordem).fit()

    # 3. One-step-ahead no TESTE: parâmetros do treino, histórico real.
    # Aplica na série completa (treino + teste) e pega só o trecho do teste,
    # para a primeira previsão do teste já usar o histórico do treino.
    y_full = pd.concat([y_train, y_test])
    applied = trained_model.apply(y_full)
    preds = applied.fittedvalues.iloc[len(y_train):]
    preds.index = y_test.index

    # 4. "yesterday" = preço real do dia anterior, alinhado ao teste
    #    (o primeiro dia do teste usa o último preço do treino)
    yesterday = y_full.shift(1).iloc[len(y_train):]
    yesterday.index = y_test.index

    # 5. Mesmas métricas dos outros modelos (RMSE, MAPE, acurácia direcional)
    result = evaluate(f"ARIMA{ordem} {stock_name}", y_test, yesterday, preds)

    return {
        "stock": stock_name,
        "order": ordem,
        "model": trained_model,
        "predictions": preds,
        "real": y_test,
        "yesterday": yesterday,
        **result,  # rmse, mape, dir_acc
    }


def run_arima_30(train_dataset, test_dataset, stock_name, target="close",
                 horizon=30, order=None, verbose=True):
    """
    ARIMA com previsão h passos à frente, no mesmo formato do run_lstm_30.
    - 'backtest': {data_da_origem: DataFrame(real, previsao)} para cada origem do teste
    - 'futuro':   DataFrame(real=NaN, previsao) depois do fim da base
    - métricas:   origem rolante, prevendo o preço de t+h com dados reais até t
    """
    col = f"{stock_name}_{target}"
    full = pd.concat([train_dataset, test_dataset])
    datas = (pd.to_datetime(full["date"]).reset_index(drop=True)
             if "date" in full.columns
             else pd.Series(pd.to_datetime(full.index)))
    y_full = full[col].astype(float).reset_index(drop=True)
    n_train, n_test = len(train_dataset), len(test_dataset)

    if n_test < horizon:
        raise ValueError(f"Teste tem {n_test} pontos, menor que o horizonte {horizon}.")

    # 1. Ordem escolhida só com o treino
    if order is None:
        order = auto_arima(y_full.iloc[:n_train], seasonal=False,
                           suppress_warnings=True, error_action="ignore").order
    if verbose:
        print(f"ARIMA{order} | horizonte = {horizon} dias")

    # 2. Parâmetros estimados só com o treino
    trained_model = ARIMA(y_full.iloc[:n_train], order=order).fit()

    # 3. Trajetória única do fim do treino (mantida da versão anterior)
    fc = trained_model.get_forecast(steps=horizon)
    path = pd.Series(fc.predicted_mean.values, index=datas.iloc[n_train:n_train + horizon])
    path_ci = fc.conf_int(alpha=0.05)
    path_ci.index = path.index
    real_path = pd.Series(y_full.iloc[n_train:n_train + horizon].values, index=path.index)

    # 4. Backtest: cada origem = último dia conhecido; prevê os 'horizon' dias seguintes
    backtest, alvo_prev, alvo_real, alvo_base, alvo_idx = {}, [], [], [], []
    for pos in range(n_train - 1, len(full) - horizon):
        known = y_full.iloc[: pos + 1]
        fc_h = trained_model.apply(known).forecast(horizon).values
        idx = pd.DatetimeIndex(datas.iloc[pos + 1: pos + 1 + horizon])
        real = y_full.iloc[pos + 1: pos + 1 + horizon].values
        backtest[datas.iloc[pos]] = pd.DataFrame({"real": real, "previsao": fc_h}, index=idx)

        alvo_prev.append(fc_h[-1]); alvo_real.append(real[-1])
        alvo_base.append(y_full.iloc[pos]); alvo_idx.append(idx[-1])

    # 5. Futuro: depois do fim da base, sem preço real (dias úteis; feriados ignorados)
    fc_fut = trained_model.apply(y_full).forecast(horizon).values
    idx_fut = pd.bdate_range(datas.iloc[-1] + pd.Timedelta(days=1), periods=horizon)
    futuro = pd.DataFrame({"real": np.nan, "previsao": fc_fut}, index=idx_fut)

    # 6. Métricas agregadas (t+h em todas as origens)
    preds = pd.Series(alvo_prev, index=alvo_idx)
    reals = pd.Series(alvo_real, index=alvo_idx)
    base = pd.Series(alvo_base, index=alvo_idx)
    result = evaluate(f"ARIMA{order} {stock_name} h={horizon}", reals, base, preds)

    return {
        "stock": stock_name, "order": order, "horizon": horizon, "model": trained_model,
        "backtest": backtest, "futuro": futuro,
        "path": path, "path_ci": path_ci, "real_path": real_path,
        "predictions": preds, "real": reals, "baseline": base,
        **result,
    }