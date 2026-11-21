import pandas as pd
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