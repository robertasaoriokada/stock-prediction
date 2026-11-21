from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .library import pd, np


def _dates(df):
    """Usa a coluna 'date' se existir; senão, o índice (que deve ser de datas)."""
    if "date" in df.columns:
        return pd.to_datetime(df["date"]).values
    return pd.to_datetime(df.index).values


def plot_forecast(
    train,
    test,
    predictions,
    stock_name,
    model_name="ARIMA",
    ticker_label=None,
    company_name=None,
    baseline=None,
    baseline_label="Baseline Naïve (Lag 1)",
    target="close",
    zoom_train=120,
    month_interval=2,
    save_path=None,
    show=True,
):
    """
    Gráfico de previsão vs. real, para qualquer ação e qualquer modelo.

    stock_name:    prefixo do dataset (ex.: "vale", "petro", "itau").
    model_name:    texto da legenda/título (ex.: "XGBoost (Tuned)", "ARIMA(1,1,0)").
    ticker_label:  código exibido no título (ex.: "PETR4"). Padrão: stock_name em maiúsculas.
    company_name:  nome exibido no título (ex.: "Petrobras"). Se None, o título usa só o ticker.
    baseline:      série do baseline naive (res["ontem"]). Se None, não desenha.
    zoom_train:    quantos dias do final do treino mostrar (None = treino inteiro).
    save_path:     se None, grava em results/<modelo>/<stock>_<modelo>.png.
    """
    col = f"{stock_name}_{target}"
    ticker_label = ticker_label or stock_name.upper()
    predictions = np.asarray(predictions, dtype=float)

    if len(predictions) != len(test):
        raise ValueError(f"Previsões ({len(predictions)}) e teste ({len(test)}) têm tamanhos diferentes.")

    zoom = train if zoom_train is None else train.iloc[-zoom_train:]

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

    # 1. Treino recente
    ax.plot(_dates(zoom), zoom[col], label="Treino (Histórico Recente)",
            color="#1f77b4", linewidth=1.5)

    # 2. Teste real
    ax.plot(_dates(test), test[col], label="Teste (Real)",
            color="#2ca02c", linewidth=2, zorder=2)

    # 3. Previsão do modelo
    ax.plot(_dates(test), predictions, label=f"Previsão {model_name}",
            color="#ff7f0e", linestyle="--", linewidth=1.8, zorder=3)

    # 4. Baseline naive (preço de ontem), por baixo das outras linhas
    if baseline is not None:
        baseline = np.asarray(baseline, dtype=float)
        if len(baseline) != len(test):
            raise ValueError(f"Baseline ({len(baseline)}) e teste ({len(test)}) têm tamanhos diferentes.")
        ax.plot(_dates(test), baseline, label=baseline_label,
                color="#7f7f7f", linestyle=":", linewidth=1.3, alpha=0.8, zorder=1)

    nome_ativo = f"{company_name} ({ticker_label})" if company_name else ticker_label
    ax.set_title(f"Previsão de Fechamento da {nome_ativo} — {model_name} vs. Dados Reais",
                 fontsize=14, pad=15, fontweight="bold")
    ax.set_xlabel("Data", fontsize=12)
    ax.set_ylabel("Preço de Fechamento (R$)", fontsize=12)
    ax.legend(loc="upper left", fontsize=11, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=month_interval))
    plt.setp(ax.get_xticklabels(), rotation=30)

    fig.tight_layout()

    if save_path is None:
        slug = model_name.split("(")[0].strip().lower().replace(" ", "_")
        save_path = Path("results") / slug / f"{stock_name}_{slug}.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
