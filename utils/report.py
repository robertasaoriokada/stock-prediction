from pathlib import Path
import matplotlib.pyplot as plt
from .library import pd, np

RESULTS_FOLDER = Path("results")


def results_frame(res, test_dataset):
    """Tabela dia a dia: data, real, ontem, previsto."""
    return pd.DataFrame({
        "date": test_dataset["date"].values,
        "real": res["real"].values,
        "ontem": res["ontem"].values,
        "previsto": res["predictions"].values,
    })


def plot_predictions(res, test_dataset, zoom_days=60, save_path=None):
    df = results_frame(res, test_dataset)
    nome = f"ARIMA{res['order']} - {res['stock'].upper()}"

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # teste completo
    axes[0].plot(df["date"], df["real"], label="Real", linewidth=1.2)
    axes[0].plot(df["date"], df["previsto"], label="Previsto (1 passo)", linewidth=1, alpha=0.8)
    axes[0].set_title(f"{nome} - período de teste")
    axes[0].legend()

    # zoom nos últimos dias: no teste completo as duas linhas ficam sobrepostas
    z = df.tail(zoom_days)
    axes[1].plot(z["date"], z["real"], label="Real", marker="o", markersize=3)
    axes[1].plot(z["date"], z["previsto"], label="Previsto", marker="x", markersize=3)
    axes[1].set_title(f"Últimos {zoom_days} pregões")
    axes[1].legend()

    for ax in axes:
        ax.set_ylabel("Preço de fechamento (R$)")
        ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def save_results(res, test_dataset, folder=RESULTS_FOLDER):
    """Grava previsões (CSV), métricas (CSV) e o modelo (pickle) de uma ação."""
    folder = Path(folder) / "arima"
    folder.mkdir(parents=True, exist_ok=True)
    s = res["stock"]

    results_frame(res, test_dataset).to_csv(folder / f"{s}_predicoes.csv", index=False)

    metricas = {k: res[k] for k in ("stock", "order", "rmse", "mape", "dir_acc")}
    pd.DataFrame([metricas]).to_csv(folder / f"{s}_metricas.csv", index=False)

    res["model"].save(folder / f"{s}_modelo.pkl")   # statsmodels: recarrega com ARIMAResults.load()
    return folder