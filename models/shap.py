import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap


def explain_xgb(res, max_display=15, bar=True, save_dir=None, show=True):
    """SHAP por feature (beeswarm + barras) nos dados de TESTE, a partir do res do run_xgbooster."""
    stock = res["stock"]
    X_test = res["X_test"]
    save_dir = Path(save_dir or "results/xgboost/shap")
    save_dir.mkdir(parents=True, exist_ok=True)

    shap_values = shap.TreeExplainer(res["model"])(X_test)

    shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)
    plt.title(f"SHAP - XGBoost ({stock.upper()})")
    plt.tight_layout()
    plt.savefig(save_dir / f"{stock}_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    if bar:
        shap.plots.bar(shap_values, max_display=max_display, show=False)
        plt.title(f"Importância média |SHAP| - XGBoost ({stock.upper()})")
        plt.tight_layout()
        plt.savefig(save_dir / f"{stock}_shap_bar.png", dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    return shap_values


def aggregate_by_variable(shap_values, features):
    """Soma o SHAP dos lags de cada variável (por dia) e tira a média do valor absoluto."""
    base = [re.sub(r"_lag_\d+$", "", f) for f in features]
    df = pd.DataFrame(shap_values.values, columns=base)
    por_variavel = df.T.groupby(level=0).sum().T
    imp = por_variavel.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({
        "variavel": imp.index,
        "mean_abs_shap": imp.values,
        "pct": imp.values / imp.sum() * 100,
    })


def explain_variables(res, shap_values=None, top=10, save_dir=None, show=True):
    """Importância SHAP por variável (lags agregados) para uma ação."""
    stock = res["stock"]
    save_dir = Path(save_dir or "results/xgboost/shap")
    save_dir.mkdir(parents=True, exist_ok=True)

    if shap_values is None:
        shap_values = shap.TreeExplainer(res["model"])(res["X_test"])

    tabela = aggregate_by_variable(shap_values, res["features"])
    tabela.to_csv(save_dir / f"{stock}_shap_variaveis.csv", index=False)

    t = tabela.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.barh(t["variavel"], t["pct"], color="#1f77b4")
    for y, v in enumerate(t["pct"]):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=10)
    ax.set_xlabel("Contribuição relativa (% do |SHAP| médio)", fontsize=12)
    ax.set_title(f"Variáveis mais importantes — XGBoost ({stock.upper()})",
                 fontsize=14, fontweight="bold", pad=12)
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(save_dir / f"{stock}_shap_variaveis.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return tabela