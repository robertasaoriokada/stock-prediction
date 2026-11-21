from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)

from .library import np
from .stock_config import TRAIN_FRAC


def split_temporal(database, train_frac=TRAIN_FRAC):
    cut = int(len(database) * train_frac)
    return database.iloc[:cut], database.iloc[cut:]


def evaluate(nome, real, ontem, previsto):
    """RMSE, MAE, MAPE e acurácia direcional (direção medida contra o preço real de ontem)."""
    real, ontem, previsto = (np.asarray(x, dtype=float) for x in (real, ontem, previsto))

    rmse = np.sqrt(mean_squared_error(real, previsto))
    mae = mean_absolute_error(real, previsto)
    mape = mean_absolute_percentage_error(real, previsto)

    if np.allclose(previsto, ontem, rtol=0, atol=1e-8):
        dir_val, direcao = np.nan, "n/a"
    else:
        dir_val = np.mean(np.sign(real - ontem) == np.sign(previsto - ontem))
        direcao = f"{dir_val:.2%}"

    print(
        f"{nome:<22} RMSE: {rmse:.4f} | MAE: {mae:.4f} | "
        f"MAPE: {mape:.2%} | Acurácia direcional: {direcao}"
    )
    return {"rmse": rmse, "mae": mae, "mape": mape, "dir_acc": dir_val}