"""Split temporal e métricas, iguais para todos os modelos."""
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from .library import np
from .stock_config import TRAIN_FRAC


def split_temporal(database, train_frac=TRAIN_FRAC):
    cut = int(len(database) * train_frac)
    return database.iloc[:cut], database.iloc[cut:]


def evaluate(name, real, past, predicted):
    """RMSE, MAE, MAPE e acurácia direcional (direção medida contra o preço real de ontem)."""
    mae = mean_absolute_error(real, predicted)
    rmse = np.sqrt(mean_squared_error(real, predicted))
    mape = mean_absolute_percentage_error(real, predicted)
    if np.all(np.asarray(predicted) == np.asarray(past)):
        direction_val, direction = np.nan, "n/a"
    else:
        direction_val = np.mean(np.sign(real - past) == np.sign(predicted - past))
        direction = f"{direction_val:.2%}"
        print(f"{name:<22} RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2%} | Acurácia direcional: {direction}")
    return {"rmse": rmse, "mae": mae, "mape": mape, "dir_acc": direction_val}