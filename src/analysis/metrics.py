import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)


def rmse(y, yhat):
    return np.sqrt(mean_squared_error(y, yhat))


def bias(y, yhat):
    et = y - yhat
    return np.sum(et) / len(et)


def forecast_congruence(forecast_error):
    x = np.var(forecast_error, axis=0, ddof=1)
    x = np.mean(x)
    return np.sqrt(x)


def reliability(y, yhat):
    err = y - yhat
    fiab = 1 - abs(err) / y
    fiab = np.where(np.isinf(fiab), 0, fiab)
    return fiab


def weighted_reliability(y, yhat, clip=True):
    err = y - yhat
    fiab = 1 - np.abs(err) / y
    if clip:
        fiab = fiab.clip(0, None)
    return np.sum(y * fiab) / np.sum(y)


def display_metrics(y, yhat, name="default"):
    metrics = pd.DataFrame()
    metrics["fname"] = [name]
    metrics["rmse"] = rmse(y, yhat)
    metrics["bias"] = bias(y, yhat)
    metrics["mape"] = mean_absolute_percentage_error(y, yhat)
    metrics["wfiab"] = weighted_reliability(y, yhat, clip=False)
    metrics["mae"] = mean_absolute_error(y, yhat)
    return metrics
