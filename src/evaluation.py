"""Regression metrics."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def metrics(y, pred):
    return {"RMSE": float(np.sqrt(mean_squared_error(y,pred))),
            "MAE": float(mean_absolute_error(y,pred)), "R2": float(r2_score(y,pred))}
