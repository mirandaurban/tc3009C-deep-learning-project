"""Loading and structural validation of the processed runner dataset."""
from pathlib import Path
import numpy as np
import pandas as pd
from .features import TARGET, GROUP, FEATURES_B, FORBIDDEN_PREDICTORS

REQUIRED = set(FEATURES_B) | {TARGET, GROUP, "distancia_km", "tiempo_ajustado_s"}

def load_data(path):
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(f"No existe el parquet: {path}")
    frame = pd.read_parquet(path)
    validate_data(frame)
    return frame

def validate_data(frame):
    missing = REQUIRED - set(frame.columns)
    if missing: raise ValueError(f"Faltan columnas: {sorted(missing)}")
    if frame.empty: raise ValueError("Dataset vacío")
    if frame[list(REQUIRED)].isna().any().any(): raise ValueError("Nulos en columnas requeridas")
    if frame[GROUP].nunique() < 2: raise ValueError("Se requieren al menos dos corredores")
    if (frame["tiempo_ajustado_s"] <= 0).any(): raise ValueError("tiempo_ajustado_s debe ser positivo")
    derived = frame["distancia_km"] / (frame["tiempo_ajustado_s"] / 3600)
    if not np.allclose(derived, frame[TARGET], rtol=0, atol=1e-9):
        raise ValueError("velocity_kmh no coincide con distancia/tiempo")
    if FORBIDDEN_PREDICTORS & set(FEATURES_B): raise AssertionError("Feature prohibida en B")
    return True
