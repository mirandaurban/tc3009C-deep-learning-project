"""
Configuraciones importantes del experimento final.

Este script:
- Permite centralizar decisiones fundamentales para la experimentación. 
- Garantiza reproducibilidad y repetibilidad. 
- Se definen rutas y valores de entrenamiento. 

Para justificación de las decisiones, véase 'Propuesta de Proyecto de Proyecto.pdf'
"""

import os
from pathlib import Path

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_PATH = Path(os.environ.get("DATA_PATH", ROOT / "data" / "processed" / "df_long.parquet"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "resultados"))

TARGET = "velocity_kmh"   # km/h, derivada de distancia_km y tiempo_ajustado_s
GROUP = "id"              # identificador del corredor

# Grupos de variables
FEATURE_GROUPS = {
    "demograficas":  ["age", "bmi", "female"],
    "entrenamiento": ["typical", "max", "sprint", "tempo", "endurancecat", "endurancespeed"],
    "carrera":       ["distancia_km", "forma_fisica", "dificultad"],
    "otras":         ["injury", "footwear"],
}

ALL_FEATURES = [c for g in FEATURE_GROUPS.values() for c in g]

FEATURES_SIN_ENTRENAMIENTO = [c for g, cols in FEATURE_GROUPS.items()
                              if g != "entrenamiento" for c in cols]

# Protocolo de evaluación del modelo
TEST_SIZE = 0.15          # tamaño de prueba final (reservada y solo se usa al final)
N_FOLDS = 5               # k-fold agrupado por corredor sobre el resto del dataset (85%) 
SEED_SPLIT = 42           # semilla de particiones
SEED_LIST = [0, 1, 2]     # semillas de inicialización para MLP

# Hiperparámetros elegidos
MLP_CONFIG = {
    "hidden": (64, 32),
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 80,
    "batch_size": 64,
}
RIDGE_ALPHA = 1.0