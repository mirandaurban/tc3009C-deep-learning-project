"""Predictor sets aligned with experimental ablation A/B."""
DEMO = ["age", "bmi", "female"]
RACE_PRE = ["distancia_km"]
OTHER = ["footwear", "injury"]
TRAINING = ["typical", "max", "sprint", "tempo", "endurancecat", "endurancespeed"]
FEATURES_A = DEMO + RACE_PRE + OTHER
FEATURES_B = FEATURES_A + TRAINING
TARGET = "velocity_kmh"
GROUP = "id"
FORBIDDEN_PREDICTORS = {"id", "tiempo_ajustado_s", "forma_fisica", "dificultad", "distancia"}

def get_features(variant: str) -> list[str]:
    if variant == "A": return FEATURES_A.copy()
    if variant == "B": return FEATURES_B.copy()
    raise ValueError("variant debe ser 'A' o 'B'")

def validate_predictors(features):
    forbidden = set(features) & FORBIDDEN_PREDICTORS
    if forbidden: raise ValueError(f"Predictores prohibidos: {sorted(forbidden)}")
