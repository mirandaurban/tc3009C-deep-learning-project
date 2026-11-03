"""
Transformación de datos raw -> processed.

Este script:
  1. Lee el archivo original de corredores (Excel).
  2. Aplica la transformación a formato largo (una fila por carrera).
  3. Calcula la variable objetivo 'velocity_kmh'.
  4. Valida la integridad del resultado.
  5. Guarda el DataFrame procesado en Parquet.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

# Rutas
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "dataset_corredores.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "df_long.parquet"

# Constantes de transformación
MILE_KM = 1.60934

RACE_PREFIXES: dict[str, tuple[str, float]] = {
    "k5":  ("5K",            5.0),
    "k10": ("10K",           10.0),
    "m5":  ("5mi",           5.0 * MILE_KM),
    "m10": ("10mi",          10.0 * MILE_KM),
    "mh":  ("Media maratón", 21.0975),
    "mf":  ("Maratón",       42.195),
}

RUNNER_COLS = [
    "id", "age", "bmi", "female", "endurancecat", "endurancespeed",
    "typical", "sprint", "tempo", "max", "injury", "footwear",
]

RACE_COLS = {
    "d":      "distancia_reportada_m",
    "ti":     "tiempo_s",
    "ti_adj": "tiempo_ajustado_s",
    "tr":     "forma_fisica",   # 1 = mejor forma de mi vida ... 4 = mal preparado
    "di":     "dificultad",     # 1 = muy difícil ... 5 = muy rápido
}


# Transformación
def build_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte el dataset de formato ancho (una fila por corredor) a formato largo
    (una fila por carrera), calculando la velocidad en km/h como variable objetivo.
    """
    registros: list[pd.DataFrame] = []

    for prefix, (nombre, dist_km) in RACE_PREFIXES.items():
        src = {f"{prefix}_{s}": new for s, new in RACE_COLS.items()}
        if not all(c in df.columns for c in src):
            print(f" | Prefijo '{prefix}' incompleto, se omite.")
            continue

        sub = df[RUNNER_COLS + list(src)].rename(columns=src)
        sub = sub.assign(distancia=nombre, distancia_km=dist_km)
        registros.append(sub)

    out = pd.concat(registros, ignore_index=True)

    # Descartar carreras sin tiempo ajustado válido
    out = out[out["tiempo_ajustado_s"].notna() & (out["tiempo_ajustado_s"] > 0)].copy()

    # Variable objetivo
    out["velocity_kmh"] = out["distancia_km"] / (out["tiempo_ajustado_s"] / 3600)

    # Eliminar columnas que no se usarán en el modelado
    out = out.drop(columns=["distancia_reportada_m", "tiempo_s"])

    return out


# Validaciones
def validate(df_long: pd.DataFrame) -> None:
    """
    Verifica que el DataFrame procesado sea consistente.
    Lanza AssertionError si algo no cuadra.
    """
    print("\n--- Validaciones ---")

    # 1. Sin nulos en columnas clave
    key_cols = [
        "id", "age", "bmi", "female", "velocity_kmh",
        "distancia_km", "tiempo_ajustado_s",
    ]
    nulos = df_long[key_cols].isnull().sum()
    assert nulos.sum() == 0, f"Hay nulos en columnas clave:\n{nulos[nulos > 0]}"
    print("  -  Sin nulos en columnas clave.")

    # 2. Velocidad en rango razonable
    v = df_long["velocity_kmh"]
    assert v.min() > 0, "Velocidad mínima no positiva."
    assert v.max() < 50, f"Velocidad máxima sospechosa: {v.max():.2f} km/h."
    print(f"  -  Velocidad en rango razonable [{v.min():.2f}, {v.max():.2f}] km/h.")

    # 3. Recalcular velocidad a mano para verificar la fórmula
    v_recalc = df_long["distancia_km"] / (df_long["tiempo_ajustado_s"] / 3600)
    max_diff = (v_recalc - v).abs().max()
    assert max_diff < 1e-9, f"La fórmula de velocidad no cierra (diff={max_diff})."
    print("  -  Fórmula de velocidad verificada.")

    # 4. Cobertura de distancias
    dist_counts = df_long["distancia"].value_counts()
    assert len(dist_counts) == len(RACE_PREFIXES), (
        f"Se esperaban {len(RACE_PREFIXES)} distancias, "
        f"se encontraron {len(dist_counts)}."
    )
    print(f"  -  {len(dist_counts)} distancias presentes.")

    # 5. Corredores únicos
    n_corredores = df_long["id"].nunique()
    n_carreras = len(df_long)
    assert n_corredores > 0, "Sin corredores únicos."
    print(f"  -  {n_carreras} carreras de {n_corredores} corredores únicos.")

    # 6. Cada corredor tiene al menos una carrera
    carreras_por_corredor = df_long.groupby("id").size()
    assert carreras_por_corredor.min() >= 1, "Hay corredores con 0 carreras."
    print(
        f"  -  Carreras por corredor: "
        f"min={carreras_por_corredor.min()}, "
        f"max={carreras_por_corredor.max()}, "
        f"media={carreras_por_corredor.mean():.2f}"
    )

    print("\nTodas las validaciones pasaron.")


# Main
def main() -> None:
    print(f"Leyendo: {RAW_PATH.relative_to(ROOT)}")
    assert RAW_PATH.exists(), (
        f"No se encontró el archivo raw en {RAW_PATH}. "
        "Colócalo en data/raw/ antes de ejecutar."
    )

    df_raw = pd.read_excel(RAW_PATH, sheet_name=0)
    print(f"  → {df_raw.shape[0]} filas × {df_raw.shape[1]} columnas (formato ancho).")

    print("\nTransformando a formato largo...")
    df_long = build_long(df_raw)

    validate(df_long)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_long.to_parquet(PROCESSED_PATH, index=False)
    print(f"\nGuardado en: {PROCESSED_PATH.relative_to(ROOT)}")
    print(f"   → {df_long.shape[0]} filas × {df_long.shape[1]} columnas.")


if __name__ == "__main__":
    main()
