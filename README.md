# Predicción de velocidad de carrera en corredores recreativos

Proyecto académico de TC3009C que analiza el valor predictivo de variables de entrenamiento sobre características demográficas para estimar velocidad de carrera. La pregunta guía es: ¿cuánto aportan kilometraje semanal, intervalos y sesiones tempo, además de edad, sexo e IMC, en modelos MLP? La comparación es predictiva y observacional: no identifica causalidad ni demuestra que la no linealidad sea necesaria.

## Datos y objetivo

Se emplea el material suplementario de Vickers y Vertosick (2016), *An empirical study of race times in recreational endurance runners*, encuesta en línea a corredores recreativos. La fuente, los datos auto reportados, y las seis distancias se describen en [la publicación](https://pmc.ncbi.nlm.nih.gov/articles/PMC5000509/). `data/raw/dataset_corredores.xlsx` es el archivo fuente y `data/processed/df_long.parquet` contiene una fila por carrera (5,153 observaciones en el archivo presente). `src/data_processing.py` transforma el Excel ancho a largo.

El objetivo `velocity_kmh` es `distancia_km / (tiempo_ajustado_s / 3600)`. La unidad de agrupación es el corredor (`id`); nunca es predictor. Las características derivadas de la carrera o posteriores al resultado (`tiempo_ajustado_s`, `forma_fisica`, `dificultad`, etiqueta de distancia) se excluyen del modelado para reducir fuga de información. Se conserva `distancia_km`, conocida antes de correr. Las variables A son `age`, `bmi`, `female`, `distancia_km`, `footwear`, `injury`; B añade `typical`, `max`, `sprint`, `tempo`, `endurancecat`, `endurancespeed`. El diseño A/B es una ablación grupal; no permite asignar contribuciones individuales.

## Preprocesamiento y diseño experimental

Se hacen particiones agrupadas por `id` con `GroupShuffleSplit` (15% test reservado, 15% validación y 70% entrenamiento; semilla 42). El escalador estándar se ajusta exclusivamente en entrenamiento. Las configuraciones congeladas del MLP original son capas `(64, 32)`, dropout 0.2, Adam, tasa 0.001, weight decay 0.0001, batch 64 y 80 épocas. Ridge usa alpha=1 en la comparación fija de ablación. La búsqueda documentada usa validation y early stopping (paciencia 15, máximo 200 épocas), con búsqueda aleatoria de 150 configuraciones de un espacio de 2,016 y cinco semillas en la segunda etapa.

La validación cruzada agrupada posterior usa cinco folds sobre el conjunto de desarrollo. En cada fold, los MLP seleccionan época mediante una validación interna agrupada y se reajustan con todo el entrenamiento del fold. Para Ridge, la CV histórica usa alpha=1e-6 para A y alpha=100 para B. Ambos alphas se seleccionaron previamente mediante validation; las configuraciones MLP también se seleccionaron previamente con validation. Por ello, la CV es una comparación interna condicionada a esa selección y no una estimación independiente de generalización del proceso de selección.

## Resultados históricos verificados

Los siguientes valores se transcriben de `resultados/03_comparacion_final_validacion.csv` y `resultados/04_cv_resumen.csv`; no se recalcularon. RMSE/MAE están en km/h y se muestran media ± desviación estándar cuando aplica.

| Experimento / modelo | A | B |
|---|---:|---:|
| Ablación Ridge, validación fija, RMSE | 1.946 | 1.460 |
| MLP original, validación fija, RMSE (80 épocas, 5 semillas) | 1.957 ± 0.016 | 1.519 ± 0.018 |
| MLP optimizada, validación fija, RMSE (5 semillas) | 1.888 ± 0.013 | 1.448 ± 0.006 |
| CV agrupada, RMSE media ± DE entre folds (MLP) | 1.856 ± 0.095 | 1.476 ± 0.060 |
| CV agrupada, RMSE media ± DE entre folds (Ridge) | 1.887 ± 0.094 | 1.430 ± 0.068 |

Los resultados documentados son compatibles con utilidad predictiva adicional del bloque de entrenamiento en A/B. Ridge B presenta menor RMSE promedio que MLP B en la CV reportada; así que los resultados no apoyan afirmar que MLP o la no linealidad sean necesarios. La CV tampoco elimina el sesgo de selección por hiperparámetros previamente elegidos con validation.

### Registro de evaluación final e integridad

Los artefactos históricos existentes bajo `resultados/` se preservan como registro. `resultados/ejecucion.json` registra una ejecución fechada el 17-11-2026 e informa 1,839 corredores de desarrollo y 325 de prueba sin solapamiento. El historial del proyecto documenta un incidente: la evaluación final se ejecutó sin autorización explícita. No se presenta como evaluación formalmente autorizada, ni se emplea aquí para seleccionar o cambiar modelos. Esta entrega no vuelve a ejecutarla ni modifica sus resultados. No se reportan aquí métricas del conjunto test.

## Código y ejecución

Los notebooks son la evidencia y el registro de la experimentación histórica. Los módulos de `src/` contienen implementaciones modulares reutilizables de carga, variables, preprocesamiento, modelos y utilidades experimentales; los notebooks históricos aún no fueron migrados a esos módulos. Los scripts de evaluación final y sus resultados son artefactos históricos protegidos y no forman parte de los comandos de uso documentados aquí.

La CLI ofrece `validate-data`, que valida los datos procesados sin entrenar modelos, y `smoke`, que ajusta un Ridge con datos sintéticos. Las pruebas unitarias incluyen además un entrenamiento MLP pequeño con datos sintéticos. Ninguna de estas comprobaciones ejecuta evaluación final ni entrena los modelos finales.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Después de activar el entorno:

```bash
python -m pip install -r requirements.txt
python main.py --help
python main.py validate-data
python main.py smoke
python -m unittest discover -s tests -v
```

`validate-data` solo verifica el Parquet procesado y sus columnas/objetivo; no crea particiones, lee métricas ni ajusta modelos. `smoke` ajusta Ridge únicamente con datos sintéticos y no accede al dataset del proyecto. Las pruebas unitarias también ajustan una MLP pequeña solo con datos sintéticos. Para reconstruir el Parquet desde el Excel, ejecutar `python -m src.data_processing`; esto sobrescribe `data/processed/df_long.parquet`, por lo que solo debe hacerse deliberadamente.

No existe un comando de evaluación test. Para estudiar los experimentos históricos, abrir los notebooks numerados en `analisis_exploratorio/` y `experimentos/`; su ejecución integral puede volver a entrenar y escribir resultados, por lo que no forma parte de la validación de este entregable.

### Alcance real de la implementación modular

La CLI integra validación de datos y un smoke test sintético. `src/data.py`, `src/features.py`, `src/preprocessing.py`, `src/models.py`, `src/training.py`, `src/evaluation.py` y `src/pipeline.py` contienen componentes reutilizables; `src/optimization.py` solo muestrea configuraciones. No hay una orquestación modular que ejecute la búsqueda histórica completa, GroupKFold o la evaluación final. Los módulos no reproducen por sí solos las tablas históricas. Gradient Boosting se menciona como posibilidad en la propuesta inicial, pero no está implementado ni forma parte de los resultados reportados.

`src/features.py` es la definición activa de predictores A/B y excluye `id`, `tiempo_ajustado_s`, `forma_fisica` y `dificultad`. `src/common.py` se conserva como helper histórico utilizado por notebooks. `src/config.py` también es configuración histórica, no se importa desde la CLI y contiene grupos de variables antiguos que incluyen `forma_fisica` y `dificultad`; no debe usarse para construir predictores del modelo actual. Los notebooks conservan su flujo experimental propio y no se migraron a los módulos nuevos.

## Estructura

```text
src/                         componentes reutilizables, procesamiento y helpers históricos
  common.py, config.py       soporte histórico para notebooks/configuración; no son ruta CLI activa
tests/                       pruebas unitarias sintéticas y de integridad
data/raw/                    archivo original XLSX
data/processed/              dataset largo Parquet
analisis_exploratorio/       EDA y propuesta inicial
experimentos/                ablación, búsqueda y CV (notebooks originales)
resultados/                  CSVs y registros históricos preservados
docs/reporte_final.md         reporte técnico final (requiere docs/figures/ para sus imágenes)
main.py                      CLI no destructiva
```

## Instalación, dependencias y reproducibilidad

Python 3.10 o posterior. Las versiones mínimas/rangos están en `requirements.txt`; no había lockfile en el repositorio, por lo que no se garantiza identidad bit a bit entre plataformas. El registro histórico indica Python 3.12.10 y CPU. PyTorch puede variar en reproducibilidad entre hardware/backend; el código fija semillas Python, NumPy y Torch. El Parquet procesado está incluido para evitar depender del reprocesamiento.

Licencia del código: MIT. Datos complementarios según los términos indicados por la publicación. Autoría académica: Miranda Urban Solano, Tecnológico de Monterrey, Campus Monterrey.
