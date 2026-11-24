# Predicción de velocidad de carrera en corredores recreativos

## Descripción del Proyecto

Este repositorio contiene el desarrollo de un proyecto de Deep Learning para la materia **TC3009C: Inteligencia Artificial Avanzada para la Ciencia de Datos**, del Tecnológico de Monterrey.

El proyecto busca predecir la velocidad de carrera de corredores recreativos a partir de sus características demográficas y hábitos de entrenamiento, utilizando modelos de aprendizaje automático y redes neuronales.

El objetivo principal es investigar en qué medida las variables relacionadas con el entrenamiento, como el kilometraje semanal, los intervalos y las sesiones de tempo, aportan información predictiva adicional frente a variables demográficas como la edad, el sexo y el IMC.

## Pregunta de Investigación

> ¿Cuánto aportan las variables de entrenamiento (kilometraje semanal, intervalos, tempo runs) sobre las demográficas (edad, sexo, BMI) para predecir la velocidad en carrera con base en una red MLP que captura dichas interacciones no lineales?

## Datos y objetivo

El proyecto utiliza el conjunto de datos publicado como material complementario del siguiente estudio:

**Fuente:** Vickers, A., & Vertosick, E. (2016). _An empirical study of race times in recreational endurance runners_. BMC Sports Science, Medicine and Rehabilitation, 8(1), 26.

- **Publicación:** [An empirical study of race times in recreational endurance runners](https://pmc.ncbi.nlm.nih.gov/articles/PMC5000509/)
- **Dataset:** archivo complementario Additional File 2 (`.xlsx`).
- **Origen:** encuesta en línea realizada entre abril y junio de 2014 a corredores recreativos.
- **Tipo de datos:** información autorreportada.
- **Variable objetivo:** velocidad de carrera, derivada de la distancia y el tiempo reportados.

`data/raw/dataset_corredores.xlsx` es el archivo fuente y `data/processed/df_long.parquet` contiene una fila por carrera (5,153 observaciones en el archivo presente). `src/data_processing.py` transforma el Excel ancho a largo.

El objetivo `velocity_kmh` es `distancia_km / (tiempo_ajustado_s / 3600)`. La unidad de agrupación es el corredor (`id`); nunca es predictor. Las características derivadas de la carrera o posteriores al resultado (`tiempo_ajustado_s`, `forma_fisica`, `dificultad`, etiqueta de distancia) se excluyen del modelado para reducir fuga de información. Se conserva `distancia_km`, conocida antes de correr. Las variables A son `age`, `bmi`, `female`, `distancia_km`, `footwear`, `injury`; B añade `typical`, `max`, `sprint`, `tempo`, `endurancecat`, `endurancespeed`. El diseño A/B es una ablación grupal; no permite asignar contribuciones individuales.

## Preprocesamiento y diseño experimental

Se hacen particiones agrupadas por `id` con `GroupShuffleSplit` (15% test reservado, 15% validación y 70% entrenamiento; semilla 42). El escalador estándar se ajusta exclusivamente en entrenamiento. Las configuraciones congeladas del MLP original son capas `(64, 32)`, dropout 0.2, Adam, tasa 0.001, weight decay 0.0001, batch 64 y 80 épocas. Ridge usa alpha=1 en la comparación fija de ablación. La búsqueda documentada usa validation y early stopping (paciencia 15, máximo 200 épocas), con búsqueda aleatoria de 150 configuraciones de un espacio de 2,016 y cinco semillas en la segunda etapa.

La validación cruzada agrupada posterior usa cinco folds sobre el conjunto de desarrollo. En cada fold, los MLP seleccionan época mediante una validación interna agrupada y se reajustan con todo el entrenamiento del fold. Para Ridge, la CV histórica usa alpha=1e-6 para A y alpha=100 para B. Ambos alphas se seleccionaron previamente mediante validation; las configuraciones MLP también se seleccionaron previamente con validation. Por ello, la CV es una comparación interna condicionada a esa selección y no una estimación independiente de generalización del proceso de selección.

## Resultados

Los siguientes valores se transcriben de `resultados/03_comparacion_final_validacion.csv` y `resultados/04_cv_resumen.csv`. RMSE/MAE están en km/h y se muestran media ± desviación estándar cuando aplica.

| Experimento / modelo                                        |             A |             B |
| ----------------------------------------------------------- | ------------: | ------------: |
| Ablación Ridge, validación fija, RMSE                       |         1.946 |         1.460 |
| MLP original, validación fija, RMSE (80 épocas, 5 semillas) | 1.957 ± 0.016 | 1.519 ± 0.018 |
| MLP optimizada, validación fija, RMSE (5 semillas)          | 1.888 ± 0.013 | 1.448 ± 0.006 |
| CV agrupada, RMSE media ± DE entre folds (MLP)              | 1.856 ± 0.095 | 1.476 ± 0.060 |
| CV agrupada, RMSE media ± DE entre folds (Ridge)            | 1.887 ± 0.094 | 1.430 ± 0.068 |

Los resultados documentados son compatibles con utilidad predictiva adicional del bloque de entrenamiento en A/B. Ridge B presenta menor RMSE promedio que MLP B en la CV reportada; así que los resultados no apoyan afirmar que MLP o la no linealidad sean necesarios. La CV tampoco elimina el sesgo de selección por hiperparámetros previamente elegidos con validation.

## Código y ejecución

Los notebooks son la evidencia y el registro de la experimentación histórica. Los módulos de `src/` contienen scripts modulares reutilizables de carga, variables, preprocesamiento, modelos y utilidades experimentales

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

## Estructura

```text
├── analisis_exploratorio/
│   ├── propuesta_proyecto.md
│   └── Propuesta_Proyecto.ipynb
│
├── data/
│   ├── raw/
│   │   └── dataset_corredores.xlsx
│   └── processed/
│
├── modelo/
│   └── README.md
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Instalación, dependencias y reproducibilidad

Python 3.10 o posterior. Las versiones mínimas/rangos están en `requirements.txt`

Licencia del código: MIT. Datos complementarios según los términos indicados por la publicación. Autoría académica: Miranda Urban Solano, Tecnológico de Monterrey, Campus Monterrey.
