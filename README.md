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

La entrega se ejecuta como notebooks, en este orden:

1. `analisis_exploratorio/00_verificacion_datos.ipynb`
2. `analisis_exploratorio/01_eda_y_baselines.ipynb`
3. `experimentos/02_ablation_experiment.ipynb`
4. `experimentos/03_hyperparameter_optimization.ipynb`
5. `experimentos/04_cross_validation.ipynb`

Instala las dependencias declaradas en `requirements.txt` y ejecuta cada notebook desde la raíz del repositorio. Los notebooks 02–04 escriben sus CSV en `resultados/`; una ejecución normal vuelve a generar esos archivos.

`src/common.py` reúne las funciones compartidas por 03 y 04. `src/data_processing.py` prepara `data/processed/df_long.parquet` desde el Excel fuente; ejecutar `python -m src.data_processing` lo sobrescribe.

## Verificación medida

Se ejecutaron los cinco notebooks en orden con `nbconvert --execute`, sobre una copia temporal del repositorio. La ejecución completa terminó correctamente. Se compararon los CSV generados con los versionados en `resultados/`: 4 de 4 archivos de 02, 5 de 5 de 03 y 10 de 10 de 04 pasaron. Las diferencias numéricas fueron cero en las métricas y tablas comparables, dentro de tolerancias de `1e-6` para Ridge y `1e-3` para MLP. La columna `segundos` de `03_busqueda_etapa1.csv` y `03_busqueda_etapa2.csv` varió como medida de tiempo de ejecución; no se encontró otra diferencia.

Versiones observadas en el entorno utilizado:

- Python 3.12.10
- PyTorch 2.15.0
- scikit-learn 1.9.1
- pandas 3.1.1

Tiempo real medido en una ejecución completa (24 de noviembre de 2026):

| Notebook | Tiempo |
| --- | ---: |
| 00 verificación de datos | 2.30 s |
| 01 EDA y baselines | 8.57 s |
| 02 ablación | 18.01 s |
| 03 búsqueda de hiperparámetros | 913.21 s (15 min 13.21 s) |
| 04 validación cruzada | 118.80 s (1 min 58.80 s) |

Los tiempos dependen del equipo y de las versiones instaladas; son los observados durante esta verificación.

## Estructura

```text
├── analisis_exploratorio/
│   ├── 00_verificacion_datos.ipynb
│   └── 01_eda_y_baselines.ipynb
├── experimentos/
│   ├── 02_ablation_experiment.ipynb
│   ├── 03_hyperparameter_optimization.ipynb
│   └── 04_cross_validation.ipynb
├── data/
│   ├── raw/dataset_corredores.xlsx
│   └── processed/df_long.parquet
├── resultados/             # CSV producidos por los notebooks 02–04
├── src/
│   ├── common.py
│   └── data_processing.py
├── archivo/                 # propuesta de Colab y duplicados
├── docs/
├── README.md
└── requirements.txt
```

## Instalación, dependencias y reproducibilidad

Python 3.10 o posterior. Las versiones mínimas/rangos están en `requirements.txt`; las versiones y tiempos medidos para esta verificación aparecen arriba.

Licencia del código: MIT. Datos complementarios según los términos indicados por la publicación. Autoría académica: Miranda Urban Solano, Tecnológico de Monterrey, Campus Monterrey.
