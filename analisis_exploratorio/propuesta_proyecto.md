# Propuesta del Proyecto y Análisis Exploratorio

Esta carpeta contiene los materiales correspondientes a la primera entrega del proyecto **Predicción del Rendimiento de Corredores Recreativos con Deep Learning**, desarrollado para la materia TC3009C: Inteligencia Artificial Avanzada para la Ciencia de Datos.

El objetivo de esta etapa es establecer el planteamiento del problema, conocer la estructura y características del dataset, evaluar la viabilidad del enfoque propuesto y realizar experimentos preliminares que permitan justificar el desarrollo de la solución.

## Contenido

| Archivo                    | Descripción                                                  |
| -------------------------- | ------------------------------------------------------------ |
| `propuesta_proyecto.md`    | Documento con el planteamiento del problema, pregunta de investigación, descripción de los datos, hipótesis y metodología propuesta. |
| `Propuesta_Proyecto.ipynb` | Notebook con la exploración inicial del dataset, análisis de variables y experimentación preliminar. |

## Alcance de la Primera Entrega

Esta etapa tiene dos objetivos principales:

1. **Exploración y entendimiento de los datos:** verificar la carga del dataset, conocer su estructura, identificar las variables disponibles, analizar la distribución de la variable objetivo y detectar posibles riesgos para el modelado.
2. **Validación preliminar de la propuesta:** implementar modelos iniciales que permitan comprobar la viabilidad del problema y establecer puntos de comparación para el desarrollo posterior.

## Análisis Exploratorio de Datos (EDA)

El análisis se realiza a partir del dataset original de corredores recreativos y contempla:

- Inspección de dimensiones, tipos de datos y estadísticas descriptivas.
- Transformación del dataset de formato ancho a formato largo, utilizando una observación por carrera.
- Construcción de la variable objetivo: velocidad de carrera en km/h, calculada a partir de la distancia y el tiempo ajustado.
- Análisis de la distribución de velocidades y su comportamiento según la distancia.
- Exploración de las variables demográficas y de entrenamiento.
- Identificación de valores faltantes y posibles valores atípicos.
- Análisis de correlaciones entre las variables de entrada y el rendimiento.

## Modelos Baseline

Antes de implementar la red neuronal, se establecen dos modelos de referencia:

**1. Media de velocidad por sexo**

Asigna a cada observación la velocidad promedio registrada para el grupo correspondiente en los datos de entrenamiento. Funciona como una referencia sencilla para evaluar la utilidad de los modelos predictivos.

**2. Ridge Regression**

Modelo de regresión lineal regularizada mediante penalización L2. Permite establecer cuánto de la variabilidad del rendimiento puede explicarse mediante relaciones lineales entre las características disponibles.

Ambos modelos utilizan el mismo conjunto de variables, preprocesamiento y partición de datos para mantener condiciones de comparación consistentes.

## Experimento Preliminar

Se propone una red neuronal tipo **Multilayer Perceptron (MLP)** para abordar el problema de regresión.

La hipótesis inicial plantea que una MLP puede capturar interacciones no lineales entre las variables demográficas y de entrenamiento que una regresión Ridge no logra representar, mejorando la predicción de velocidad.

La arquitectura preliminar contempla:

- Dos capas ocultas de 64 y 32 neuronas.
- Función de activación ReLU.
- Dropout de 0.2.
- Optimizador Adam.
- Función de pérdida Mean Squared Error (MSE).
- Regularización mediante weight decay.

La comparación preliminar entre Ridge y MLP utiliza:

- **RMSE:** métrica principal para evaluar el error de predicción.
- **MAE:** métrica complementaria para interpretar el error absoluto promedio.

Se analizan también las curvas de aprendizaje, las predicciones frente a los valores reales y la distribución de los residuales.

**Importante:** este experimento es exploratorio y no constituye una respuesta definitiva a la pregunta de investigación. Su propósito es verificar la viabilidad del pipeline y obtener una referencia inicial del desempeño predictivo.

## Consideraciones Metodológicas

Una consideración central es evitar la fuga de información entre los conjuntos de entrenamiento, validación y prueba.

Debido a que un mismo corredor puede tener registros correspondientes a distintas distancias, las particiones se realizan agrupando por identificador (`id`). De esta manera, las observaciones de un mismo participante no aparecen simultáneamente en diferentes conjuntos.

Se utiliza `GroupShuffleSplit` para generar las particiones y `StandardScaler` para estandarizar las variables numéricas, ajustando el escalador exclusivamente con los datos de entrenamiento.

## Próxima Etapa

La entrega final contempla el desarrollo experimental completo, incluyendo:

- Implementación y evaluación de la MLP.
- Ablation study para medir la contribución de los grupos de variables demográficas y de entrenamiento.
- Validación cruzada agrupada por corredor.
- Evaluación con múltiples semillas aleatorias.
- Comparación entre modelos bajo condiciones experimentales controladas.
- Análisis de resultados y conclusiones respecto a la pregunta de investigación.

La documentación detallada de esta etapa se encontrará en `../modelo/README.md`.

## Referencia Principal

Vickers, A., & Vertosick, E. (2016). *An empirical study of race times in recreational endurance runners*. BMC Sports Science, Medicine and Rehabilitation, 8(1), 26.

https://doi.org/10.1186/s13102-016-0052-y

**Acceso al dataset:** [PubMed Central – Additional File 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC5000509/)

*Proyecto académico desarrollado para TC3009C — Tecnológico de Monterrey, Campus Monterrey.*