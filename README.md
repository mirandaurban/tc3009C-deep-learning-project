# Predicción del Rendimiento de Corredores Recreativos con Deep Learning

## Descripción del Proyecto

Este repositorio contiene el desarrollo de un proyecto de Deep Learning para la materia **TC3009C: Inteligencia Artificial Avanzada para la Ciencia de Datos**, del Tecnológico de Monterrey.

El proyecto busca predecir la velocidad de carrera de corredores recreativos a partir de sus características demográficas y hábitos de entrenamiento, utilizando modelos de aprendizaje automático y redes neuronales.

El objetivo principal es investigar en qué medida las variables relacionadas con el entrenamiento, como el kilometraje semanal, los intervalos y las sesiones de tempo, aportan información predictiva adicional frente a variables demográficas como la edad, el sexo y el IMC.

Para abordar el problema, se propone utilizar una red neuronal tipo **Multilayer Perceptron (MLP)**, capaz de modelar relaciones no lineales entre las variables de entrada.

## Pregunta de Investigación

> ¿Cuánto aportan las variables de entrenamiento (kilometraje semanal, intervalos, tempo runs) sobre las demográficas (edad, sexo, BMI) para predecir la velocidad en carrera con base en una red MLP que captura dichas interacciones no lineales?

## Datos

El proyecto utiliza el conjunto de datos publicado como material complementario del siguiente estudio:

**Fuente:** Vickers, A., & Vertosick, E. (2016). *An empirical study of race times in recreational endurance runners*. BMC Sports Science, Medicine and Rehabilitation, 8(1), 26.

- **Publicación:** [An empirical study of race times in recreational endurance runners](https://pmc.ncbi.nlm.nih.gov/articles/PMC5000509/)
- **Dataset:** archivo complementario Additional File 2 (`.xlsx`).
- **Origen:** encuesta en línea realizada entre abril y junio de 2014 a corredores recreativos.
- **Tipo de datos:** información autorreportada.
- **Variable objetivo:** velocidad de carrera, derivada de la distancia y el tiempo reportados.

El conjunto de datos contiene información de distintas distancias de carrera por participante. Para el modelado, se contempla su transformación a un formato donde cada observación representa una carrera.

## Estructura del Repositorio

El repositorio se organiza en módulos que separan la propuesta inicial, el análisis de los datos y el desarrollo del modelo.

```
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

### Organización

- `analisis_exploratorio/`: contiene la documentación de la propuesta inicial y el notebook correspondiente a la primera entrega, incluyendo la exploración preliminar y la evaluación de viabilidad.
- `data/raw/`: almacena los datos originales.
- `data/processed/`: contiene los datos transformados y preparados para el modelado.
- `modelo/`: concentra el desarrollo de la solución, los experimentos y la evaluación de los modelos.
- `requirements.txt`: especifica las dependencias necesarias para ejecutar el proyecto.

## Documentación del Proyecto

El desarrollo se documenta en dos etapas principales:

**1. Propuesta y análisis exploratorio**

Se presenta el planteamiento del problema, la descripción del dataset, la exploración inicial de los datos, los modelos baseline y la justificación de la metodología propuesta.

[Consultar documentación de la propuesta](analisis_exploratorio/propuesta_proyecto.md)

**2. Desarrollo y entrega final**

Documenta la implementación de los modelos, el proceso experimental, la evaluación del rendimiento y los resultados obtenidos durante el desarrollo del proyecto.

[Consultar documentación del modelo](modelo/modelo.md)

## Tecnologías

- **Python:** lenguaje principal de desarrollo.
- **PyTorch:** implementación y entrenamiento de redes neuronales.
- **scikit-learn:** preprocesamiento, modelos baseline y evaluación.
- **Pandas y NumPy:** manipulación y procesamiento de datos.
- **Matplotlib y Seaborn:** análisis exploratorio y visualización.

## Configuración del Entorno

Para replicar el entorno de desarrollo:

1. Clonar el repositorio:

```
git clone https://github.com/tu-usuario/prediccion-rendimiento-corredores.git
cd prediccion-rendimiento-corredores
```

1. Crear y activar un entorno virtual:

```
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

1. Instalar las dependencias:

```
pip install -r requirements.txt
```

## Licencia y Atribución de los Datos

El código de este proyecto se distribuye bajo la licencia MIT.

Los datos utilizados provienen del material complementario del estudio de Vickers y Vertosick (2016). De acuerdo con la documentación de la publicación, los archivos de datos complementarios se encuentran bajo la renuncia CC0, salvo indicación contraria.

Se reconoce a los autores originales del estudio por la recopilación y publicación de los datos.

## Autor

**Miranda Urban Solano**
Tecnológico de Monterrey — Campus Monterrey
Proyecto individual de Deep Learning, TC3009C.
