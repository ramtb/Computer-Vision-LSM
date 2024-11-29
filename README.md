# Visión artificial para el reconocimiento de la Lengua de señas Mexicana

Este repositorio contiene...

## Estructura del Repositorio

La estructura del repositorio está organizada en varias carpetas para facilitar la modularización y reutilización del código. Aquí se describe el contenido de cada carpeta:

### 1. `notebooks/` - Jupyter Notebooks
Esta carpeta contiene los **Jupyter Notebooks** utilizados para la exploración, entrenamiento, evaluación e inferencia de los modelos.

### 2. `models/` - Modelos de Inteligencia Artificial
Esta carpeta contiene todo lo relacionado con los modelos de IA, tanto los **modelos entrenados** como las **definiciones de las arquitecturas**.

- **`entrenados/`**: Modelos ya entrenados, incluidos los archivos de pesos y configuraciones.
  - **`README.md`**: Descripción de los modelos entrenados y sus características.
  
- **`definiciones/`**: Definiciones de las arquitecturas de los modelos.

- **`utils.py`**: Funciones auxiliares para trabajar con los modelos, como la carga de pesos y la realización de predicciones.

### 3. `data/` - Datos
Esta carpeta contiene los datos utilizados en el proyecto, organizados en diferentes subcarpetas dependiendo de su estado.

- **`features/`**: Características extraídas de los datos procesados, listas para la entrada en los modelos.
- **`README.md`**: Descripción de los datos 

### 4. `pipelines/` - Pipelines de Procesamiento
En esta carpeta se encuentran los scripts utilizados para **preprocesar** los datos, **entrenar** los modelos, y realizar **evaluaciones** e **inferencias**.

- **`preprocesamiento.py`**: Script que realiza el preprocesamiento de los datos.
- **`entrenamiento.py`**: Script para entrenar los modelos de IA.
- **`evaluacion.py`**: Script para evaluar los modelos utilizando métricas estándar.
- **`inferencia.py`**: Script para realizar inferencias con el modelo entrenado sobre nuevos datos.

### 5. `visualization/` - Visualización y Análisis
Esta carpeta contiene todo lo relacionado con la **visualización de datos** y resultados.

- **`graficos.py`**: Código para generar gráficos y visualizaciones utilizando herramientas como Matplotlib y Seaborn.
- **`dashboards/`**: Dashboards interactivos (usando Streamlit, Dash, etc.).
- **`plots/`**: Carpeta donde se guardan las salidas gráficas generadas.
- **`README.md`**: Guía sobre cómo generar y visualizar gráficos en el proyecto.

### 6. `gui/` - Interfaz Gráfica de Usuario (GUI)
Esta carpeta contiene todo lo relacionado con la **interfaz gráfica de usuario (GUI)** del proyecto.

- **`main.py`**: Script principal que inicializa y ejecuta la GUI.
- **`components/`**: Componentes reutilizables de la GUI, como botones, cuadros de texto, etc.
- **`assets/`**: Recursos estáticos, como imágenes, íconos y archivos de estilo (CSS, JSON, etc.).

### 7. `modules/` - Módulos Reutilizables
Esta carpeta contiene **módulos de código reutilizables** que encapsulan funcionalidades específicas y son utilizados en diferentes partes del proyecto.

- **`utils.py`**: Funciones auxiliares generales que no encajan en otros módulos.

### 8. `tests/` - Pruebas Automatizadas
Contiene pruebas automatizadas para garantizar que el código funciona como se espera. Las pruebas pueden ser unitarias o de integración.

### 9. Archivos de Configuración
- **`README.md`**: Este archivo, donde se proporciona información sobre el proyecto y su uso.
- **`requirements.txt`**: Archivo que lista las dependencias necesarias para ejecutar el proyecto.
- **`setup.py`**: Configuración para la instalación del proyecto como paquete Python.
- **`.gitignore`**: Archivo que lista los archivos que deben ser ignorados por Git.
- **`LICENSE`**: Archivo que describe la licencia bajo la cual se distribuye el proyecto.

