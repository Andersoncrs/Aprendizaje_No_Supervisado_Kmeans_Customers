# 🏪 Mall Customers - Aprendizaje No Supervisado - K Means - Clustering Jerárquico

Este proyecto se centra en el análisis de datos de clientes de un centro comercial utilizando técnicas de aprendizaje no supervisado, específicamente K Means y clustering jerárquico. El objetivo es segmentar a los clientes en grupos homogéneos para entender mejor sus comportamientos y características, lo cual puede ser útil para estrategias de marketing y personalización de servicios.

## 📖 Contenido

- [Introducción](#introducción)
- [Análisis Exploratorio de Datos (EDA)](#análisis-exploratorio-de-datos-eda)
  - [Librerías](#librerías)
  - [Ingesta de Datos](#ingesta-de-datos)
  - [Resúmenes de Datos](#resúmenes-de-datos)
  - [Transformación de Características](#transformación-de-características)
  - [Valores Atípicos](#valores-atípicos)
  - [Análisis Univariable](#análisis-univariable)
  - [Análisis Bivariado](#análisis-bivariado)
  - [Análisis Multivariado](#análisis-multivariado)
  - [Selección de Características](#selección-de-características)
  - [Preprocesamiento de Características](#preprocesamiento-de-características)
- [Modelo de Machine Learning No Supervisado - K-means](#modelo-de-machine-learning-no-supervisado---k-means)
  - [Generación del Modelo](#generación-del-modelo)
  - [Evaluación del Modelo K-Means](#evaluación-del-modelo-k-means)
- [Conclusión](#conclusión)
- [Contacto](#contacto)
- [Licencia](#licencia)

## 🖋️ Introducción

En este ejercicio, nos enfocamos en el aprendizaje no supervisado utilizando técnicas como K Means y clustering jerárquico para analizar datos relacionados con clientes de un centro comercial. El objetivo principal es identificar grupos homogéneos de clientes y entender sus características y comportamientos. Este tipo de análisis puede ser muy útil para diseñar estrategias de marketing personalizado, mejorar la satisfacción del cliente, y aumentar la fidelidad del cliente.

## 🕵️ Análisis Exploratorio de Datos (EDA)

### 📚 Librerías

Para llevar a cabo el análisis y modelado de datos, utilizamos un conjunto de librerías de Python que son esenciales para la manipulación de datos, visualización y aplicación de algoritmos de machine learning:

- **pandas**: Utilizada para la manipulación y análisis de datos.
- **numpy**: Utilizada para operaciones matemáticas y manejo de arreglos.
- **matplotlib y seaborn**: Utilizadas para la visualización de datos.
- **scipy**: Utilizada para estadística y análisis jerárquico.
- **scikit-learn**: Utilizada para la implementación de algoritmos de machine learning.
- **tabulate**: Utilizada para la tabulación de datos en la salida.
- **warnings**: Utilizada para el manejo de advertencias en el código.

### 📄 Ingesta de Datos

Cargamos los datos del archivo `Mall_Customers.csv` y los mostramos en pantalla para entender su estructura y contenido. Este paso es crucial para familiarizarnos con el conjunto de datos con el que trabajaremos, asegurándonos de que entendemos cada columna y su significado.

### 🔮 Resúmenes de Datos

Realizamos resúmenes descriptivos del conjunto de datos para iniciar la comprensión del mismo. Esto incluye la visualización de las primeras filas del dataframe, el tipo de datos de cada columna, y la identificación de valores nulos o faltantes.

### ⚒️ Transformación de Características

En esta etapa, analizamos algunas variables y realizamos transformaciones que podrían beneficiar su análisis posterior y la generación de modelos. Por ejemplo, eliminamos la columna `ID` ya que no proporciona información relevante para el análisis de clustering.

### 🔎 Valores Atípicos

Generamos diagramas de caja para visualizar la distribución de cada variable y detectar la presencia de valores atípicos o extremos. Aunque se identificaron algunos valores atípicos, se consideraron mediciones legítimas y no se eliminaron.

### 🔭 Análisis Univariable

Realizamos un análisis univariable de cada variable en nuestro conjunto de datos para comprender sus estadísticos descriptivos más relevantes y la distribución de cada una. Esto nos ayuda a identificar patrones iniciales y características destacadas de los datos.

### 🔬 Análisis Bivariado

Exploramos las relaciones entre la variable `Genero` y las variables numéricas mediante diagramas de caja. Esto nos proporciona una comprensión más profunda de cómo se distribuyen las variables numéricas en función del género.

### 🩺 Análisis Multivariado

Exploramos las relaciones entre las variables numéricas mediante diagramas de dispersión. Esto nos ayuda a identificar posibles relaciones o patrones entre múltiples variables.

### ✔️ Selección de Características

Seleccionamos las variables `Edad`, `Ingresos` y `Puntuacion_de_gasto` para el análisis, eliminando la variable `Genero` ya que no se considera relevante para el análisis de clustering.

### 📜Preprocesamiento de Características

Estandarizamos las características utilizando `StandardScaler` para asegurarnos de que todas las características están en la misma escala, lo cual es importante para evitar sesgos en los algoritmos de clustering.

## 🤖 Modelo de Machine Learning No Supervisado - K-means

### 🪄 Generación del Modelo

El modelo k-means es un algoritmo de agrupamiento o clustering utilizado en el análisis de datos y aprendizaje automático no supervisado. Su objetivo principal es dividir un conjunto de datos en k grupos según su similitud. En esta etapa, creamos y entrenamos varios modelos K-means con diferentes números de clusters (de 1 a 10) y utilizamos la gráfica de codo para determinar el número óptimo de clusters, que resultó ser 6.

### 📈 Evaluación del Modelo K-Means

Una vez generado el modelo K-Means, procedemos a evaluar su calidad de ajuste utilizando varias métricas:
- **Coeficiente de Silhouette**: Mide qué tan similares son los objetos dentro de un mismo cluster en comparación con objetos de otros clusters.
- **Índice de Calinski-Harabasz**: Mide la relación entre la suma de la dispersión dentro de los clusters y la dispersión entre los clusters.
- **Inercia**: Sumatoria de las distancias cuadráticas dentro de cada cluster.
- **Índice de Davies-Bouldin**: Mide la media de la relación entre la dispersión dentro del cluster y la distancia entre clusters.

Estas métricas nos permiten evaluar la calidad de los clusters generados y ajustar el modelo si es necesario.

## 🏆 Conclusión

Este análisis nos permitió segmentar a los clientes del centro comercial en grupos homogéneos utilizando técnicas de aprendizaje no supervisado. Los resultados obtenidos pueden ser útiles para diseñar estrategias de marketing y personalización de servicios para diferentes segmentos de clientes. La identificación de grupos específicos de clientes puede ayudar a mejorar la satisfacción del cliente y aumentar su fidelidad.

## ✍️ Contacto
Si tienes alguna pregunta o sugerencia, no dudes en contactarme a traves de los siguientes canales:

Linkedin: [Anderson Rodríguez](https://www.linkedin.com/in/andersoncrs)

Email: andersoncamilo.rodriguez.s@gmail.com

## 📓 Licencia

Este proyecto está bajo la licencia Apache 2.0. Ver el archivo [LICENSE](LICENSE) para más detalles.