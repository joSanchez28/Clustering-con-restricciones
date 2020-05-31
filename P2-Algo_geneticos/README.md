# Metaheurísticas UGR 2020 - Práctica 2.b, Problema de Agrupamiento con Restricciones (PAR)

Segunda práctica del curso Metaheurísticas 2020 de la UGR; se trata del Problema de Agrupamiento con Restricciones. En este documento, puedes ver una vista general del proyecto y cómo ejecutarlo, el entorno usado, los paquetes necesarios y sus versiones.


## Carpetas del proyecto
Estructura de carpetas y archivos:


* `memoriaP2.pdf` Memoria final. Incluye la parte desarrollada con latex y la parte de análisis de resultados (exportada desde una de las Jupyter Notebooks).
* `Análisis_resultados.html` Parte de la memoria que se centra en el análisis de resultados presentada en formato HTML (se muestran las tablas de forma más elegante).
* `Software/` Archivos de código en Python
    * `Ejecución_y_resultados.ipynb` La notebook de ejecución principal donde encontrarás la lectura de los datos, la ejecución de los algoritmos y la obtención de las tablas.
    * `Análisis_resultados.ipynb` Notebook usada para analizar los resultados y realizar varias visualizaciones; se adjunta también en pdf como parte de la memoria.
    * `algoritmos.py` Fichero de código donde encontrás las funciones que definen todos los algoritmos implementados y usados (ejecutados en `Ejecución_y_resultados.ipynb`)
    * `funciones_auxiliares_y_estadisticos.py` Fichero de código donde encontraras las funciones comunes a todos los algoritmos (los estadísticos, la función objetivo, alguna función para la visualización de los resultados,...).
    * `leer_datos.py`. Fichero donde se definen las funciones dedicadas a la lectura y carga de los conjuntos de datos
* `Software/data/`. Directorio con los conjuntos de datos a usar.
* `docs/`. Directorio con la parte de memoria en código en latex.
* `Results/` Directorio que contiene los resultados (tablas) obtenidos en la notebook `Ejecución_y_resultados.ipynb`, tanto en formato .npy como en Excel.

## Entorno y paquetes
Para el desarrollo del proyecto se ha trabajado con [Anaconda](https://www.anaconda.com/download/); en concreto con Python y Jupyter notebook. Un ordenador con una versión instalada de Python 3 será requerido. Específicamente, se ha usado la version Python 3.7.3. Para instalarla, puedes ejecutar en anaconda prompt:
* `conda install python=3.7.3`

Los paquetes NumPy (1.16.2) y Matplotlib (3.0.3) son usados, los puedes obtener con la herramienta pip:
* `pip3 install numpy`
* `pip3 install matplotlib`

Los siguientes paquetes son también requeridos para ejecutar todo el código:
- [seaborn](https://seaborn.pydata.org/) (0.9.0) para mejorar las visualizaciones.
- [pandas](https://pandas.pydata.org/) (0.24.2) para tratar con los resultados, crear las tablas y trabajar con ellas.
Los puedes instalar con conda:
* `conda install -c anaconda seaborn`
* `conda install -c anaconda pandas`

## Ejecutando el proyecto
### Notebooks
Basta con abrir y ejecutar la notebook principal `Ejecución_y_resultados.ipynb` con Jupyter Notebook. También puede ejecutar la notebook `Análisis_resultados.ipynb`, que carga directamente los resultados obtenidos de las ejecuciones en `Ejecución_y_resultados.ipynb`. Los demás ficheros no han sido desarrollados para ser ejecutados (solo para definir funciones que usaremos en las Notebooks).
