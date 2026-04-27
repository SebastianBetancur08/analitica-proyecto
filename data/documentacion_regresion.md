<<<<<<< HEAD
# Documentación: sp_volume.csv

**a) Nombre de la base de datos**
S&P 500 Historical Trading Volume

**b) Fuente (URL)**
https://www.kaggle.com/datasets/kapakudaibergenov/sp-5001 .

**c) Descripción general del problema**
En los mercados financieros, el volumen de transacciones (cantidad de acciones compradas y vendidas en un día) es un indicador crucial de la liquidez, la convicción detrás de los movimientos de precios y el impacto institucional. Para los analistas cuantitativos ("quants"), predecir de forma precisa cuánto volumen operará una acción en el futuro es vital para diseñar algoritmos de ejecución de órdenes que minimicen los costos de deslizamiento (slippage).

**d) Objetivo del análisis**
Entrenar un modelo de regresión para predecir el volumen de transacciones diario de un activo específico de alta liquidez (por ejemplo, `AAPL` - Apple), utilizando como variables predictoras el volumen transaccional histórico de las demás acciones del índice S&P 500.

**e) Variable objetivo (variable respuesta)**
`AAPL` (u otra acción elegida a criterio del analista). 
*(Representa el volumen diario de transacciones para esa acción en particular).*

**f) Diccionario de variables**
*(Nota: El dataset es una matriz ancha con 505 tickers bursátiles. Aquí se detalla la estructura general).*

* **Nombre de la variable:** `Date`
    * **Descripción:** Fecha de la jornada de negociación en la bolsa.
    * **Tipo de variable:** Categórica ordinal (Temporal).
* **Nombre de la variable:** `AAPL` (Variable Objetivo de ejemplo)
    * **Descripción:** Cantidad total de acciones de Apple transadas en ese día.
    * **Tipo de variable:** Numérica continua.
* **Nombre de la variable:** `MSFT`, `GOOGL`, `AMZN`, `TSLA`, etc. (Resto de columnas)
    * **Descripción:** Volumen de transacciones diario de cada una de las empresas individuales que conforman el índice S&P 500.
    * **Tipo de variable:** Numérica continua.

**g) Número de observaciones**
15,096 jornadas (registros diarios).

**h) Número de variables**
506 variables (1 columna de Fecha + 505 tickers de acciones).

**i) Posibles hipótesis de estudio**

1.  **Hipótesis de correlación sectorial:** El volumen diario de `AAPL` tiene una relación lineal positiva más fuerte con el volumen de empresas de su mismo sector tecnológico (como `MSFT` o `GOOGL`) que con el índice en su totalidad.
2.  **Hipótesis de estacionalidad:** Existe un patrón cíclico donde el volumen general del mercado presenta picos consistentes durante ciertos períodos (ej. temporadas de reportes de ganancias financieras o fines de trimestre).
=======
# Documentación: sp_volume.csv

**a) Nombre de la base de datos**
S&P 500 Historical Trading Volume

**b) Fuente (URL)**
https://www.kaggle.com/datasets/kapakudaibergenov/sp-5001 .

**c) Descripción general del problema**
En los mercados financieros, el volumen de transacciones (cantidad de acciones compradas y vendidas en un día) es un indicador crucial de la liquidez, la convicción detrás de los movimientos de precios y el impacto institucional. Para los analistas cuantitativos ("quants"), predecir de forma precisa cuánto volumen operará una acción en el futuro es vital para diseñar algoritmos de ejecución de órdenes que minimicen los costos de deslizamiento (slippage).

**d) Objetivo del análisis**
Entrenar un modelo de regresión para predecir el volumen de transacciones diario de un activo específico de alta liquidez (por ejemplo, `AAPL` - Apple), utilizando como variables predictoras el volumen transaccional histórico de las demás acciones del índice S&P 500.

**e) Variable objetivo (variable respuesta)**
`AAPL` (u otra acción elegida a criterio del analista). 
*(Representa el volumen diario de transacciones para esa acción en particular).*

**f) Diccionario de variables**
*(Nota: El dataset es una matriz ancha con 505 tickers bursátiles. Aquí se detalla la estructura general).*

* **Nombre de la variable:** `Date`
    * **Descripción:** Fecha de la jornada de negociación en la bolsa.
    * **Tipo de variable:** Categórica ordinal (Temporal).
* **Nombre de la variable:** `AAPL` (Variable Objetivo de ejemplo)
    * **Descripción:** Cantidad total de acciones de Apple transadas en ese día.
    * **Tipo de variable:** Numérica continua.
* **Nombre de la variable:** `MSFT`, `GOOGL`, `AMZN`, `TSLA`, etc. (Resto de columnas)
    * **Descripción:** Volumen de transacciones diario de cada una de las empresas individuales que conforman el índice S&P 500.
    * **Tipo de variable:** Numérica continua.

**g) Número de observaciones**
15,096 jornadas (registros diarios).

**h) Número de variables**
506 variables (1 columna de Fecha + 505 tickers de acciones).

**i) Posibles hipótesis de estudio**

1.  **Hipótesis de correlación sectorial:** El volumen diario de `AAPL` tiene una relación lineal positiva más fuerte con el volumen de empresas de su mismo sector tecnológico (como `MSFT` o `GOOGL`) que con el índice en su totalidad.
2.  **Hipótesis de estacionalidad:** Existe un patrón cíclico donde el volumen general del mercado presenta picos consistentes durante ciertos períodos (ej. temporadas de reportes de ganancias financieras o fines de trimestre).
>>>>>>> ba1180c4c90277ad3eac15f7e9fbaba70ea8b59a
3.  **Hipótesis de efecto derrame (Spillover):** Días con anomalías extremas de volumen (picos por encima de 3 desviaciones estándar) en el top 5 de acciones de mayor capitalización actúan como predictores significativos de un aumento de volumen generalizado para el día posterior (T+1).