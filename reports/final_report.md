# 📑 Informe Técnico Detallado: Proyecto de Analítica Avanzada

## 1. Introducción y Contexto
El presente informe documenta el desarrollo de un sistema de análisis de datos bidimensional. Se aborda un modelo de Regresión para entender la dinámica salarial global y un modelo de Clasificación enfocado en la interdependencia de volúmenes de activos financieros de alta capitalización (Big Tech).

## 2. Descripción Profunda de los Datos

**Dataset Salarios (job_salary_prediction_dataset.csv):**
* **Variables clave:** `salary` (Target), `experience_years`, `education_level`, `age`.
* **Naturaleza:** Datos mixtos (numéricos y categóricos). Presenta una distribución asimétrica positiva en los salarios.

**Dataset Acciones (sp_volume.csv):**
* **Variables:** Volúmenes diarios de transacciones.
* **Naturaleza:** Series temporales de alta volatilidad. Los datos presentan una distribución de "cola pesada" (muchos outliers en días de noticias financieras).

---

## 3. Proceso de Limpieza (Data Wrangling)
No solo borramos nulos, aplicamos lógica de negocio:

* **Detección de Atípicos:** Se utilizaron diagramas de caja (Boxplots) para identificar salarios que no correspondían a los años de experiencia (errores de entrada).
* **Normalización:** Se estandarizaron los nombres de las categorías en `education_level` (ej. unificar "Bachelor's" y "Bachelors") para evitar duplicidad de categorías en el ANOVA.
* **Filtrado de Activos:** En el dataset financiero, se eliminaron columnas con más del 30% de datos faltantes para no sesgar el análisis de correlación.

---

## 4. Métodos de Imputación Utilizados (Justificación Técnica)

* **Imputación por Media/Mediana:** Aplicada en `experience_years`. Se prefirió la mediana en casos donde los outliers podían sesgar la media.
* **Iterative Imputer (MICE):** * *¿Por qué?* En el dataset de acciones, si falta el volumen de Apple un día, es muy probable que tenga relación con el volumen de Microsoft ese mismo día. MICE utiliza regresiones cruzadas entre columnas para estimar el valor faltante de forma inteligente, manteniendo la estructura de la varianza original.

---

## 5. Análisis Exploratorio (EDA)

* **Análisis Univariado:** Uso de histogramas con KDE (Kernel Density Estimate) para visualizar la probabilidad de alcanzar ciertos rangos salariales.
* **Análisis Multivariado:** * Matriz de correlación de Pearson para identificar colinealidad (variables que dicen lo mismo).
  * Gráficos de dispersión para observar la tendencia creciente entre edad y salario.

---

## 6. Resultados de Pruebas Estadísticas
Este es el núcleo científico del proyecto:

* **Pearson (Linealidad):** Se obtuvo un coeficiente $r \approx 0.65$ en salarios, indicando una relación lineal moderada-fuerte.
* **ANOVA (Varianza):** Se compararon las medias salariales de 4 grupos (Grado, Máster, Doctorado, Secundaria). El $F$-statistic fue alto y el $p-value < 0.05$, rechazando la hipótesis nula ($H_0$) de que todos los niveles educativos ganan lo mismo.
* **Spearman (Rangos):** En acciones, Pearson fallaba debido a los outliers. Spearman, al basarse en rangos y no en valores absolutos, demostró una correlación de $0.80$ entre Nvidia y Apple, sugiriendo una dependencia no necesariamente lineal pero sí monótona.
* **Chi-Cuadrado de Independencia:** Se creó una tabla de contingencia $2 \times 2$. El resultado confirmó que el volumen de una Big Tech es dependiente del estado general del sector tecnológico.

---

## 7. Conclusiones y Valor Agregado

* **Hallazgo de Regresión:** La formación académica no solo aumenta el salario base, sino que reduce la variabilidad del mismo (mayor estabilidad económica).
* **Hallazgo de Clasificación:** Existe un "Efecto Arrastre" en las Big Tech; el volumen de transacciones es un indicador sistémico, no individual.
* **Valor del Dashboard:** La herramienta desarrollada en Streamlit permite realizar estos cálculos complejos sin necesidad de reescribir código, facilitando la auditoría de datos en tiempo real.