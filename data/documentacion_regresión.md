<<<<<<< HEAD
# Documentación: job_salary_prediction_dataset.csv

**a) Nombre de la base de datos**
Job Salary Prediction Dataset

**b) Fuente (URL)**
https://www.kaggle.com/datasets/nalisha/job-salary-prediction-dataset .

**c) Descripción general del problema**
En el mercado laboral actual, la compensación económica de un profesional está influenciada por múltiples factores, desde su nivel educativo y experiencia hasta el tamaño de la empresa y la modalidad de trabajo. El problema para los reclutadores y analistas de recursos humanos es poder clasificar y estructurar de manera justa en qué "banda salarial" debería ubicarse un candidato en función de su perfil competitivo.

**d) Objetivo del análisis**
Desarrollar un modelo de clasificación multiclase que permita predecir a qué categoría o banda salarial (ej. *Bajo, Medio, Alto*) pertenece un profesional basándose en sus habilidades, años de experiencia, industria y nivel educativo.

**e) Variable objetivo (variable respuesta)**
`categoria_salarial` (Variable derivada).
*(Nota: Dado que este es un problema de clasificación y la columna original `salary` es continua, se deberá crear esta variable agrupando los salarios en categorías/cuartiles como 'Junior/Bajo', 'Mid/Medio', 'Senior/Alto'). 

**f) Diccionario de variables**

* **Nombre de la variable:** `job_title`
    * **Descripción:** Cargo o rol del empleado (ej. AI Engineer, Data Analyst).
    * **Tipo de variable:** Categórica nominal.
* **Nombre de la variable:** `experience_years`
    * **Descripción:** Cantidad de años de experiencia profesional.
    * **Tipo de variable:** Numérica discreta.
* **Nombre de la variable:** `education_level`
    * **Descripción:** Máximo grado académico alcanzado (ej. Bachelor, PhD).
    * **Tipo de variable:** Categórica ordinal.
* **Nombre de la variable:** `skills_count`
    * **Descripción:** Número de habilidades técnicas reportadas en el perfil.
    * **Tipo de variable:** Numérica discreta.
* **Nombre de la variable:** `industry` / `company_size` / `location`
    * **Descripción:** Sector económico, tamaño de la empresa (Small, Medium, Large) y país donde opera el profesional.
    * **Tipo de variable:** Categórica nominal (`industry`, `location`) y Categórica ordinal (`company_size`).
* **Nombre de la variable:** `remote_work`
    * **Descripción:** Indica si el trabajo es remoto, presencial o híbrido.
    * **Tipo de variable:** Categórica nominal.
* **Nombre de la variable:** `certifications`
    * **Descripción:** Cantidad de certificaciones oficiales obtenidas.
    * **Tipo de variable:** Numérica discreta.
* **Nombre de la variable:** `salary`
    * **Descripción:** Salario anual continuo (se utilizará para derivar la clase objetivo).
    * **Tipo de variable:** Numérica continua.

**g) Número de observaciones**
250,000 registros.

**h) Número de variables**
10 variables.

**i) Posibles hipótesis de estudio**

1.  **Hipótesis de especialización:** Los profesionales con un nivel educativo de "PhD" y un `skills_count` superior a 10 tienen una probabilidad significativamente mayor de pertenecer a la categoría salarial más alta (Senior/Alto).
=======
# Documentación: job_salary_prediction_dataset.csv

**a) Nombre de la base de datos**
Job Salary Prediction Dataset

**b) Fuente (URL)**
https://www.kaggle.com/datasets/nalisha/job-salary-prediction-dataset .

**c) Descripción general del problema**
En el mercado laboral actual, la compensación económica de un profesional está influenciada por múltiples factores, desde su nivel educativo y experiencia hasta el tamaño de la empresa y la modalidad de trabajo. El problema para los reclutadores y analistas de recursos humanos es poder clasificar y estructurar de manera justa en qué "banda salarial" debería ubicarse un candidato en función de su perfil competitivo.

**d) Objetivo del análisis**
Desarrollar un modelo de clasificación multiclase que permita predecir a qué categoría o banda salarial (ej. *Bajo, Medio, Alto*) pertenece un profesional basándose en sus habilidades, años de experiencia, industria y nivel educativo.

**e) Variable objetivo (variable respuesta)**
`categoria_salarial` (Variable derivada).
*(Nota: Dado que este es un problema de clasificación y la columna original `salary` es continua, se deberá crear esta variable agrupando los salarios en categorías/cuartiles como 'Junior/Bajo', 'Mid/Medio', 'Senior/Alto'). 

**f) Diccionario de variables**

* **Nombre de la variable:** `job_title`
    * **Descripción:** Cargo o rol del empleado (ej. AI Engineer, Data Analyst).
    * **Tipo de variable:** Categórica nominal.
* **Nombre de la variable:** `experience_years`
    * **Descripción:** Cantidad de años de experiencia profesional.
    * **Tipo de variable:** Numérica discreta.
* **Nombre de la variable:** `education_level`
    * **Descripción:** Máximo grado académico alcanzado (ej. Bachelor, PhD).
    * **Tipo de variable:** Categórica ordinal.
* **Nombre de la variable:** `skills_count`
    * **Descripción:** Número de habilidades técnicas reportadas en el perfil.
    * **Tipo de variable:** Numérica discreta.
* **Nombre de la variable:** `industry` / `company_size` / `location`
    * **Descripción:** Sector económico, tamaño de la empresa (Small, Medium, Large) y país donde opera el profesional.
    * **Tipo de variable:** Categórica nominal (`industry`, `location`) y Categórica ordinal (`company_size`).
* **Nombre de la variable:** `remote_work`
    * **Descripción:** Indica si el trabajo es remoto, presencial o híbrido.
    * **Tipo de variable:** Categórica nominal.
* **Nombre de la variable:** `certifications`
    * **Descripción:** Cantidad de certificaciones oficiales obtenidas.
    * **Tipo de variable:** Numérica discreta.
* **Nombre de la variable:** `salary`
    * **Descripción:** Salario anual continuo (se utilizará para derivar la clase objetivo).
    * **Tipo de variable:** Numérica continua.

**g) Número de observaciones**
250,000 registros.

**h) Número de variables**
10 variables.

**i) Posibles hipótesis de estudio**

1.  **Hipótesis de especialización:** Los profesionales con un nivel educativo de "PhD" y un `skills_count` superior a 10 tienen una probabilidad significativamente mayor de pertenecer a la categoría salarial más alta (Senior/Alto).
>>>>>>> ba1180c4c90277ad3eac15f7e9fbaba70ea8b59a
2.  **Hipótesis de tamaño corporativo:** Trabajar en una empresa de tamaño "Large" aumenta estadísticamente la probabilidad de pertenecer a una banda salarial superior frente a empresas "Small", independientemente de los años de experiencia.