"""
Reentrena y guarda el modelo TUNED de Regresión Logística para predicción de
Attrition (renuncia de empleados).

Reproduce exactamente el pipeline del notebook
`notebooks/modelos_clasificacion/1.Regresion_Logistica.ipynb`:
  - Mismo preprocesamiento (OneHotEncoder + StandardScaler)
  - Misma búsqueda de hiperparámetros (RandomizedSearchCV, F1, n_iter=60)
  - Reentrena el mejor estimador sobre todo el train
  - Calcula el umbral óptimo de decisión (máx F1 out-of-fold)

Guarda en models/:
  - model_regresion_logistica.joblib   (Pipeline completo listo para predecir)
  - model_regresion_logistica_meta.json (umbrales, métricas, mejores params)
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_validate, cross_val_predict,
                                     RandomizedSearchCV)
from sklearn.metrics import precision_recall_curve
from scipy.stats import loguniform

RAIZ = Path(__file__).resolve().parent.parent
DATA_PATH = RAIZ / "data" / "raw" / "dataset_clasificacion.csv"
MODELS_DIR = RAIZ / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── 1. Carga y limpieza (idéntico al notebook) ────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])

y = (df["Attrition"] == "Yes").astype(int)
X = df.drop(columns=["Attrition"])

COLS_NOMINALES = ["BusinessTravel", "Department", "EducationField",
                  "Gender", "JobRole", "MaritalStatus", "OverTime"]
COLS_NUMERICAS = [c for c in X.columns if c not in COLS_NOMINALES]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# ── 2. Pipeline base ──────────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), COLS_NOMINALES),
        ("num", StandardScaler(), COLS_NUMERICAS),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

modelo_base = Pipeline(steps=[
    ("preprocesamiento", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=2000,
                               solver="lbfgs", random_state=42)),
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# ── 3. Búsqueda de hiperparámetros (idéntica al notebook) ─────────────────
param_dist = {
    "clf__C": loguniform(1e-4, 1e2),
    "clf__penalty": ["l1", "l2"],
    "clf__solver": ["saga"],
    "clf__class_weight": ["balanced", {0: 1, 1: 3}, {0: 1, 1: 5}, {0: 1, 1: 7}],
}

print("Entrenando RandomizedSearchCV (n_iter=60)... esto puede tardar.")
t0 = time.time()
search = RandomizedSearchCV(
    estimator=modelo_base, param_distributions=param_dist,
    n_iter=60, cv=skf, scoring="f1", return_train_score=True,
    random_state=42, n_jobs=-1, refit=True, verbose=0,
)
search.fit(X_train, y_train)
print(f"Búsqueda completada en {time.time()-t0:.1f}s")
print(f"Mejor F1 (CV): {search.best_score_:.4f}")
print(f"Mejores params: {search.best_params_}")

modelo_tuned = search.best_estimator_

# ── 4. Métricas en validación cruzada del modelo ajustado ─────────────────
cv_tuned = cross_validate(modelo_tuned, X_train, y_train, cv=skf,
                          scoring=scoring, return_train_score=True, n_jobs=-1)

# ── 5. Umbral óptimo (máx F1 out-of-fold sobre el train) ──────────────────
y_proba_cv = cross_val_predict(modelo_tuned, X_train, y_train, cv=skf,
                               method="predict_proba", n_jobs=-1)[:, 1]
prec, rec, thr = precision_recall_curve(y_train, y_proba_cv)
prec_t, rec_t = prec[:-1], rec[:-1]
denom = prec_t + rec_t
f1_vals = np.where(denom > 0, 2 * prec_t * rec_t / denom, 0.0)
tau_f1 = float(thr[np.argmax(f1_vals)])

# ── 6. Reentrenamos sobre todo el train y persistimos ─────────────────────
modelo_tuned.fit(X_train, y_train)

ruta_modelo = MODELS_DIR / "model_regresion_logistica.joblib"
joblib.dump(modelo_tuned, ruta_modelo)

meta = {
    "modelo": "Regresión Logística (tuned)",
    "target": "Attrition (renuncia)",
    "clase_positiva": "Yes (renuncia)",
    "features_nominales": COLS_NOMINALES,
    "features_numericas": COLS_NUMERICAS,
    "best_params": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                        else str(v)) for k, v in search.best_params_.items()},
    "umbrales": {"tau_default": 0.50, "tau_f1": tau_f1},
    "metricas_cv_tuned": {
        m: {"mu": float(cv_tuned[f"test_{m}"].mean()),
            "sigma": float(cv_tuned[f"test_{m}"].std())}
        for m in scoring
    },
}
ruta_meta = MODELS_DIR / "model_regresion_logistica_meta.json"
with open(ruta_meta, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"\n✅ Modelo guardado en: {ruta_modelo}")
print(f"✅ Metadatos guardados en: {ruta_meta}")
print(f"   Umbral óptimo (máx F1): {tau_f1:.3f}")