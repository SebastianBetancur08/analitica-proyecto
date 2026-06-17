"""
🎯 Aplicación de Predicción — Salarios (Regresión) y Renuncia de Empleados (Clasificación)

Permite usar los modelos finales del proyecto para predecir:
  · Salario estimado de un perfil profesional  → LightGBM (regresión)
  · Probabilidad de renuncia de un empleado     → Regresión Logística tuned (clasificación)

Funcionalidades:
  · Predicción individual mediante formulario (con carga de ejemplos reales)
  · Rango de error en la predicción de salario (basado en el RMSE/MAE del modelo)
  · Probabilidad de renuncia con velocímetro y umbral de decisión ajustable
  · Importancia de variables (coeficientes / odds ratio del modelo logístico)
  · Predicción por lotes subiendo un CSV (con umbral ajustable y aviso de categorías desconocidas)
  · Historial de predicciones de la sesión
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────
#  Configuración y rutas
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Predicción ML — Salarios & Renuncia",
                   page_icon="🎯", layout="wide")

RAIZ = Path(__file__).resolve().parent.parent
MODELS_DIR = RAIZ / "models"
DATA_DIR = RAIZ / "data" / "raw"

MODELO_REGRESION = MODELS_DIR / "model_lightgbm.joblib"
META_REGRESION = MODELS_DIR / "model_lightgbm_meta.json"
MODELO_CLASIF = MODELS_DIR / "model_regresion_logistica.joblib"
META_CLASIF = MODELS_DIR / "model_regresion_logistica_meta.json"

DATA_REGRESION = DATA_DIR / "job_salary_prediction_dataset.csv"
DATA_CLASIF = DATA_DIR / "dataset_clasificacion.csv"


# ──────────────────────────────────────────────────────────────────────────
#  Carga de recursos (cacheada)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo(ruta):
    return joblib.load(ruta)


@st.cache_data
def cargar_meta(ruta):
    if Path(ruta).exists():
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def cargar_datos_regresion():
    return pd.read_csv(DATA_REGRESION)


@st.cache_data
def cargar_datos_clasif():
    df = pd.read_csv(DATA_CLASIF)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"],
                 errors="ignore")
    return df


# Esquemas de features
REG_CAT = ["job_title", "education_level", "industry", "company_size", "location", "remote_work"]
REG_NUM = ["experience_years", "skills_count", "certifications"]
CLF_CAT = ["BusinessTravel", "Department", "EducationField", "Gender",
           "JobRole", "MaritalStatus", "OverTime"]


def opciones_categoricas(df, columnas):
    return {c: sorted(df[c].dropna().unique().tolist()) for c in columnas}


def rango_numerico(df, columnas):
    return {c: (float(df[c].min()), float(df[c].max()), float(df[c].median()))
            for c in columnas}


def registrar_historial(tipo, entrada, resultado):
    """Guarda una predicción en el historial de la sesión."""
    if "historial" not in st.session_state:
        st.session_state.historial = []
    fila = {"Tipo": tipo, "Resultado": resultado, **entrada}
    st.session_state.historial.insert(0, fila)
    st.session_state.historial = st.session_state.historial[:50]


def gauge_probabilidad(proba, umbral):
    """Velocímetro semicircular para la probabilidad de renuncia."""
    fig, ax = plt.subplots(figsize=(6, 3.4), subplot_kw={"aspect": "equal"})
    # Zonas de color (verde / amarillo / rojo)
    zonas = [(0.0, 0.33, "#2ecc71"), (0.33, 0.66, "#f1c40f"), (0.66, 1.0, "#e74c3c")]
    for ini, fin, color in zonas:
        ang_ini = 180 * (1 - fin)
        ang_fin = 180 * (1 - ini)
        ax.add_patch(plt.matplotlib.patches.Wedge(
            (0, 0), 1.0, ang_ini, ang_fin, width=0.35, facecolor=color, alpha=0.85))
    # Aguja
    ang = np.radians(180 * (1 - proba))
    ax.plot([0, 0.78 * np.cos(ang)], [0, 0.78 * np.sin(ang)],
            color="#2c3e50", lw=4, solid_capstyle="round")
    ax.add_patch(plt.Circle((0, 0), 0.05, color="#2c3e50"))
    # Marca del umbral
    ang_u = np.radians(180 * (1 - umbral))
    ax.plot([0.65 * np.cos(ang_u), 1.0 * np.cos(ang_u)],
            [0.65 * np.sin(ang_u), 1.0 * np.sin(ang_u)],
            color="black", lw=2, ls="--")
    ax.text(0, -0.18, f"{proba*100:.1f} %", ha="center", va="center",
            fontsize=26, fontweight="bold", color="#2c3e50")
    ax.text(0, -0.40, f"umbral τ = {umbral:.0%}", ha="center", va="center",
            fontsize=10, color="#555")
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.5, 1.15)
    ax.axis("off")
    return fig


# ──────────────────────────────────────────────────────────────────────────
#  Navegación
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.title("🎯 Navegación")
pagina = st.sidebar.radio(
    "Selecciona una sección:",
    ["🏠 Inicio",
     "💰 Predecir Salario",
     "🚪 Predecir Renuncia",
     "📦 Predicción por lotes (CSV)",
     "🕑 Historial"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Modelos finales del proyecto:")
st.sidebar.caption("· Regresión → **LightGBM**")
st.sidebar.caption("· Clasificación → **Regresión Logística (tuned)**")


# ══════════════════════════════════════════════════════════════════════════
#  PÁGINA: INICIO
# ══════════════════════════════════════════════════════════════════════════
if pagina == "🏠 Inicio":
    st.title("🎯 Aplicación de Predicción con Modelos de Machine Learning")
    st.markdown("""
    Bienvenido. Esta herramienta permite usar los **modelos finales** entrenados
    en el proyecto para realizar predicciones sobre nuevos datos.
    """)

    meta_r = cargar_meta(META_REGRESION)
    meta_c = cargar_meta(META_CLASIF)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("💰 Regresión — Salario")
        st.markdown("""
        Predice el **salario estimado** de un perfil profesional a partir de su
        experiencia, formación, industria, ubicación y habilidades.

        - **Modelo:** LightGBM Regressor
        - **Target:** `salary` (USD)
        """)
        if meta_r:
            m = meta_r["metricas_test"]
            st.caption(f"R² = {m['r2']:.3f} · RMSE = ${m['rmse']:,.0f} · MAE = ${m['mae']:,.0f}")
    with col2:
        st.subheader("🚪 Clasificación — Renuncia")
        st.markdown("""
        Estima la **probabilidad de que un empleado renuncie** (attrition) a partir
        de su perfil laboral y de satisfacción.

        - **Modelo:** Regresión Logística (ajustada)
        - **Target:** `Attrition` (Yes / No)
        """)
        if meta_c:
            m = meta_c["metricas_cv_tuned"]
            st.caption(f"F1 = {m['f1']['mu']:.3f} · AUC = {m['roc_auc']['mu']:.3f} "
                       f"· Recall = {m['recall']['mu']:.3f}")

    st.markdown("---")
    st.info("👈 Usa el menú lateral para elegir qué quieres predecir. "
            "También puedes predecir **muchos registros a la vez** subiendo un CSV.")

    st.subheader("Estado de los modelos")
    estado = []
    for nombre, ruta in [("Regresión (LightGBM)", MODELO_REGRESION),
                         ("Clasificación (Reg. Logística)", MODELO_CLASIF)]:
        estado.append({"Modelo": nombre, "Archivo": ruta.name,
                       "Disponible": "✅" if ruta.exists() else "❌ No encontrado"})
    st.dataframe(pd.DataFrame(estado), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
#  PÁGINA: PREDECIR SALARIO (REGRESIÓN)
# ══════════════════════════════════════════════════════════════════════════
elif pagina == "💰 Predecir Salario":
    st.title("💰 Predicción de Salario")
    st.caption("Modelo: LightGBM Regressor")

    if not MODELO_REGRESION.exists():
        st.error("No se encontró el modelo de regresión.")
        st.stop()

    modelo = cargar_modelo(MODELO_REGRESION)
    meta = cargar_meta(META_REGRESION)
    df = cargar_datos_regresion()
    cats = opciones_categoricas(df, REG_CAT)
    nums = rango_numerico(df, REG_NUM)

    # Defaults en sesión
    if "perfil_reg" not in st.session_state:
        st.session_state.perfil_reg = {c: cats[c][0] for c in REG_CAT}
        st.session_state.perfil_reg.update({c: int(nums[c][2]) for c in REG_NUM})

    cE1, cE2 = st.columns([1, 3])
    with cE1:
        if st.button("🎲 Cargar perfil de ejemplo", use_container_width=True):
            fila = df.sample(1).iloc[0]
            st.session_state.perfil_reg = {
                **{c: fila[c] for c in REG_CAT},
                **{c: int(fila[c]) for c in REG_NUM}}
            st.rerun()
    d = st.session_state.perfil_reg

    with st.form("form_salario"):
        c1, c2, c3 = st.columns(3)
        with c1:
            job_title = st.selectbox("Puesto (job_title)", cats["job_title"],
                                     index=cats["job_title"].index(d["job_title"]))
            education_level = st.selectbox("Nivel educativo", cats["education_level"],
                                           index=cats["education_level"].index(d["education_level"]))
        with c2:
            industry = st.selectbox("Industria", cats["industry"],
                                    index=cats["industry"].index(d["industry"]))
            company_size = st.selectbox("Tamaño de empresa", cats["company_size"],
                                        index=cats["company_size"].index(d["company_size"]))
        with c3:
            location = st.selectbox("Ubicación", cats["location"],
                                    index=cats["location"].index(d["location"]))
            remote_work = st.selectbox("Trabajo remoto", cats["remote_work"],
                                       index=cats["remote_work"].index(d["remote_work"]))

        st.markdown("**Variables numéricas**")
        n1, n2, n3 = st.columns(3)
        with n1:
            experience_years = st.slider("Años de experiencia", int(nums["experience_years"][0]),
                                         int(nums["experience_years"][1]), int(d["experience_years"]))
        with n2:
            skills_count = st.slider("N.º de habilidades", int(nums["skills_count"][0]),
                                     int(nums["skills_count"][1]), int(d["skills_count"]))
        with n3:
            certifications = st.slider("N.º de certificaciones", int(nums["certifications"][0]),
                                       int(nums["certifications"][1]), int(d["certifications"]))

        enviar = st.form_submit_button("💰 Predecir salario", use_container_width=True)

    if enviar:
        entrada = {
            "job_title": job_title, "experience_years": experience_years,
            "education_level": education_level, "skills_count": skills_count,
            "industry": industry, "company_size": company_size,
            "location": location, "remote_work": remote_work,
            "certifications": certifications,
        }
        X_in = pd.DataFrame([entrada])
        pred = float(modelo.predict(X_in)[0])

        st.markdown("---")
        # Resumen del perfil
        with st.expander("📋 Perfil ingresado", expanded=False):
            st.dataframe(pd.DataFrame([entrada]), use_container_width=True, hide_index=True)

        # Rango de error usando el MAE/RMSE del modelo
        rmse = meta.get("metricas_test", {}).get("rmse")
        mae = meta.get("metricas_test", {}).get("mae")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.metric("Salario anual estimado", f"$ {pred:,.0f} USD")
        with c2:
            if mae:
                st.metric("Rango típico (± MAE)",
                          f"$ {pred-mae:,.0f}  —  $ {pred+mae:,.0f}")
        if rmse:
            st.caption(f"El error medio del modelo es ~${mae:,.0f} (MAE) / ±${rmse:,.0f} (RMSE). "
                       f"La predicción puntual es la estimación central; el rango refleja la "
                       f"incertidumbre típica del modelo.")

        sal = df["salary"].dropna()
        pct = (sal < pred).mean() * 100
        st.caption(f"Este salario supera al **{pct:.0f}%** de los registros del dataset "
                   f"(mediana: $ {sal.median():,.0f}).")

        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.hist(sal, bins=50, color="#bdc3c7", edgecolor="white")
        if rmse:
            ax.axvspan(pred - rmse, pred + rmse, color="#27ae60", alpha=0.15,
                       label=f"± RMSE (${rmse:,.0f})")
        ax.axvline(pred, color="#27ae60", lw=2.5, label=f"Predicción: ${pred:,.0f}")
        ax.axvline(sal.median(), color="#34495e", ls="--", lw=1.5,
                   label=f"Mediana: ${sal.median():,.0f}")
        ax.set_xlabel("Salario (USD)"); ax.set_ylabel("Frecuencia")
        ax.legend(); ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

        # Descargar resultado individual
        res = {**entrada, "salario_estimado": round(pred, 2)}
        if mae:
            res["rango_min"] = round(pred - mae, 2)
            res["rango_max"] = round(pred + mae, 2)
        st.download_button("⬇️ Descargar esta predicción",
                           pd.DataFrame([res]).to_csv(index=False).encode("utf-8"),
                           file_name="prediccion_salario.csv", mime="text/csv")
        registrar_historial("Salario", entrada, f"${pred:,.0f}")


# ══════════════════════════════════════════════════════════════════════════
#  PÁGINA: PREDECIR RENUNCIA (CLASIFICACIÓN)
# ══════════════════════════════════════════════════════════════════════════
elif pagina == "🚪 Predecir Renuncia":
    st.title("🚪 Predicción de Renuncia (Attrition)")
    st.caption("Modelo: Regresión Logística ajustada")

    if not MODELO_CLASIF.exists():
        st.error("No se encontró el modelo de clasificación. "
                 "Ejecuta `python src/train_logistica.py` primero.")
        st.stop()

    modelo = cargar_modelo(MODELO_CLASIF)
    meta = cargar_meta(META_CLASIF)
    df = cargar_datos_clasif()
    CLF_NUM = meta["features_numericas"]
    cats = opciones_categoricas(df, CLF_CAT)
    nums = rango_numerico(df, CLF_NUM)
    tau_f1 = meta["umbrales"]["tau_f1"]

    # Umbral ajustable
    st.markdown("#### ⚙️ Umbral de decisión")
    cu1, cu2 = st.columns([3, 1])
    with cu1:
        umbral = st.slider("Umbral τ — probabilidad mínima para clasificar como 'renuncia'",
                           0.0, 1.0, float(round(tau_f1, 2)), 0.01)
    with cu2:
        st.metric("τ óptimo (máx F1)", f"{tau_f1:.3f}")
    st.caption("Bajar el umbral detecta más renuncias (↑ recall) pero genera más falsas alarmas. "
               "Subirlo es más conservador (↑ precisión).")

    st.markdown("---")

    # Defaults en sesión
    if "perfil_clf" not in st.session_state:
        st.session_state.perfil_clf = {c: cats[c][0] for c in CLF_CAT}
        st.session_state.perfil_clf.update({c: nums[c][2] for c in CLF_NUM})

    cE1, cE2 = st.columns([1, 3])
    with cE1:
        if st.button("🎲 Cargar empleado de ejemplo", use_container_width=True):
            fila = df.drop(columns=["Attrition"], errors="ignore").sample(1).iloc[0]
            st.session_state.perfil_clf = {
                **{c: fila[c] for c in CLF_CAT},
                **{c: float(fila[c]) for c in CLF_NUM}}
            st.rerun()
    d = st.session_state.perfil_clf

    with st.form("form_renuncia"):
        st.markdown("**Información categórica**")
        cols_cat = st.columns(3)
        entrada = {}
        for i, c in enumerate(CLF_CAT):
            with cols_cat[i % 3]:
                entrada[c] = st.selectbox(c, cats[c], index=cats[c].index(d[c]))

        st.markdown("**Información numérica**")
        cols_num = st.columns(4)
        for i, c in enumerate(CLF_NUM):
            mn, mx, _ = nums[c]
            with cols_num[i % 4]:
                entrada[c] = st.number_input(c, min_value=mn, max_value=mx,
                                             value=float(d[c]), step=1.0)

        enviar = st.form_submit_button("🚪 Predecir renuncia", use_container_width=True)

    if enviar:
        X_in = pd.DataFrame([entrada])
        proba = float(modelo.predict_proba(X_in)[0, 1])
        pred = int(proba >= umbral)

        st.markdown("---")
        with st.expander("📋 Perfil ingresado", expanded=False):
            st.dataframe(pd.DataFrame([entrada]), use_container_width=True, hide_index=True)

        r1, r2 = st.columns([1, 1])
        with r1:
            st.pyplot(gauge_probabilidad(proba, umbral))
        with r2:
            if pred == 1:
                st.error(f"### ⚠️ RIESGO DE RENUNCIA\nProb {proba*100:.1f}% ≥ umbral {umbral:.0%}")
            else:
                st.success(f"### ✅ Probablemente PERMANECE\nProb {proba*100:.1f}% < umbral {umbral:.0%}")

        # Importancia de variables para esta predicción
        st.markdown("#### 🔍 Variables que más influyen en esta predicción")
        try:
            prep = modelo.named_steps["preprocesamiento"]
            clf = modelo.named_steps["clf"]
            nombres = prep.get_feature_names_out()
            coefs = clf.coef_[0]
            x_trans = prep.transform(X_in)
            if hasattr(x_trans, "toarray"):
                x_trans = x_trans.toarray()
            contrib = coefs * x_trans.ravel()

            df_c = pd.DataFrame({"Variable": nombres, "Contribución": contrib})
            df_c = df_c[df_c["Contribución"].abs() > 1e-6]
            df_c = df_c.reindex(df_c["Contribución"].abs().sort_values(ascending=False).index)
            df_c = df_c.head(12).sort_values("Contribución")

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            colores = ["#e74c3c" if v > 0 else "#3498db" for v in df_c["Contribución"]]
            ax2.barh(df_c["Variable"], df_c["Contribución"], color=colores, edgecolor="white")
            ax2.axvline(0, color="black", lw=0.8)
            ax2.set_xlabel("Contribución al log-odds de renuncia (β × valor)")
            ax2.set_title("Rojo = empuja hacia renuncia · Azul = empuja hacia permanecer")
            ax2.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig2)
            st.caption("Efecto de cada variable **para este empleado concreto** "
                       "(coeficiente del modelo × valor estandarizado).")
        except Exception as e:
            st.warning(f"No se pudo calcular la importancia de variables: {e}")

        res = {**entrada, "prob_renuncia": round(proba, 4),
               "prediccion": "Renuncia" if pred else "Permanece", "umbral": umbral}
        st.download_button("⬇️ Descargar esta predicción",
                           pd.DataFrame([res]).to_csv(index=False).encode("utf-8"),
                           file_name="prediccion_renuncia.csv", mime="text/csv")
        registrar_historial("Renuncia", entrada,
                            f"{proba*100:.1f}% ({'Renuncia' if pred else 'Permanece'})")


# ══════════════════════════════════════════════════════════════════════════
#  PÁGINA: PREDICCIÓN POR LOTES (CSV)
# ══════════════════════════════════════════════════════════════════════════
elif pagina == "📦 Predicción por lotes (CSV)":
    st.title("📦 Predicción por lotes")
    st.markdown("Sube un archivo **CSV** con varios registros y obtén todas las predicciones de una vez.")

    tipo = st.radio("¿Qué quieres predecir?",
                    ["💰 Salario (Regresión)", "🚪 Renuncia (Clasificación)"],
                    horizontal=True)

    if tipo.startswith("💰"):
        ruta_modelo = MODELO_REGRESION
        df_ref = cargar_datos_regresion()
        cols_cat, cols_necesarias = REG_CAT, REG_CAT + REG_NUM
    else:
        ruta_modelo = MODELO_CLASIF
        meta = cargar_meta(META_CLASIF)
        df_ref = cargar_datos_clasif()
        cols_cat = CLF_CAT
        cols_necesarias = CLF_CAT + meta["features_numericas"]
        # Umbral ajustable también en lote (arregla el bug anterior)
        umbral_b = st.slider("Umbral τ para clasificar como 'renuncia'", 0.0, 1.0,
                             float(round(meta["umbrales"]["tau_f1"], 2)), 0.01)

    if not ruta_modelo.exists():
        st.error("El modelo correspondiente no está disponible.")
        st.stop()

    with st.expander("ℹ️ Columnas que debe tener tu CSV"):
        st.write(cols_necesarias)
        plantilla = df_ref[cols_necesarias].head(5).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Descargar plantilla de ejemplo", plantilla,
                           file_name="plantilla.csv", mime="text/csv")

    archivo = st.file_uploader("Sube tu CSV", type=["csv"])

    if archivo is not None:
        try:
            datos = pd.read_csv(archivo)
            datos.columns = datos.columns.str.strip()
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")
            st.stop()

        faltantes = [c for c in cols_necesarias if c not in datos.columns]
        if faltantes:
            st.error(f"Faltan columnas obligatorias: {faltantes}")
            st.stop()

        # Aviso de categorías desconocidas (no vistas en entrenamiento)
        validos = opciones_categoricas(df_ref, cols_cat)
        avisos = []
        for c in cols_cat:
            desconocidas = set(datos[c].dropna().unique()) - set(validos[c])
            if desconocidas:
                avisos.append(f"**{c}**: {sorted(map(str, desconocidas))[:8]}")
        if avisos:
            st.warning("⚠️ Se encontraron valores de categoría no vistos en el entrenamiento "
                       "(el modelo los tratará como desconocidos):\n\n- " + "\n- ".join(avisos))

        modelo = cargar_modelo(ruta_modelo)
        X_in = datos[cols_necesarias].copy()

        with st.spinner("Generando predicciones..."):
            if tipo.startswith("💰"):
                datos["salario_estimado"] = modelo.predict(X_in).round(2)
                resumen = f"Salario medio estimado: $ {datos['salario_estimado'].mean():,.0f}"
            else:
                proba = modelo.predict_proba(X_in)[:, 1]
                datos["prob_renuncia"] = proba.round(4)
                datos["prediccion"] = np.where(proba >= umbral_b, "Renuncia", "Permanece")
                en_riesgo = (datos["prediccion"] == "Renuncia").sum()
                resumen = f"Empleados en riesgo (τ={umbral_b:.2f}): {en_riesgo} de {len(datos)}"

        st.success(f"✅ {len(datos)} registros procesados. {resumen}")
        st.dataframe(datos.head(50), use_container_width=True)

        st.download_button("⬇️ Descargar resultados completos",
                           datos.to_csv(index=False).encode("utf-8"),
                           file_name="predicciones.csv", mime="text/csv",
                           use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
#  PÁGINA: HISTORIAL
# ══════════════════════════════════════════════════════════════════════════
elif pagina == "🕑 Historial":
    st.title("🕑 Historial de predicciones de la sesión")
    hist = st.session_state.get("historial", [])
    if not hist:
        st.info("Aún no has hecho ninguna predicción en esta sesión.")
    else:
        resumen = pd.DataFrame([{"Tipo": h["Tipo"], "Resultado": h["Resultado"]} for h in hist])
        st.dataframe(resumen, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Descargar historial completo",
                           pd.DataFrame(hist).to_csv(index=False).encode("utf-8"),
                           file_name="historial_predicciones.csv", mime="text/csv")
        if st.button("🗑️ Limpiar historial"):
            st.session_state.historial = []
            st.rerun()