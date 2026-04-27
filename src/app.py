import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import pearsonr, chi2_contingency

# Configuración inicial
st.set_page_config(page_title="Dashboard Big Tech", layout="wide")
st.title("📊 Aplicación de Análisis Exploratorio de Datos")

# --- FUNCIÓN DE CARGA ---
@st.cache_data 
def cargar_datos(dataset):
    ruta_src = os.path.dirname(os.path.abspath(__file__))
    ruta_raiz = os.path.dirname(ruta_src)
    
    if dataset == "Regresión (Salarios)":
        nombre = 'job_salary_prediction_dataset.csv'
    else:
        nombre = 'sp_volume.csv'
        
    ruta_final = os.path.join(ruta_raiz, 'data', 'database', nombre)
    return pd.read_csv(ruta_final)

# --- SIDEBAR ---
opcion = st.sidebar.selectbox(
    "Selecciona la base de datos:",
    ("Regresión (Salarios)", "Clasificación (Volumen Acciones)")
)

try:
    df = cargar_datos(opcion)
    # --- FILTRADO ESTRATÉGICO PARA CLASIFICACIÓN ---
    if opcion == "Clasificación (Volumen Acciones)":
        # Lista de las empresas más importantes (Big Tech)
        big_tech = ['date', 'AAPL', 'MSFT', 'NVDA', 'META', 'AMZN', 'GOOG', 'TSLA']
        # Solo nos quedamos con las que existan en el CSV para evitar errores
        columnas_existentes = [col for col in big_tech if col in df.columns]
        df = df[columnas_existentes]
    
    st.sidebar.success(f"✅ {opcion} cargado")
except Exception as e:
    st.error(f"Error: No se encontró el archivo en la ruta esperada.")
    st.stop()

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📈 Estadísticas", "🧼 Limpieza", "🧪 Pruebas"])

with tab1:
    st.subheader("📋 Resumen Descriptivo")
    st.write(df.describe(include='all' if opcion == "Regresión (Salarios)" else 'number'))
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribución Numérica**")
        # Quitamos 'date' de las opciones de gráfica si existe
        cols_graf = df.select_dtypes(include=['number']).columns
        c_num = st.selectbox("Selecciona Variable:", cols_graf)
        fig1, ax1 = plt.subplots()
        sns.histplot(df[c_num].dropna(), kde=True, color="skyblue", ax=ax1)
        st.pyplot(fig1)
    
    with col2:
        if opcion == "Regresión (Salarios)":
            st.write("**Nivel de Formación (Estudios)**")
            if 'education_level' in df.columns:
                fig2, ax2 = plt.subplots()
                df['education_level'].value_counts().plot(kind='bar', color='salmon', ax=ax2)
                plt.xticks(rotation=45)
                st.pyplot(fig2)
        else:
            st.write("**Evolución del Volumen (Time Series)**")
            fig_line, ax_line = plt.subplots()
            # Graficamos Apple y Microsoft como referencia
            sns.lineplot(data=df, x=df.index, y='AAPL', label='Apple', ax=ax_line)
            sns.lineplot(data=df, x=df.index, y='MSFT', label='Microsoft', ax=ax_line)
            plt.title("Comparativa de Volumen Histórico")
            st.pyplot(fig_line)

with tab2:
    st.subheader("🧼 Imputación y Correlaciones")
    st.write(f"Nulos totales encontrados: {df.isnull().sum().sum()}")
    
    st.subheader("🔗 Matriz de Correlación: Gigantes Tecnológicos")
    # Ahora la matriz es pequeña y rápida
    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

with tab3:
    st.subheader("🧪 Pruebas Estadísticas")
    
    if opcion == "Regresión (Salarios)":
        # 1. Pearson
        if 'experience_years' in df.columns and 'salary' in df.columns:
            st.write("#### Pearson: Experiencia vs Salario")
            d_p = df[['experience_years', 'salary']].dropna()
            r, p = stats.pearsonr(d_p['experience_years'], d_p['salary'])
            st.write(f"Coeficiente: **{r:.3f}** | P-value: **{p:.2e}**")

        # 2. ANOVA (Educación)
        st.write("#### ANOVA: Impacto de la Formación")
        if 'education_level' in df.columns and 'salary' in df.columns:
            try:
                d_a = df[['salary', 'education_level']].dropna()
                validos = d_a['education_level'].value_counts()
                d_a = d_a[d_a['education_level'].isin(validos[validos > 2].index)]
                
                model = ols('salary ~ C(education_level)', data=d_a).fit()
                table = sm.stats.anova_lm(model, typ=2)
                st.dataframe(table)
                
                p_anova = table.iloc[0, 3] 
                if p_anova < 0.05:
                    st.success(f"La formación influye en el salario (p={p_anova:.4f})")
                else:
                    st.info("No hay diferencia significativa.")
            except:
                st.warning("No hay suficientes datos categóricos para el ANOVA.")
    
    else:
        st.write("#### Spearman: Dependencia entre Gigantes")
        # Comparamos Apple y Nvidia que suelen estar muy relacionadas
        if 'AAPL' in df.columns and 'NVDA' in df.columns:
            s_r, s_p = stats.spearmanr(df['AAPL'].fillna(0), df['NVDA'].fillna(0))
            st.write(f"Relación entre **Apple** y **NVIDIA**")
            st.write(f"Coeficiente Spearman: **{s_r:.3f}** (p={s_p:.2e})")
            
            st.write("#### Chi-Cuadrado (Dependencia Apple vs Microsoft)")
            m1, m2 = df['AAPL'].median(), df['MSFT'].median()
            tabla = pd.crosstab(df['AAPL'] > m1, df['MSFT'] > m2)
            c2, pc, dof, ex = chi2_contingency(tabla)
            st.write(f"P-value Chi-Cuadrado: **{pc:.4e}**")
            st.info("Esta prueba indica si cuando Apple tiene mucho volumen, Microsoft también suele tenerlo.")