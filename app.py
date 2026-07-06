"""
Sistema de Predicción de Riesgo de Morosidad
Modelo Random Forest - Universidad Privada Antenor Orrego (UPAO)
Curso: Aprendizaje Estadístico

Esta versión entrena el modelo al iniciar (usando data/data.csv),
replicando el pipeline del notebook, para no depender de archivos .pkl.
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------------------------
# Configuración de la página
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Morosidad",
    page_icon="💳",
    layout="centered",
)

# ----------------------------------------------------------------------
# Entrenamiento del modelo (cacheado: solo se ejecuta una vez)
# ----------------------------------------------------------------------
@st.cache_resource
def entrenar_modelo():
    # 1. Cargar dataset
    df = pd.read_csv("data/data.csv")

    # 2. Separar variables numéricas y categóricas
    vars_categoricas = df.select_dtypes(include=["object"]).columns.tolist()
    vars_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    vars_numericas_features = [v for v in vars_numericas if v != "mora"]

    # 3. Imputar nulos: mediana para numéricas, moda para categóricas
    for col in vars_numericas_features:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    for col in vars_categoricas:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 4. One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=vars_categoricas, drop_first=False)

    # 5. Separar X e y
    X = df_encoded.drop("mora", axis=1)
    y = df_encoded["mora"]
    features = X.columns.tolist()

    # 6. Normalización
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 7. Balanceo con SMOTE (si está disponible)
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_scaled, y = smote.fit_resample(X_scaled, y)
    except Exception:
        pass  # si imblearn no está, entrena sin SMOTE

    # 8. Entrenar Random Forest (mismos hiperparámetros del proyecto)
    modelo = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    modelo.fit(X_scaled, y)

    return modelo, scaler, features


with st.spinner("Cargando el modelo por primera vez..."):
    modelo, scaler, features = entrenar_modelo()

# Extraer opciones categóricas de las features
viviendas = sorted([f.replace("vivienda_", "") for f in features if f.startswith("vivienda_")])
zonas = sorted([f.replace("zona_", "") for f in features if f.startswith("zona_")])
niveles_educ = sorted([f.replace("nivel_educ_", "") for f in features if f.startswith("nivel_educ_")])

# ----------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------
st.title("💳 Predicción de Riesgo de Morosidad")
st.markdown(
    "Sistema basado en **Random Forest** que estima si un cliente del sistema "
    "financiero peruano incurrirá en morosidad. Ingrese el perfil del cliente y "
    "presione **Predecir**."
)
st.divider()

# ----------------------------------------------------------------------
# Formulario de entrada
# ----------------------------------------------------------------------
st.subheader("Perfil del cliente")

col1, col2 = st.columns(2)

with col1:
    edad = st.number_input("Edad (años)", min_value=18, max_value=90, value=35)
    ingreso = st.number_input("Ingreso mensual (S/.)", min_value=0.0, value=2500.0, step=100.0)
    dias_lab = st.number_input("Días laborados", min_value=0, max_value=25000, value=5000)
    exp_sf = st.number_input("Experiencia en sist. financiero (meses)", min_value=0.0, value=40.0)
    nivel_ahorro = st.slider("Nivel de ahorro", min_value=0, max_value=12, value=6)
    score = st.number_input("Score crediticio", min_value=100, max_value=300, value=200)

with col2:
    atraso = st.number_input("Días de atraso histórico", min_value=0, max_value=245, value=0)
    linea_sf = st.number_input("Línea de crédito (S/.)", min_value=0.0, value=5000.0, step=100.0)
    deuda_sf = st.number_input("Deuda actual (S/.)", min_value=0.0, value=3000.0, step=100.0)
    clasif_sbs = st.selectbox(
        "Clasificación SBS",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: f"{x} - {['Normal','CPP','Deficiente','Dudoso','Pérdida'][x]}",
    )
    vivienda = st.selectbox("Tipo de vivienda", options=viviendas)
    zona = st.selectbox("Zona (región)", options=zonas, index=zonas.index("Lima") if "Lima" in zonas else 0)

nivel_educ = st.selectbox("Nivel educativo", options=niveles_educ)

st.divider()

# ----------------------------------------------------------------------
# Construcción del vector de entrada
# ----------------------------------------------------------------------
def construir_vector():
    row = {f: 0 for f in features}
    for k, v in [("atraso", atraso), ("edad", edad), ("dias_lab", dias_lab),
                 ("exp_sf", exp_sf), ("nivel_ahorro", nivel_ahorro), ("ingreso", ingreso),
                 ("linea_sf", linea_sf), ("deuda_sf", deuda_sf), ("score", score),
                 ("clasif_sbs", clasif_sbs)]:
        if k in row:
            row[k] = v
    for col in ["vivienda_" + vivienda, "zona_" + zona, "nivel_educ_" + nivel_educ]:
        if col in row:
            row[col] = 1
    return np.array([row[f] for f in features]).reshape(1, -1)

# ----------------------------------------------------------------------
# Predicción
# ----------------------------------------------------------------------
if st.button("🔍 Predecir", type="primary", use_container_width=True):
    x = construir_vector()
    x_scaled = scaler.transform(x)
    pred = modelo.predict(x_scaled)[0]
    prob = modelo.predict_proba(x_scaled)[0][1]

    if prob < 0.40:
        riesgo, color = "BAJO", "green"
    elif prob < 0.70:
        riesgo, color = "MEDIO", "orange"
    else:
        riesgo, color = "ALTO", "red"

    st.subheader("Resultado")

    if pred == 1:
        st.error("### ⚠️ Cliente MOROSO (mora = 1)")
    else:
        st.success("### ✅ Cliente NO MOROSO (mora = 0)")

    c1, c2 = st.columns(2)
    c1.metric("Probabilidad de morosidad", f"{prob*100:.1f}%")
    c2.markdown(f"### Nivel de riesgo: :{color}[{riesgo}]")

    st.progress(float(prob))

    st.caption(
        "Interpretación: BAJO (< 40%), MEDIO (40%–70%), ALTO (> 70%). "
        "Modelo Random Forest entrenado sobre el dataset BankDefaultAnalysis."
    )

# ----------------------------------------------------------------------
# Pie de página
# ----------------------------------------------------------------------
st.divider()
st.caption(
    "Proyecto académico — Aprendizaje Estadístico — UPAO 2026 | "
    "Rodríguez Correa, Gutiérrez Cossa, Ticlla Silva"
)
