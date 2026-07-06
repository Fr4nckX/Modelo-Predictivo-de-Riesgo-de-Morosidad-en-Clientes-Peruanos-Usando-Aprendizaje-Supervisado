"""
Sistema de Predicción de Riesgo de Morosidad
Modelo Random Forest - Universidad Privada Antenor Orrego (UPAO)
Curso: Aprendizaje Estadístico
"""

import streamlit as st
import numpy as np
import joblib

# ----------------------------------------------------------------------
# Configuración de la página
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Morosidad",
    page_icon="💳",
    layout="centered",
)

# ----------------------------------------------------------------------
# Carga del modelo, scaler y features (con caché para eficiencia)
# ----------------------------------------------------------------------
@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load("models/modelo_crediticio.pkl")
    scaler = joblib.load("models/scaler_crediticio.pkl")
    features = joblib.load("models/features_crediticio.pkl")
    return modelo, scaler, features

modelo, scaler, features = cargar_artefactos()

# Extraer las opciones categóricas directamente de las features del modelo
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
    row["atraso"] = atraso
    row["edad"] = edad
    row["dias_lab"] = dias_lab
    row["exp_sf"] = exp_sf
    row["nivel_ahorro"] = nivel_ahorro
    row["ingreso"] = ingreso
    row["linea_sf"] = linea_sf
    row["deuda_sf"] = deuda_sf
    row["score"] = score
    row["clasif_sbs"] = clasif_sbs
    # Variables categóricas (One-Hot)
    for col, val in [("vivienda_" + vivienda, 1),
                     ("zona_" + zona, 1),
                     ("nivel_educ_" + nivel_educ, 1)]:
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
    prob = modelo.predict_proba(x_scaled)[0][1]  # probabilidad de mora=1

    # Nivel de riesgo
    if prob < 0.40:
        riesgo, color = "BAJO", "green"
    elif prob < 0.70:
        riesgo, color = "MEDIO", "orange"
    else:
        riesgo, color = "ALTO", "red"

    st.subheader("Resultado")

    if pred == 1:
        st.error(f"### ⚠️ Cliente MOROSO (mora = 1)")
    else:
        st.success(f"### ✅ Cliente NO MOROSO (mora = 0)")

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
