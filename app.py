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
# Estilos (paleta UPAO)
# ----------------------------------------------------------------------
PRIMARY = "#0067AE"
DARK = "#1F497D"
INK = "#0E1B2A"
BG_SOFT = "#F4F8FD"

st.markdown(
    f"""
    <style>
    /* Tipografía y fondo general */
    html, body, [class*="css"] {{
        font-family: 'Segoe UI', 'Inter', system-ui, sans-serif;
    }}
    .stApp {{
        background: linear-gradient(180deg, #FFFFFF 0%, {BG_SOFT} 100%);
    }}
    .block-container {{
        padding-top: 1.6rem;
        max-width: 880px;
    }}
    /* Ocultar chrome por defecto para una vista más limpia */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Cabecera tipo banner */
    .hero {{
        background: linear-gradient(120deg, {PRIMARY} 0%, {DARK} 100%);
        border-radius: 18px;
        padding: 26px 30px;
        color: #FFFFFF;
        box-shadow: 0 10px 26px rgba(0,103,174,0.28);
        margin-bottom: 22px;
    }}
    .hero h1 {{
        margin: 0;
        font-size: 30px;
        font-weight: 800;
        letter-spacing: 0.2px;
    }}
    .hero p {{
        margin: 8px 0 0 0;
        font-size: 15px;
        line-height: 1.5;
        color: #DCEBFA;
    }}
    .hero .tag {{
        display: inline-block;
        margin-top: 14px;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.35);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 600;
    }}

    /* Títulos de sección */
    .section {{
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 17px;
        font-weight: 700;
        color: {INK};
        margin: 6px 0 4px 0;
    }}
    .section::before {{
        content: "";
        width: 6px; height: 20px;
        background: {PRIMARY};
        border-radius: 4px;
        display: inline-block;
    }}

    /* Inputs: etiquetas más legibles */
    label p {{ font-weight: 600 !important; color: {DARK} !important; }}

    /* Botón principal */
    .stButton > button {{
        background: {PRIMARY};
        color: #FFFFFF;
        border: 0;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        font-weight: 700;
        font-size: 16px;
        transition: all .15s ease-in-out;
    }}
    .stButton > button:hover {{
        background: {DARK};
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(0,103,174,0.30);
    }}

    /* Tarjeta de resultado */
    .result-card {{
        border-radius: 16px;
        padding: 18px 22px;
        margin-top: 6px;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 6px 18px rgba(20,40,70,0.08);
    }}
    .result-card h2 {{ margin: 0; font-size: 22px; font-weight: 800; }}
    .result-card p {{ margin: 4px 0 0 0; font-size: 14px; color: #435468; }}

    .pill {{
        display: inline-block;
        padding: 6px 16px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 15px;
        color: #FFFFFF;
    }}
    .prob-num {{ font-size: 40px; font-weight: 800; line-height: 1; color: {DARK}; }}
    .prob-lbl {{ font-size: 13px; color: #5B6B7C; font-weight: 600; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def section(title: str):
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)


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
# Encabezado (banner, sin emojis en el título)
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>Predicción de Riesgo de Morosidad</h1>
        <p>Sistema predictivo que estima si un cliente del
        sistema financiero peruano incurrirá en morosidad. Ingrese el perfil del
        cliente y presione <b>Predecir</b>.</p>
        <span class="tag">Aprendizaje Estadístico · UPAO</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Formulario de entrada
# ----------------------------------------------------------------------
section("Perfil del cliente")

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

st.write("")

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
if st.button("Predecir", type="primary", use_container_width=True):
    x = construir_vector()
    x_scaled = scaler.transform(x)
    pred = modelo.predict(x_scaled)[0]
    prob = modelo.predict_proba(x_scaled)[0][1]

    if prob < 0.40:
        riesgo, color = "BAJO", "#2E7D32"
    elif prob < 0.70:
        riesgo, color = "MEDIO", "#E8820C"
    else:
        riesgo, color = "ALTO", "#C62828"

    section("Resultado")

    # Tarjeta principal según la clase predicha
    if pred == 1:
        card_bg, card_bd, card_tx = "#FDECEC", "#F3C0C0", "#B3261E"
        titulo = "Cliente MOROSO  (mora = 1)"
        detalle = "El modelo estima que el cliente tiene alta probabilidad de incumplir sus pagos."
    else:
        card_bg, card_bd, card_tx = "#E9F7EE", "#BEE6CB", "#1E7A3D"
        titulo = "Cliente NO MOROSO  (mora = 0)"
        detalle = "El modelo estima que el cliente pagará sus obligaciones al día."

    st.markdown(
        f"""
        <div class="result-card" style="background:{card_bg}; border-color:{card_bd};">
            <h2 style="color:{card_tx};">{titulo}</h2>
            <p>{detalle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # Probabilidad + nivel de riesgo
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            f"""
            <div style="text-align:left;">
                <div class="prob-lbl">Probabilidad de morosidad</div>
                <div class="prob-num">{prob*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div style="text-align:left;">
                <div class="prob-lbl">Nivel de riesgo</div>
                <div style="margin-top:6px;">
                    <span class="pill" style="background:{color};">{riesgo}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.progress(float(prob))

    st.caption(
        "Interpretación: BAJO (< 40%), MEDIO (40%–70%), ALTO (> 70%). "
        "Modelo Random Forest entrenado sobre el dataset BankDefaultAnalysis."
    )

# ----------------------------------------------------------------------
# Pie de página
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <hr style="margin-top:28px; border:none; border-top:1px solid #E1E8F0;">
    <div style="text-align:center; color:#7A8794; font-size:12.5px; padding-top:6px;">
        Proyecto académico — Aprendizaje Estadístico — UPAO 2026<br>
        Rodríguez Correa · Gutiérrez Cossa · Ticlla Silva
    </div>
    """,
    unsafe_allow_html=True,
)
