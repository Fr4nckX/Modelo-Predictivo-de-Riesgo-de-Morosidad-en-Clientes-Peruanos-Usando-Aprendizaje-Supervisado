# Modelo Predictivo de Riesgo de Morosidad en Clientes Peruanos Usando Aprendizaje Supervisado

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JOwcv-Pnv5On9ka5jmHBpBMhFW-RoPld?usp=sharing)

Proyecto del curso **Aprendizaje Estadístico** — Universidad Privada Antenor Orrego (UPAO).
Modelo de clasificación binaria que predice si un cliente del sistema financiero peruano incurrirá en morosidad (`mora = 1`) o pagará puntualmente (`mora = 0`), usando Aprendizaje Supervisado.

## Descripción

Se comparan cuatro modelos (Regresión Lineal, Regresión Logística, Árbol de Decisión y Random Forest) implementados en **Google Colab** (Python / scikit-learn) y validados en **Weka**. El modelo ganador es **Random Forest**.

- **Dataset:** BankDefaultAnalysis — Morosidad del Sistema Financiero Peruano (Kaggle)
- **Registros:** 8,399 | **Variables:** 14
- **Variable objetivo:** `mora` (0 = paga al día, 1 = moroso)

## Estructura del repositorio

| Carpeta / Archivo | Descripción |
|---|---|
| `notebook/` | Notebook de Google Colab (.ipynb) con EDA, preprocesamiento, entrenamiento y evaluación |
| `data/` | Dataset en formato `.arff` para Weka y `.csv` original |
| `models/` | Modelo exportado (`modelo_crediticio.pkl`) y scaler (`scaler_crediticio.pkl`) |
| `docs/` | Informe técnico en PDF, capturas de Weka y gráficos de Colab |

## Cómo ejecutar

### En Google Colab
1. Abre el notebook con el botón **Open in Colab** de arriba.
2. Ejecuta las celdas en orden (Entorno de ejecución → Ejecutar todas).
3. Sube el archivo `data.csv` cuando la celda de carga lo solicite.

### En Weka
1. Abre Weka Explorer.
2. Carga el archivo `.arff` de la carpeta `data/`.
3. Aplica los filtros `ReplaceMissingValues` y `NumericToNominal`.
4. Ejecuta los clasificadores Logistic, J48 y RandomForest.

## Resultados (Random Forest — modelo ganador)

| Herramienta / Método | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| Google Colab (CV k=5) | 80.41% | 0.8029 | 0.8837 |
| Weka (CV 5-fold) | 87.84% | 0.918 | 0.932 |

## Autores

- Rodríguez Correa, Franck Williams
- Gutiérrez Cossa, César Alberto
- Ticlla Silva, Abel Gerardo

**Docente:** Sagástegui Chigne, Teobaldo Hernán
**2026**
