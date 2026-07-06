# Casos de Prueba del Sistema de Predicción

Este documento presenta casos de prueba documentados para validar el funcionamiento del modelo **Random Forest** de predicción de riesgo de morosidad. Cada caso corresponde a un perfil de cliente con distintas características socioeconómicas y crediticias, junto con la predicción real generada por el modelo (`modelo_crediticio.pkl`).

## Cómo se ejecutan las pruebas

Cada cliente se procesa mediante la función `predecir_cliente()` implementada en el notebook, la cual:

1. Construye el vector de 42 variables (numéricas + One-Hot Encoding de vivienda, zona y nivel educativo).
2. Normaliza los datos con el `scaler_crediticio.pkl` (StandardScaler).
3. Predice con el modelo Random Forest y devuelve la clase (`0` = no moroso, `1` = moroso) y la probabilidad de morosidad.

El nivel de riesgo se interpreta así:

| Probabilidad de morosidad | Nivel de riesgo |
|---|---|
| Menor a 40% | BAJO |
| Entre 40% y 70% | MEDIO |
| Mayor a 70% | ALTO |

## Caso 1 — Cliente de bajo riesgo (esperado: NO moroso)

| Variable | Valor |
|---|---|
| Edad | 45 años |
| Días laborados | 8,000 |
| Experiencia en sist. financiero | 120 meses |
| Nivel de ahorro | 10 |
| Ingreso mensual | S/. 6,500 |
| Línea de crédito | S/. 25,000 |
| Deuda actual | S/. 3,000 |
| Score crediticio | 255 |
| Clasificación SBS | 0 (normal) |
| Días de atraso histórico | 0 |
| Vivienda | Propia |
| Zona | Lima |
| Nivel educativo | Universitaria |

**Resultado del modelo:**
- Predicción: **NO MOROSO (mora = 0)**
- Probabilidad de morosidad: **27.3%**
- Nivel de riesgo: **BAJO**

## Caso 2 — Cliente de alto riesgo (esperado: moroso)

| Variable | Valor |
|---|---|
| Edad | 24 años |
| Días laborados | 3,000 |
| Experiencia en sist. financiero | 6 meses |
| Nivel de ahorro | 1 |
| Ingreso mensual | S/. 1,200 |
| Línea de crédito | S/. 800 |
| Deuda actual | S/. 7,000 |
| Score crediticio | 140 |
| Clasificación SBS | 4 (pérdida) |
| Días de atraso histórico | 120 |
| Vivienda | Alquilada |
| Zona | Loreto |
| Nivel educativo | Secundaria |

**Resultado del modelo:**
- Predicción: **MOROSO (mora = 1)**
- Probabilidad de morosidad: **83.1%**
- Nivel de riesgo: **ALTO**

## Caso 3 — Cliente intermedio (perfil favorable)

| Variable | Valor |
|---|---|
| Edad | 38 años |
| Días laborados | 5,500 |
| Experiencia en sist. financiero | 60 meses |
| Nivel de ahorro | 7 |
| Ingreso mensual | S/. 3,500 |
| Línea de crédito | S/. 8,000 |
| Deuda actual | S/. 4,000 |
| Score crediticio | 210 |
| Clasificación SBS | 0 (normal) |
| Días de atraso histórico | 5 |
| Vivienda | Familiar |
| Zona | Arequipa |
| Nivel educativo | Técnica |

**Resultado del modelo:**
- Predicción: **MOROSO (mora = 1)**
- Probabilidad de morosidad: **50.3%**
- Nivel de riesgo: **MEDIO**

## Caso 4 — Cliente intermedio (perfil riesgoso)

| Variable | Valor |
|---|---|
| Edad | 29 años |
| Días laborados | 4,000 |
| Experiencia en sist. financiero | 18 meses |
| Nivel de ahorro | 3 |
| Ingreso mensual | S/. 2,000 |
| Línea de crédito | S/. 2,500 |
| Deuda actual | S/. 5,500 |
| Score crediticio | 175 |
| Clasificación SBS | 2 |
| Días de atraso histórico | 45 |
| Vivienda | Familiar |
| Zona | Piura |
| Nivel educativo | Secundaria |

**Resultado del modelo:**
- Predicción: **MOROSO (mora = 1)**
- Probabilidad de morosidad: **82.8%**
- Nivel de riesgo: **ALTO**

## Resumen de resultados

| Caso | Perfil | Predicción | Prob. morosidad | Riesgo |
|---|---|---|---|---|
| 1 | Bajo riesgo | No moroso | 27.3% | BAJO |
| 2 | Alto riesgo | Moroso | 83.1% | ALTO |
| 3 | Intermedio favorable | Moroso | 50.3% | MEDIO |
| 4 | Intermedio riesgoso | Moroso | 82.8% | ALTO |

## Interpretación

Los resultados demuestran que el modelo discrimina correctamente entre perfiles de distinto riesgo. El cliente con historial limpio, buen score, alto ingreso y experiencia financiera consolidada (Caso 1) es clasificado como no moroso con baja probabilidad de incumplimiento. En contraste, los perfiles con atraso histórico elevado, score bajo, clasificación SBS desfavorable y poca experiencia financiera (Casos 2 y 4) son identificados como morosos con alta probabilidad.

El Caso 3 se ubica en la frontera de decisión (50.3%), lo que refleja la sensibilidad del modelo ante perfiles ambiguos y la utilidad de reportar la probabilidad además de la clase, tal como se implementa en el sistema de despliegue propuesto.
