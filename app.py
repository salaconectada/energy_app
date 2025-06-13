# ────────────────────────────────────────────────────────────────────
# Energy_APP  |  Predicción de consumo energético multivariable
# ────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
# ───────────────────────── 1. CONFIG  ───────────────────────────────
st.set_page_config(
    page_title="Energy-APP • Pronóstico de Consumo Energético Industrial",
    layout="wide",
    initial_sidebar_state="expanded")
sns.set_style("whitegrid")

# ───────────────────────── 2. HEADER  ───────────────────────────────
st.title("🔌 Energy-APP – Pronóstico de Consumo Energético en Industrias")

st.markdown(
"""
Sube un **CSV** o **Excel** con al menos las columnas:

| columna | descripción | ejemplo |
|---------|-------------|---------|
| `fecha` | Fecha YYYY-MM-DD | 2024-05-01 |
| `valor` | Consumo GWh     | 123.4 |
| otras   | (opcionales) Temperatura, producción, precio spot, etc. |

Selecciona variables exógenas, modelo (**SARIMAX** o **Random-Forest**),
horizonte (6-24 meses) y descarga el pronóstico.
""")

# ───────────────────────── 3. LOAD DATA  ────────────────────────────
upl = st.file_uploader("📤 Cargar archivo", type=["csv", "xlsx"])
if not upl:
    st.stop()

ext = Path(upl.name).suffix
df = (pd.read_csv(upl, parse_dates=["fecha"])
      if ext == ".csv"
      else pd.read_excel(upl, parse_dates=["fecha"]))
df = df.sort_values("fecha").reset_index(drop=True)

st.success(f"Datos cargados → {df.shape[0]:,} filas • {df.shape[1]} columnas")

# ───────────────────────── 4. SIDEBAR  ──────────────────────────────
st.sidebar.header("⚙️ Configuración")

num_cols = [c for c in df.columns
            if c not in ("fecha", "valor")
            and np.issubdtype(df[c].dtype, np.number)]

exog_sel = st.sidebar.multiselect("Variables exógenas", num_cols)

model_sel = st.sidebar.radio("Modelo de pronóstico", ["SARIMAX", "Random-Forest"])

periods = st.sidebar.slider("Meses a predecir", 6, 24, 12, step=6)

if st.sidebar.button("🔄 Recalcular"):
    st.experimental_rerun()

# ───────────────────────── 5. KPIs BÁSICOS  ─────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Total histórico (GWh)", f"{df['valor'].sum():,.0f}")

trend_pct = (df.set_index("fecha")["valor"].resample("A").sum()
             .pct_change().mean()*100)
col2.metric("Tendencia anual media", f"{trend_pct:+.1f}%")

seasonality = (df.groupby(df['fecha'].dt.month)["valor"].mean().max() /
               df.groupby(df['fecha'].dt.month)["valor"].mean().min())
col3.metric("Factor estacional pico/valle", f"{seasonality:.2f}")

# ───────────────────────── 6. EXPLORACIÓN  ─────────────────────────
with st.expander("🔍 Exploración de datos"):
    fig_h, ax_h = plt.subplots(figsize=(10, 3))
    sns.lineplot(data=df, x="fecha", y="valor", ax=ax_h)
    ax_h.set_title("Histórico de consumo")
    st.pyplot(fig_h)

    if exog_sel:
        st.write("### Matriz de correlación")
        fig_c, ax_c = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[["valor"]+exog_sel].corr(), annot=True, cmap="coolwarm", ax=ax_c)
        st.pyplot(fig_c)

# ───────────────────────── 7. STL  ──────────────────────────────────
with st.expander("📈 Descomposición STL"):
    stl = STL(df.set_index("fecha")["valor"], period=12).fit()
    st.pyplot(stl.plot().figure)

# ───────────────────────── 8. MODELOS  ──────────────────────────────
@st.cache_data(show_spinner="Entrenando SARIMAX…")
def train_sarimax(y, exog):
    return SARIMAX(
        y, exog=exog,
        order=(1,1,1), seasonal_order=(0,1,1,12),
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)

@st.cache_data(show_spinner="Entrenando Random-Forest…")
def train_rf(df_, lags, exog_vars):
    X, y = [], []
    for i in range(lags, len(df_)):
        feat = df_["valor"].shift(range(1, lags+1)).iloc[i].values
        if exog_vars:
            feat = np.concatenate([feat, df_[exog_vars].iloc[i].values])
        X.append(feat)
        y.append(df_["valor"].iloc[i])
    rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(X, y)
    return rf, lags

# Entrenamiento y pronóstico
if model_sel == "SARIMAX":
    model = train_sarimax(df["valor"], df[exog_sel] if exog_sel else None)

    # repetir último valor exógeno 'periods' veces
    fut_exog = (pd.concat([df[exog_sel].iloc[[-1]]] * periods, ignore_index=True)
                if exog_sel else None)

    forecast = model.get_forecast(periods, exog=fut_exog)
    pred_series = pd.Series(
        forecast.predicted_mean,
        index=pd.date_range(df["fecha"].iloc[-1] + pd.offsets.MonthBegin(),
                            periods=periods, freq="MS"))
else:  # Random-Forest
    rf, lags = train_rf(df, lags=12, exog_vars=exog_sel)
    last = df.copy()
    preds = []
    for _ in range(periods):
        feat = last["valor"].iloc[-lags:][::-1].values
        if exog_sel:
            feat = np.concatenate([feat, last[exog_sel].iloc[-1].values])
        y_hat = rf.predict(feat.reshape(1, -1))[0]
        preds.append(y_hat)
        new_row = {"fecha": last["fecha"].iloc[-1] + pd.offsets.MonthBegin(),
                   "valor": y_hat, **{c: last[c].iloc[-1] for c in exog_sel}}
        last = pd.concat([last, pd.DataFrame([new_row])], ignore_index=True)

    pred_series = pd.Series(
        preds,
        index=pd.date_range(df["fecha"].iloc[-1] + pd.offsets.MonthBegin(),
                            periods=periods, freq="MS"))

# ───────────────────────── 9. PLOT PRONÓSTICO  ─────────────────────
st.subheader("🔮 Pronóstico")
fig_p, ax_p = plt.subplots(figsize=(11, 3))
# histórico
sns.lineplot(x=df["fecha"], y=df["valor"], label="Histórico", ax=ax_p,
             color="#1f77b4")
# pronóstico
sns.lineplot(x=pred_series.index, y=pred_series.values, label="Pronóstico",
             ax=ax_p, color="#ff7f0e", marker="o", linewidth=2)
ax_p.set_xlim(df["fecha"].min(), pred_series.index.max())
st.pyplot(fig_p)

# ───────────── 10. MÉTRICAS + INTERPRETACIÓN  ──────────────────────
def explain_errors(mae, rmse, mean_val):
    rel_mae  = mae  / mean_val * 100
    rel_rmse = rmse / mean_val * 100
    txt  = f"MAE = {mae:,.2f}  ({rel_mae:.1f}% del promedio mensual)\n"
    txt += f"RMSE = {rmse:,.2f} ({rel_rmse:.1f}% del promedio mensual)\n"
    if rel_mae < 5:
        txt += "🔹 **Muy buen ajuste** (error < 5 %)."
    elif rel_mae < 10:
        txt += "🔸 **Ajuste aceptable** (error 5-10 %)."
    else:
        txt += "🔴 **Ajuste pobre** (error > 10 %): revisa modelo/datos."
    return txt

if model_sel == "SARIMAX":
    fitted = model.fittedvalues
    mae  = mean_absolute_error(df["valor"].iloc[-12:], fitted.iloc[-12:])
    rmse = np.sqrt(mean_squared_error(df["valor"].iloc[-12:], fitted.iloc[-12:]))
    st.markdown("#### Métricas (últimos 12 meses)")
    st.code(explain_errors(mae, rmse, df["valor"].mean()))

# ───────────── 11. CONCLUSIONES & SUGERENCIAS  ─────────────────────
growth = trend_pct
conclu = (f"• Consumo total proyectado próximos **{periods} meses** ≈ "
          f"**{pred_series.sum():,.0f} GWh**\n"
          f"• Crecimiento anual histórico: **{growth:+.1f}%**\n"
          f"• Estacionalidad pico/valle ≈ **{seasonality:.2f}**")
st.markdown("### 📝 Conclusiones")
st.info(conclu)

tips = []
if seasonality > 1.2:
    tips.append("• Revisar turnos/mantenciones en meses pico.")
if growth > 5:
    tips.append("• Auditar procesos para contener crecimiento (>5 %).")
if tips:
    st.markdown("### 💡 Sugerencias")
    st.write("\n".join(tips))

# ───────────── 12. DESCARGA CSV  ───────────────────────────────────
csv_out = (pred_series.reset_index()
           .rename(columns={"index": "fecha", 0: "pronostico_GWh"})
           .to_csv(index=False).encode())
st.download_button("💾 Descargar pronóstico CSV", csv_out, "pronostico.csv")

st.caption("Powered by ☀Felipe Leiva • Sala Conectada ")
