# ──────────────────────────────────────────────────────────────
#  Energy-APP | Pronóstico de Consumo Energético Multivariable
#  Felipe Leiva — Sala Conectada © 2025
# ──────────────────────────────────────────────────────────────
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

# ╭────────────────── 1. CONFIG GLOBAL ───────────────────────╮
st.set_page_config(page_title="Energy-APP · Forecast",
                   layout="wide", initial_sidebar_state="expanded")
sns.set_style("whitegrid")

# ╭────────────────── 2. SIDEBAR / NAV ───────────────────────╮
st.sidebar.title("Energy-APP")
st.sidebar.info("""
**Bienvenido**

1. Explorar y limpiar datos de consumo energético.  
2. Entrenar modelos SARIMAX o Random-Forest con variables exógenas.  
3. Generar y descargar pronósticos de 6-24 meses.  
""")

page = st.sidebar.radio("Secciones",
                        ["Pronóstico", "Datos & Metadatos"], index=0)

# ╭────────────────── 3. CARGA DE DATOS ──────────────────────╮
st.sidebar.header("📤 Datos de entrada")
upl = st.sidebar.file_uploader("Sube CSV / Excel", type=["csv", "xlsx"])

if upl is None:
    st.title("🔌 Energy-APP — Pronóstico Energético")
    st.markdown("""
Cargue un archivo para comenzar.

| columna | descripción | ejemplo |
|---------|-------------|---------|
| `fecha` | **Fecha** YYYY-MM-DD | 2024-05-01 |
| `valor` | Consumo / demanda (GWh) | 123.4 |
| demás   | (opcional) exógenas numéricas |
""")
    st.stop()

ext = Path(upl.name).suffix.lower()
df = (pd.read_csv(upl, parse_dates=["fecha"])
      if ext == ".csv" else
      pd.read_excel(upl, parse_dates=["fecha"]))

df.columns = df.columns.str.strip().str.lower()

if "fecha" not in df.columns:
    st.error("❌ Falta columna **fecha**."); st.stop()

if "valor" not in df.columns:
    num_first = df.select_dtypes("number").columns
    if num_first.empty:
        st.error("❌ No se encontró columna numérica para `valor`."); st.stop()
    df = df.rename(columns={num_first[0]: "valor"})
    st.warning(f"Se usó **{num_first[0]}** como `valor`.")

df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
for c in df.columns.difference(["fecha", "valor"]):
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.sort_values("fecha").reset_index(drop=True)
dataset_clean = df.copy()

num_cols = [c for c in df.columns
            if c not in ("fecha", "valor")
            and np.issubdtype(df[c].dtype, np.number)]

# ╭────────────── 4. PÁGINA DATOS & METADATOS ────────────────╮
if page == "Datos & Metadatos":
    st.title("📑 Procedencia, limpieza y metadatos")

    with st.expander("Fuentes de datos"):
        src = pd.DataFrame({
            "Fuente": ["Mina El Toqui — Consumo",
                       "NASA POWER — Clima",
                       "OASIS CEN — Precio spot"],
            "URL": ["https://eltoqui.cl/transparencia/energia",
                    "https://power.larc.nasa.gov/api/",
                    "https://oasis.cen.cl"],
            "Frecuencia": ["Mensual", "Diaria", "Horaria"],
            "Cobertura": ["2014-01 → 2024-06"]*3
        })
        st.dataframe(src, use_container_width=True)

    with st.expander("Pasos de limpieza & merge"):
        st.markdown("""
* **Merge outer** por `fecha`.  
* **Resample mensual (MS)**.  
* `valor`: interpolación lineal ≤ 1 mes.  
* Exógenas: **ffill + bfill**.  
* **Outliers** (> 3 σ) reemplazados por mediana mensual.
""")

    st.subheader("Vista previa del dataset limpio")
    st.dataframe(dataset_clean.head(), use_container_width=True)

    csv_clean = dataset_clean.to_csv(index=False).encode()
    st.download_button("💾 Descargar dataset limpio",
                       csv_clean, "dataset_limpio.csv")
    st.stop()

# ╭────────────────── 5. PÁGINA PRONÓSTICO ───────────────────╮
st.title("🔌 Energy-APP — Pronóstico de Consumo Industrial")

st.sidebar.header("⚙️ Configuración del modelo")
exog_sel = st.sidebar.multiselect("Variables exógenas", num_cols)
model_sel = st.sidebar.radio("Modelo", ["SARIMAX", "Random-Forest"])
periods   = st.sidebar.slider("Meses a predecir", 6, 24, 12, step=6)
if st.sidebar.button("🔄 Ejecutar / refrescar"):
    st.experimental_rerun()

# KPIs rápidos
c1, c2, c3 = st.columns(3)
c1.metric("Total histórico (GWh)", f"{df['valor'].sum():,.0f}")
trend_pct = (df.set_index("fecha")["valor"].resample("A").sum()
             .pct_change().mean()*100)
c2.metric("Tendencia anual media", f"{trend_pct:+.1f}%")
seasonality = (df.groupby(df['fecha'].dt.month)["valor"].mean().max() /
               df.groupby(df['fecha'].dt.month)["valor"].mean().min())
c3.metric("Pico / valle", f"{seasonality:.2f}")

# Exploración rápida
with st.expander("🔍 Exploración rápida"):
    fig_hist, ax_hist = plt.subplots(figsize=(10,3))
    sns.lineplot(data=df, x="fecha", y="valor", ax=ax_hist)
    st.pyplot(fig_hist)

    if exog_sel:
        fig_corr, ax_corr = plt.subplots(figsize=(5,4))
        sns.heatmap(df[["valor"]+exog_sel].corr(),
                    annot=True, cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)

# Resample mensual para modelado
df_m = (df.set_index("fecha")
          .resample("MS").mean()
          .interpolate("linear", limit_direction="both")
          .reset_index())

# Descomposición STL
with st.expander("📈 Descomposición STL"):
    stl = STL(df_m.set_index("fecha")["valor"], period=12).fit()
    st.pyplot(stl.plot().figure)

# ─────────────── Funciones de entrenamiento ────────────────
@st.cache_data(show_spinner="Entrenando SARIMAX…")
def fit_sarimax(y, exog):
    return SARIMAX(y, exog=exog,
                   order=(1,1,1), seasonal_order=(0,1,1,12),
                   enforce_stationarity=False,
                   enforce_invertibility=False).fit(disp=False)

@st.cache_data(show_spinner="Entrenando Random-Forest…")
def fit_rf(frame, lags, exog):
    X, y = [], []
    for i in range(lags, len(frame)):
        lag_feats = frame["valor"].shift(range(1, lags+1)).iloc[i].values
        if exog:
            lag_feats = np.concatenate([lag_feats,
                                        frame[exog].iloc[i].values])
        X.append(lag_feats); y.append(frame["valor"].iloc[i])
    rf = RandomForestRegressor(n_estimators=400, random_state=0).fit(X, y)
    return rf, lags

# ─────────────── Pronóstico ────────────────────────────────
if model_sel == "SARIMAX":
    y_train = df_m["valor"].astype(float).fillna(method="ffill")
    exog_train = df_m[exog_sel] if exog_sel else None
    mod = fit_sarimax(y_train, exog_train)

    fut_exog = (pd.concat([exog_train.iloc[[-1]]] * periods,
                          ignore_index=True)
                if exog_sel else None)

    fc = mod.get_forecast(periods, exog=fut_exog)
    pred_values = np.asarray(fc.predicted_mean).astype(float)  # evitar NaN
    pred = pd.Series(pred_values,
                     index=pd.date_range(df_m["fecha"].iloc[-1] +
                                         pd.offsets.MonthBegin(),
                                         periods=periods, freq="MS"))
else:
    rf, lags = fit_rf(df_m, 12, exog_sel)
    tmp = df_m.copy()
    preds = []
    for _ in range(periods):
        feats = tmp["valor"].iloc[-lags:][::-1].values
        if exog_sel:
            feats = np.concatenate([feats,
                                    tmp[exog_sel].iloc[-1].values])
        y_hat = rf.predict(feats.reshape(1,-1))[0]
        preds.append(y_hat)
        new = {"fecha": tmp["fecha"].iloc[-1] + pd.offsets.MonthBegin(),
               "valor": y_hat, **{c: tmp[c].iloc[-1] for c in exog_sel}}
        tmp = pd.concat([tmp, pd.DataFrame([new])], ignore_index=True)
    pred = pd.Series(preds,
                     index=pd.date_range(df_m["fecha"].iloc[-1] +
                                         pd.offsets.MonthBegin(),
                                         periods=periods, freq="MS"))

# Plot principal
st.subheader("🔮 Pronóstico")
fig_p, ax_p = plt.subplots(figsize=(11,3))
sns.lineplot(x=df_m["fecha"], y=df_m["valor"], label="Histórico", ax=ax_p)
sns.lineplot(x=pred.index, y=pred.values, label="Pronóstico",
             marker="o", linewidth=2, color="#ff7f0e", ax=ax_p)
ax_p.set_xlim(df_m["fecha"].min(), pred.index.max())
st.pyplot(fig_p)

# Métricas (solo SARIMAX)
if model_sel == "SARIMAX":
    fitted = mod.fittedvalues
    mae  = mean_absolute_error(df_m["valor"].iloc[-12:], fitted.iloc[-12:])
    rmse = np.sqrt(mean_squared_error(df_m["valor"].iloc[-12:], fitted.iloc[-12:]))
    perc = mae / df_m["valor"].mean() * 100
    badge = "🟢" if perc < 5 else "🟡" if perc < 10 else "🔴"
    st.markdown("#### Calidad de ajuste (12 meses)")
    st.info(f"MAE {mae:,.2f} ({perc:.1f} %) · RMSE {rmse:,.2f} {badge}")

# Conclusiones y recomendaciones
st.markdown("### 📝 Conclusiones")
msg = (f"• Proyección próximos **{periods} meses** ≈ "
       f"**{pred.sum():,.0f} GWh**\n"
       f"• Crecimiento anual histórico: **{trend_pct:+.1f}%**\n"
       f"• Estacionalidad pico/valle ≈ **{seasonality:.2f}**")
if exog_sel:
    msg += f"\n• Exógenas empleadas: `{', '.join(exog_sel)}`"
st.success(msg)

recs = []
if seasonality > 1.2:
    recs.append("• Ajustar turnos/mantenciones en meses pico.")
if trend_pct > 5:
    recs.append("• Auditar procesos para contener el crecimiento (> 5 %).")
if recs:
    st.markdown("### 💡 Recomendaciones")
    st.write("\n".join(recs))

# Descarga pronóstico
csv_out = (pred.reset_index()
           .rename(columns={"index": "fecha",
                            0: "pronostico_gwh"})
           .to_csv(index=False).encode())
st.download_button("💾 Descargar pronóstico",
                   csv_out, "pronostico_energy_app.csv")

st.caption("Energy-APP · Felipe Leiva A. / Sala Conectada © 2025")
