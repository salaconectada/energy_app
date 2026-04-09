from __future__ import annotations

from io import BytesIO
import os

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from statsmodels.tsa.seasonal import STL

from backend.services.forecast import run_forecast
from backend.services.preprocessing import (
    DataValidationError,
    get_exogenous_columns,
    load_dataset,
)
from backend.services.utils import forecast_to_frame

sns.set_style("whitegrid")
st.set_page_config(page_title="Energy-APP · Forecast", layout="wide", initial_sidebar_state="expanded")


st.sidebar.title("Energy-APP")
st.sidebar.info(
    """
**Modo modular**

- Arquitectura separada en `frontend` y `backend`.
- Puedes ejecutar pronóstico en local o vía API FastAPI.
"""
)

default_api_url = os.getenv("ENERGY_API_URL", "http://localhost:8000")
default_mode_index = 1 if os.getenv("ENERGY_FORCE_API", "0") == "1" else 0
connection_mode = st.sidebar.radio("Modo de conexión", ["Local", "API"], index=default_mode_index)
api_url = st.sidebar.text_input("URL API", default_api_url) if connection_mode == "API" else None

page = st.sidebar.radio("Secciones", ["Pronóstico", "Datos & Metadatos"], index=0)
upl = st.sidebar.file_uploader("Sube CSV / Excel", type=["csv", "xlsx"])

if upl is None:
    st.title("🔌 Energy-APP — Pronóstico Energético")
    st.markdown("Sube un archivo con `fecha`, `valor` y exógenas opcionales.")
    st.stop()

try:
    raw = upl.getvalue()
    df = load_dataset(raw, upl.name)
except DataValidationError as exc:
    st.error(f"❌ {exc}")
    st.stop()

exog_available = get_exogenous_columns(df)

if page == "Datos & Metadatos":
    st.title("📑 Procedencia, limpieza y metadatos")
    st.dataframe(df.head(20), use_container_width=True)
    st.download_button("💾 Descargar dataset limpio", df.to_csv(index=False).encode(), "dataset_limpio.csv")
    st.stop()

st.title("🔌 Energy-APP — Pronóstico de Consumo Industrial")
model_sel = st.sidebar.radio("Modelo", ["SARIMAX", "Random-Forest"])
periods = st.sidebar.slider("Meses a predecir", 6, 24, 12, step=6)
exog_sel = st.sidebar.multiselect("Variables exógenas", exog_available)

if connection_mode == "API":
    files = {"file": (upl.name, BytesIO(raw), upl.type or "application/octet-stream")}
    data = {
        "model": model_sel,
        "periods": str(periods),
        "exog": ",".join(exog_sel),
    }
    try:
        response = requests.post(f"{api_url.rstrip('/')}/forecast", files=files, data=data, timeout=45)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        st.error(f"No se pudo conectar a la API: {exc}")
        st.stop()

    pred_df = pd.DataFrame(payload["forecast"])
    pred_df["fecha"] = pd.to_datetime(pred_df["fecha"])
    pred = pred_df.set_index("fecha")["pronostico_gwh"]
    monthly_hist = pd.DataFrame(payload["history_monthly"])
    monthly_hist["fecha"] = pd.to_datetime(monthly_hist["fecha"])
    kpis = payload["kpis"]
    metrics = payload["metrics"]
else:
    result = run_forecast(df, model=model_sel, periods=periods, exog_sel=exog_sel)
    pred = result.pred
    monthly_hist = result.monthly_df
    kpis = result.kpis
    metrics = result.metrics

c1, c2, c3 = st.columns(3)
c1.metric("Total histórico (GWh)", f"{kpis['total_historico_gwh']:,.0f}")
c2.metric("Tendencia anual media", f"{kpis['tendencia_anual_media_pct']:+.1f}%")
c3.metric("Pico / valle", f"{kpis['indice_estacionalidad']:.2f}")

fig_p, ax_p = plt.subplots(figsize=(11, 3))
sns.lineplot(x=monthly_hist["fecha"], y=monthly_hist["valor"], label="Histórico", ax=ax_p)
sns.lineplot(x=pred.index, y=pred.values, label="Pronóstico", marker="o", linewidth=2, color="#ff7f0e", ax=ax_p)
st.pyplot(fig_p)

with st.expander("📈 Descomposición STL"):
    stl = STL(monthly_hist.set_index("fecha")["valor"], period=12).fit()
    st.pyplot(stl.plot().figure)

if metrics:
    badge = "🟢" if metrics["mae_pct"] < 5 else "🟡" if metrics["mae_pct"] < 10 else "🔴"
    st.info(
        f"MAE {metrics['mae']:,.2f} ({metrics['mae_pct']:.1f} %) · "
        f"RMSE {metrics['rmse']:,.2f} {badge}"
    )

st.download_button(
    "💾 Descargar pronóstico",
    forecast_to_frame(pred).to_csv(index=False).encode(),
    "pronostico_energy_app.csv",
)
