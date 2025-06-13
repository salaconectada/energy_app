# app.py  (versión ampliada)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ML / TS
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# --------------------------------------------------
st.set_page_config(page_title="Energy-ML Demo", layout="centered")
st.title("Modelo de Demostración • Consumo Energético (Chile)")

st.markdown(
"""
Sube un archivo **CSV** o **Excel** con las columnas:

* `fecha` — tipo fecha (YYYY-MM-DD)
* `valor` — consumo en **GWh / kWh**

La aplicación mostrará:

1. KPIs descriptivos  
2. Gráfico histórico  
3. Descomposición estacional (STL)  
4. Pronóstico SARIMAX + métricas  
5. Descarga del pronóstico en CSV
"""
)

# --------------------------------------------------
# 1. CARGA DE DATOS
# --------------------------------------------------
uploaded = st.file_uploader(
    "Sube tu archivo de consumo energético",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False
)

if uploaded:

    ext = Path(uploaded.name).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(uploaded, parse_dates=["fecha"])
    else:
        df = pd.read_excel(uploaded, parse_dates=["fecha"])

    df = df.sort_values("fecha").reset_index(drop=True)

    # 1.1 Vista rápida
    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    # --------------------------------------------------
    # 2. INDICADORES DESCRIPTIVOS (KPI)
    # --------------------------------------------------
    st.subheader("Indicadores Clave")
    col1, col2, col3 = st.columns(3)
    col1.metric("Prom. mensual (GWh)", f"{df['valor'].mean():.1f}")
    col2.metric("Máximo histórico", f"{df['valor'].max():.0f}")
    if len(df) >= 13:
        growth = (df["valor"].iloc[-1] / df["valor"].iloc[-13] - 1) * 100
        col3.metric("Crecimiento 12m", f"{growth:.1f}%")
    else:
        col3.metric("Crecimiento 12m", "N/D")

    # --------------------------------------------------
    # 3. GRÁFICO HISTÓRICO
    # --------------------------------------------------
    fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=df, x="fecha", y="valor", marker="o", ax=ax_hist)
    ax_hist.set_title("Consumo energético histórico")
    ax_hist.set_xlabel("")
    ax_hist.set_ylabel("Consumo (GWh)")
    st.pyplot(fig_hist)

    # --------------------------------------------------
    # 4. DESCOMPOSICIÓN STL
    # --------------------------------------------------
    st.subheader("Descomposición estacional (STL)")
    try:
        stl = STL(df.set_index("fecha")["valor"], period=12).fit()
        fig_stl = stl.plot()
        st.pyplot(fig_stl)
    except Exception as e:
        st.info(f"No se pudo descomponer la serie: {e}")

    # --------------------------------------------------
    # 5. MODELO SARIMAX + PRONÓSTICO
    # --------------------------------------------------
    st.subheader("Pronóstico SARIMAX (12 meses)")
    try:
        # Ajuste rápido; puedes cambiar órdenes
        model = SARIMAX(
            df["valor"],
            order=(1, 1, 1),
            seasonal_order=(0, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

        # Pronóstico futuro
        pred = model.get_forecast(steps=12)
        pred_index = pd.date_range(
            start=df["fecha"].iloc[-1] + pd.offsets.MonthBegin(),
            periods=12,
            freq="MS"
        )
        pred_series = pd.Series(pred.predicted_mean, index=pred_index)

        # Gráfico
        fig_pred, ax_pred = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=df, x="fecha", y="valor", label="Histórico", ax=ax_pred)
        sns.lineplot(x=pred_series.index, y=pred_series.values,
                     label="Pronóstico", ax=ax_pred)
        ax_pred.set_ylabel("Consumo (GWh)")
        st.pyplot(fig_pred)

        # --------------------------------------------------
        # 5.1 Métricas y residuos
        # --------------------------------------------------
        if len(df) >= 24:
            y_true = df["valor"].iloc[-12:]
            y_pred = model.predict(start=len(df) - 12, end=len(df) - 1)
            mae  = mean_absolute_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred, squared=False)

            st.markdown(f"**MAE (últimos 12 m):** {mae:.2f} &nbsp;&nbsp; "
                        f"**RMSE:** {rmse:.2f}")

            fig_res, ax_res = plt.subplots()
            sns.lineplot(x=y_true.index, y=model.resid[-12:], ax=ax_res)
            ax_res.axhline(0, ls="--", c="k"); ax_res.set_title("Residuos 12m")
            st.pyplot(fig_res)

        # --------------------------------------------------
        # 5.2 Descargar pronóstico
        # --------------------------------------------------
        pred_df = pred_series.reset_index()
        pred_df.columns = ["fecha", "pronostico_GWh"]
        st.download_button(
            "⬇️ Descargar pronóstico CSV",
            data=pred_df.to_csv(index=False).encode(),
            file_name="pronostico_energia.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.warning(f"No se pudo ajustar SARIMAX rápidamente: {e}")

else:
    st.info("⬆️ Esperando que subas un archivo…")

