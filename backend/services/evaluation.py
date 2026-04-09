from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_kpis(df: pd.DataFrame) -> dict[str, float]:
    trend_pct = (
        df.set_index("fecha")["valor"].resample("YE").sum().pct_change().mean() * 100
    )
    monthly_means = df.groupby(df["fecha"].dt.month)["valor"].mean()
    seasonality = monthly_means.max() / monthly_means.min() if len(monthly_means) > 0 else np.nan

    return {
        "total_historico_gwh": float(df["valor"].sum()),
        "tendencia_anual_media_pct": float(trend_pct),
        "indice_estacionalidad": float(seasonality),
    }


def sarimax_fit_metrics(real: pd.Series, fitted: pd.Series, window: int = 12) -> dict[str, float]:
    real_w = real.iloc[-window:]
    fitted_w = fitted.iloc[-window:]

    mae = mean_absolute_error(real_w, fitted_w)
    rmse = np.sqrt(mean_squared_error(real_w, fitted_w))
    perc = mae / max(real.mean(), 1e-9) * 100

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "mae_pct": float(perc),
    }
