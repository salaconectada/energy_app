from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backend.services.evaluation import compute_kpis, sarimax_fit_metrics
from backend.services.preprocessing import get_exogenous_columns, monthly_resample
from backend.services.rf_model import forecast_random_forest
from backend.services.sarimax_model import forecast_sarimax


@dataclass
class ForecastResult:
    pred: pd.Series
    kpis: dict[str, float]
    metrics: dict[str, float] | None
    monthly_df: pd.DataFrame


def run_forecast(df: pd.DataFrame, model: str, periods: int, exog_sel: list[str] | None = None) -> ForecastResult:
    exog_sel = exog_sel or []
    available_exog = get_exogenous_columns(df)
    exog_sel = [c for c in exog_sel if c in available_exog]

    df_m = monthly_resample(df, fill_cols=exog_sel)
    kpis = compute_kpis(df)

    if model.upper() == "SARIMAX":
        pred, fitted_model = forecast_sarimax(df_m, exog_cols=exog_sel, periods=periods)
        metrics = sarimax_fit_metrics(df_m["valor"], fitted_model.fittedvalues)
    else:
        pred = forecast_random_forest(df_m, exog_cols=exog_sel, periods=periods)
        metrics = None

    return ForecastResult(pred=pred, kpis=kpis, metrics=metrics, monthly_df=df_m)
