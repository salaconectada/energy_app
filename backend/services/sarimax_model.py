from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def forecast_sarimax(df_m: pd.DataFrame, exog_cols: list[str], periods: int) -> tuple[pd.Series, object]:
    y_train = df_m["valor"].astype(float).ffill()
    exog_train = df_m[exog_cols] if exog_cols else None

    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=(1, 1, 1),
        seasonal_order=(0, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    fut_exog = (
        pd.concat([exog_train.iloc[[-1]]] * periods, ignore_index=True) if exog_cols else None
    )

    fc = model.get_forecast(periods, exog=fut_exog)
    pred_values = np.asarray(fc.predicted_mean).astype(float)
    pred_index = pd.date_range(
        df_m["fecha"].iloc[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS"
    )

    return pd.Series(pred_values, index=pred_index, name="pronostico"), model
