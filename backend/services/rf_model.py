from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def _build_features(frame: pd.DataFrame, lags: int, exog_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lags, len(frame)):
        lag_feats = frame["valor"].shift(range(1, lags + 1)).iloc[i].values
        if exog_cols:
            lag_feats = np.concatenate([lag_feats, frame[exog_cols].iloc[i].values])
        X.append(lag_feats)
        y.append(frame["valor"].iloc[i])

    return np.array(X), np.array(y)


def forecast_random_forest(
    df_m: pd.DataFrame, exog_cols: list[str], periods: int, lags: int = 12
) -> pd.Series:
    X, y = _build_features(df_m, lags=lags, exog_cols=exog_cols)
    rf = RandomForestRegressor(n_estimators=400, random_state=0)
    rf.fit(X, y)

    tmp = df_m.copy()
    preds = []

    for _ in range(periods):
        feats = tmp["valor"].iloc[-lags:][::-1].values
        if exog_cols:
            feats = np.concatenate([feats, tmp[exog_cols].iloc[-1].values])

        y_hat = float(rf.predict(feats.reshape(1, -1))[0])
        preds.append(y_hat)

        new = {
            "fecha": tmp["fecha"].iloc[-1] + pd.offsets.MonthBegin(),
            "valor": y_hat,
            **{c: tmp[c].iloc[-1] for c in exog_cols},
        }
        tmp = pd.concat([tmp, pd.DataFrame([new])], ignore_index=True)

    pred_index = pd.date_range(
        df_m["fecha"].iloc[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS"
    )
    return pd.Series(preds, index=pred_index, name="pronostico")
