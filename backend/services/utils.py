from __future__ import annotations

import pandas as pd


def forecast_to_frame(pred: pd.Series) -> pd.DataFrame:
    return (
        pred.reset_index()
        .rename(columns={"index": "fecha", pred.name or 0: "pronostico_gwh"})
        .assign(fecha=lambda x: x["fecha"].dt.strftime("%Y-%m-%d"))
    )


def serialize_frame(df: pd.DataFrame) -> list[dict]:
    records = df.copy()
    if "fecha" in records.columns:
        records["fecha"] = pd.to_datetime(records["fecha"], errors="coerce").dt.strftime("%Y-%m-%d")
    return records.to_dict(orient="records")
