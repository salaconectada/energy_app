from __future__ import annotations

from io import BytesIO
from typing import Iterable

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Error de validación para datasets de entrada."""


def load_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    buffer = BytesIO(file_bytes)

    if ext == "csv":
        df = pd.read_csv(buffer)
    elif ext in {"xlsx", "xls"}:
        df = pd.read_excel(buffer)
    else:
        raise DataValidationError("Formato no soportado. Use CSV o Excel.")

    return normalize_dataset(df)


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = clean.columns.str.strip().str.lower()

    if "fecha" not in clean.columns:
        raise DataValidationError("Falta columna 'fecha'.")

    clean["fecha"] = pd.to_datetime(clean["fecha"], errors="coerce")
    clean = clean.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

    if "valor" not in clean.columns:
        numeric_candidates = clean.select_dtypes(include=[np.number]).columns
        if len(numeric_candidates) == 0:
            raise DataValidationError("No hay columna numérica para usar como 'valor'.")
        clean = clean.rename(columns={numeric_candidates[0]: "valor"})

    clean["valor"] = pd.to_numeric(clean["valor"], errors="coerce")

    for c in clean.columns.difference(["fecha", "valor"]):
        clean[c] = pd.to_numeric(clean[c], errors="coerce")

    clean = clean.dropna(subset=["valor"])
    return clean


def get_exogenous_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in ("fecha", "valor") and np.issubdtype(df[c].dtype, np.number)
    ]


def monthly_resample(df: pd.DataFrame, fill_cols: Iterable[str]) -> pd.DataFrame:
    out = (
        df.set_index("fecha")
        .resample("MS")
        .mean(numeric_only=True)
        .interpolate("linear", limit_direction="both")
        .reset_index()
    )

    fill_cols = list(fill_cols)
    if fill_cols:
        out[fill_cols] = out[fill_cols].ffill().bfill()

    return out
