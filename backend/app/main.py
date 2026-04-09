from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from backend.services.forecast import run_forecast
from backend.services.preprocessing import DataValidationError, load_dataset
from backend.services.utils import forecast_to_frame, serialize_frame

app = FastAPI(title="Energy-APP API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/forecast")
async def forecast(
    file: UploadFile = File(...),
    model: str = Form("SARIMAX"),
    periods: int = Form(12),
    exog: str = Form(""),
) -> dict:
    try:
        raw = await file.read()
        df = load_dataset(raw, file.filename or "dataset.csv")
    except DataValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    exog_cols = [x.strip() for x in exog.split(",") if x.strip()]
    result = run_forecast(df, model=model, periods=periods, exog_sel=exog_cols)

    return {
        "model": model,
        "periods": periods,
        "kpis": result.kpis,
        "metrics": result.metrics,
        "forecast": forecast_to_frame(result.pred).to_dict(orient="records"),
        "history_monthly": serialize_frame(result.monthly_df.tail(24)),
    }
