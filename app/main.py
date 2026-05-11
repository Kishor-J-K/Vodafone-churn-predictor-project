from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Vodafone Churn Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "model", "preprocessor.pkl")

model = None
preprocessor = None


@app.on_event("startup")
def load_model():
    global model, preprocessor
    if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("✓ Model and preprocessor loaded.")
    else:
        print("WARNING: model files not found. Run train.py first.")


# ── Pydantic schema — matches voice_customer_churn.csv exactly ──────────────
class CustomerInput(BaseModel):
    gender: Optional[str] = Field("Male", example="Male")
    SeniorCitizen: int = Field(0, ge=0, le=1, example=0)
    Dependents: str = Field("No", example="No")
    tenure: int = Field(6, ge=0, le=72, example=6)
    PhoneService: str = Field("Yes", example="Yes")
    MultipleLines: str = Field("No", example="No")
    InternetService: str = Field("Fiber optic", example="Fiber optic")
    OnlineSecurity: str = Field("No", example="No")
    OnlineBackup: str = Field("No", example="No")
    DeviceProtection: str = Field("No", example="No")
    TechSupport: str = Field("No", example="No")
    StreamingTV: Optional[str] = Field("No", example="No")
    StreamingMovies: Optional[str] = Field("No", example="No")
    Contract: str = Field("Month-to-month", example="Month-to-month")
    PaperlessBilling: str = Field("Yes", example="Yes")
    PaymentMethod: str = Field("Electronic check", example="Electronic check")
    MonthlyCharges: float = Field(85.0, ge=0, example=85.0)
    TotalCharges: float = Field(510.0, ge=0, example=510.0)
    numAdminTickets: Optional[float] = Field(0, ge=0, example=0)
    numTechTickets: int = Field(0, ge=0, example=0)
    Location: Optional[str] = Field("North - New York", example="North - New York")


class PredictionResponse(BaseModel):
    churn: bool
    probability: float
    risk_level: str
    top_factors: list[str]


@app.get("/")
def root():
    return {"message": "Vodafone Churn Predictor API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    # Build dataframe in the same column order as training (minus customerID and Churn)
    data = {
        "gender":            [customer.gender],
        "SeniorCitizen":     [customer.SeniorCitizen],
        "Dependents":        [customer.Dependents],
        "tenure":            [customer.tenure],
        "PhoneService":      [customer.PhoneService],
        "MultipleLines":     [customer.MultipleLines],
        "InternetService":   [customer.InternetService],
        "OnlineSecurity":    [customer.OnlineSecurity],
        "OnlineBackup":      [customer.OnlineBackup],
        "DeviceProtection":  [customer.DeviceProtection],
        "TechSupport":       [customer.TechSupport],
        "StreamingTV":       [customer.StreamingTV],
        "StreamingMovies":   [customer.StreamingMovies],
        "Contract":          [customer.Contract],
        "PaperlessBilling":  [customer.PaperlessBilling],
        "PaymentMethod":     [customer.PaymentMethod],
        "MonthlyCharges":    [customer.MonthlyCharges],
        "TotalCharges":      [str(customer.TotalCharges)],   # trained as string → coerced
        "numAdminTickets":   [customer.numAdminTickets],
        "numTechTickets":    [customer.numTechTickets],
        "Location":          [customer.Location],
    }

    df = pd.DataFrame(data)

    # TotalCharges was object in training, coerce here
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    try:
        X = preprocessor.transform(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

    try:
        prob = float(model.predict_proba(X)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    churn = prob >= 0.5
    risk = "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"
    factors = _explain(customer, prob)

    return PredictionResponse(
        churn=churn,
        probability=round(prob, 4),
        risk_level=risk,
        top_factors=factors,
    )


def _explain(c: CustomerInput, prob: float) -> list[str]:
    factors = []
    if c.Contract == "Month-to-month":
        factors.append("Month-to-month contract is the #1 churn driver")
    elif c.Contract == "Two year":
        factors.append("Two-year contract strongly reduces churn risk")
    if c.tenure < 12:
        factors.append(f"Short tenure ({c.tenure} months) — still in high-risk window")
    elif c.tenure > 36:
        factors.append(f"Long tenure ({c.tenure} months) — strong loyalty signal")
    if c.InternetService == "Fiber optic":
        factors.append("Fiber optic users churn at 42% — monitor satisfaction")
    if c.OnlineSecurity == "No" and c.InternetService != "No":
        factors.append("No online security — protective service missing")
    if c.TechSupport == "No" and c.InternetService != "No":
        factors.append("No tech support — correlates with dissatisfaction")
    if c.MonthlyCharges > 65:
        factors.append(f"High monthly charges (${c.MonthlyCharges:.0f}) increase price-sensitivity risk")
    if c.PaymentMethod == "Electronic check":
        factors.append("Electronic check payment correlates with higher churn")
    if c.SeniorCitizen == 1:
        factors.append("Senior citizens churn at higher rates")
    if c.Dependents == "Yes":
        factors.append("Has dependents — protective retention factor")
    if c.numAdminTickets and c.numAdminTickets > 2:
        factors.append(f"{int(c.numAdminTickets)} admin tickets — elevated service friction")
    if c.numTechTickets > 2:
        factors.append(f"{c.numTechTickets} tech tickets — suggests recurring technical issues")
    if not factors:
        factors.append("No strong individual risk factors identified")
    return factors[:4]


# Serve React frontend in production
frontend_dist = os.path.join(BASE_DIR, "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/assets", StaticFiles(directory=f"{frontend_dist}/assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend(full_path: str):
        return FileResponse(os.path.join(frontend_dist, "index.html"))