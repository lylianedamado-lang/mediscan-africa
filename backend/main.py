import joblib
import pandas as pd
import numpy as np
import os
import json
import hashlib
from typing import List
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="MediScan Africa API", description="Plateforme Intelligente de Detection de Maladies", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
models = {}
diagnostics_history = []
cache = {}

def load_models():
    try:
        models["cancer"] = joblib.load(os.path.join(MODELS_DIR, "breast_cancer_best_model.joblib"))
        models["cancer_scaler"] = joblib.load(os.path.join(MODELS_DIR, "breast_cancer_scaler.joblib"))
        print("Modele Cancer charge")
    except Exception as e:
        print(f"Erreur Cancer : {e}")
    try:
        models["heart"] = joblib.load(os.path.join(MODELS_DIR, "heart_disease_best_model.joblib"))
        models["heart_scaler"] = joblib.load(os.path.join(MODELS_DIR, "heart_disease_scaler.joblib"))
        print("Modele Heart charge")
    except Exception as e:
        print(f"Erreur Heart : {e}")

load_models()

class CancerInput(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float

class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalch: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: int

class DiagnosticResponse(BaseModel):
    pathology: str
    prediction: str
    probability: float
    severity: str
    recommendations: list

def get_cache_key(prefix, data_dict):
    payload_str = json.dumps(data_dict, sort_keys=True)
    return f"{prefix}_{hashlib.md5(payload_str.encode()).hexdigest()}"

def log_diagnostic(result):
    result_copy = dict(result)
    result_copy["timestamp"] = datetime.now().isoformat()
    diagnostics_history.append(result_copy)

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l API MediScan Africa", "status": "online", "docs": "/docs"}

@app.get("/health")
def health_check():
    return {"status": "healthy" if all(k in models for k in ["cancer","heart"]) else "degraded", "models": {"cancer": "loaded" if "cancer" in models else "missing", "heart": "loaded" if "heart" in models else "missing"}, "total_diagnostics": len(diagnostics_history)}

@app.post("/diagnostic/cancer", response_model=DiagnosticResponse)
def predict_cancer(data: CancerInput):
    if "cancer" not in models:
        raise HTTPException(status_code=500, detail="Modele Cancer non charge")
    input_dict = data.model_dump()
    cache_key = get_cache_key("cancer", input_dict)
    if cache_key in cache:
        return cache[cache_key]
    input_data = pd.DataFrame([input_dict])
    input_data.columns = [c.replace("concave_points", "concave points") for c in input_data.columns]
    try:
        scaled = models["cancer_scaler"].transform(input_data)
        pred = int(models["cancer"].predict(scaled)[0])
        proba = float(models["cancer"].predict_proba(scaled)[0][1])
        if pred == 1:
            label = "MALIGNANT (CANCER DETECTE)"
            sev = "HIGH" if proba > 0.7 else "MEDIUM"
            recs = ["Consultation oncologique urgente.", "Biopsie complementaire necessaire.", "Imagerie medicale a programmer.", "Orientation vers centre cancerologie."]
        else:
            label = "BENIN (PAS DE CANCER)"
            sev = "LOW"
            recs = ["Resultat rassurant.", "Surveillance annuelle recommandee.", "Consulter si nouvelle masse."]
        result = {"pathology": "Breast Cancer", "prediction": label, "probability": round(proba, 4), "severity": sev, "recommendations": recs}
        cache[cache_key] = result
        log_diagnostic(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur: {str(e)}")

@app.post("/diagnostic/heart", response_model=DiagnosticResponse)
def predict_heart(data: HeartInput):
    if "heart" not in models:
        raise HTTPException(status_code=500, detail="Modele Heart non charge")
    input_dict = data.model_dump()
    cache_key = get_cache_key("heart", input_dict)
    if cache_key in cache:
        return cache[cache_key]
    input_data = pd.DataFrame([input_dict])
    try:
        scaled = models["heart_scaler"].transform(input_data)
        pred = int(models["heart"].predict(scaled)[0])
        proba = float(models["heart"].predict_proba(scaled)[0][1])
        if pred == 1:
            label = "MALADIE CARDIOVASCULAIRE DETECTEE"
            sev = "CRITICAL" if proba > 0.8 else "MODERATE"
            recs = ["Consultation cardiologique immediate.", "ECG et echocardiographie a programmer.", "Bilan lipidique a controler."]
        else:
            label = "SAIN"
            sev = "LOW"
            recs = ["Resultat rassurant.", "Activite physique reguliere.", "Controle annuel recommande."]
        result = {"pathology": "Cardiovascular Disease", "prediction": label, "probability": round(proba, 4), "severity": sev, "recommendations": recs}
        cache[cache_key] = result
        log_diagnostic(result)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur: {str(e)}")

@app.get("/patient/{patient_id}/historique")
def get_patient_history(patient_id: str):
    return {"patient_id": patient_id, "total": len(diagnostics_history), "diagnostics": diagnostics_history[-10:]}

@app.get("/stats/aggregees")
def get_stats():
    total = len(diagnostics_history)
    cancer = sum(1 for d in diagnostics_history if "Cancer" in d["pathology"])
    heart = total - cancer
    return {"total_diagnostics": total, "cancer": cancer, "cardiovasculaire": heart}

@app.get("/diagnostic/history")
def get_history():
    return {"total": len(diagnostics_history), "diagnostics": diagnostics_history[-20:]}

if __name__ == "__main__":
    import uvicorn
    print("MediScan Africa API")
    uvicorn.run(app, host="0.0.0.0", port=8000)
