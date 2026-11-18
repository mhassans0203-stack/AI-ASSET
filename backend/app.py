from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import json
import os
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

app = FastAPI(
    title="Loan Prediction KNN API",
    description="REST API serving a K-Nearest Neighbors Classifier for loan approval prediction.",
    version="1.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "knn_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.joblib")
META_PATH = os.path.join(BASE_DIR, "model", "model_meta.json")

# Fallbacks to workspace root if needed
if not os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(os.path.dirname(BASE_DIR), "knn_model.joblib")):
    MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "knn_model.joblib")
    SCALER_PATH = os.path.join(os.path.dirname(BASE_DIR), "scaler.joblib")
    META_PATH = os.path.join(os.path.dirname(BASE_DIR), "model_meta.json")

knn_model = None
scaler = None
model_meta = {}

def load_artifacts():
    global knn_model, scaler, model_meta
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            knn_model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            if os.path.exists(META_PATH):
                with open(META_PATH, "r") as f:
                    model_meta = json.load(f)
            print("Model and Scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

load_artifacts()

class LoanApplicationRequest(BaseModel):
    Gender: str = Field(default="Male", description="Gender: 'Male' or 'Female'")
    Married: str = Field(default="Yes", description="Marital status: 'Yes' or 'No'")
    Dependents: str = Field(default="0", description="Dependents: '0', '1', '2', or '3+'")
    Education: str = Field(default="Graduate", description="Education: 'Graduate' or 'Not Graduate'")
    Self_Employed: str = Field(default="No", description="Self-employed: 'Yes' or 'No'")
    ApplicantIncome: float = Field(default=5000.0, ge=0, description="Primary applicant monthly income")
    CoapplicantIncome: float = Field(default=0.0, ge=0, description="Co-applicant monthly income")
    LoanAmount: float = Field(default=120.0, gt=0, description="Loan amount in thousands (e.g. 120 = $120,000)")
    Loan_Amount_Term: float = Field(default=360.0, gt=0, description="Loan term in months (e.g. 360, 180)")
    Credit_History: float = Field(default=1.0, description="Credit history meets guidelines: 1.0 or 0.0")
    Property_Area: str = Field(default="Urban", description="Property area: 'Urban', 'Semiurban', or 'Rural'")

class PredictionResponse(BaseModel):
    approved: bool
    prediction_label: str
    confidence: float
    probability_approved: float
    probability_rejected: float
    neighbors_vote: Dict[str, int]
    features_received: Dict[str, Any]
    analysis: List[str]

@app.get("/")
def read_root():
    return {
        "service": "Loan Prediction KNN API",
        "status": "online",
        "docs_url": "/docs",
        "endpoints": ["/api/health", "/api/model-info", "/api/predict"]
    }

@app.get("/api/health")
def health_check():
    if knn_model is None or scaler is None:
        load_artifacts()
    return {
        "status": "healthy" if knn_model is not None else "degraded",
        "model_loaded": knn_model is not None,
        "scaler_loaded": scaler is not None,
        "optimal_k": model_meta.get("optimal_k", 16) if knn_model else None
    }

@app.get("/api/model-info")
def get_model_info():
    if not model_meta:
        load_artifacts()
    return model_meta

@app.post("/api/predict", response_model=PredictionResponse)
def predict_loan(application: LoanApplicationRequest):
    global knn_model, scaler
    if knn_model is None or scaler is None:
        load_artifacts()
        if knn_model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Machine learning model is not yet loaded.")

    # Categorical encoding maps matching model_training.py
    gender_map = {"Female": 0, "Male": 1}
    married_map = {"No": 0, "Yes": 1}
    dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3, "3": 3}
    education_map = {"Graduate": 1, "Not Graduate": 0}
    self_emp_map = {"No": 0, "Yes": 1}
    property_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}

    try:
        gender_val = gender_map.get(application.Gender, 1)
        married_val = married_map.get(application.Married, 1)
        dep_val = dependents_map.get(str(application.Dependents), 0)
        edu_val = education_map.get(application.Education, 1)
        self_emp_val = self_emp_map.get(application.Self_Employed, 0)
        prop_val = property_map.get(application.Property_Area, 2)
        credit_val = float(application.Credit_History)
        
        # Feature order must match model training exactly:
        # ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
        #  'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
        #  'Credit_History', 'Property_Area']
        features_vector = [
            gender_val,
            married_val,
            dep_val,
            edu_val,
            self_emp_val,
            float(application.ApplicantIncome),
            float(application.CoapplicantIncome),
            float(application.LoanAmount),
            float(application.Loan_Amount_Term),
            credit_val,
            prop_val
        ]

        # Wrap as DataFrame with feature names to match StandardScaler fit
        feature_cols = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
            'Credit_History', 'Property_Area'
        ]
        X_df = pd.DataFrame([features_vector], columns=feature_cols)
        X_scaled = scaler.transform(X_df)

        # KNN prediction and probabilities
        pred = int(knn_model.predict(X_scaled)[0])
        proba = knn_model.predict_proba(X_scaled)[0]
        prob_rejected = float(proba[0])
        prob_approved = float(proba[1])

        # Get nearest neighbor votes
        distances, indices = knn_model.kneighbors(X_scaled)
        # Reconstruct votes from neighbors in training data if accessible or from proba
        n_neighbors = knn_model.n_neighbors
        approved_votes = int(round(prob_approved * n_neighbors))
        rejected_votes = n_neighbors - approved_votes

        is_approved = bool(pred == 1)
        confidence = prob_approved if is_approved else prob_rejected

        # Generate intelligent decision insights
        analysis_points = []
        if credit_val == 1.0:
            analysis_points.append("Positive credit history strongly supports approval eligibility.")
        else:
            analysis_points.append("Adverse credit history significantly elevates default risk.")

        total_income = application.ApplicantIncome + application.CoapplicantIncome
        if total_income > 0:
            loan_to_income = (application.LoanAmount * 1000) / (total_income * 12)
            if loan_to_income < 2.5:
                analysis_points.append(f"Healthy loan-to-annual-income ratio ({loan_to_income:.2f}x).")
            elif loan_to_income > 5.0:
                analysis_points.append(f"Elevated loan-to-annual-income ratio ({loan_to_income:.2f}x) requests high leverage.")

        if application.Property_Area == "Semiurban":
            analysis_points.append("Semiurban property locations correlate with higher historical loan approval rates.")

        analysis_points.append(
            f"Based on {n_neighbors} nearest historical applicants, {approved_votes} were approved and {rejected_votes} were rejected."
        )

        return PredictionResponse(
            approved=is_approved,
            prediction_label="Approved" if is_approved else "Rejected",
            confidence=round(confidence, 4),
            probability_approved=round(prob_approved, 4),
            probability_rejected=round(prob_rejected, 4),
            neighbors_vote={
                "approved": approved_votes,
                "rejected": rejected_votes
            },
            features_received=application.model_dump(),
            analysis=analysis_points
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
