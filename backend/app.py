from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
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

# Robust artifact path finder for local & Vercel serverless environments
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_artifact(filename):
    possible_paths = [
        os.path.join(BASE_DIR, "model", filename),
        os.path.join(BASE_DIR, filename),
        os.path.join(os.path.dirname(BASE_DIR), "backend", "model", filename),
        os.path.join(os.path.dirname(BASE_DIR), "api", "model", filename),
        os.path.join(os.path.dirname(BASE_DIR), "model", filename),
        os.path.join(os.path.dirname(BASE_DIR), filename),
        os.path.join(os.getcwd(), "backend", "model", filename),
        os.path.join(os.getcwd(), "api", "model", filename),
        os.path.join(os.getcwd(), "model", filename),
        os.path.join(os.getcwd(), filename),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return os.path.join(BASE_DIR, "model", filename)

MODEL_PATH = find_artifact("knn_model.joblib")
SCALER_PATH = find_artifact("scaler.joblib")
META_PATH = find_artifact("model_meta.json")

knn_model = None
scaler = None
model_meta = {}

def load_artifacts():
    global knn_model, scaler, model_meta
    try:
        m_path = find_artifact("knn_model.joblib")
        s_path = find_artifact("scaler.joblib")
        meta_path = find_artifact("model_meta.json")
        
        if os.path.exists(m_path) and os.path.exists(s_path):
            knn_model = joblib.load(m_path)
            scaler = joblib.load(s_path)
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    model_meta = json.load(f)
            print(f"Model and Scaler loaded successfully from: {m_path}")
        else:
            print(f"Model artifacts not found at: {m_path}, {s_path}")
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

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Loan Prediction AI | K-Nearest Neighbors</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { background-color: #020617; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  </style>
</head>
<body class="min-h-screen py-8 px-4 sm:px-6 lg:px-8">
  <div class="max-w-6xl mx-auto space-y-6">
    <!-- Header -->
    <header class="flex flex-col sm:flex-row sm:items-center justify-between pb-6 border-b border-slate-800 gap-4">
      <div class="flex items-center gap-3">
        <div class="h-11 w-11 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center font-black text-xl shadow-lg shadow-blue-500/20 text-white">
          KNN
        </div>
        <div>
          <h1 class="text-2xl sm:text-3xl font-extrabold text-white">Loan Prediction AI Engine</h1>
          <p class="text-xs text-slate-400">Tuned K-Nearest Neighbors Classifier (k=16, 85.37% Accuracy)</p>
        </div>
      </div>
      <div class="flex items-center gap-2">
        <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-950/70 text-emerald-400 border border-emerald-800">
          <span class="h-2 w-2 rounded-full bg-emerald-400 animate-pulse"></span>
          FastAPI Online
        </span>
        <a href="/docs" target="_blank" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-blue-400 border border-slate-700 transition">
          Swagger Docs &rarr;
        </a>
      </div>
    </header>

    <!-- Presets -->
    <div class="bg-slate-900 border border-slate-800 rounded-xl p-3 flex flex-wrap items-center justify-between gap-2">
      <span class="text-xs text-slate-400 font-medium">⚡ Quick Test Presets:</span>
      <div class="flex flex-wrap gap-2">
        <button onclick="setPreset('prime')" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-blue-600 text-slate-300 hover:text-white transition">Prime Approved</button>
        <button onclick="setPreset('risky')" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-rose-600 text-slate-300 hover:text-white transition">High-Risk Applicant</button>
        <button onclick="setPreset('urban')" class="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-indigo-600 text-slate-300 hover:text-white transition">Moderate Urban</button>
      </div>
    </div>

    <!-- Main Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
      <!-- Form (7 cols) -->
      <form id="loanForm" class="lg:col-span-7 bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-5 shadow-xl">
        <h2 class="text-base font-bold text-white border-b border-slate-800 pb-3 flex items-center justify-between">
          <span>Applicant & Loan Parameters</span>
          <span class="text-xs text-slate-400 font-normal">11 Features</span>
        </h2>

        <!-- Demographics -->
        <div class="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
          <div>
            <label class="block text-slate-400 mb-1">Gender</label>
            <select id="Gender" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-slate-200">
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </div>
          <div>
            <label class="block text-slate-400 mb-1">Married</label>
            <select id="Married" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-slate-200">
              <option value="Yes">Yes</option>
              <option value="No">No</option>
            </select>
          </div>
          <div>
            <label class="block text-slate-400 mb-1">Dependents</label>
            <select id="Dependents" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-slate-200">
              <option value="0">0</option>
              <option value="1">1</option>
              <option value="2">2</option>
              <option value="3+">3+</option>
            </select>
          </div>
          <div>
            <label class="block text-slate-400 mb-1">Education</label>
            <select id="Education" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-slate-200">
              <option value="Graduate">Graduate</option>
              <option value="Not Graduate">Not Graduate</option>
            </select>
          </div>
          <div>
            <label class="block text-slate-400 mb-1">Self Employed</label>
            <select id="Self_Employed" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-slate-200">
              <option value="No">No</option>
              <option value="Yes">Yes</option>
            </select>
          </div>
          <div>
            <label class="block text-slate-400 mb-1">Property Area</label>
            <select id="Property_Area" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-slate-200">
              <option value="Semiurban">Semiurban</option>
              <option value="Urban">Urban</option>
              <option value="Rural">Rural</option>
            </select>
          </div>
        </div>

        <!-- Financials -->
        <div class="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs pt-2 border-t border-slate-800/80">
          <div>
            <label class="block text-slate-400 mb-1">Applicant Income ($/mo)</label>
            <input type="number" id="ApplicantIncome" value="5500" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-slate-200" required>
          </div>
          <div>
            <label class="block text-slate-400 mb-1">Co-applicant Income ($/mo)</label>
            <input type="number" id="CoapplicantIncome" value="2500" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-slate-200">
          </div>
          <div>
            <label class="block text-slate-400 mb-1">Loan Amount ($ in thousands)</label>
            <input type="number" id="LoanAmount" value="130" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-slate-200 font-bold text-emerald-400" required>
          </div>
          <div>
            <label class="block text-slate-400 mb-1">Loan Term (Months)</label>
            <select id="Loan_Amount_Term" class="w-full bg-slate-800 border border-slate-700 rounded-lg p-2 text-slate-200">
              <option value="360">360 Months (30 Years)</option>
              <option value="240">240 Months (20 Years)</option>
              <option value="180">180 Months (15 Years)</option>
              <option value="120">120 Months (10 Years)</option>
            </select>
          </div>
          <div class="sm:col-span-2">
            <label class="block text-slate-400 mb-1">Credit History Guideline</label>
            <select id="Credit_History" class="w-full bg-slate-800 border border-emerald-700 rounded-lg p-2 text-emerald-300 font-semibold">
              <option value="1.0">1.0 - Meets Guidelines (Good Credit Record)</option>
              <option value="0.0">0.0 - Has Defaults / Unmet Guidelines</option>
            </select>
          </div>
        </div>

        <button type="submit" id="submitBtn" class="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-4 rounded-xl shadow-lg transition flex items-center justify-center gap-2 cursor-pointer text-sm">
          <span>Predict Loan Application Approval</span>
        </button>
      </form>

      <!-- Results (5 cols) -->
      <div class="lg:col-span-5 space-y-4">
        <div id="resultBox" class="bg-slate-900 border border-slate-800 rounded-2xl p-6 text-center space-y-4">
          <div class="h-16 w-16 mx-auto rounded-full bg-slate-800 flex items-center justify-center text-slate-500 font-bold text-2xl">
            ?
          </div>
          <div>
            <h3 class="text-base font-bold text-slate-300">Awaiting Prediction</h3>
            <p class="text-xs text-slate-400 mt-1">Adjust parameters and click Predict Loan Approval to see real-time KNN decision.</p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const presets = {
      prime: { Gender:'Male', Married:'Yes', Dependents:'1', Education:'Graduate', Self_Employed:'No', ApplicantIncome:5500, CoapplicantIncome:2500, LoanAmount:130, Loan_Amount_Term:360, Credit_History:1.0, Property_Area:'Semiurban' },
      risky: { Gender:'Male', Married:'No', Dependents:'0', Education:'Not Graduate', Self_Employed:'Yes', ApplicantIncome:1800, CoapplicantIncome:0, LoanAmount:260, Loan_Amount_Term:360, Credit_History:0.0, Property_Area:'Rural' },
      urban: { Gender:'Female', Married:'No', Dependents:'0', Education:'Graduate', Self_Employed:'No', ApplicantIncome:4200, CoapplicantIncome:0, LoanAmount:110, Loan_Amount_Term:360, Credit_History:1.0, Property_Area:'Urban' }
    };

    function setPreset(key) {
      const p = presets[key];
      for (const k in p) {
        const el = document.getElementById(k);
        if (el) el.value = p[k];
      }
    }

    document.getElementById('loanForm').onsubmit = async (e) => {
      e.preventDefault();
      const btn = document.getElementById('submitBtn');
      btn.innerText = 'Analyzing with KNN Model (k=16)...';
      btn.disabled = true;

      const payload = {
        Gender: document.getElementById('Gender').value,
        Married: document.getElementById('Married').value,
        Dependents: document.getElementById('Dependents').value,
        Education: document.getElementById('Education').value,
        Self_Employed: document.getElementById('Self_Employed').value,
        ApplicantIncome: parseFloat(document.getElementById('ApplicantIncome').value) || 0,
        CoapplicantIncome: parseFloat(document.getElementById('CoapplicantIncome').value) || 0,
        LoanAmount: parseFloat(document.getElementById('LoanAmount').value) || 0,
        Loan_Amount_Term: parseFloat(document.getElementById('Loan_Amount_Term').value) || 360,
        Credit_History: parseFloat(document.getElementById('Credit_History').value) || 0,
        Property_Area: document.getElementById('Property_Area').value
      };

      try {
        const endpoints = ['/api/predict', '/predict'];
        let res = null;
        for (const ep of endpoints) {
          try {
            res = await fetch(ep, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
            });
            if (res.ok) break;
          } catch(err){}
        }

        if (!res || !res.ok) throw new Error('API request failed');
        const data = await res.json();
        renderResult(data);
      } catch (err) {
        document.getElementById('resultBox').innerHTML = `
          <div class="text-rose-400 font-bold text-sm">Error running prediction: ${err.message}</div>
        `;
      } finally {
        btn.innerText = 'Predict Loan Application Approval';
        btn.disabled = false;
      }
    };

    function renderResult(data) {
      const isApproved = data.approved;
      const box = document.getElementById('resultBox');
      box.className = `rounded-2xl p-6 border shadow-2xl space-y-4 ${
        isApproved
          ? 'bg-gradient-to-b from-slate-900 to-emerald-950/40 border-emerald-800'
          : 'bg-gradient-to-b from-slate-900 to-rose-950/40 border-rose-800'
      }`;

      box.innerHTML = `
        <div class="inline-flex items-center justify-center w-16 h-16 rounded-full ${
          isApproved ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'
        } font-black text-2xl">
          ${isApproved ? '✓' : '✕'}
        </div>
        <div>
          <span class="text-[10px] font-bold uppercase tracking-wider px-2.5 py-0.5 rounded-full ${
            isApproved ? 'bg-emerald-500/20 text-emerald-300' : 'bg-rose-500/20 text-rose-300'
          }">Outcome</span>
          <h2 class="text-2xl font-black mt-1 ${isApproved ? 'text-emerald-400' : 'text-rose-400'}">
            ${isApproved ? 'LOAN APPROVED' : 'LOAN REJECTED'}
          </h2>
          <p class="text-xs text-slate-400 mt-1">Confidence: <strong class="text-white">${(data.confidence*100).toFixed(1)}%</strong></p>
        </div>

        <div class="py-3 border-y border-slate-800 space-y-2 text-xs text-left">
          <div class="flex justify-between">
            <span class="text-slate-400">Approval Probability</span>
            <span class="text-emerald-400 font-mono font-bold">${(data.probability_approved*100).toFixed(1)}%</span>
          </div>
          <div class="w-full bg-slate-800 rounded-full h-2">
            <div class="bg-emerald-500 h-2 rounded-full" style="width:${data.probability_approved*100}%"></div>
          </div>
          <div class="flex justify-between pt-1">
            <span class="text-slate-400">Rejection Probability</span>
            <span class="text-rose-400 font-mono font-bold">${(data.probability_rejected*100).toFixed(1)}%</span>
          </div>
          <div class="w-full bg-slate-800 rounded-full h-2">
            <div class="bg-rose-500 h-2 rounded-full" style="width:${data.probability_rejected*100}%"></div>
          </div>
        </div>

        <div class="grid grid-cols-2 gap-2 text-center text-xs">
          <div class="bg-slate-950/80 p-2.5 rounded-lg border border-emerald-900/50">
            <span class="text-lg font-bold text-emerald-400">${data.neighbors_vote.approved}</span>
            <p class="text-[10px] text-slate-400">Approved Neighbors</p>
          </div>
          <div class="bg-slate-950/80 p-2.5 rounded-lg border border-rose-900/50">
            <span class="text-lg font-bold text-rose-400">${data.neighbors_vote.rejected}</span>
            <p class="text-[10px] text-slate-400">Rejected Neighbors</p>
          </div>
        </div>

        <div class="text-left pt-2 text-xs">
          <p class="text-[11px] font-bold text-slate-400 uppercase tracking-wider mb-2">Decision Factors:</p>
          <ul class="space-y-1 text-slate-300">
            ${data.analysis.map(pt => `<li class="flex items-start gap-1.5"><span class="text-blue-400">•</span><span>${pt}</span></li>`).join('')}
          </ul>
        </div>
      `;
    }
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
def read_root(request: Request = None):
    if request:
        accept = request.headers.get("accept", "")
        if "application/json" in accept and "text/html" not in accept:
            return JSONResponse({
                "service": "Loan Prediction KNN API",
                "status": "online",
                "docs_url": "/docs",
                "endpoints": ["/api/health", "/api/model-info", "/api/predict"]
            })
    return HTMLResponse(content=INDEX_HTML)

@app.get("/api")
@app.get("/api/")
def api_root():
    return {
        "service": "Loan Prediction KNN API",
        "status": "online",
        "docs_url": "/docs",
        "endpoints": ["/api/health", "/api/model-info", "/api/predict"]
    }

@app.get("/api/health")
@app.get("/health")
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
@app.get("/model-info")
def get_model_info():
    if not model_meta:
        load_artifacts()
    return model_meta

@app.post("/api/predict", response_model=PredictionResponse)
@app.post("/predict", response_model=PredictionResponse)
def predict_loan(application: LoanApplicationRequest):
    global knn_model, scaler
    if knn_model is None or scaler is None:
        load_artifacts()
        if knn_model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Machine learning model is not yet loaded.")

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

        feature_cols = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 
            'Credit_History', 'Property_Area'
        ]
        X_df = pd.DataFrame([features_vector], columns=feature_cols)
        X_scaled = scaler.transform(X_df)

        pred = int(knn_model.predict(X_scaled)[0])
        proba = knn_model.predict_proba(X_scaled)[0]
        prob_rejected = float(proba[0])
        prob_approved = float(proba[1])

        n_neighbors = knn_model.n_neighbors
        approved_votes = int(round(prob_approved * n_neighbors))
        rejected_votes = n_neighbors - approved_votes

        is_approved = bool(pred == 1)
        confidence = prob_approved if is_approved else prob_rejected

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
