# Loan Application Approval Prediction using K-Nearest Neighbors (KNN)

An end-to-end Machine Learning and Full-Stack Web Application to predict whether a loan application will be **Approved** or **Rejected** based on applicant profiles, credit history, and financial metrics.

Dataset: [Kaggle Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)

---

## 🌟 Key Features

- **Data Preprocessing & Imputation**:
  - Automatically loads and inspects the dataset (`train.csv`).
  - Missing values imputed using **median** for numerical features and **mode** for categorical features.
  - Categorical variables encoded into numerical values.
  - Stratified 80/20 train/test split.
  - Feature standardization using scikit-learn's `StandardScaler`.
- **K-Nearest Neighbors (KNN) Modeling**:
  - Default KNN classifier ($k=5$).
  - Systematic empirical exploration across various neighborhood sizes ($k \in [1, 3, 5, 7, 9, 11, 15, 17, 21, 25]$).
  - Hyperparameter tuning for `n_neighbors` via 5-fold cross-validation grid search ($k=16$ selected as optimal).
- **Comprehensive Evaluation & Visualizations**:
  - Accuracy, Precision, Recall, F1-Score, and Confusion Matrix analysis.
  - Side-by-side Confusion Matrix comparison plot (Default vs Tuned).
  - Hyperparameter tuning curve ($k$ vs Accuracy).
- **FastAPI REST API Backend**:
  - Fast, typed REST API serving predictions, confidence scores, and nearest-neighbor voting distribution.
  - Automatic Swagger documentation at `http://localhost:8000/docs`.
- **Modern Next.js Web Application**:
  - Interactive UI built with Next.js 14 and Tailwind CSS.
  - Form with 11 applicant features, real-time ratio calculations, and 1-click test presets (Prime Approved, High-Risk, Moderate Urban).
  - Real-time loan prediction results with visual status badges, probability meters, neighbor vote counters, and underwriting risk analysis.
  - Built-in Model Performance modal showing evaluation metrics and high-resolution visualization charts.

---

## 📊 Model Performance & Results

### 1. Default vs Tuned KNN Comparison

| Metric | Default KNN ($k=5$) | Tuned KNN ($k=16$) | Improvement |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **84.55%** | **85.37%** | **+0.82%** |
| **True Positives (Approved)** | 82 / 85 (96.5%) | 84 / 85 (98.8%) | +2 correct approvals |
| **True Negatives (Rejected)** | 22 / 38 (57.9%) | 21 / 38 (55.3%) | High precision (95%) |
| **False Negatives (Type II)** | 3 | 1 | **-66.7% error reduction** |
| **False Positives (Type I)** | 16 | 17 | Controlled |

### 2. Empirical Performance across Different $k$ Values

| $k$ | Train Accuracy | Test Accuracy | 5-Fold Cross Validation Mean |
| :---: | :---: | :---: | :---: |
| 1 | 100.00% | 76.42% | 67.84% ($\pm$ 4.9%) |
| 3 | 83.30% | 82.93% | 74.34% ($\pm$ 3.7%) |
| 5 | 81.67% | 84.55% | 78.21% ($\pm$ 1.3%) |
| 7 | 81.26% | 84.55% | 78.00% ($\pm$ 1.1%) |
| 9 | 80.24% | 85.37% | 78.41% ($\pm$ 0.8%) |
| 11 | 79.63% | 84.55% | 79.02% ($\pm$ 1.5%) |
| 15 | 79.43% | 85.37% | 78.82% ($\pm$ 1.2%) |
| **16 (Tuned)** | **79.23%** | **85.37%** | **79.63% ($\pm$ 1.0%)** |
| 21 | 78.82% | 83.74% | 79.22% ($\pm$ 1.1%) |
| 25 | 79.23% | 82.93% | 79.02% ($\pm$ 1.0%) |

---

## 📁 Repository Structure

```
KNN Model/
├── backend/
│   ├── app.py                     # FastAPI REST API application
│   └── model/
│       ├── knn_model.joblib       # Tuned KNN Classifier
│       ├── scaler.joblib          # Fitted StandardScaler
│       └── model_meta.json        # Preprocessing rules & metrics
├── frontend/
│   ├── app/
│   │   ├── globals.css            # Tailwind custom styles
│   │   ├── layout.tsx             # Root layout & page metadata
│   │   └── page.tsx               # Main loan prediction UI
│   ├── public/
│   │   └── visualizations/        # Plots served in the web application
│   ├── package.json
│   └── tailwind.config.ts
├── visualizations/
│   ├── confusion_matrix_comparison.png
│   └── knn_k_accuracy_tuning.png
├── train.csv                      # Kaggle Loan Prediction Dataset
├── model_training.py              # ML pipeline (training, tuning, evaluation, plots)
├── loan_prediction_knn.ipynb      # Step-by-step Jupyter Notebook
├── requirements.txt               # Python dependencies
├── run_servers.py                 # Multi-server concurrent launcher
└── start.bat                      # 1-click Windows launcher
```

---

## 🚀 How to Run

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Re-train the Model & Generate Visualizations
```bash
python model_training.py
```

### Step 3: Launch Full-Stack Application (Backend + Frontend)
You can start both servers simultaneously using:
```bash
python run_servers.py
```
Or double-click `start.bat`.

Once launched:
- **Next.js Web UI**: `http://localhost:3000`
- **FastAPI REST API**: `http://localhost:8000`
- **Swagger Interactive API Documentation**: `http://localhost:8000/docs`

---

## 🔌 API Reference (`POST /api/predict`)

### Request Example
```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "1",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000.0,
  "CoapplicantIncome": 2000.0,
  "LoanAmount": 130.0,
  "Loan_Amount_Term": 360.0,
  "Credit_History": 1.0,
  "Property_Area": "Semiurban"
}
```

### Response Example
```json
{
  "approved": true,
  "prediction_label": "Approved",
  "confidence": 0.875,
  "probability_approved": 0.875,
  "probability_rejected": 0.125,
  "neighbors_vote": {
    "approved": 14,
    "rejected": 2
  },
  "analysis": [
    "Positive credit history strongly supports approval eligibility.",
    "Healthy loan-to-annual-income ratio (1.55x).",
    "Semiurban property locations correlate with higher historical loan approval rates.",
    "Based on 16 nearest historical applicants, 14 were approved and 2 were rejected."
  ]
}
```



---

## ⚡ Deploying to Vercel

This repository is pre-configured for instant deployment on **Vercel** with full support for both the **Next.js Frontend** and the **FastAPI Backend Serverless Functions**.

### How It Works on Vercel
1. **Entrypoint Resolution**:
   - `pyproject.toml` defines `[tool.vercel] entrypoint = "backend.app:app"` to tell Vercel's Python runtime where the FastAPI app lives.
   - `api/index.py`, `app.py`, `main.py`, and `index.py` are provided to support all Vercel detection methods.
2. **Unified Routing (`vercel.json`)**:
   - Requests matching `/api/*` route to the Python FastAPI backend (`api/index.py`).
   - All other routes serve the Next.js production frontend (`frontend/package.json`).
3. **Environment Adaptation**:
   - `frontend/app/page.tsx` automatically switches to relative `/api/...` calls in production, eliminating localhost connection issues.
   - `requirements.txt` is optimized for fast, lightweight serverless installation without heavy non-runtime dependencies.

### Deploy Steps
1. Push your changes to your GitHub repository:
   ```bash
   git add .
   git commit -m "Configure Vercel deployment for FastAPI and Next.js"
   git push origin main
   ```
2. In the [Vercel Dashboard](https://vercel.com/new):
   - Import your GitHub repository (`AI-ASSET`).
   - Keep the Root Directory as `./` (default).
   - Click **Deploy**.
3. Or deploy via Vercel CLI:
   ```bash
   vercel
   ```
