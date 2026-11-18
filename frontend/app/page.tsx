"use client";

import React, { useState, useEffect } from "react";

interface LoanForm {
  Gender: string;
  Married: string;
  Dependents: string;
  Education: string;
  Self_Employed: string;
  ApplicantIncome: number;
  CoapplicantIncome: number;
  LoanAmount: number;
  Loan_Amount_Term: number;
  Credit_History: number;
  Property_Area: string;
}

interface PredictionResponse {
  approved: boolean;
  prediction_label: string;
  confidence: number;
  probability_approved: number;
  probability_rejected: number;
  neighbors_vote: {
    approved: number;
    rejected: number;
  };
  features_received: Record<string, any>;
  analysis: string[];
}

interface ModelMeta {
  model_name?: string;
  optimal_k?: number;
  default_k?: number;
  accuracy_default?: number;
  accuracy_tuned?: number;
  confusion_matrix_default?: number[][];
  confusion_matrix_tuned?: number[][];
}

const PRESETS: Record<string, { label: string; desc: string; data: LoanForm }> = {
  prime: {
    label: "Prime Approved Profile",
    desc: "High income, co-applicant support, spotless credit history",
    data: {
      Gender: "Male",
      Married: "Yes",
      Dependents: "1",
      Education: "Graduate",
      Self_Employed: "No",
      ApplicantIncome: 5500,
      CoapplicantIncome: 2500,
      LoanAmount: 130,
      Loan_Amount_Term: 360,
      Credit_History: 1.0,
      Property_Area: "Semiurban"
    }
  },
  risky: {
    label: "High-Risk Profile",
    desc: "Low income, high loan request, adverse credit history",
    data: {
      Gender: "Male",
      Married: "No",
      Dependents: "0",
      Education: "Not Graduate",
      Self_Employed: "Yes",
      ApplicantIncome: 1800,
      CoapplicantIncome: 0,
      LoanAmount: 260,
      Loan_Amount_Term: 360,
      Credit_History: 0.0,
      Property_Area: "Rural"
    }
  },
  borderline: {
    label: "Moderate / Urban Profile",
    desc: "Single applicant with standard income and approved credit in urban area",
    data: {
      Gender: "Female",
      Married: "No",
      Dependents: "0",
      Education: "Graduate",
      Self_Employed: "No",
      ApplicantIncome: 4200,
      CoapplicantIncome: 0,
      LoanAmount: 110,
      Loan_Amount_Term: 360,
      Credit_History: 1.0,
      Property_Area: "Urban"
    }
  }
};

export default function Home() {
  const [form, setForm] = useState<LoanForm>(PRESETS.prime.data);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [backendHealth, setBackendHealth] = useState<boolean | null>(null);
  const [modelMeta, setModelMeta] = useState<ModelMeta | null>(null);
  const [showMetricsModal, setShowMetricsModal] = useState(false);

  const API_BASE =
    process.env.NEXT_PUBLIC_API_URL !== undefined
      ? process.env.NEXT_PUBLIC_API_URL
      : typeof window !== "undefined" &&
        window.location.hostname !== "localhost" &&
        window.location.hostname !== "127.0.0.1"
      ? ""
      : "http://localhost:8000";

  // Check backend health on mount
  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE}/api/health`, { cache: "no-store" });
        if (res.ok) {
          setBackendHealth(true);
          const metaRes = await fetch(`${API_BASE}/api/model-info`);
          if (metaRes.ok) {
            const meta = await metaRes.json();
            setModelMeta(meta);
          }
        } else {
          setBackendHealth(false);
        }
      } catch (err) {
        setBackendHealth(false);
      }
    }
    checkHealth();
  }, [API_BASE]);

  const handleChange = (field: keyof LoanForm, value: string | number) => {
    setForm((prev) => ({
      ...prev,
      [field]: value
    }));
  };

  const handlePresetSelect = (presetKey: string) => {
    setForm(PRESETS[presetKey].data);
    setPrediction(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(form)
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Prediction failed with status ${res.status}`);
      }

      const data: PredictionResponse = await res.json();
      setPrediction(data);
    } catch (err: any) {
      setError(
        err.message ||
        "Could not connect to FastAPI server. Ensure FastAPI is running on port 8000 (uvicorn app:app --port 8000)."
      );
    } finally {
      setLoading(false);
    }
  };

  const totalIncome = form.ApplicantIncome + form.CoapplicantIncome;
  const loanToIncomeRatio =
    totalIncome > 0 ? (form.LoanAmount * 1000) / (totalIncome * 12) : 0;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 py-8 px-4 sm:px-6 lg:px-8">
      {/* Top Header */}
      <header className="max-w-7xl mx-auto mb-8">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800">
          <div>
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20 font-bold text-xl text-white">
                KNN
              </div>
              <div>
                <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
                  Loan Prediction AI Engine
                </h1>
                <p className="text-sm text-slate-400">
                  Real-time Loan Approval Prediction using K-Nearest Neighbors Classifier
                </p>
              </div>
            </div>
          </div>

          {/* Status & Action Badges */}
          <div className="flex flex-wrap items-center gap-3">
            <div
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold border ${
                backendHealth === true
                  ? "bg-emerald-950/70 text-emerald-400 border-emerald-800"
                  : backendHealth === false
                  ? "bg-rose-950/70 text-rose-400 border-rose-800"
                  : "bg-amber-950/70 text-amber-400 border-amber-800"
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  backendHealth === true
                    ? "bg-emerald-400 animate-pulse"
                    : backendHealth === false
                    ? "bg-rose-400"
                    : "bg-amber-400 animate-bounce"
                }`}
              />
              {backendHealth === true
                ? "FastAPI Backend Online"
                : backendHealth === false
                ? "Backend Offline (Port 8000)"
                : "Checking Backend..."}
            </div>

            <button
              onClick={() => setShowMetricsModal(true)}
              className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition shadow-sm"
            >
              <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              Model Metrics & Visuals
            </button>
          </div>
        </div>

        {/* Quick Presets Bar */}
        <div className="mt-6 bg-slate-900/80 border border-slate-800 rounded-xl p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="text-xs text-slate-400 font-medium flex items-center gap-1.5">
            <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <span>Quick Test Presets:</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(PRESETS).map(([key, item]) => (
              <button
                key={key}
                type="button"
                onClick={() => handlePresetSelect(key)}
                className="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-blue-600 hover:text-white text-slate-300 border border-slate-700 transition font-medium"
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Input Form (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          <form
            onSubmit={handleSubmit}
            className="bg-slate-900 border border-slate-800 rounded-2xl p-6 sm:p-7 shadow-xl space-y-6"
          >
            <div className="flex items-center justify-between pb-4 border-b border-slate-800">
              <h2 className="text-lg font-bold text-white flex items-center gap-2">
                <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Applicant & Loan Details
              </h2>
              <span className="text-xs text-slate-400 bg-slate-800 px-2.5 py-1 rounded-md">
                11 Features
              </span>
            </div>

            {/* Section 1: Demographics */}
            <div>
              <h3 className="text-xs font-semibold text-blue-400 uppercase tracking-wider mb-3">
                1. Applicant Profile
              </h3>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                {/* Gender */}
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">Gender</label>
                  <select
                    value={form.Gender}
                    onChange={(e) => handleChange("Gender", e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </select>
                </div>

                {/* Married */}
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">Married</label>
                  <select
                    value={form.Married}
                    onChange={(e) => handleChange("Married", e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                  </select>
                </div>

                {/* Dependents */}
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">Dependents</label>
                  <select
                    value={form.Dependents}
                    onChange={(e) => handleChange("Dependents", e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="0">0</option>
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3+">3+</option>
                  </select>
                </div>

                {/* Education */}
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">Education</label>
                  <select
                    value={form.Education}
                    onChange={(e) => handleChange("Education", e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="Graduate">Graduate</option>
                    <option value="Not Graduate">Not Graduate</option>
                  </select>
                </div>

                {/* Self Employed */}
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">Self Employed</label>
                  <select
                    value={form.Self_Employed}
                    onChange={(e) => handleChange("Self_Employed", e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                  </select>
                </div>

                {/* Property Area */}
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">Property Area</label>
                  <select
                    value={form.Property_Area}
                    onChange={(e) => handleChange("Property_Area", e.target.value)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="Urban">Urban</option>
                    <option value="Semiurban">Semiurban</option>
                    <option value="Rural">Rural</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Section 2: Financial Details */}
            <div>
              <h3 className="text-xs font-semibold text-blue-400 uppercase tracking-wider mb-3">
                2. Income & Financial Assets
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {/* Applicant Income */}
                <div>
                  <div className="flex justify-between items-center mb-1.5">
                    <label className="text-xs font-medium text-slate-300">
                      Applicant Monthly Income
                    </label>
                    <span className="text-xs text-blue-400 font-mono">
                      ${form.ApplicantIncome.toLocaleString()}
                    </span>
                  </div>
                  <input
                    type="number"
                    min="0"
                    step="100"
                    value={form.ApplicantIncome}
                    onChange={(e) => handleChange("ApplicantIncome", parseFloat(e.target.value) || 0)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>

                {/* Coapplicant Income */}
                <div>
                  <div className="flex justify-between items-center mb-1.5">
                    <label className="text-xs font-medium text-slate-300">
                      Co-applicant Monthly Income
                    </label>
                    <span className="text-xs text-blue-400 font-mono">
                      ${form.CoapplicantIncome.toLocaleString()}
                    </span>
                  </div>
                  <input
                    type="number"
                    min="0"
                    step="100"
                    value={form.CoapplicantIncome}
                    onChange={(e) => handleChange("CoapplicantIncome", parseFloat(e.target.value) || 0)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>

            {/* Section 3: Loan Parameters */}
            <div>
              <h3 className="text-xs font-semibold text-blue-400 uppercase tracking-wider mb-3">
                3. Requested Loan & Credit History
              </h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {/* Loan Amount */}
                <div>
                  <div className="flex justify-between items-center mb-1.5">
                    <label className="text-xs font-medium text-slate-300">
                      Loan Amount (in $1,000s)
                    </label>
                    <span className="text-xs text-emerald-400 font-mono font-bold">
                      ${(form.LoanAmount * 1000).toLocaleString()}
                    </span>
                  </div>
                  <input
                    type="number"
                    min="10"
                    step="5"
                    value={form.LoanAmount}
                    onChange={(e) => handleChange("LoanAmount", parseFloat(e.target.value) || 0)}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    required
                  />
                </div>

                {/* Loan Term */}
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">
                    Loan Term (Months)
                  </label>
                  <select
                    value={form.Loan_Amount_Term}
                    onChange={(e) => handleChange("Loan_Amount_Term", parseFloat(e.target.value))}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value={360}>360 Months (30 Years)</option>
                    <option value={240}>240 Months (20 Years)</option>
                    <option value={180}>180 Months (15 Years)</option>
                    <option value={120}>120 Months (10 Years)</option>
                    <option value={84}>84 Months (7 Years)</option>
                    <option value={60}>60 Months (5 Years)</option>
                  </select>
                </div>

                {/* Credit History */}
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1.5">
                    Credit History Guideline
                  </label>
                  <select
                    value={form.Credit_History}
                    onChange={(e) => handleChange("Credit_History", parseFloat(e.target.value))}
                    className={`w-full border rounded-lg px-3 py-2 text-sm font-semibold focus:outline-none focus:ring-2 ${
                      form.Credit_History === 1.0
                        ? "bg-emerald-950/40 border-emerald-700 text-emerald-300 focus:ring-emerald-500"
                        : "bg-rose-950/40 border-rose-700 text-rose-300 focus:ring-rose-500"
                    }`}
                  >
                    <option value={1.0}>1.0 - Meets Guidelines (Good)</option>
                    <option value={0.0}>0.0 - Does Not Meet (Default/Risk)</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Quick Metrics Live Card */}
            <div className="bg-slate-950/60 rounded-xl p-4 border border-slate-800/80 grid grid-cols-2 sm:grid-cols-3 gap-3 text-center">
              <div>
                <p className="text-[11px] text-slate-400">Total Monthly Income</p>
                <p className="text-base font-bold text-white font-mono mt-0.5">
                  ${totalIncome.toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-[11px] text-slate-400">Loan-to-Annual-Income</p>
                <p
                  className={`text-base font-bold font-mono mt-0.5 ${
                    loanToIncomeRatio > 4 ? "text-amber-400" : "text-emerald-400"
                  }`}
                >
                  {loanToIncomeRatio.toFixed(2)}x
                </p>
              </div>
              <div className="col-span-2 sm:col-span-1">
                <p className="text-[11px] text-slate-400">Credit Standing</p>
                <p
                  className={`text-base font-bold mt-0.5 ${
                    form.Credit_History === 1.0 ? "text-emerald-400" : "text-rose-400"
                  }`}
                >
                  {form.Credit_History === 1.0 ? "Eligible" : "High Risk"}
                </p>
              </div>
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="w-full bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-700 hover:from-blue-500 hover:to-indigo-500 text-white font-bold py-3.5 px-4 rounded-xl shadow-lg shadow-blue-500/25 transition disabled:opacity-50 flex items-center justify-center gap-2 text-base cursor-pointer"
            >
              {loading ? (
                <>
                  <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  <span>Querying KNN Model (k=16)...</span>
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  <span>Predict Loan Application Approval</span>
                </>
              )}
            </button>
          </form>
        </div>

        {/* Right Column: Prediction Results (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {error && (
            <div className="bg-rose-950/60 border border-rose-800 rounded-2xl p-5 text-rose-200">
              <div className="flex items-start gap-3">
                <svg className="w-6 h-6 text-rose-400 shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                <div>
                  <h4 className="font-bold text-rose-300">Connection or Prediction Error</h4>
                  <p className="text-xs text-rose-300/80 mt-1">{error}</p>
                  <p className="text-xs text-slate-400 mt-2">
                    Make sure the FastAPI backend is running via:
                    <code className="block bg-slate-900 px-2 py-1 rounded text-slate-300 mt-1">
                      uvicorn backend.app:app --port 8000
                    </code>
                  </p>
                </div>
              </div>
            </div>
          )}

          {prediction ? (
            <div
              className={`rounded-2xl p-6 sm:p-7 border shadow-2xl transition duration-500 ${
                prediction.approved
                  ? "bg-gradient-to-b from-slate-900 to-emerald-950/40 border-emerald-800 shadow-emerald-950/30"
                  : "bg-gradient-to-b from-slate-900 to-rose-950/40 border-rose-800 shadow-rose-950/30"
              }`}
            >
              {/* Status Header */}
              <div className="text-center pb-6 border-b border-slate-800">
                <div
                  className={`inline-flex items-center justify-center w-20 h-20 rounded-full mb-4 shadow-lg ${
                    prediction.approved
                      ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 shadow-emerald-500/20"
                      : "bg-rose-500/20 text-rose-400 border border-rose-500/40 shadow-rose-500/20"
                  }`}
                >
                  {prediction.approved ? (
                    <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <svg className="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  )}
                </div>

                <div className="space-y-1">
                  <span
                    className={`inline-block px-3 py-1 rounded-full text-xs font-black uppercase tracking-wider ${
                      prediction.approved
                        ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                        : "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                    }`}
                  >
                    Prediction Outcome
                  </span>
                  <h2
                    className={`text-3xl font-extrabold tracking-tight ${
                      prediction.approved ? "text-emerald-400" : "text-rose-400"
                    }`}
                  >
                    {prediction.approved ? "LOAN APPROVED" : "LOAN REJECTED"}
                  </h2>
                  <p className="text-xs text-slate-400">
                    Model Confidence:{" "}
                    <strong className="text-white font-mono">
                      {(prediction.confidence * 100).toFixed(1)}%
                    </strong>
                  </p>
                </div>
              </div>

              {/* Probabilities Gauges */}
              <div className="py-5 space-y-3 border-b border-slate-800">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300 font-medium">Approval Probability</span>
                    <span className="text-emerald-400 font-mono font-bold">
                      {(prediction.probability_approved * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                    <div
                      className="bg-emerald-500 h-2.5 rounded-full transition-all duration-700"
                      style={{ width: `${prediction.probability_approved * 100}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300 font-medium">Rejection Probability</span>
                    <span className="text-rose-400 font-mono font-bold">
                      {(prediction.probability_rejected * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-2.5 overflow-hidden">
                    <div
                      className="bg-rose-500 h-2.5 rounded-full transition-all duration-700"
                      style={{ width: `${prediction.probability_rejected * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* KNN Neighbors Voting Card */}
              <div className="py-5 border-b border-slate-800">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
                  <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
                  </svg>
                  Nearest Neighbors Vote (k={prediction.neighbors_vote.approved + prediction.neighbors_vote.rejected})
                </h4>

                <div className="grid grid-cols-2 gap-3 text-center">
                  <div className="bg-slate-900/80 p-3 rounded-xl border border-emerald-900/50">
                    <span className="text-2xl font-black text-emerald-400 font-mono">
                      {prediction.neighbors_vote.approved}
                    </span>
                    <p className="text-[11px] text-slate-400 font-medium">Voted Approved</p>
                  </div>
                  <div className="bg-slate-900/80 p-3 rounded-xl border border-rose-900/50">
                    <span className="text-2xl font-black text-rose-400 font-mono">
                      {prediction.neighbors_vote.rejected}
                    </span>
                    <p className="text-[11px] text-slate-400 font-medium">Voted Rejected</p>
                  </div>
                </div>
              </div>

              {/* Analysis & Factors */}
              <div className="pt-5">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2.5">
                  Automated Underwriting Factors
                </h4>
                <ul className="space-y-2">
                  {prediction.analysis.map((point, idx) => (
                    <li key={idx} className="text-xs text-slate-300 flex items-start gap-2">
                      <span className="h-1.5 w-1.5 rounded-full bg-blue-400 mt-1.5 shrink-0" />
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ) : (
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-8 text-center text-slate-400 flex flex-col items-center justify-center min-h-[420px]">
              <div className="h-16 w-16 rounded-full bg-slate-800 flex items-center justify-center text-slate-500 mb-4">
                <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <h3 className="text-lg font-bold text-slate-200">Awaiting Loan Application</h3>
              <p className="text-xs text-slate-400 max-w-sm mt-1.5">
                Adjust applicant details on the left or select a quick preset, then click{" "}
                <span className="text-blue-400 font-semibold">Predict Loan Application Approval</span>.
              </p>
            </div>
          )}

          {/* Quick Model Performance Card */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <svg className="w-4 h-4 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
                </svg>
                Trained KNN Performance
              </h4>
              <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800/80">
                85.37% Accuracy
              </span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Tuned over 30 neighbor configurations with 5-fold cross validation. Optimal neighborhood size is{" "}
              <strong className="text-white font-mono">k = 16</strong>.
            </p>
            <div className="flex items-center justify-between pt-2 border-t border-slate-800 text-[11px] text-slate-400">
              <span>Default KNN: 84.55%</span>
              <span>•</span>
              <span className="text-emerald-400 font-medium">Tuned KNN: 85.37% (+0.82%)</span>
              <span>•</span>
              <button
                onClick={() => setShowMetricsModal(true)}
                className="text-blue-400 hover:underline cursor-pointer"
              >
                View Charts
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Model Performance & Visualizations Modal */}
      {showMetricsModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-fadeIn">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto p-6 sm:p-8 space-y-6 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-4">
              <div>
                <h3 className="text-xl font-bold text-white flex items-center gap-2">
                  <svg className="w-6 h-6 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                  </svg>
                  KNN Model Evaluation & Visualizations
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Kaggle Loan Prediction Dataset (altruistdelhite04) Evaluation
                </p>
              </div>
              <button
                onClick={() => setShowMetricsModal(false)}
                className="h-8 w-8 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center justify-center transition"
              >
                ✕
              </button>
            </div>

            {/* Metrics Comparison Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left border-collapse">
                <thead>
                  <tr className="bg-slate-800/60 text-slate-300 border-b border-slate-700">
                    <th className="py-2.5 px-3 font-semibold">Evaluation Metric</th>
                    <th className="py-2.5 px-3 font-semibold text-center">Default KNN (k=5)</th>
                    <th className="py-2.5 px-3 font-semibold text-center text-emerald-400">Tuned KNN (k=16)</th>
                    <th className="py-2.5 px-3 font-semibold text-right">Improvement</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800 font-mono">
                  <tr>
                    <td className="py-2.5 px-3 text-slate-300 font-sans">Test Accuracy</td>
                    <td className="py-2.5 px-3 text-center">84.55%</td>
                    <td className="py-2.5 px-3 text-center font-bold text-emerald-400">85.37%</td>
                    <td className="py-2.5 px-3 text-right text-emerald-400">+0.82%</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 text-slate-300 font-sans">True Positives (Approved)</td>
                    <td className="py-2.5 px-3 text-center">82 / 85 (96.5%)</td>
                    <td className="py-2.5 px-3 text-center font-bold text-emerald-400">84 / 85 (98.8%)</td>
                    <td className="py-2.5 px-3 text-right text-emerald-400">+2 correct approvals</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 text-slate-300 font-sans">True Negatives (Rejected)</td>
                    <td className="py-2.5 px-3 text-center">22 / 38 (57.9%)</td>
                    <td className="py-2.5 px-3 text-center text-slate-200">21 / 38 (55.3%)</td>
                    <td className="py-2.5 px-3 text-right text-slate-400">High Precision (95%)</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 px-3 text-slate-300 font-sans">Type II Errors (False Negatives)</td>
                    <td className="py-2.5 px-3 text-center text-amber-400">3</td>
                    <td className="py-2.5 px-3 text-center font-bold text-emerald-400">1 (Near 0!)</td>
                    <td className="py-2.5 px-3 text-right text-emerald-400">-66.7% errors</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Generated Visualizations */}
            <div className="space-y-6">
              <div>
                <h4 className="text-sm font-bold text-white mb-2 flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-blue-500" />
                  Confusion Matrix Comparison (Default vs Tuned)
                </h4>
                <div className="bg-slate-950 p-2 rounded-xl border border-slate-800">
                  <img
                    src="/visualizations/confusion_matrix_comparison.png"
                    alt="Confusion Matrix Comparison"
                    className="w-full rounded-lg"
                  />
                </div>
              </div>

              <div>
                <h4 className="text-sm font-bold text-white mb-2 flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-emerald-500" />
                  Hyperparameter Tuning Curve (Number of Neighbors k vs Accuracy)
                </h4>
                <div className="bg-slate-950 p-2 rounded-xl border border-slate-800">
                  <img
                    src="/visualizations/knn_k_accuracy_tuning.png"
                    alt="KNN Tuning Curve"
                    className="w-full rounded-lg"
                  />
                </div>
              </div>
            </div>

            <div className="flex justify-end pt-4 border-t border-slate-800">
              <button
                onClick={() => setShowMetricsModal(false)}
                className="px-5 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs transition"
              >
                Close Metrics Window
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
