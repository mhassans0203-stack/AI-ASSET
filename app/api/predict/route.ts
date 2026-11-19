import { NextRequest, NextResponse } from "next/server";
import modelData from "../knn_trained_data.json";

export async function POST(req: NextRequest) {
  try {
    const app = await req.json();

    const gender = app.Gender === "Female" ? 0 : 1;
    const married = app.Married === "No" ? 0 : 1;
    const dep = app.Dependents === "3+" ? 3 : parseInt(app.Dependents || "0", 10);
    const edu = app.Education === "Not Graduate" ? 0 : 1;
    const selfEmp = app.Self_Employed === "Yes" ? 1 : 0;
    const prop = app.Property_Area === "Rural" ? 0 : (app.Property_Area === "Semiurban" ? 1 : 2);
    const credit = typeof app.Credit_History === "number" ? app.Credit_History : parseFloat(app.Credit_History || "1.0");

    const raw = [
      gender,
      married,
      dep,
      edu,
      selfEmp,
      Number(app.ApplicantIncome) || 0,
      Number(app.CoapplicantIncome) || 0,
      Number(app.LoanAmount) || 0,
      Number(app.Loan_Amount_Term) || 360,
      credit,
      prop
    ];

    // Standardize features using the exact fitted StandardScaler parameters
    const scaled = raw.map((val, i) => (val - modelData.mean[i]) / modelData.scale[i]);

    // Calculate Euclidean distances to all 491 training samples
    const distances = modelData.X_train.map((trainSample: number[], idx: number) => {
      let sumSq = 0;
      for (let i = 0; i < scaled.length; i++) {
        const d = scaled[i] - trainSample[i];
        sumSq += d * d;
      }
      return { dist: Math.sqrt(sumSq), label: modelData.y_train[idx] };
    });

    // Sort by distance and pick top k neighbors (k=16)
    distances.sort((a, b) => a.dist - b.dist);
    const kNeighbors = distances.slice(0, modelData.n_neighbors);
    const approvedCount = kNeighbors.filter(n => n.label === 1).length;
    const rejectedCount = modelData.n_neighbors - approvedCount;

    const probApproved = approvedCount / modelData.n_neighbors;
    const probRejected = rejectedCount / modelData.n_neighbors;
    const isApproved = approvedCount >= rejectedCount;
    const confidence = isApproved ? probApproved : probRejected;

    // Decision factor explanations
    const analysis: string[] = [];
    if (credit === 1.0) {
      analysis.push("Positive credit history strongly supports approval eligibility.");
    } else {
      analysis.push("Adverse credit history significantly elevates default risk.");
    }

    const totalIncome = (Number(app.ApplicantIncome) || 0) + (Number(app.CoapplicantIncome) || 0);
    if (totalIncome > 0) {
      const loanToIncome = (Number(app.LoanAmount) * 1000) / (totalIncome * 12);
      if (loanToIncome < 2.5) {
        analysis.push(`Healthy loan-to-annual-income ratio (${loanToIncome.toFixed(2)}x).`);
      } else if (loanToIncome > 5.0) {
        analysis.push(`Elevated loan-to-annual-income ratio (${loanToIncome.toFixed(2)}x) requests high leverage.`);
      }
    }

    if (app.Property_Area === "Semiurban") {
      analysis.push("Semiurban property locations correlate with higher historical loan approval rates.");
    }

    analysis.push(
      `Based on ${modelData.n_neighbors} nearest historical applicants, ${approvedCount} were approved and ${rejectedCount} were rejected.`
    );

    return NextResponse.json({
      approved: isApproved,
      prediction_label: isApproved ? "Approved" : "Rejected",
      confidence: Math.round(confidence * 10000) / 10000,
      probability_approved: Math.round(probApproved * 10000) / 10000,
      probability_rejected: Math.round(probRejected * 10000) / 10000,
      neighbors_vote: {
        approved: approvedCount,
        rejected: rejectedCount
      },
      features_received: app,
      analysis
    });
  } catch (error: any) {
    return NextResponse.json(
      { detail: error.message || "Failed to process prediction." },
      { status: 400 }
    );
  }
}
