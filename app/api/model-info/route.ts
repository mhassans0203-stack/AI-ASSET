import { NextResponse } from "next/server";

export async function GET() {
  return NextResponse.json({
    model_name: "K-Nearest Neighbors Classifier (Tuned)",
    optimal_k: 16,
    default_k: 5,
    accuracy_default: 0.8455284552845529,
    accuracy_tuned: 0.8536585365853658,
    features: [
      "Gender", "Married", "Dependents", "Education", "Self_Employed",
      "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
      "Credit_History", "Property_Area"
    ],
    confusion_matrix_default: [[22, 16], [3, 82]],
    confusion_matrix_tuned: [[21, 17], [1, 84]]
  });
}
