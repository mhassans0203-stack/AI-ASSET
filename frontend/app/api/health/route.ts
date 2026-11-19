import { NextResponse } from "next/server";
import modelData from "../knn_trained_data.json";

export async function GET() {
  return NextResponse.json({
    status: "healthy",
    model_loaded: true,
    scaler_loaded: true,
    optimal_k: modelData.n_neighbors || 16,
    engine: "Next.js Native ML Engine"
  });
}
