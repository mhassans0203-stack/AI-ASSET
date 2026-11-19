import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Loan Prediction AI | K-Nearest Neighbors Classifier",
  description: "Real-time loan application approval prediction system powered by a tuned K-Nearest Neighbors (KNN) model.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-900 text-slate-100 antialiased selection:bg-blue-500 selection:text-white">
        {children}
      </body>
    </html>
  );
}
