import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CareRoute | Geospatial Carer Visit Optimiser & SHAP Risk Engine",
  description:
    "Multi-objective VRPTW scheduling over OpenStreetMap road networks with XGBoost + TreeSHAP late-visit explainability.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}