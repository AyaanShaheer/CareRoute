"use client";

import React, { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  HeartHandshake,
  Navigation,
  RefreshCw,
  ShieldAlert,
  Sliders,
  Users,
} from "lucide-react";
import {
  AssignedVisit,
  ComparisonResponse,
  MetadataResponse,
  Visit,
} from "@/types/api";

const RouteMap = dynamic(() => import("@/components/RouteMap"), {
  ssr: false,
  loading: () => (
    <div className="h-full w-full flex items-center justify-center bg-slate-100 rounded-xl text-slate-500 text-sm">
      Loading OpenStreetMap Camden road network...
    </div>
  ),
});

// Relative `/api/v1` path is proxied to the FastAPI backend via Next.js rewrites
// (frontend/next.config.mjs). Override with NEXT_PUBLIC_API_URL for remote deployments.
const RAW_API_BASE = process.env.NEXT_PUBLIC_API_URL;
const API_BASE = RAW_API_BASE
  ? `${RAW_API_BASE.replace(/\/+$/, "")}/api/v1`
  : "/api/v1";

function formatMins(mins: number): string {
  const h = Math.floor(mins / 60);
  const m = Math.round(mins % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

async function readApiError(resp: Response, fallback: string): Promise<string> {
  try {
    const body = await resp.json();
    if (body?.detail) return String(body.detail);
    if (typeof body?.error === "string") return body.error;
    if (body?.message) return String(body.message);
  } catch {
    // Non-JSON error body (e.g. proxy errors) -> fall through
  }
  return `${fallback} (HTTP ${resp.status})`;
}

export default function CareRouteDashboard() {
  const [metadata, setMetadata] = useState<MetadataResponse | null>(null);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [continuityWeight, setContinuityWeight] = useState<number>(1.5);
  const [activeMode, setActiveMode] = useState<"optimised" | "naive">(
    "optimised"
  );
  const [selectedCarerId, setSelectedCarerId] = useState<string>("ALL");
  const [selectedVisit, setSelectedVisit] = useState<AssignedVisit | null>(
    null
  );
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const visitsMap = useMemo(() => {
    const map: Record<string, Visit> = {};
    if (metadata) {
      metadata.visits.forEach((v) => {
        map[v.visit_id] = v;
      });
    }
    return map;
  }, [metadata]);

  const runOptimisation = async (weight: number) => {
    setLoading(true);
    setError(null);
    try {
      const optResp = await fetch(`${API_BASE}/optimise`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          continuity_weight: weight,
          time_limit_seconds: 5,
          include_geometry: true,
        }),
      });
      if (!optResp.ok) throw new Error(await readApiError(optResp, "Optimisation request failed"));
      const compData: ComparisonResponse = await optResp.json();

      const metaResp = metadata ? Promise.resolve(null) : fetch(`${API_BASE}/metadata`);
      const meta = await metaResp;
      if (meta) {
        if (!meta.ok) throw new Error(await readApiError(meta, "Failed to fetch metadata"));
        const metaData: MetadataResponse = await meta.json();
        setMetadata(metaData);
      }
      setComparison(compData);

      // Default select the highest-risk visit in the active schedule
      const targetSched =
        activeMode === "optimised"
          ? compData.optimised_schedule
          : compData.naive_schedule;
      const allVisits = targetSched.routes.flatMap((r) => r.assigned_visits);
      if (allVisits.length > 0) {
        const topRisk = [...allVisits].sort(
          (a, b) => b.late_risk_prob - a.late_risk_prob
        )[0];
        setSelectedVisit(topRisk);
      }
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? err.message
          : `Cannot reach CareRoute FastAPI backend at ${API_BASE}`
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runOptimisation(continuityWeight);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const currentSchedule =
    activeMode === "optimised"
      ? comparison?.optimised_schedule
      : comparison?.naive_schedule;

  const highRiskVisits = useMemo(() => {
    if (!currentSchedule) return [];
    return currentSchedule.routes
      .flatMap((r) => r.assigned_visits)
      .sort((a, b) => b.late_risk_prob - a.late_risk_prob)
      .slice(0, 8);
  }, [currentSchedule]);

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      {/* Top Navigation Bar */}
      <header className="bg-slate-900 text-white px-6 py-4 shadow-md flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="bg-blue-600 p-2 rounded-lg">
            <Navigation className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">
              CareRoute: Geospatial Carer Scheduling & Risk Optimiser
            </h1>
            <p className="text-xs text-slate-400">
              Camden Road Network (OSMnx) • Google OR-Tools VRPTW • XGBoost +
              TreeSHAP Explainability
            </p>
          </div>
        </div>

        {/* Mode Toggle & Controls */}
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex bg-slate-800 p-1 rounded-lg border border-slate-700">
            <button
              onClick={() => setActiveMode("optimised")}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition ${
                activeMode === "optimised"
                  ? "bg-blue-600 text-white shadow"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              OR-Tools VRPTW (Optimised)
            </button>
            <button
              onClick={() => setActiveMode("naive")}
              className={`px-3 py-1.5 rounded-md text-xs font-semibold transition ${
                activeMode === "naive"
                  ? "bg-amber-600 text-white shadow"
                  : "text-slate-300 hover:text-white"
              }`}
            >
              Naive Baseline (Greedy)
            </button>
          </div>

          {/* Pareto Continuity Slider */}
          <div className="flex items-center gap-3 bg-slate-800 px-3 py-1.5 rounded-lg border border-slate-700">
            <Sliders className="h-4 w-4 text-blue-400" />
            <div className="flex flex-col">
              <label className="text-[11px] text-slate-300 font-medium">
                Continuity Weight (β):{" "}
                <span className="text-blue-400 font-bold">
                  {continuityWeight.toFixed(1)}
                </span>
              </label>
              <input
                type="range"
                min={0.0}
                max={3.5}
                step={0.5}
                value={continuityWeight}
                onChange={(e) =>
                  setContinuityWeight(parseFloat(e.target.value))
                }
                className="w-28 accent-blue-500 cursor-pointer h-1.5"
              />
            </div>
            <button
              onClick={() => runOptimisation(continuityWeight)}
              disabled={loading}
              className="bg-blue-600 hover:bg-blue-500 disabled:bg-slate-600 text-white px-3 py-1.5 rounded-md text-xs font-semibold flex items-center gap-1.5 transition"
            >
              <RefreshCw
                className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`}
              />
              {loading ? "Solving..." : "Re-Solve"}
            </button>
          </div>
        </div>
      </header>

      {error && (
        <div className="bg-red-50 border-b border-red-200 px-6 py-3 text-red-700 text-sm flex items-center gap-2">
          <AlertTriangle className="h-4 w-4" />
          <span>{error}</span>
        </div>
      )}

      {/* Main Content */}
      <main className="flex-1 p-6 max-w-[1600px] w-full mx-auto space-y-6">
        {/* 1. KPI Comparison Strip */}
        {comparison && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
            {/* Total Travel Time */}
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
              <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
                <span>TOTAL FLEET TRAVEL TIME</span>
                <Clock className="h-4 w-4 text-blue-600" />
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-2xl font-bold text-slate-900">
                  {currentSchedule?.kpis.total_travel_mins}m
                </span>
                <span className="text-xs font-semibold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full">
                  -{comparison.optimised_schedule.kpis.travel_time_reduction_pct}
                  % vs Naive (
                  {comparison.naive_schedule.kpis.total_travel_mins}m)
                </span>
              </div>
              <p className="text-[11px] text-slate-500 mt-1">
                Saves{" "}
                {(
                  (comparison.naive_schedule.kpis.total_travel_mins -
                    comparison.optimised_schedule.kpis.total_travel_mins) /
                  60
                ).toFixed(1)}{" "}
                hours of carer driving per day
              </p>
            </div>

            {/* Continuity of Care */}
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
              <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
                <span>CONTINUITY OF CARE</span>
                <HeartHandshake className="h-4 w-4 text-purple-600" />
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-2xl font-bold text-slate-900">
                  {currentSchedule?.kpis.continuity_score_pct}%
                </span>
                <span className="text-xs text-slate-500">
                  (Naive: {comparison.naive_schedule.kpis.continuity_score_pct}
                  %)
                </span>
              </div>
              <p className="text-[11px] text-slate-500 mt-1">
                Visits matched to primary regular carer (≥15 visits)
              </p>
            </div>

            {/* Workload Std Dev */}
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
              <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
                <span>WORKLOAD STD DEV</span>
                <Users className="h-4 w-4 text-indigo-600" />
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-2xl font-bold text-slate-900">
                  ±{currentSchedule?.kpis.workload_std_dev_mins}m
                </span>
                <span className="text-xs text-emerald-600 font-medium">
                  (Naive: ±{comparison.naive_schedule.kpis.workload_std_dev_mins}
                  m)
                </span>
              </div>
              <p className="text-[11px] text-slate-500 mt-1">
                Shift equity across all 20 active carers
              </p>
            </div>

            {/* Unassigned Visits */}
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
              <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
                <span>UNASSIGNED VISITS</span>
                <CheckCircle2 className="h-4 w-4 text-emerald-600" />
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-2xl font-bold text-slate-900">
                  {currentSchedule?.kpis.unassigned_visits_count} / 200
                </span>
                <span className="text-xs text-slate-500">
                  (Naive:{" "}
                  {comparison.naive_schedule.kpis.unassigned_visits_count})
                </span>
              </div>
              <p className="text-[11px] text-slate-500 mt-1">
                Hard time-window & skill satisfaction rate
              </p>
            </div>

            {/* High Risk Visits (XGBoost) */}
            <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm">
              <div className="flex items-center justify-between text-slate-500 text-xs font-medium">
                <span>HIGH LATE-RISK VISITS</span>
                <ShieldAlert className="h-4 w-4 text-red-600" />
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-2xl font-bold text-red-600">
                  {currentSchedule?.kpis.high_risk_visits_count}
                </span>
                <span className="text-xs text-slate-500">
                  (Naive:{" "}
                  {comparison.naive_schedule.kpis.high_risk_visits_count} | AUC:{" "}
                  {metadata?.ml_metrics.roc_auc})
                </span>
              </div>
              <p className="text-[11px] text-slate-500 mt-1">
                Visits with P(Late &gt; 5m) ≥ 50% via XGBoost
              </p>
            </div>
          </div>
        )}

        {/* 2. Main Split Workspace: Leaflet Map (Left 7 cols) + SHAP Inspector (Right 5 cols) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Map Column */}
          <div className="lg:col-span-7 bg-white rounded-xl border border-slate-200 shadow-sm p-4 flex flex-col h-[640px]">
            <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-slate-100">
              <div>
                <h2 className="text-sm font-bold text-slate-900">
                  Street-Network Route Map —{" "}
                  {activeMode === "optimised"
                    ? "OR-Tools VRPTW Schedule"
                    : "Naive Greedy Baseline"}
                </h2>
                <p className="text-xs text-slate-500">
                  Click any visit circle to inspect its XGBoost lateness risk &
                  local TreeSHAP drivers
                </p>
              </div>

              {/* Carer Route Filter */}
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-600 font-medium">
                  Filter Carer:
                </span>
                <select
                  value={selectedCarerId}
                  onChange={(e) => setSelectedCarerId(e.target.value)}
                  className="text-xs border border-slate-300 rounded-lg px-2.5 py-1.5 bg-white font-medium text-slate-800"
                >
                  <option value="ALL">All 20 Carers (200 Visits)</option>
                  {currentSchedule?.routes.map((r) => (
                    <option key={r.carer_id} value={r.carer_id}>
                      {r.carer_name} ({r.assigned_visits.length} visits,{" "}
                      {r.total_travel_mins}m travel)
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="flex-1 mt-3 relative">
              {currentSchedule && metadata && (
                <RouteMap
                  schedule={currentSchedule}
                  carers={metadata.carers}
                  visitsMap={visitsMap}
                  selectedCarerId={selectedCarerId}
                  selectedVisit={selectedVisit}
                  onSelectVisit={setSelectedVisit}
                />
              )}
            </div>

            {/* Map Legend */}
            <div className="pt-3 flex flex-wrap items-center justify-between text-xs text-slate-600 gap-2">
              <div className="flex items-center gap-4">
                <span className="flex items-center gap-1.5">
                  <span className="h-3 w-3 rounded-full bg-emerald-500 inline-block" />
                  Low Risk (&lt;30%)
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-3 w-3 rounded-full bg-amber-500 inline-block" />
                  Medium Risk (30–49%)
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-3 w-3 rounded-full bg-red-500 inline-block" />
                  High Late Risk (≥50%)
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-3 w-3 rounded-full border-2 border-slate-900 bg-white inline-block" />
                  Carer Home Depot
                </span>
              </div>
            </div>
          </div>

          {/* Right Column: SHAP Explainability Inspector + Priority Risk Queue */}
          <div className="lg:col-span-5 flex flex-col gap-6">
            {/* Local TreeSHAP Visit Inspector */}
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
              <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-blue-600" />
                  <h2 className="text-sm font-bold text-slate-900">
                    SHAP Late-Visit Risk Explainability
                  </h2>
                </div>
                {selectedVisit && (
                  <span
                    className={`px-2.5 py-0.5 rounded-full text-xs font-bold ${
                      selectedVisit.late_risk_prob >= 0.5
                        ? "bg-red-100 text-red-700"
                        : selectedVisit.late_risk_prob >= 0.3
                        ? "bg-amber-100 text-amber-700"
                        : "bg-emerald-100 text-emerald-700"
                    }`}
                  >
                    Late Risk: {(selectedVisit.late_risk_prob * 100).toFixed(1)}
                    %
                  </span>
                )}
              </div>

              {selectedVisit ? (
                <div className="mt-4 space-y-4">
                  {/* Visit Operational Metadata */}
                  <div className="grid grid-cols-3 gap-2 bg-slate-50 p-3 rounded-lg text-xs">
                    <div>
                      <span className="text-slate-400 block">Visit / Client</span>
                      <span className="font-bold text-slate-800">
                        {selectedVisit.visit_id}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 block">Assigned Carer</span>
                      <span className="font-bold text-slate-800">
                        {selectedVisit.carer_id}{" "}
                        {selectedVisit.is_preferred_carer ? "★" : ""}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 block">
                        Arrival / Slack
                      </span>
                      <span className="font-bold text-slate-800">
                        {formatMins(selectedVisit.arrival_min)} (
                        {selectedVisit.slack_mins}m slack)
                      </span>
                    </div>
                  </div>

                  {/* Top 3 Local SHAP Drivers */}
                  <div className="space-y-2.5">
                    <div className="text-xs font-semibold text-slate-700">
                      Top 3 Local TreeSHAP Drivers (Log-Odds Impact):
                    </div>
                    {selectedVisit.top_shap_drivers.map((driver) => {
                      const isPositive = driver.shap_value > 0;
                      const widthPct = Math.min(
                        Math.abs(driver.shap_value) * 55,
                        100
                      );
                      return (
                        <div
                          key={driver.feature_name}
                          className="bg-slate-50 p-2.5 rounded-lg border border-slate-100"
                        >
                          <div className="flex justify-between text-xs mb-1">
                            <span className="font-mono font-semibold text-slate-800">
                              {driver.feature_name} ={" "}
                              <span className="text-blue-600">
                                {driver.feature_value}
                              </span>
                            </span>
                            <span
                              className={`font-bold ${
                                isPositive ? "text-red-600" : "text-emerald-600"
                              }`}
                            >
                              {driver.shap_value > 0 ? "+" : ""}
                              {driver.shap_value.toFixed(3)}{" "}
                              {isPositive ? "↑ Risk" : "↓ Risk"}
                            </span>
                          </div>
                          <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                            <div
                              style={{ width: `${widthPct}%` }}
                              className={`h-full rounded-full ${
                                isPositive ? "bg-red-500" : "bg-emerald-500"
                              }`}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : (
                <p className="text-xs text-slate-500 py-6 text-center">
                  Select a visit marker on the map to inspect its SHAP risk
                  breakdown.
                </p>
              )}
            </div>

            {/* Priority High-Risk Visit Queue */}
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 flex-1">
              <div className="flex items-center justify-between pb-3 border-b border-slate-100">
                <h3 className="text-sm font-bold text-slate-900">
                  Highest Late-Risk Visits in Active Rota
                </h3>
                <span className="text-[11px] text-slate-500">
                  Click row to locate & explain
                </span>
              </div>

              <div className="mt-3 divide-y divide-slate-100 max-h-[250px] overflow-y-auto">
                {highRiskVisits.map((av) => (
                  <button
                    key={av.visit_id}
                    onClick={() => {
                      setSelectedVisit(av);
                    }}
                    className={`w-full text-left py-2.5 px-2.5 flex items-center justify-between rounded-lg transition text-xs ${
                      selectedVisit?.visit_id === av.visit_id
                        ? "bg-blue-50 border border-blue-200"
                        : "hover:bg-slate-50"
                    }`}
                  >
                    <div>
                      <span className="font-bold text-slate-900">
                        {av.visit_id}
                      </span>{" "}
                      <span className="text-slate-500">
                        • {av.carer_id} • Arrival {formatMins(av.arrival_min)}
                      </span>
                      <div className="text-[11px] text-slate-500">
                        Top driver:{" "}
                        <span className="font-mono">
                          {av.top_shap_drivers[0]?.feature_name}=
                          {av.top_shap_drivers[0]?.feature_value}
                        </span>
                      </div>
                    </div>
                    <span
                      className={`px-2 py-0.5 rounded font-bold ${
                        av.late_risk_prob >= 0.5
                          ? "bg-red-100 text-red-700"
                          : "bg-amber-100 text-amber-700"
                      }`}
                    >
                      {(av.late_risk_prob * 100).toFixed(1)}%
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}