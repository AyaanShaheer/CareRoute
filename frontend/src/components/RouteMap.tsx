"use client";

import React from "react";
import {
  MapContainer,
  TileLayer,
  Polyline,
  CircleMarker,
  Tooltip,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import {
  AssignedVisit,
  Carer,
  OptimisationResponse,
  Visit,
} from "@/types/api";

const CARER_COLORS = [
  "#2563eb", "#7c3aed", "#db2777", "#ea580c", "#0d9488",
  "#4f46e5", "#0284c7", "#65a30d", "#9333ea", "#059669",
  "#d97706", "#e11d48", "#0891b2", "#4338ca", "#16a34a",
  "#c026d3", "#ca8a04", "#dc2626", "#2dd4bf", "#6366f1",
];

interface RouteMapProps {
  schedule: OptimisationResponse;
  carers: Carer[];
  visitsMap: Record<string, Visit>;
  selectedCarerId: string;
  selectedVisit: AssignedVisit | null;
  onSelectVisit: (av: AssignedVisit) => void;
}

function formatMins(mins: number): string {
  const h = Math.floor(mins / 60);
  const m = Math.round(mins % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

export default function RouteMap({
  schedule,
  carers,
  visitsMap,
  selectedCarerId,
  selectedVisit,
  onSelectVisit,
}: RouteMapProps) {
  const visibleRoutes =
    selectedCarerId === "ALL"
      ? schedule.routes
      : schedule.routes.filter((r) => r.carer_id === selectedCarerId);

  return (
    <MapContainer
      center={[51.545, -0.16]}
      zoom={13}
      className="h-full w-full rounded-xl z-0"
    >
      <TileLayer
  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
/>

      {/* 1. Render Street-Level Route Polylines */}
      {visibleRoutes.map((route) => {
        const carerIdx = parseInt(route.carer_id.split("_")[1], 10) - 1;
        const color = CARER_COLORS[carerIdx % CARER_COLORS.length];

        return route.assigned_visits.map((av) => {
          if (!av.route_geometry || av.route_geometry.length < 2) return null;
          return (
            <Polyline
              key={`${route.carer_id}-${av.visit_id}`}
              positions={av.route_geometry}
              pathOptions={{
                color,
                weight: selectedCarerId === "ALL" ? 2.5 : 4,
                opacity: selectedCarerId === "ALL" ? 0.65 : 0.9,
              }}
            />
          );
        });
      })}

      {/* 2. Render Carer Home Depots */}
      {carers
        .filter(
          (c) => selectedCarerId === "ALL" || c.carer_id === selectedCarerId
        )
        .map((carer) => (
          <CircleMarker
            key={carer.carer_id}
            center={[carer.home_location.lat, carer.home_location.lon]}
            radius={5}
            pathOptions={{
              color: "#0f172a",
              fillColor: "#f8fafc",
              fillOpacity: 1,
              weight: 2,
            }}
          >
            <Tooltip direction="top">
              <div className="text-xs font-semibold">
                {carer.name} (Home Depot)
              </div>
              <div className="text-[11px] text-slate-600">
                Skills: {carer.skills.join(", ")}
              </div>
            </Tooltip>
          </CircleMarker>
        ))}

      {/* 3. Render Client Visits Coloured by Late-Visit Risk */}
      {visibleRoutes.map((route) =>
        route.assigned_visits.map((av, idx) => {
          const vMeta = visitsMap[av.visit_id];
          if (!vMeta) return null;

          const isHighRisk = av.late_risk_prob >= 0.5;
          const isMedRisk = av.late_risk_prob >= 0.3 && av.late_risk_prob < 0.5;
          const isSelected = selectedVisit?.visit_id === av.visit_id;

          const fillColor = isHighRisk
            ? "#ef4444"
            : isMedRisk
            ? "#f59e0b"
            : "#10b981";

          return (
            <CircleMarker
              key={av.visit_id}
              center={[vMeta.location.lat, vMeta.location.lon]}
              radius={isSelected ? 10 : isHighRisk ? 7.5 : 5.5}
              eventHandlers={{
                click: () => onSelectVisit(av),
              }}
              pathOptions={{
                color: isSelected ? "#1e293b" : "#ffffff",
                weight: isSelected ? 3 : 1.5,
                fillColor,
                fillOpacity: 0.92,
              }}
            >
              <Tooltip direction="top">
                <div className="text-xs font-bold">
                  {av.visit_id} — Stop #{idx + 1} ({av.carer_id})
                </div>
                <div className="text-[11px]">
                  Arrival: {formatMins(av.arrival_min)} (Window:{" "}
                  {formatMins(vMeta.window_start_min)}–
                  {formatMins(vMeta.window_end_min)})
                </div>
                <div className="text-[11px] font-semibold">
                  Late Risk: {(av.late_risk_prob * 100).toFixed(1)}% | Slack:{" "}
                  {av.slack_mins}m
                </div>
              </Tooltip>
            </CircleMarker>
          );
        })
      )}
    </MapContainer>
  );
}