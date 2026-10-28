export interface Location {
  lat: number;
  lon: number;
  osm_node_id: number;
}

export interface Carer {
  carer_id: string;
  name: string;
  home_location: Location;
  skills: string[];
  shift_start_min: number;
  shift_end_min: number;
  max_working_mins: number;
}

export interface Visit {
  visit_id: string;
  client_id: string;
  location: Location;
  window_start_min: number;
  window_end_min: number;
  duration_mins: number;
  required_skill: string;
  historical_carer_counts: Record<string, number>;
}

export interface ShapDriver {
  feature_name: string;
  feature_value: number;
  shap_value: number;
}

export interface AssignedVisit {
  visit_id: string;
  client_id: string;
  carer_id: string;
  arrival_min: number;
  departure_min: number;
  travel_time_from_prev_mins: number;
  slack_mins: number;
  is_preferred_carer: boolean;
  late_risk_prob: number;
  top_shap_drivers: ShapDriver[];
  route_geometry: [number, number][];
}

export interface CarerRoute {
  carer_id: string;
  carer_name: string;
  assigned_visits: AssignedVisit[];
  total_travel_mins: number;
  total_service_mins: number;
  continuity_matches: number;
}

export interface ScheduleKPIs {
  solver_type: string;
  total_travel_mins: number;
  travel_time_reduction_pct: number | null;
  continuity_score_pct: number;
  workload_std_dev_mins: number;
  unassigned_visits_count: number;
  high_risk_visits_count: number;
}

export interface OptimisationResponse {
  kpis: ScheduleKPIs;
  routes: CarerRoute[];
  unassigned_visit_ids: string[];
}

export interface ComparisonResponse {
  naive_schedule: OptimisationResponse;
  optimised_schedule: OptimisationResponse;
}

export interface MLMetrics {
  train_rows: number;
  test_rows: number;
  historical_late_rate_pct: number;
  roc_auc: number;
  f1_score: number;
  brier_score: number;
  global_mean_abs_shap: Record<string, number>;
}

export interface MetadataResponse {
  carers: Carer[];
  visits: Visit[];
  ml_metrics: MLMetrics;
}