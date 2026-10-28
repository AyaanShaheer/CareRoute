from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class Skill(str, Enum):
    GENERAL = "general_care"
    MEDICATION = "medication_admin"
    DEMENTIA = "dementia_specialist"
    MOBILITY = "mobility_hoist"


class Location(BaseModel):
    lat: float
    lon: float
    osm_node_id: int = Field(..., description="Snapped nearest OpenStreetMap node ID")


class Carer(BaseModel):
    carer_id: str
    name: str
    home_location: Location
    skills: List[Skill]
    shift_start_min: int = Field(420, description="Shift start in minutes from midnight (e.g., 420 = 07:00)")
    shift_end_min: int = Field(1200, description="Shift end in minutes from midnight (e.g., 1200 = 20:00)")
    max_working_mins: int = Field(480, description="Max active duty time (8 hours)")


class Visit(BaseModel):
    visit_id: str
    client_id: str
    location: Location
    window_start_min: int = Field(..., description="Earliest arrival in minutes from midnight")
    window_end_min: int = Field(..., description="Latest arrival in minutes from midnight")
    duration_mins: int = Field(..., description="Service duration in minutes (e.g., 30, 45, 60)")
    required_skill: Skill
    # Maps carer_id -> number of past visits with this client (for Continuity of Care)
    historical_carer_counts: Dict[str, int] = Field(default_factory=dict)


class ShapDriver(BaseModel):
    feature_name: str
    feature_value: float
    shap_value: float = Field(..., description="Log-odds or probability impact on lateness")


class AssignedVisit(BaseModel):
    visit_id: str
    client_id: str
    carer_id: str
    arrival_min: int
    departure_min: int
    travel_time_from_prev_mins: float
    slack_mins: float = Field(..., description="window_end_min - arrival_min")
    is_preferred_carer: bool
    late_risk_prob: float = Field(0.0, description="XGBoost predicted probability of lateness")
    top_shap_drivers: List[ShapDriver] = Field(default_factory=list)
    route_geometry: List[Tuple[float, float]] = Field(
        default_factory=list, description="List of (lat, lon) coordinates along the street network"
    )


class CarerRoute(BaseModel):
    carer_id: str
    carer_name: str
    assigned_visits: List[AssignedVisit]
    total_travel_mins: float
    total_service_mins: float
    continuity_matches: int


class ScheduleKPIs(BaseModel):
    solver_type: str = Field(..., description="'naive_baseline' or 'ortools_vrptw'")
    total_travel_mins: float
    travel_time_reduction_pct: Optional[float] = None
    continuity_score_pct: float = Field(..., description="% of visits assigned to a top historical carer")
    workload_std_dev_mins: float = Field(..., description="Workload variance/std-dev across carers")
    unassigned_visits_count: int
    high_risk_visits_count: int


class OptimisationResponse(BaseModel):
    kpis: ScheduleKPIs
    routes: List[CarerRoute]
    unassigned_visit_ids: List[str]

class OptimiseRequest(BaseModel):
    continuity_weight: float = Field(
        1.5, ge=0.0, le=5.0, description="Trade-off weight for Continuity of Care (0.0 = pure speed, 3.5 = high continuity)"
    )
    time_limit_seconds: int = Field(
        5, ge=1, le=15, description="OR-Tools Guided Local Search time limit in seconds"
    )
    include_geometry: bool = Field(
        True, description="Whether to attach street-network (lat, lon) polylines"
    )


class ComparisonResponse(BaseModel):
    naive_schedule: OptimisationResponse
    optimised_schedule: OptimisationResponse