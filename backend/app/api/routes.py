import copy
from typing import Any, Dict

from fastapi import APIRouter, Request

from backend.app.data.routing_geometry import attach_route_geometries
from backend.app.ml.risk_model import score_schedule_with_shap
from backend.app.optimiser.ortools_vrptw import solve_ortools_vrptw
from backend.app.schemas.models import ComparisonResponse, OptimiseRequest

router = APIRouter(prefix="/api/v1", tags=["CareRoute Optimiser"])


@router.get("/health")
def health_check(request: Request) -> Dict[str, Any]:
    state = request.app.state.careroute
    return {
        "status": "healthy",
        "carers_loaded": len(state["carers"]),
        "visits_loaded": len(state["visits"]),
        "graph_nodes": len(state["graph"].nodes),
        "graph_edges": len(state["graph"].edges),
        "ml_roc_auc": state["ml_metrics"].get("roc_auc"),
    }


@router.get("/metadata")
def get_metadata(request: Request) -> Dict[str, Any]:
    """Returns carers, client visits, and trained XGBoost/SHAP evaluation metrics."""
    state = request.app.state.careroute
    return {
        "carers": [c.model_dump() for c in state["carers"]],
        "visits": [v.model_dump() for v in state["visits"]],
        "ml_metrics": state["ml_metrics"],
    }


@router.post("/optimise", response_model=ComparisonResponse)
def optimise_schedule(
    payload: OptimiseRequest, request: Request
) -> ComparisonResponse:
    """
    Runs OR-Tools VRPTW with the requested continuity_weight, scores all visits
    using XGBoost + TreeSHAP, attaches street-level polylines, and compares against
    the pre-computed Naive Baseline.
    """
    state = request.app.state.careroute
    carers = state["carers"]
    visits = state["visits"]
    matrix = state["matrix"]
    model = state["model"]
    G = state["graph"]

    # 1. Retrieve pre-scored Naive Baseline
    naive_res = copy.deepcopy(state["naive_baseline"])

    # 2. Solve OR-Tools VRPTW with user's continuity trade-off weight
    ortools_res = solve_ortools_vrptw(
        carers=carers,
        visits=visits,
        travel_matrix=matrix,
        continuity_weight=payload.continuity_weight,
        time_limit_seconds=payload.time_limit_seconds,
    )

    # 3. Compute Travel Time Reduction % vs Naive Baseline
    base_travel = naive_res.kpis.total_travel_mins
    opt_travel = ortools_res.kpis.total_travel_mins
    ortools_res.kpis.travel_time_reduction_pct = round(
        ((base_travel - opt_travel) / base_travel) * 100, 1
    )

    # 4. Score OR-Tools schedule with XGBoost + TreeSHAP
    ortools_res = score_schedule_with_shap(ortools_res, visits, model)

    # 5. Attach OSM street polylines if requested
    if payload.include_geometry:
        attach_route_geometries(G, ortools_res, carers, visits)
    else:
        for r in naive_res.routes:
            for av in r.assigned_visits:
                av.route_geometry = []

    return ComparisonResponse(
        naive_schedule=naive_res,
        optimised_schedule=ortools_res,
    )