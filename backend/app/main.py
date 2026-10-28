from contextlib import asynccontextmanager
import json
from typing import AsyncGenerator

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.core.config import (
    DATASET_PATH,
    MATRIX_PATH,
    ML_METRICS_PATH,
    MODEL_PATH,
)
from backend.app.data.generator import (
    compute_travel_time_matrix,
    generate_synthetic_dataset,
    load_or_create_graph,
)
from backend.app.data.routing_geometry import attach_route_geometries
from backend.app.ml.risk_model import (
    load_risk_model,
    score_schedule_with_shap,
    train_and_save_risk_model,
)
from backend.app.optimiser.baseline import solve_naive_baseline
from backend.app.schemas.models import Carer, Visit


def _initialise_state() -> dict:
    """Loads cached artifacts (or builds them if missing) into memory on startup."""
    G = load_or_create_graph()

    if not DATASET_PATH.exists() or not MATRIX_PATH.exists():
        carers, visits = generate_synthetic_dataset(G)
        matrix = compute_travel_time_matrix(G, carers, visits)
    else:
        raw = json.loads(DATASET_PATH.read_text())
        carers = [Carer(**c) for c in raw["carers"]]
        visits = [Visit(**v) for v in raw["visits"]]
        matrix = np.load(MATRIX_PATH)["matrix_mins"]

    if not MODEL_PATH.exists() or not ML_METRICS_PATH.exists():
        ml_metrics = train_and_save_risk_model(carers, visits, matrix)
    else:
        ml_metrics = json.loads(ML_METRICS_PATH.read_text())

    model = load_risk_model()

    # Pre-compute, score, and attach street geometry to the Naive Baseline once
    naive_baseline = solve_naive_baseline(carers, visits, matrix)
    naive_baseline = score_schedule_with_shap(naive_baseline, visits, model)
    attach_route_geometries(G, naive_baseline, carers, visits)

    return {
        "graph": G,
        "carers": carers,
        "visits": visits,
        "matrix": matrix,
        "model": model,
        "ml_metrics": ml_metrics,
        "naive_baseline": naive_baseline,
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("[FastAPI] Initialising CareRoute geospatial & ML artifacts...")
    app.state.careroute = _initialise_state()
    print("[FastAPI] Startup complete. Ready to serve requests.")
    yield


app = FastAPI(
    title="CareRoute: Geospatial Carer Visit Optimiser & Risk API",
    version="1.0.0",
    description="Multi-objective VRPTW scheduling over OpenStreetMap road networks with XGBoost + TreeSHAP late-visit explainability.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)