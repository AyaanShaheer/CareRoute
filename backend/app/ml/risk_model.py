import json
from typing import Dict, List

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from backend.app.core.config import ML_METRICS_PATH, MODEL_PATH, RANDOM_SEED
from backend.app.schemas.models import (
    Carer,
    OptimisationResponse,
    ShapDriver,
    Visit,
)

FEATURE_COLUMNS = [
    "travel_time_from_prev_mins",
    "slack_mins",
    "visit_order_in_shift",
    "cumulative_work_mins",
    "is_peak_traffic_hour",
    "visit_duration_mins",
    "carer_familiarity_count",
    "rota_volatility_score",
]


def _is_rush_hour(arrival_min: int) -> float:
    """Returns 1.0 if arrival falls in London morning (07:30-09:30) or evening (16:30-19:00) peak."""
    morning_peak = 450 <= arrival_min <= 570
    evening_peak = 990 <= arrival_min <= 1140
    return 1.0 if (morning_peak or evening_peak) else 0.0


def _carer_volatility(carer_id: str) -> float:
    """Deterministic 7-day rolling rota volatility score per carer in [0.10, 0.85]."""
    idx = int(carer_id.split("_")[1])
    rng = np.random.default_rng(RANDOM_SEED + idx * 17)
    return round(float(rng.uniform(0.10, 0.85)), 2)


def generate_historical_training_data(
    carers: List[Carer],
    visits: List[Visit],
    travel_matrix: np.ndarray,
    num_days: int = 60,
) -> pd.DataFrame:
    """
    Simulates 60 days of historical care rota executions (12,000 visit logs)
    with realistic operational delay mechanics.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n_carers = len(carers)
    records = []

    for day in range(num_days):
        for v_idx, visit in enumerate(visits):
            travel_mins = float(
                rng.choice(travel_matrix[travel_matrix > 0])
                * rng.uniform(0.75, 1.35)
            )
            slack_mins = float(rng.triangular(0.0, 12.0, 55.0))
            visit_order = int(rng.integers(1, 12))
            cum_work_mins = float(
                visit_order * rng.uniform(32.0, 48.0) + rng.normal(0, 10)
            )
            arrival_min = int(visit.window_end_min - slack_mins)
            peak_hour = _is_rush_hour(arrival_min)

            # Pick a random carer (sometimes familiar, sometimes unfamiliar cover)
            cid = carers[int(rng.integers(0, n_carers))].carer_id
            familiarity = float(visit.historical_carer_counts.get(cid, 0))
            volatility = float(
                np.clip(_carer_volatility(cid) + rng.normal(0, 0.08), 0.05, 0.95)
            )

            # Ground-truth structural log-odds of a late/missed visit (>5 mins past window)
            logit = (
                -1.85
                + 0.14 * (travel_mins - 8.0)
                - 0.11 * (slack_mins - 15.0)
                + 0.16 * max(visit_order - 5, 0)
                + 0.005 * max(cum_work_mins - 240.0, 0)
                + 0.65 * peak_hour
                - 0.045 * familiarity
                + 1.45 * (volatility - 0.40)
                + rng.normal(0, 0.45)
            )
            prob_late = 1.0 / (1.0 + np.exp(-logit))
            is_late = int(rng.random() < prob_late)

            records.append(
                {
                    "travel_time_from_prev_mins": round(travel_mins, 2),
                    "slack_mins": round(slack_mins, 2),
                    "visit_order_in_shift": float(visit_order),
                    "cumulative_work_mins": round(max(cum_work_mins, 15.0), 1),
                    "is_peak_traffic_hour": peak_hour,
                    "visit_duration_mins": float(visit.duration_mins),
                    "carer_familiarity_count": familiarity,
                    "rota_volatility_score": round(volatility, 2),
                    "target_late": is_late,
                }
            )

    return pd.DataFrame(records)


def train_and_save_risk_model(
    carers: List[Carer],
    visits: List[Visit],
    travel_matrix: np.ndarray,
) -> Dict[str, float]:
    """Trains the XGBoost Late-Visit Classifier, computes SHAP importances, and saves artifacts."""
    df = generate_historical_training_data(carers, visits, travel_matrix, num_days=60)
    X = df[FEATURE_COLUMNS]
    y = df["target_late"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        base_score=0.5,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    roc_auc = float(roc_auc_score(y_test, y_prob))
    f1 = float(f1_score(y_test, y_pred))
    brier = float(brier_score_loss(y_test, y_prob))
    late_rate = float(y.mean())

    # Compute global mean(|SHAP|) across test set
    shap_matrix = _compute_shap_matrix(model, X_test)
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    global_shap = {
        col: round(float(val), 4)
        for col, val in sorted(
            zip(FEATURE_COLUMNS, mean_abs_shap), key=lambda x: x[1], reverse=True
        )
    }

    model.save_model(str(MODEL_PATH))

    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "historical_late_rate_pct": round(late_rate * 100, 1),
        "roc_auc": round(roc_auc, 4),
        "f1_score": round(f1, 4),
        "brier_score": round(brier, 4),
        "global_mean_abs_shap": global_shap,
    }
    ML_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def _compute_shap_matrix(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Computes exact TreeSHAP values for each row in X."""
    try:
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(X)
        if isinstance(vals, list):
            vals = vals[1]
        return np.asarray(vals, dtype=np.float32)
    except Exception:
        # Native XGBoost C++ TreeSHAP fallback (last column is bias term)
        booster = model.get_booster()
        dmat = xgb.DMatrix(X, feature_names=FEATURE_COLUMNS)
        contribs = booster.predict(dmat, pred_contribs=True)
        return np.asarray(contribs[:, :-1], dtype=np.float32)


def load_risk_model() -> xgb.XGBClassifier:
    """Loads the trained XGBoost model from disk."""
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def score_schedule_with_shap(
    schedule: OptimisationResponse,
    visits: List[Visit],
    model: xgb.XGBClassifier,
) -> OptimisationResponse:
    """
    Scores every assigned visit in a schedule with XGBoost lateness probability
    and attaches the top 3 local TreeSHAP drivers.
    """
    visit_lookup: Dict[str, Visit] = {v.visit_id: v for v in visits}
    rows = []
    visit_refs = []

    for route in schedule.routes:
        cum_work = 0.0
        volatility = _carer_volatility(route.carer_id)

        for order_idx, av in enumerate(route.assigned_visits, start=1):
            v_meta = visit_lookup[av.visit_id]
            cum_work += av.travel_time_from_prev_mins + v_meta.duration_mins
            familiarity = float(
                v_meta.historical_carer_counts.get(route.carer_id, 0)
            )

            rows.append(
                {
                    "travel_time_from_prev_mins": round(
                        av.travel_time_from_prev_mins, 2
                    ),
                    "slack_mins": round(av.slack_mins, 2),
                    "visit_order_in_shift": float(order_idx),
                    "cumulative_work_mins": round(cum_work, 1),
                    "is_peak_traffic_hour": _is_rush_hour(av.arrival_min),
                    "visit_duration_mins": float(v_meta.duration_mins),
                    "carer_familiarity_count": familiarity,
                    "rota_volatility_score": volatility,
                }
            )
            visit_refs.append(av)

    if not rows:
        return schedule

    X_sched = pd.DataFrame(rows)[FEATURE_COLUMNS]
    probs = model.predict_proba(X_sched)[:, 1]
    shap_mat = _compute_shap_matrix(model, X_sched)

    high_risk_count = 0
    for idx, av in enumerate(visit_refs):
        prob = round(float(probs[idx]), 3)
        av.late_risk_prob = prob
        if prob >= 0.50:
            high_risk_count += 1

        row_shap = shap_mat[idx]
        # Top 3 features by absolute SHAP magnitude for this specific visit
        top_indices = np.argsort(np.abs(row_shap))[::-1][:3]
        av.top_shap_drivers = [
            ShapDriver(
                feature_name=FEATURE_COLUMNS[f_idx],
                feature_value=round(float(X_sched.iloc[idx, f_idx]), 2),
                shap_value=round(float(row_shap[f_idx]), 3),
            )
            for f_idx in top_indices
        ]

    schedule.kpis.high_risk_visits_count = high_risk_count
    return schedule