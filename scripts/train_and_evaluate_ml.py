import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import DATASET_PATH, MATRIX_PATH
from backend.app.ml.risk_model import (
    load_risk_model,
    score_schedule_with_shap,
    train_and_save_risk_model,
)
from backend.app.optimiser.baseline import solve_naive_baseline
from backend.app.optimiser.ortools_vrptw import solve_ortools_vrptw
from backend.app.schemas.models import Carer, Visit


def main() -> None:
    raw_data = json.loads(DATASET_PATH.read_text())
    carers = [Carer(**c) for c in raw_data["carers"]]
    visits = [Visit(**v) for v in raw_data["visits"]]
    matrix = np.load(MATRIX_PATH)["matrix_mins"]

    print("[MLOps] 1. Generating 60-day historical dataset (12,000 visits) & training XGBoost...")
    metrics = train_and_save_risk_model(carers, visits, matrix)

    print("\n--- XGBoost Late-Visit Classifier Metrics (Held-Out Test Set) ---")
    print(f"Train / Test Split:     {metrics['train_rows']} / {metrics['test_rows']} rows")
    print(f"Base Late Visit Rate:   {metrics['historical_late_rate_pct']}%")
    print(f"ROC-AUC Score:          {metrics['roc_auc']}")
    print(f"F1 Score:               {metrics['f1_score']}")
    print(f"Brier Calibration Loss: {metrics['brier_score']}")
    print("\nGlobal Feature Importance (Mean |SHAP| in log-odds):")
    for feat, val in metrics["global_mean_abs_shap"].items():
        print(f"  - {feat:<28}: {val:.4f}")

    # Load model and score Naive vs OR-Tools Pareto Frontier
    model = load_risk_model()

    print("\n[Pareto + Risk] Evaluating Naive Baseline vs OR-Tools Pareto Frontier...")
    naive = score_schedule_with_shap(
        solve_naive_baseline(carers, visits, matrix), visits, model
    )
    ort_speed = score_schedule_with_shap(
        solve_ortools_vrptw(carers, visits, matrix, continuity_weight=0.0, time_limit_seconds=5),
        visits,
        model,
    )
    ort_bal = score_schedule_with_shap(
        solve_ortools_vrptw(carers, visits, matrix, continuity_weight=1.5, time_limit_seconds=5),
        visits,
        model,
    )
    ort_cont = score_schedule_with_shap(
        solve_ortools_vrptw(carers, visits, matrix, continuity_weight=3.5, time_limit_seconds=5),
        visits,
        model,
    )

    base_t = naive.kpis.total_travel_mins
    for res in (ort_speed, ort_bal, ort_cont):
        res.kpis.travel_time_reduction_pct = round(
            ((base_t - res.kpis.total_travel_mins) / base_t) * 100, 1
        )

    print("\n" + "=" * 88)
    print("               PARETO TRADE-OFF & ML LATE-RISK BENCHMARK (200 VISITS)")
    print("=" * 88)
    print(
        f"{'Metric':<25} | {'Naive Base':<12} | {'OR-Tools (Speed)':<16} | "
        f"{'OR-Tools (Bal)':<15} | {'OR-Tools (Cont)':<15}"
    )
    print("-" * 88)
    print(
        f"{'Total Travel (mins)':<25} | {naive.kpis.total_travel_mins:<12} | "
        f"{ort_speed.kpis.total_travel_mins:<16} | {ort_bal.kpis.total_travel_mins:<15} | "
        f"{ort_cont.kpis.total_travel_mins:<15}"
    )
    print(
        f"{'Travel Reduction (%)':<25} | {'0.0%':<12} | "
        f"{f'{ort_speed.kpis.travel_time_reduction_pct}%':<16} | "
        f"{f'{ort_bal.kpis.travel_time_reduction_pct}%':<15} | "
        f"{f'{ort_cont.kpis.travel_time_reduction_pct}%':<15}"
    )
    print(
        f"{'Continuity Score (%)':<25} | {f'{naive.kpis.continuity_score_pct}%':<12} | "
        f"{f'{ort_speed.kpis.continuity_score_pct}%':<16} | "
        f"{f'{ort_bal.kpis.continuity_score_pct}%':<15} | "
        f"{f'{ort_cont.kpis.continuity_score_pct}%':<15}"
    )
    print(
        f"{'Workload Std Dev (mins)':<25} | {naive.kpis.workload_std_dev_mins:<12} | "
        f"{ort_speed.kpis.workload_std_dev_mins:<16} | "
        f"{ort_bal.kpis.workload_std_dev_mins:<15} | "
        f"{ort_cont.kpis.workload_std_dev_mins:<15}"
    )
    print(
        f"{'Unassigned Visits':<25} | {naive.kpis.unassigned_visits_count:<12} | "
        f"{ort_speed.kpis.unassigned_visits_count:<16} | "
        f"{ort_bal.kpis.unassigned_visits_count:<15} | "
        f"{ort_cont.kpis.unassigned_visits_count:<15}"
    )
    print(
        f"{'High Late-Risk Visits':<25} | {naive.kpis.high_risk_visits_count:<12} | "
        f"{ort_speed.kpis.high_risk_visits_count:<16} | "
        f"{ort_bal.kpis.high_risk_visits_count:<15} | "
        f"{ort_cont.kpis.high_risk_visits_count:<15}"
    )
    print("=" * 88)

    # Show an example SHAP explanation for the highest-risk visit in the balanced schedule
    all_assigned = [av for r in ort_bal.routes for av in r.assigned_visits]
    highest_risk = max(all_assigned, key=lambda x: x.late_risk_prob)
    print(
        f"\nSample Local SHAP Explanation -> Visit {highest_risk.visit_id} "
        f"(Carer: {highest_risk.carer_id}, Late Risk: {highest_risk.late_risk_prob * 100:.1f}%):"
    )
    for d in highest_risk.top_shap_drivers:
        direction = "increases risk" if d.shap_value > 0 else "reduces risk"
        print(
            f"  * {d.feature_name} = {d.feature_value} "
            f"(SHAP: {d.shap_value:+.3f} -> {direction})"
        )


if __name__ == "__main__":
    main()