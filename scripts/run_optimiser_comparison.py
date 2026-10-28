import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.core.config import DATASET_PATH, MATRIX_PATH
from backend.app.optimiser.baseline import solve_naive_baseline
from backend.app.optimiser.ortools_vrptw import solve_ortools_vrptw
from backend.app.schemas.models import Carer, Visit


def main() -> None:
    # 1. Load generated synthetic data and road matrix
    raw_data = json.loads(DATASET_PATH.read_text())
    carers = [Carer(**c) for c in raw_data["carers"]]
    visits = [Visit(**v) for v in raw_data["visits"]]
    matrix = np.load(MATRIX_PATH)["matrix_mins"]

    print("\n[Optimiser] 1. Solving with Naive Baseline Heuristic...")
    naive_res = solve_naive_baseline(carers, visits, matrix)

    print("[Optimiser] 2. Solving with Google OR-Tools Guided Local Search...")
    ortools_res = solve_ortools_vrptw(carers, visits, matrix, time_limit_seconds=5)

    # Compute percentage travel time reduction
    baseline_trav = naive_res.kpis.total_travel_mins
    ortools_trav = ortools_res.kpis.total_travel_mins
    reduction_pct = round(((baseline_trav - ortools_trav) / baseline_trav) * 100, 1)
    ortools_res.kpis.travel_time_reduction_pct = reduction_pct

    # Print results comparison table
    col_w = 26
    print("\n" + "=" * 65)
    print("           CAREROUTE OPTIMISATION PERFORMANCE BENCHMARK")
    print("=" * 65)
    print(f"{'Metric':<{col_w}} | {'Naive Baseline':<16} | {'OR-Tools VRPTW':<16}")
    print("-" * 65)
    print(
        f"{'Total Travel Time (mins)':<{col_w}} | "
        f"{naive_res.kpis.total_travel_mins:<16} | "
        f"{ortools_res.kpis.total_travel_mins:<16}"
    )
    print(
        f"{'Travel Time Reduction':<{col_w}} | "
        f"{'-':<16} | "
        f"{f'{reduction_pct}%':<16}"
    )
    print(
        f"{'Continuity of Care':<{col_w}} | "
        f"{f'{naive_res.kpis.continuity_score_pct}%':<16} | "
        f"{f'{ortools_res.kpis.continuity_score_pct}%':<16}"
    )
    print(
        f"{'Workload Std Dev (mins)':<{col_w}} | "
        f"{naive_res.kpis.workload_std_dev_mins:<16} | "
        f"{ortools_res.kpis.workload_std_dev_mins:<16}"
    )
    print(
        f"{'Unassigned Visits':<{col_w}} | "
        f"{naive_res.kpis.unassigned_visits_count:<16} | "
        f"{ortools_res.kpis.unassigned_visits_count:<16}"
    )
    print("=" * 65)


if __name__ == "__main__":
    main()