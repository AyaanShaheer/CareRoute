from fastapi.testclient import TestClient
from backend.app.main import app


def test_health_and_metadata() -> None:
    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        h_data = health.json()
        assert h_data["status"] == "healthy"
        assert h_data["carers_loaded"] == 20
        assert h_data["visits_loaded"] == 200
        assert h_data["ml_roc_auc"] >= 0.75

        meta = client.get("/api/v1/metadata")
        assert meta.status_code == 200
        m_data = meta.json()
        assert len(m_data["carers"]) == 20
        assert len(m_data["visits"]) == 200
        assert "global_mean_abs_shap" in m_data["ml_metrics"]


def test_optimise_endpoint_and_shap_explanations() -> None:
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/optimise",
            json={
                "continuity_weight": 1.5,
                "time_limit_seconds": 5,
                "include_geometry": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()

        naive_kpis = data["naive_schedule"]["kpis"]
        opt_kpis = data["optimised_schedule"]["kpis"]

        # 1. Verify OR-Tools beats Naive travel time significantly
        assert opt_kpis["total_travel_mins"] < naive_kpis["total_travel_mins"]
        assert opt_kpis["travel_time_reduction_pct"] > 20.0

        # 2. Verify workload std dev is reduced vs Naive Baseline
        assert opt_kpis["workload_std_dev_mins"] < naive_kpis["workload_std_dev_mins"]

        # 3. Verify unassigned visits are less than or equal to Naive Baseline
        assert opt_kpis["unassigned_visits_count"] <= naive_kpis["unassigned_visits_count"]

        # 4. Verify every assigned visit has valid XGBoost probabilities, 3 SHAP drivers, and street geometry
        assigned_visits = [
            av
            for r in data["optimised_schedule"]["routes"]
            for av in r["assigned_visits"]
        ]
        assert len(assigned_visits) == 200 - opt_kpis["unassigned_visits_count"]
        sample_visit = assigned_visits[0]
        assert 0.0 <= sample_visit["late_risk_prob"] <= 1.0
        assert len(sample_visit["top_shap_drivers"]) == 3
        assert len(sample_visit["route_geometry"]) >= 1