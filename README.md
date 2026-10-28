# CareRoute: Carer Visit Scheduling Optimiser & Risk Engine

CareRoute is an end-to-end **Operations Research + MLOps** web application designed to solve complex home-care scheduling challenges. It optimises daily carer rotas across real city street networks while respecting strict time windows, skill requirements, workload equity, and client-carer continuity of care, backed by an **XGBoost + TreeSHAP** late-visit risk and explainability model.

<img width="1872" height="912" alt="Screenshot 2026-10-28 143930" src="https://github.com/user-attachments/assets/a996580a-08c6-456d-8568-deeeda792313" />


---

## ✨ Key Features

- **Geospatial Optimisation:** Real OpenStreetMap road network (Camden, London) via `OSMnx` + `NetworkX` shortest-path travel-time matrices.
- **Multi-Objective VRPTW:** `Google OR-Tools` Guided Local Search balancing fleet travel time, carer continuity of care (Pareto weight β), and workload equity (duty-time soft bounds).
- **Greedy Baseline:** Naive Earliest-Deadline-First heuristic for benchmark comparison.
- **ML Risk Engine:** `XGBoost` late-visit classifier trained on 60 days of synthetic historical rota data, scored with `SHAP` TreeExplainer local feature attributions.
- **Operations Dashboard:** `Next.js` + `TypeScript` + `React-Leaflet` interactive street-network map, risk-coloured visit markers, Pareto continuity slider, SHAP inspector, and priority risk queue.

---

## Architecture & Tech Stack

```
+-----------------------------------------------------------------------------------+
|                            OFFLINE MLOPS & DATA PIPELINE                          |
|  [OSMnx Road Graph] ---> [NetworkX Shortest Paths] ---> [Travel-Time Matrix (.npz)]|
|  [Synthetic Generator] -> [60-Day Historical DB] ------> [XGBoost + SHAP Artifacts] |
+-----------------------------------+---------------------------------------------------+
                                    | (Artifacts / Model / Matrix)
                                    v
+-----------------------------------------------------------------------------------+
|                        FASTAPI BACKEND SERVICE (CONTAINERISED)                    |
|   POST /api/v1/optimise                                                           |
|     ├──> Naive Baseline (Greedy Earliest-Deadline-First)                            |
|     ├──> Google OR-Tools VRPTW Solver (Guided Local Search + DutyTime Equity)    |
|     └──> XGBoost Risk Scorer + TreeSHAP Local Feature Attributions                |
+-----------------------------------+---------------------------------------------------+
                                    | (REST JSON / GeoJSON Street Polylines)
                                    v
+-----------------------------------------------------------------------------------+
|                       NEXT.JS + TYPESCRIPT + LEAFLET DASHBOARD                    |
|  - Interactive Leaflet Map (Multi-carer street polylines & risk-coloured nodes)   |
|  - Pareto Trade-Off Controls (Continuity Weight β slider)                         |
|  - Local SHAP Explainability Inspector & Priority Risk Queue                      |
+-----------------------------------------------------------------------------------+

```

* **Geospatial & Routing:** `OSMnx`, `NetworkX`, `GeoPandas`, `Shapely` (OpenStreetMap Camden, London road network extraction, nearest-node coordinate snapping, and Dijkstra travel matrices).
* **Mathematical Optimisation:** `Google OR-Tools` (Vehicle Routing Problem with Time Windows / VRPTW, disjunction soft penalties, and multi-objective cost callbacks).
* **Machine Learning & Explainability:** `XGBoost`, `SHAP` (`TreeExplainer` capturing non-linear feature interactions like schedule slack and cumulative shift fatigue).
* **Backend API:** `FastAPI`, `Uvicorn`, `Pydantic`, `Pytest`.
* **Frontend:** `Next.js (React 18)`, `TypeScript`, `Tailwind CSS`, `React-Leaflet`.

---

## Empirical Benchmark Results (200 Visits, 20 Carers)

Evaluated against a greedy Earliest-Deadline-First naive baseline across 200 daily visits:

| Metric | Naive Baseline | OR-Tools (Speed) | OR-Tools (Balanced) | OR-Tools (High Continuity) |
| --- | --- | --- | --- | --- |
| **Total Fleet Travel Time** | 1,966.3 mins | 1,063.2 mins | 1,415.5 mins | 1,668.4 mins |
| **Travel Time Reduction** | 0.0% | **-45.9%** | **-28.0%** | **-15.2%** |
| **Continuity Score (%)** | 72.1% | 8.1% | **57.8%** | **77.8%** |
| **Workload Std Dev** | ±103.4 mins | ±44.6 mins | ±54.5 mins | ±67.7 mins |
| **Unassigned Visits** | 3 | 3 | **1** | 2 |
| **High Late-Risk Visits (≥ 50%)** | 9 | 8 | **6** | 6 |

---

## Project Structure

```text
careroute/
├── artifacts/                # Cached OSM road graph, travel matrices, & XGBoost models
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI route handlers (/health, /metadata, /optimise)
│   │   ├── core/             # Config & constants
│   │   ├── data/             # OSMnx graph builder, synthetic generator & route geometries
│   │   ├── ml/               # XGBoost training & TreeSHAP local explainer
│   │   ├── optimiser/        # Naive greedy baseline & OR-Tools VRPTW solver
│   │   ├── schemas/          # Pydantic data contracts
│   │   └── main.py           # FastAPI entrypoint & lifespan state manager
│   └── tests/                # Pytest integration suite
├── frontend/                 # Next.js + TypeScript + Leaflet Dashboard
├── scripts/                  # CLI data generation & ML training scripts
└── pytest.ini                # Pytest configuration

```

---

## Getting Started & Local Development

### 1. Backend Setup (FastAPI)

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows Git Bash: source .venv/Scripts/activate

# Install dependencies
pip install -r backend/requirements.txt

# Build offline OSM graph, synthetic dataset, travel matrix, and train XGBoost
python scripts/build_data_layer.py
python scripts/train_and_evaluate_ml.py

# Run Pytest test suite
pytest -v

# Start FastAPI server
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

```

### 2. Frontend Setup (Next.js Dashboard)

Open a separate terminal window:

```bash
cd frontend
npm install
npm run dev

```

Open **`http://localhost:3000`** in your browser.

---

## Quick Start with Docker

```bash
docker compose up --build
```

This builds and starts the FastAPI backend (exposed on `http://localhost:8000`) and the Next.js dashboard (exposed on `http://localhost:3000`), with the backend health-gated startup.

---

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Service health, loaded carers/visits, graph size, and ML ROC-AUC. |
| `GET` | `/api/v1/metadata` | Carers, client visits, and trained ML evaluation metrics. |
| `POST` | `/api/v1/optimise` | Solve OR-Tools VRPTW schedule, score with XGBoost + TreeSHAP, compare to naive baseline. |

---

## ⚠️ Known Environment Note (For OpenCode / AI Coding Agents)

### Issue: WSL2 Host-to-Browser Cross-Origin or `ECONNREFUSED` Fetch Failure

When running the stack inside a **WSL2 Linux environment** while opening the browser on the **Windows host machine**, Next.js frontend requests to backend endpoints (`http://localhost:8000/api/v1/...`) can occasionally fail with `Failed to fetch` or `ECONNREFUSED` due to local socket loopback translation or IPv4/IPv6 mismatches between Node.js (`127.0.0.1`) and Uvicorn (`0.0.0.0`).

#### Instructions for OpenCode Agent to Fix:

1. **Verify Backend Binding:** Ensure Uvicorn is explicitly bound to `0.0.0.0:8000` (`uvicorn backend.app.main:app --host 0.0.0.0 --port 8000`).

2. **Configure Next.js Rewrites (`frontend/next.config.mjs`):** Implement a robust server-side proxy so browser API calls to `/api/v1/*` are transparently proxied by Next.js to FastAPI without triggering cross-origin or loopback refusal:

```javascript
const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: "http://127.0.0.1:8000/api/v1/:path*",
      },
    ];
  },
};
export default nextConfig;

```

3. **Frontend Client Fetch Base:** Ensure `frontend/src/app/page.tsx` uses relative paths (`/api/v1`) or checks an environment variable (`NEXT_PUBLIC_API_URL`) so production and local Docker setups work seamlessly without hardcoded host IPs.

---

## Final Verification Checklist (for the Open-Source Agent)

- [ ] Backend test suite passes: `pytest -v` (FastAPI + OR-Tools + XGBoost integration tests).
- [ ] Lint passes: `ruff check backend scripts`.
- [ ] Frontend compiles: `npx tsc --noEmit` and `npm run build`.
- [ ] Docker images build: `docker compose build`.
- [ ] Live stack healthy: FastAPI `/api/v1/health` → 200, Next.js proxy → 200, `POST /api/v1/optimise` → 200.

---

*Built with OR-Tools, XGBoost, FastAPI, Next.js, and OpenStreetMap data.*
