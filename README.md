# CareRoute 🏥
### Intelligent Home-Care Scheduling · Operations Research · MLOps

**Optimising care delivery across real-world street networks with mathematical optimisation, machine learning, and explainable AI.**

CareRoute is an end-to-end scheduling optimisation and risk intelligence system for home-care operations. It combines **real-world geospatial routing, Vehicle Routing Problems with Time Windows (VRPTW), workload balancing, continuity-of-care optimisation, and explainable late-visit risk prediction** into an interactive operations dashboard.

Built around the Camden, London, street network, CareRoute helps explore a critical operational question:

> How can home-care providers reduce travel time, deliver visits within their required time windows, distribute workloads fairly, and preserve continuity of care?

## 📸 Application Preview

<img width="1872" height="912" alt="CareRoute Operations Dashboard" src="https://github.com/user-attachments/assets/a996580a-08c6-456d-8568-deeeda792313" />

*Interactive scheduling dashboard with geospatial routes, visit risk indicators, and optimisation controls.*

---

## ⚡ Benchmark at a Glance

Evaluated against a greedy Earliest-Deadline-First baseline across **200 daily visits and 20 carers**.

| Metric | Greedy Baseline | OR-Tools Speed | OR-Tools Balanced | OR-Tools High Continuity |
|:--|--:|--:|--:|--:|
| Fleet travel time | 1,966.3 min | **1,063.2 min** | 1,415.5 min | 1,668.4 min |
| Travel-time reduction | — | **45.9%** | 28.0% | 15.2% |
| Continuity score | 72.1% | 8.1% | 57.8% | **77.8%** |
| Workload standard deviation | ±103.4 min | **±44.6 min** | ±54.5 min | ±67.7 min |
| Unassigned visits | 3 | 3 | **1** | 2 |
| High late-risk visits (≥50%) | 9 | 8 | **6** | 6 |

**The trade-off:** The Speed configuration prioritises travel efficiency, while High Continuity prioritises retaining familiar carers for clients. The Balanced configuration offers a compromise between these competing objectives.

These results are from the supplied benchmark scenario, not a claim of measured improvement in a live care service.

---

## 🧠 What Makes CareRoute Different?

CareRoute brings three engineering disciplines together in one operational workflow.

| Capability | Implementation | Purpose |
|:--|:--|:--|
| Geospatial intelligence | OSMnx + NetworkX | Model real streets and calculate travel times |
| Combinatorial optimisation | Google OR-Tools | Build schedules under time-window constraints |
| Machine learning | XGBoost | Estimate the risk of late visits |
| Explainable AI | SHAP TreeExplainer | Explain individual visit risk predictions |
| API engineering | FastAPI + Pydantic | Expose scheduling and risk-scoring capabilities |
| Operations dashboard | Next.js + TypeScript + Leaflet | Inspect routes, trade-offs, and visit priorities |
| Testing | Pytest | Validate backend behaviour and integration workflows |

## ✨ Key Features

### 🗺️ 1. Real-World Geospatial Routing

CareRoute uses the OpenStreetMap road network for Camden, London, rather than relying on straight-line distances.

- Extract street-network data with **OSMnx**.
- Snap client and carer coordinates to nearby road-network nodes.
- Calculate shortest paths with **NetworkX and Dijkstra's algorithm**.
- Build reusable travel-time matrices for the optimisation engine.
- Return route geometries for visualisation on the interactive map.

**Why it matters:** Travel between two addresses depends on the road network, not simply the distance between their coordinates.

### 🧮 2. Multi-Objective Scheduling Optimisation

The scheduling engine formulates home-care rota planning as a **Vehicle Routing Problem with Time Windows (VRPTW)** using Google OR-Tools.

The optimisation considers:

- **Visit time windows:** Schedule visits within their specified windows.
- **Skill compatibility:** Respect required carer skills.
- **Travel efficiency:** Reduce cumulative fleet travel time.
- **Continuity of care:** Prefer familiar carer-client assignments.
- **Workload equity:** Penalise undesirable imbalances in carer duty time.
- **Visit assignment:** Account for infeasible or unassigned visits through configurable penalties.

A continuity-weight parameter, β, lets users explore the trade-off between operational efficiency and familiar-carer assignments.

CareRoute also includes a greedy Earliest-Deadline-First baseline, providing a comparison against a simpler scheduling heuristic.

### 🤖 3. Late-Visit Risk Intelligence

The ML component uses **XGBoost** to classify the likelihood of a visit running late.

The model is trained on 60 days of synthetic historical rota data and evaluates scheduling-related features such as:

- Available schedule slack.
- Travel-time estimates.
- Visit timing and time-window pressure.
- Cumulative shift workload and fatigue proxies.

Predictions are surfaced in the dashboard to help operators identify visits that may need attention.

### 🔍 4. Local Explainability with TreeSHAP

Each scored visit can be inspected using SHAP-based feature attributions.

The explanation layer helps answer questions such as:

- Which features contributed most to a visit's predicted risk?
- Was the visit affected by limited schedule slack or demanding travel?
- Which high-risk visits should an operator review first?

The SHAP inspector makes individual predictions easier to investigate. Attributions explain model behaviour; they should not be interpreted as proof of causation.

### 📊 5. Interactive Operations Dashboard

The Next.js dashboard brings scheduling, route visualisation, and risk inspection together.

- **Multi-carer route map:** Explore scheduled routes on real street networks.
- **Risk-coloured visit markers:** Identify visits with elevated predicted lateness risk.
- **Continuity trade-off slider:** Explore different continuity-weight settings.
- **SHAP inspector:** Investigate local model explanations.
- **Priority risk queue:** Review visits flagged for operational attention.
- **Benchmark comparison:** Compare the optimised solution against the greedy baseline.

---

## 🏗️ System Architecture

```text
                         OFFLINE DATA & MLOPS PIPELINE
 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │  OpenStreetMap ──► OSMnx ──► NetworkX Shortest Paths          │
 │                                   │                          │
 │                                   ▼                          │
 │                       Travel-Time Matrix (.npz)               │
 │                                                              │
 │  Synthetic Rota Generator ──► 60-Day Historical Dataset      │
 │                                   │                          │
 │                                   ▼                          │
 │                          XGBoost Model + SHAP                │
 │                                                              │
 │                Persisted Data and Model Artifacts             │
 └──────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
                       FASTAPI BACKEND SERVICE
 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │                       POST /optimise                         │
 │                              │                               │
 │               ┌──────────────┼──────────────┐                │
 │               ▼              ▼              ▼                │
 │          Greedy EDF      OR-Tools       XGBoost Risk          │
 │          Baseline        VRPTW Solver   + TreeSHAP            │
 │               │              │              │                │
 │               └──────────────┼──────────────┘                │
 │                              ▼                               │
 │                   Schedule + Risk + Routes                   │
 └──────────────────────────────┬───────────────────────────────┘
                                │
                          REST / JSON
                                │
                                ▼
                      NEXT.JS OPERATIONS UI
 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │   React + TypeScript + Tailwind CSS + React-Leaflet          │
 │                                                              │
 │   Route Map │ Continuity Slider │ SHAP Inspector             │
 │             │                   │                            │
 │             └──── Priority Risk Queue ────────────────────── │
 └──────────────────────────────────────────────────────────────┘
```

### Technology Stack

**Geospatial and routing**
- OSMnx
- NetworkX
- GeoPandas
- Shapely
- OpenStreetMap

**Operations research**
- Google OR-Tools
- Vehicle Routing Problem with Time Windows
- Guided Local Search
- Configurable assignment penalties
- Duty-time equity penalties
- Multi-objective cost weighting

**Machine learning and explainability**
- XGBoost
- SHAP TreeExplainer
- scikit-learn evaluation utilities
- Synthetic historical rota generation

**Backend and testing**
- Python
- FastAPI
- Uvicorn
- Pydantic
- Pytest

**Frontend**
- Next.js
- React 18
- TypeScript
- Tailwind CSS
- React-Leaflet

**Deployment and runtime**
- Docker
- Docker Compose
- Persisted model and routing artifacts

---

## 📈 Understanding the Optimisation Trade-Off

CareRoute supports three optimisation configurations to demonstrate the competing objectives in rota planning.

| Configuration | Primary objective | Observed benchmark outcome |
|:--|:--|:--|
| **Speed** | Minimise fleet travel time | 45.9% lower travel time than baseline |
| **Balanced** | Balance travel, continuity, and workload | 28.0% lower travel time and 57.8% continuity |
| **High Continuity** | Prioritise familiar carer-client assignments | 77.8% continuity with 15.2% lower travel time |

The benchmark illustrates why there is no single universally optimal rota: a solution can perform well on travel time and still perform poorly on continuity.

**Important:** The continuity score and travel-time figures describe the supplied synthetic evaluation scenario. Their operational value depends on the assumptions, constraints, penalties, and evaluation methodology used.

---

## 📂 Project Structure

```text
careroute/
├── artifacts/
│   ├── road_graph/              # Cached OSM street-network data
│   ├── travel_matrix/           # Precomputed travel-time matrices
│   └── models/                  # Trained ML model artifacts
│
├── backend/
│   ├── app/
│   │   ├── api/                 # Health, metadata, optimisation endpoints
│   │   ├── core/                # Configuration and constants
│   │   ├── data/                # OSMnx graph and synthetic data pipeline
│   │   ├── ml/                  # XGBoost scoring and SHAP explanations
│   │   ├── optimiser/           # Greedy baseline and OR-Tools solver
│   │   ├── schemas/              # Pydantic data contracts
│   │   └── main.py               # FastAPI entry point and lifespan
│   │
│   ├── tests/                    # Backend and integration tests
│   └── requirements.txt
│
├── frontend/                     # Next.js and TypeScript dashboard
├── scripts/
│   ├── build_data_layer.py       # Geospatial/data artifact generation
│   └── train_and_evaluate_ml.py  # Model training and evaluation
│
├── pytest.ini
├── docker-compose.yml
└── README.md
```

*The tree illustrates the intended project layout; artifact and configuration filenames may differ slightly in the repository.*

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible version supported by the project dependencies.
- Node.js and npm.
- Git.
- Docker Desktop and Docker Compose, if using the containerised setup.
- Internet access for the initial OpenStreetMap data download.

OSM data extraction and shortest-path matrix construction can be resource-intensive. Once the data artifacts have been generated, they can be reused for subsequent local runs.

### 1. Clone the Repository

```bash
git clone https://github.com/AyaanShaheer/CareRoute.git
cd careroute
```

### 2. Set Up the Python Environment

```bash
python -m venv .venv
```

Activate the environment.

**Linux / macOS / Git Bash**

```bash
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```bash
pip install -r backend/requirements.txt
```

### 3. Build the Data and ML Artifacts

Run these commands from the repository root:

```bash
python scripts/build_data_layer.py
python scripts/train_and_evaluate_ml.py
```

The data pipeline prepares the street-network and travel-time artifacts, generates the synthetic historical rota dataset, and trains and evaluates the ML model.

The first run may require internet access to retrieve OpenStreetMap data.

### 4. Run the Test Suite

```bash
pytest -v
```

Use the test results to verify the current backend implementation before running the application.

### 5. Start the Backend

```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend API:

- API health: `http://localhost:8000/api/v1/health`
- Metadata: `http://localhost:8000/api/v1/metadata`
- Interactive API documentation: `http://localhost:8000/docs`

### 6. Start the Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000** to explore the dashboard.

---

## 🐳 Run with Docker Compose

For the containerised workflow, run the following command from the repository root:

```bash
docker compose up --build
```

Expected service URLs:

| Service | Local URL |
|:--|:--|
| Next.js dashboard | `http://localhost:3000` |
| FastAPI backend | `http://localhost:8000` |
| API documentation | `http://localhost:8000/docs` |

The Compose setup is intended to start the backend before the frontend depends on it. Actual readiness behaviour depends on the configured health check and service dependencies.

---

## 🔌 API Reference

| Method | Endpoint | Purpose |
|:--|:--|:--|
| `GET` | `/api/v1/health` | Report service health and available runtime/model metadata |
| `GET` | `/api/v1/metadata` | Return carer, visit, and model evaluation information |
| `POST` | `/api/v1/optimise` | Generate a rota, score visit risks, and compare scheduling approaches |

The optimisation endpoint brings together the greedy baseline, OR-Tools scheduling engine, ML risk predictions, SHAP explanations, and route information for the dashboard.

Refer to the live FastAPI documentation at `/docs` for request and response schemas.

---

## 🧪 Data, Evaluation & Limitations

CareRoute is an engineering and research prototype built around synthetic care-visit data and an OpenStreetMap street network.

- **Synthetic training data:** The historical rota dataset is generated for the project; it is not a real care provider's operational history.
- **Estimated travel times:** The shortest-path matrix depends on the configured network and travel-speed assumptions. It should not be treated as live traffic data.
- **Predictive evaluation:** Model metrics such as ROC-AUC measure performance on the selected evaluation split; they do not, by themselves, establish real-world predictive accuracy.
- **Optimisation quality:** Results depend on solver settings, objective weights, penalties, the feasibility constraints, and the time available to search for solutions.
- **Operational review:** A predicted late-visit risk is a decision-support signal, not a certainty. Human oversight remains important.

This distinction matters because a promising synthetic benchmark is a starting point for operational validation, not a substitute for it.

## 🔭 Potential Next Steps

- Add live traffic-aware travel-time estimates.
- Introduce uncertainty-aware scheduling using distributions of travel and service times.
- Track model versions and evaluation metrics across training runs.
- Add data-drift and prediction-quality monitoring.
- Build schedule replay and what-if scenario analysis.
- Add role-based access, audit logs, and secure handling of client information.
- Evaluate on appropriately governed, anonymised real-world data before any production use.

## 🎯 Engineering Concepts Demonstrated

CareRoute is designed to demonstrate practical engineering across:

- Operations research and constrained optimisation.
- Graph algorithms and geospatial data processing.
- Machine-learning model training and inference.
- Explainable AI with local feature attributions.
- REST API design and typed data contracts.
- Interactive data visualisation.
- Automated testing and containerised deployment.
- Benchmark-driven evaluation and multi-objective trade-off analysis.

## 👨‍💻 Author

**Ayaan Shaheer**

GitHub: [@AyaanShaheer](https://github.com/AyaanShaheer)

---

*Built with Python, Google OR-Tools, OSMnx, XGBoost, SHAP, FastAPI, Next.js, and OpenStreetMap data.*
