from pathlib import Path

# Base paths
ROOT_DIR = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Geospatial target: Camden, London, UK (Dense urban road network, ~22 km^2)
TARGET_PLACE = "Camden, London, United Kingdom"
CENTER_LAT = 51.5450
CENTER_LON = -0.1600
GRAPH_DIST_METERS = 3500  # 3.5km radius around center (~38 km^2 bounding box)

# Simulation constants
RANDOM_SEED = 42
NUM_CARERS = 20
NUM_VISITS = 200

# Artifact file paths
GRAPH_PATH = ARTIFACTS_DIR / "road_network.graphml"
DATASET_PATH = ARTIFACTS_DIR / "synthetic_dataset.json"
MATRIX_PATH = ARTIFACTS_DIR / "travel_time_matrix.npz"
MODEL_PATH = ARTIFACTS_DIR / "xgb_late_model.json"
ML_METRICS_PATH = ARTIFACTS_DIR / "ml_metrics.json"
