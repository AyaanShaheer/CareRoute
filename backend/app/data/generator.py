import json
import time
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import osmnx as ox

from backend.app.core.config import (
    CENTER_LAT,
    CENTER_LON,
    DATASET_PATH,
    GRAPH_DIST_METERS,
    GRAPH_PATH,
    MATRIX_PATH,
    NUM_CARERS,
    NUM_VISITS,
    RANDOM_SEED,
)
from backend.app.schemas.models import Carer, Location, Skill, Visit

OVERPASS_MIRRORS = [
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://overpass.private.coffee/api",
    "https://overpass-api.de/api",
]


def _build_fallback_camden_graph() -> nx.MultiDiGraph:
    """Builds a resilient local Camden street network if mirrors are offline."""
    print("[Geospatial] Building resilient local Camden street network (25x25 urban grid)...")
    rng = np.random.default_rng(RANDOM_SEED)
    G = nx.MultiDiGraph(crs="epsg:4326")

    grid_size = 25
    lat_span = np.linspace(CENTER_LAT - 0.025, CENTER_LAT + 0.025, grid_size)
    lon_span = np.linspace(CENTER_LON - 0.035, CENTER_LON + 0.035, grid_size)

    for r in range(grid_size):
        for c in range(grid_size):
            node_id = int(100000 + r * grid_size + c)
            lat = float(lat_span[r] + rng.normal(0, 0.0003))
            lon = float(lon_span[c] + rng.normal(0, 0.0004))
            G.add_node(node_id, y=lat, x=lon, street_count=4)

    def add_street(u: int, v: int, is_arterial: bool) -> None:
        u_y, u_x = G.nodes[u]["y"], G.nodes[u]["x"]
        v_y, v_x = G.nodes[v]["y"], G.nodes[v]["x"]
        dy_m = (v_y - u_y) * 111_320.0
        dx_m = (v_x - u_x) * 69_400.0
        length_m = float(np.hypot(dx_m, dy_m))
        speed_kph = 38.0 if is_arterial else 22.0
        travel_time_sec = float((length_m / (speed_kph * 1000.0 / 3600.0)) * 1.25)

        G.add_edge(
            u,
            v,
            key=0,
            osmid=u * 1000 + v,
            highway="primary" if is_arterial else "residential",
            length=round(length_m, 2),
            speed_kph=speed_kph,
            travel_time=round(travel_time_sec, 2),
        )

    for r in range(grid_size):
        for c in range(grid_size):
            u = int(100000 + r * grid_size + c)
            is_arterial = (r % 5 == 0) or (c % 5 == 0)
            if r + 1 < grid_size:
                v = int(100000 + (r + 1) * grid_size + c)
                add_street(u, v, is_arterial)
                add_street(v, u, is_arterial)
            if c + 1 < grid_size:
                v = int(100000 + r * grid_size + (c + 1))
                add_street(u, v, is_arterial)
                add_street(v, u, is_arterial)

    return G


def load_or_create_graph() -> nx.MultiDiGraph:
    """Loads cached road network or downloads from OSM with mirror failover."""
    if GRAPH_PATH.exists():
        print(f"[Geospatial] Loading cached road network from {GRAPH_PATH}...")
        return ox.load_graphml(GRAPH_PATH)

    G = None
    ox.settings.requests_timeout = 45
    ox.settings.overpass_rate_limit = False

    for mirror in OVERPASS_MIRRORS:
        try:
            print(f"[Geospatial] Trying Overpass mirror: {mirror} ...")
            ox.settings.overpass_url = mirror
            G = ox.graph_from_point(
                (CENTER_LAT, CENTER_LON),
                dist=GRAPH_DIST_METERS,
                network_type="drive",
                simplify=True,
            )
            G = ox.truncate.largest_component(G, strongly=True)
            G = ox.add_edge_speeds(G, fallback=25.0)
            G = ox.add_edge_travel_times(G)
            for _, _, _, data in G.edges(keys=True, data=True):
                data["travel_time"] = float(data.get("travel_time", 10.0)) * 1.33
            print(f"[Geospatial] Successfully downloaded OSM graph from {mirror}!")
            break
        except Exception as exc:
            print(f"[Geospatial] Mirror {mirror} failed ({type(exc).__name__}). Retrying next...")
            time.sleep(1)

    if G is None:
        G = _build_fallback_camden_graph()

    ox.save_graphml(G, GRAPH_PATH)
    print(f"[Geospatial] Saved graph with {len(G.nodes)} nodes and {len(G.edges)} edges.")
    return G


def _sample_clustered_locations(
    G: nx.MultiDiGraph, count: int, rng: np.random.Generator
) -> List[Location]:
    """Samples coordinates around 4 neighbourhood clusters and snaps to OSM nodes."""
    cluster_offsets = [
        (0.012, -0.015),
        (-0.010, 0.012),
        (0.008, 0.015),
        (-0.012, -0.010),
    ]

    raw_lats, raw_lons = [], []
    for _ in range(count):
        c_lat_off, c_lon_off = cluster_offsets[rng.integers(0, len(cluster_offsets))]
        lat = CENTER_LAT + c_lat_off + rng.normal(0, 0.006)
        lon = CENTER_LON + c_lon_off + rng.normal(0, 0.008)
        raw_lats.append(lat)
        raw_lons.append(lon)

    nearest_nodes = ox.distance.nearest_nodes(G, X=raw_lons, Y=raw_lats)

    locations: List[Location] = []
    for node_id in nearest_nodes:
        node_data = G.nodes[node_id]
        locations.append(
            Location(
                lat=float(node_data["y"]),
                lon=float(node_data["x"]),
                osm_node_id=int(node_id),
            )
        )
    return locations


def generate_synthetic_dataset(G: nx.MultiDiGraph) -> Tuple[List[Carer], List[Visit]]:
    """Generates 20 carers and 200 visits with skills, time windows, and continuity histories."""
    rng = np.random.default_rng(RANDOM_SEED)

    carer_locations = _sample_clustered_locations(G, NUM_CARERS, rng)
    visit_locations = _sample_clustered_locations(G, NUM_VISITS, rng)

    # 1. Generate 20 Carers
    carers: List[Carer] = []
    all_specialized_skills = [Skill.MEDICATION, Skill.DEMENTIA, Skill.MOBILITY]

    for i in range(NUM_CARERS):
        num_extra = int(rng.integers(1, 3))
        chosen_indices = rng.choice(len(all_specialized_skills), size=num_extra, replace=False)
        skills = [Skill.GENERAL] + [all_specialized_skills[idx] for idx in chosen_indices]

        carers.append(
            Carer(
                carer_id=f"CARER_{i+1:02d}",
                name=f"Carer {i+1:02d}",
                home_location=carer_locations[i],
                skills=skills,
                shift_start_min=420,  # 07:00
                shift_end_min=1260,   # 21:00
                max_working_mins=480, # 8 hours max duty
            )
        )

    # 2. Generate 200 Visits across realistic care bands
    time_bands = [
        ("morning_peak", 420, 600, 60, 0.40),   # 07:00 - 10:00 (40%)
        ("lunch_band", 690, 810, 75, 0.25),     # 11:30 - 13:30 (25%)
        ("afternoon_tea", 930, 1050, 90, 0.15), # 15:30 - 17:30 (15%)
        ("evening_bed", 1080, 1200, 60, 0.20),  # 18:00 - 20:00 (20%)
    ]
    band_probs = [b[4] for b in time_bands]
    skill_choices = [Skill.GENERAL, Skill.MEDICATION, Skill.DEMENTIA, Skill.MOBILITY]
    skill_probs = [0.45, 0.25, 0.15, 0.15]
    durations = [30, 45, 60]
    duration_probs = [0.55, 0.30, 0.15]

    visits: List[Visit] = []
    for j in range(NUM_VISITS):
        band_idx = int(rng.choice(len(time_bands), p=band_probs))
        _, min_start, max_start, width, _ = time_bands[band_idx]

        raw_start = int(rng.integers(min_start // 15, (max_start // 15) + 1) * 15)
        window_start = int(raw_start)
        window_end = int(window_start + width)

        req_skill = skill_choices[int(rng.choice(len(skill_choices), p=skill_probs))]
        duration = int(durations[int(rng.choice(len(durations), p=duration_probs))])

        qualified_carers = [c.carer_id for c in carers if req_skill in c.skills]
        sample_size = min(3, len(qualified_carers))
        chosen_carer_indices = rng.choice(len(qualified_carers), size=sample_size, replace=False)
        preferred_sample = [qualified_carers[idx] for idx in chosen_carer_indices]

        history: Dict[str, int] = {
            str(preferred_sample[0]): int(rng.integers(15, 35)),
            str(preferred_sample[1]): int(rng.integers(5, 14)),
        }
        if len(preferred_sample) > 2:
            history[str(preferred_sample[2])] = int(rng.integers(1, 5))

        visits.append(
            Visit(
                visit_id=f"VISIT_{j+1:03d}",
                client_id=f"CLIENT_{j+1:03d}",
                location=visit_locations[j],
                window_start_min=window_start,
                window_end_min=window_end,
                duration_mins=duration,
                required_skill=req_skill,
                historical_carer_counts=history,
            )
        )

    payload = {
        "carers": [c.model_dump(mode="json") for c in carers],
        "visits": [v.model_dump(mode="json") for v in visits],
    }
    DATASET_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[Data] Saved {len(carers)} carers and {len(visits)} visits to {DATASET_PATH}.")
    return carers, visits


def compute_travel_time_matrix(
    G: nx.MultiDiGraph, carers: List[Carer], visits: List[Visit]
) -> np.ndarray:
    """Computes a (220 x 220) road-network travel time matrix in minutes."""
    ordered_nodes = [c.home_location.osm_node_id for c in carers] + [
        v.location.osm_node_id for v in visits
    ]
    n_total = len(ordered_nodes)
    unique_nodes = set(ordered_nodes)

    print(f"[Matrix] Computing shortest-path travel times across {n_total} locations...")
    shortest_paths: Dict[int, Dict[int, float]] = {}
    for u in unique_nodes:
        lengths_sec = nx.single_source_dijkstra_path_length(G, u, weight="travel_time")
        shortest_paths[u] = lengths_sec

    matrix_mins = np.zeros((n_total, n_total), dtype=np.float32)
    for i, u in enumerate(ordered_nodes):
        for j, v in enumerate(ordered_nodes):
            if i == j:
                matrix_mins[i, j] = 0.0
            else:
                sec = shortest_paths[u].get(v, 3600.0)
                matrix_mins[i, j] = round((sec / 60.0) + 2.0, 2)

    np.savez_compressed(
        MATRIX_PATH,
        matrix_mins=matrix_mins,
        osm_node_ids=np.array(ordered_nodes, dtype=np.int64),
    )
    print(f"[Matrix] Saved {matrix_mins.shape} matrix to {MATRIX_PATH}.")
    return matrix_mins