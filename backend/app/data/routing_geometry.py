from typing import Dict, List, Tuple
import networkx as nx

from backend.app.schemas.models import Carer, OptimisationResponse, Visit

# Persistent in-memory cache of shortest-path street polylines between OSM node pairs
_GLOBAL_PATH_CACHE: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}


def attach_route_geometries(
    G: nx.MultiDiGraph,
    schedule: OptimisationResponse,
    carers: List[Carer],
    visits: List[Visit],
) -> OptimisationResponse:
    """
    Populates `route_geometry` on each AssignedVisit with the exact sequence of
    (lat, lon) street intersection coordinates along the shortest travel-time path.
    """
    carer_map: Dict[str, Carer] = {c.carer_id: c for c in carers}
    visit_map: Dict[str, Visit] = {v.visit_id: v for v in visits}

    for route in schedule.routes:
        carer = carer_map[route.carer_id]
        prev_osm_node = carer.home_location.osm_node_id

        for av in route.assigned_visits:
            visit = visit_map[av.visit_id]
            target_osm_node = visit.location.osm_node_id
            pair = (prev_osm_node, target_osm_node)

            if pair not in _GLOBAL_PATH_CACHE:
                if prev_osm_node == target_osm_node:
                    coords = [(visit.location.lat, visit.location.lon)]
                else:
                    try:
                        node_path = nx.shortest_path(
                            G, prev_osm_node, target_osm_node, weight="travel_time"
                        )
                        coords = [
                            (float(G.nodes[n]["y"]), float(G.nodes[n]["x"]))
                            for n in node_path
                        ]
                    except nx.NetworkXNoPath:
                        prev_lat = float(G.nodes[prev_osm_node]["y"])
                        prev_lon = float(G.nodes[prev_osm_node]["x"])
                        coords = [
                            (prev_lat, prev_lon),
                            (visit.location.lat, visit.location.lon),
                        ]
                _GLOBAL_PATH_CACHE[pair] = coords

            av.route_geometry = _GLOBAL_PATH_CACHE[pair]
            prev_osm_node = target_osm_node

    return schedule