from typing import Dict, List
import numpy as np

from backend.app.schemas.models import (
    AssignedVisit,
    Carer,
    CarerRoute,
    OptimisationResponse,
    ScheduleKPIs,
    Visit,
)


def solve_naive_baseline(
    carers: List[Carer],
    visits: List[Visit],
    travel_matrix: np.ndarray,
) -> OptimisationResponse:
    """
    Greedy Earliest-Deadline-First heuristic:
    1. Sorts visits by window start time.
    2. For each visit, prioritises carers who have the required skill and highest
       continuity history with the client, ignoring global route geometry.
    3. Checks if the carer can physically reach the client within the time window.
    """
    sorted_visits = sorted(visits, key=lambda v: (v.window_start_min, v.window_end_min))
    n_carers = len(carers)

    # Track operational state per carer
    carer_times: Dict[str, int] = {c.carer_id: c.shift_start_min for c in carers}
    carer_nodes: Dict[str, int] = {c.carer_id: i for i, c in enumerate(carers)}
    carer_assigned: Dict[str, List[AssignedVisit]] = {c.carer_id: [] for c in carers}
    carer_travel_mins: Dict[str, float] = {c.carer_id: 0.0 for c in carers}
    carer_service_mins: Dict[str, float] = {c.carer_id: 0.0 for c in carers}

    unassigned_ids: List[str] = []
    total_continuity_matches = 0

    for v_idx, visit in enumerate(sorted_visits):
        visit_matrix_idx = n_carers + int(visit.visit_id.split("_")[1]) - 1

        # Qualified carers sorted by client familiarity (descending)
        qualified = [c for c in carers if visit.required_skill in c.skills]
        qualified.sort(
            key=lambda c: visit.historical_carer_counts.get(c.carer_id, 0),
            reverse=True,
        )

        assigned = False
        for carer in qualified:
            cid = carer.carer_id
            curr_node = carer_nodes[cid]
            travel_mins = float(travel_matrix[curr_node, visit_matrix_idx])
            arrival_min = int(carer_times[cid] + travel_mins)

            # Wait if arriving before the client's window starts
            service_start = max(arrival_min, visit.window_start_min)
            departure_min = service_start + visit.duration_mins

            # Validate time window and maximum duty hours
            can_serve = (
                arrival_min <= visit.window_end_min
                and departure_min <= carer.shift_end_min
                and (carer_service_mins[cid] + visit.duration_mins) <= carer.max_working_mins
            )

            if can_serve:
                slack = float(visit.window_end_min - arrival_min)
                is_preferred = visit.historical_carer_counts.get(cid, 0) >= 15
                if is_preferred:
                    total_continuity_matches += 1

                carer_assigned[cid].append(
                    AssignedVisit(
                        visit_id=visit.visit_id,
                        client_id=visit.client_id,
                        carer_id=cid,
                        arrival_min=arrival_min,
                        departure_min=departure_min,
                        travel_time_from_prev_mins=travel_mins,
                        slack_mins=slack,
                        is_preferred_carer=is_preferred,
                        late_risk_prob=0.0,
                    )
                )

                carer_times[cid] = departure_min
                carer_nodes[cid] = visit_matrix_idx
                carer_travel_mins[cid] += travel_mins
                carer_service_mins[cid] += visit.duration_mins
                assigned = True
                break

        if not assigned:
            unassigned_ids.append(visit.visit_id)

    # Add return-home travel time for each carer
    routes: List[CarerRoute] = []
    total_travel_all = 0.0
    active_workloads = []

    for idx, carer in enumerate(carers):
        cid = carer.carer_id
        final_node = carer_nodes[cid]
        home_node = idx
        return_travel = float(travel_matrix[final_node, home_node])
        carer_travel_mins[cid] += return_travel

        t_trav = round(carer_travel_mins[cid], 2)
        t_serv = round(carer_service_mins[cid], 2)
        total_travel_all += t_trav
        active_workloads.append(t_trav + t_serv)

        routes.append(
            CarerRoute(
                carer_id=cid,
                carer_name=carer.name,
                assigned_visits=carer_assigned[cid],
                total_travel_mins=t_trav,
                total_service_mins=t_serv,
                continuity_matches=sum(
                    1 for av in carer_assigned[cid] if av.is_preferred_carer
                ),
            )
        )

    assigned_count = len(visits) - len(unassigned_ids)
    continuity_pct = (
        round((total_continuity_matches / assigned_count) * 100, 1)
        if assigned_count > 0
        else 0.0
    )

    kpis = ScheduleKPIs(
        solver_type="naive_baseline",
        total_travel_mins=round(total_travel_all, 1),
        travel_time_reduction_pct=0.0,
        continuity_score_pct=continuity_pct,
        workload_std_dev_mins=round(float(np.std(active_workloads)), 1),
        unassigned_visits_count=len(unassigned_ids),
        high_risk_visits_count=0,
    )

    return OptimisationResponse(
        kpis=kpis,
        routes=routes,
        unassigned_visit_ids=unassigned_ids,
    )