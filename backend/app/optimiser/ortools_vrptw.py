from typing import List
import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from backend.app.schemas.models import (
    AssignedVisit,
    Carer,
    CarerRoute,
    OptimisationResponse,
    ScheduleKPIs,
    Visit,
)


def solve_ortools_vrptw(
    carers: List[Carer],
    visits: List[Visit],
    travel_matrix: np.ndarray,
    continuity_weight: float = 1.5,
    time_limit_seconds: int = 5,
) -> OptimisationResponse:
    """
    Multi-objective VRPTW solver balancing:
      1. Total road-network travel time
      2. Client-carer continuity of care (tiered primary vs backup carer reward)
      3. Carer workload equity (via DutyTime soft upper bounds)
    """
    n_carers = len(carers)
    n_visits = len(visits)
    total_locations = n_carers + n_visits

    starts = list(range(n_carers))
    ends = list(range(n_carers))

    manager = pywrapcp.RoutingIndexManager(total_locations, n_carers, starts, ends)
    routing = pywrapcp.RoutingModel(manager)

    # 1. Vehicle-Specific Arc Cost Callbacks (Travel Time - Tiered Continuity Bonus)
    for v_idx, carer in enumerate(carers):
        def vehicle_cost_callback(
            from_index: int, to_index: int, c_id: str = carer.carer_id
        ) -> int:
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            # Base cost in tenths of a minute + 500 fixed offset so arc costs stay positive
            base_cost = int(travel_matrix[from_node, to_node] * 10) + 500

            if to_node >= n_carers:
                visit = visits[to_node - n_carers]
                history_count = visit.historical_carer_counts.get(c_id, 0)
                if history_count >= 15:
                    # Primary regular carer tier
                    bonus = int(min(continuity_weight * 120, 480))
                elif history_count > 0:
                    # Secondary backup carer tier
                    bonus = int(min(continuity_weight * 30, 150))
                else:
                    bonus = 0
                return max(base_cost - bonus, 10)
            return base_cost

        cost_cb_idx = routing.RegisterTransitCallback(vehicle_cost_callback)
        routing.SetArcCostEvaluatorOfVehicle(cost_cb_idx, v_idx)

    # 2. Time-of-Day Dimension (includes waiting slack for hard time windows)
    def transit_time_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel = int(round(travel_matrix[from_node, to_node]))
        service = (
            visits[from_node - n_carers].duration_mins if from_node >= n_carers else 0
        )
        return travel + service

    transit_cb_idx = routing.RegisterTransitCallback(transit_time_callback)
    routing.AddDimension(
        transit_cb_idx,
        1440,
        1440,
        False,
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    # 3. Active Duty Workload Dimension (tracks travel + service work)
    routing.AddDimension(
        transit_cb_idx,
        0,
        600,  # allow up to 10h ceiling so no visits get dropped, penalise above 400m
        True,
        "DutyTime",
    )
    duty_dimension = routing.GetDimensionOrDie("DutyTime")

    # Soft upper bound at 400 mins per carer to compress workload variance
    for v_idx in range(n_carers):
        end_idx = routing.End(v_idx)
        duty_dimension.SetCumulVarSoftUpperBound(end_idx, 400, 8)

    # 4. Visit Time Windows & High Disjunction Penalty (prioritise 100% visit coverage)
    disjunction_penalty = 1_000_000
    for j, visit in enumerate(visits):
        node = n_carers + j
        index = manager.NodeToIndex(node)
        time_dimension.CumulVar(index).SetRange(
            visit.window_start_min, visit.window_end_min
        )
        routing.AddDisjunction([index], disjunction_penalty)

    # 5. Carer Shift Windows
    for v_idx, carer in enumerate(carers):
        start_idx = routing.Start(v_idx)
        end_idx = routing.End(v_idx)
        time_dimension.CumulVar(start_idx).SetRange(
            carer.shift_start_min, carer.shift_end_min
        )
        time_dimension.CumulVar(end_idx).SetRange(
            carer.shift_start_min, carer.shift_end_min
        )

    # 6. Hard Skill Constraints
    for j, visit in enumerate(visits):
        node = n_carers + j
        index = manager.NodeToIndex(node)
        allowed_vehicles = [
            v_idx
            for v_idx, carer in enumerate(carers)
            if visit.required_skill in carer.skills
        ]
        routing.SetAllowedVehiclesForIndex(allowed_vehicles, index)

    # 7. Solve with Guided Local Search
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = time_limit_seconds

    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        raise RuntimeError("OR-Tools failed to find a feasible solution.")

    # 8. Extract Routes & KPIs
    routes: List[CarerRoute] = []
    unassigned_ids: List[str] = []
    total_travel_all = 0.0
    total_continuity_matches = 0
    active_workloads = []

    for j, visit in enumerate(visits):
        node = n_carers + j
        index = manager.NodeToIndex(node)
        if solution.Value(routing.NextVar(index)) == index:
            unassigned_ids.append(visit.visit_id)

    for v_idx, carer in enumerate(carers):
        cid = carer.carer_id
        index = routing.Start(v_idx)
        carer_assigned: List[AssignedVisit] = []
        carer_travel = 0.0
        carer_service = 0.0
        prev_node = v_idx

        while not routing.IsEnd(index):
            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)

            if next_node >= n_carers:
                visit = visits[next_node - n_carers]
                leg_travel = float(travel_matrix[prev_node, next_node])
                arrival_min = int(solution.Min(time_dimension.CumulVar(next_index)))
                service_start = max(arrival_min, visit.window_start_min)
                departure_min = service_start + visit.duration_mins
                slack = float(visit.window_end_min - arrival_min)

                is_preferred = visit.historical_carer_counts.get(cid, 0) >= 15
                if is_preferred:
                    total_continuity_matches += 1

                carer_assigned.append(
                    AssignedVisit(
                        visit_id=visit.visit_id,
                        client_id=visit.client_id,
                        carer_id=cid,
                        arrival_min=arrival_min,
                        departure_min=departure_min,
                        travel_time_from_prev_mins=leg_travel,
                        slack_mins=slack,
                        is_preferred_carer=is_preferred,
                        late_risk_prob=0.0,
                    )
                )

                carer_travel += leg_travel
                carer_service += visit.duration_mins
                prev_node = next_node

            index = next_index

        return_travel = float(travel_matrix[prev_node, v_idx])
        carer_travel += return_travel

        t_trav = round(carer_travel, 2)
        t_serv = round(carer_service, 2)
        total_travel_all += t_trav
        active_workloads.append(t_trav + t_serv)

        routes.append(
            CarerRoute(
                carer_id=cid,
                carer_name=carer.name,
                assigned_visits=carer_assigned,
                total_travel_mins=t_trav,
                total_service_mins=t_serv,
                continuity_matches=sum(
                    1 for av in carer_assigned if av.is_preferred_carer
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
        solver_type="ortools_vrptw",
        total_travel_mins=round(total_travel_all, 1),
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