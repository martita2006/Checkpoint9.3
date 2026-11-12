"""WP3 Tasks 1-3: assign aircraft to arrival slots with a binary MILP."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


def solveGHP(arrivals, flights, Exempt, slot_times, max_air=None, limit=None,
             penalty=1, air_weight=1, emission_rates=None, cost_function=None):
    """Minimize delay, CO2, or economic cost with one flight per slot.

    The mathematical model and feasible flight-slot combinations are unchanged.
    """
    if emission_rates is not None and cost_function is not None:
        raise ValueError('Choose either emissions or economic cost objective.')
    if not flights or not len(slot_times):
        raise ValueError('GHP needs affected flights and arrival slots.')

    exempt = set(Exempt)
    pairs, costs = [], []
    for fid in flights:
        eta = arrivals.loc[fid, 'ETA_min']
        for slot_id, arrival_time in enumerate(slot_times):
            delay = arrival_time - eta
            if delay < 0 or (fid in exempt and max_air is not None and delay > max_air):
                continue
            if cost_function is not None:
                cost = float(cost_function(fid, delay))
            elif emission_rates is not None:
                cost = delay * emission_rates[fid]
                # Same penalty as Task 1: ground minutes beyond 'limit' count 'penalty' times
                if fid not in exempt and limit is not None and delay > limit:
                    cost += (penalty - 1) * (delay - limit) * emission_rates[fid]
            else:
                cost = delay * (air_weight if fid in exempt else 1)
                if fid not in exempt and limit is not None and delay > limit:
                    cost += (penalty - 1) * (delay - limit)
            pairs.append((fid, slot_id))
            costs.append(cost)

    if not np.isfinite(costs).all():
        raise ValueError('GHP objective has non-finite coefficients.')
    if not pairs or set(flights) - {fid for fid, _ in pairs}:
        raise ValueError('Some flights have no feasible flight-slot assignment.')

    # Every binary decision x[fid,slot] is represented by one column.
    flight_row = {fid: row for row, fid in enumerate(flights)}
    flights_matrix = lil_matrix((len(flights), len(pairs)))
    slots_matrix = lil_matrix((len(slot_times), len(pairs)))
    for col, (fid, slot_id) in enumerate(pairs):
        flights_matrix[flight_row[fid], col] = 1
        slots_matrix[slot_id, col] = 1

    constraints = [
        LinearConstraint(flights_matrix.tocsr(), 1, 1),
        LinearConstraint(slots_matrix.tocsr(), 0, 1),
    ]
    result = milp(c=np.asarray(costs), constraints=constraints,
                  integrality=np.ones(len(pairs)), bounds=Bounds(0, 1))
    if not result.success:
        print('GHP: no feasible solution found ->', result.message)
        return None, None

    slots = np.array([[time, 0, 0] for time in slot_times], dtype=object)
    for selected, (fid, slot_id) in zip(result.x, pairs):
        if selected > 0.5:
            slots[slot_id, 1] = fid
            slots[slot_id, 2] = arrivals.loc[fid, 'Compañía']
    return slots, result.fun
