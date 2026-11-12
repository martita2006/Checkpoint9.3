"""Unit tests for EUROCONTROL-option B curves; independent of unavailable LIRF CSV."""
import numpy as np
import pandas as pd
from costs import TABLES_EUR2014, _interpolate_eur, cost_components, flight_delay_cost
from ghp import solveGHP

assert TABLES_EUR2014['GATE_PRIMARY']['A320'][3] == 4590
assert TABLES_EUR2014['GATE_FULL']['A320'][3] == 8350
assert TABLES_EUR2014['AIR_PRIMARY']['B738'][3] == 6830
assert TABLES_EUR2014['AIR_FULL']['B738'][3] == 10790
assert _interpolate_eur(TABLES_EUR2014['GATE_PRIMARY']['A320'], 0) == 0
assert _interpolate_eur(TABLES_EUR2014['GATE_PRIMARY']['A320'], 60) == 4590
assert _interpolate_eur(TABLES_EUR2014['GATE_PRIMARY']['A320'], 10) == 270
frame=pd.DataFrame({'reference_aircraft':['A320','B738'], 'is_exempt':[False,True], 'turnaround_min':[70,100]},index=[1,2])
assert abs(cost_components(1,60,frame)['total_eur2014']-7514.4444444)<0.0001
assert cost_components(1,10,frame)['reactionary_eur2014']==0
assert flight_delay_cost(1,60,frame) >= flight_delay_cost(1,30,frame)
arrivals=pd.DataFrame({'ETA_min':[480.,480.], 'Compañía':['AAA','BBB']},index=[1,2])
slots,objective=solveGHP(arrivals,[1,2],[2],[480,485,490],max_air=45,
                        cost_function=lambda fid,d: flight_delay_cost(fid,d,frame))
assert slots is not None and np.isfinite(objective)
print('OK: source knots, interpolation, turnaround costs, monotonicity and MILP integration')
