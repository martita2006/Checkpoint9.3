"""Synthetic regression: unchanged objective/feasibility against pre-refactor solver."""
import io
import zipfile
import types
from pathlib import Path
import numpy as np
import pandas as pd
from ghp import solveGHP
from costs import _interpolate_eur, TABLES_EUR2014, cost_components

archive = Path(__file__).parent.parent.parent / 'LIRF_WP3_Option_B_EUROCONTROL(1).zip'
if archive.exists():
    with zipfile.ZipFile(archive) as z:
        baseline = types.ModuleType('baseline_ghp')
        exec(compile(z.read('LIRF_modular_checkpoint7/ghp.py'), 'baseline_ghp.py', 'exec'), baseline.__dict__)
        for seed in range(6):
            rng = np.random.default_rng(seed)
            flights = [1,2,3,4]
            arrivals = pd.DataFrame({'ETA_min': [400.,401.,402.,403.], 'Compañía':['AA','BB','CC','DD']}, index=flights)
            times = np.array([403.,407.,411.,415.,419.,423.])
            rates = {f:float(rng.uniform(1,10)) for f in flights}
            scenarios = [{}, {'max_air':20,'limit':8,'penalty':2.3,'air_weight':1.7},
                         {'max_air':20,'emission_rates':rates},
                         {'max_air':20,'cost_function':lambda fid,d: (d**2)*rates[fid]+fid*0.01}]
            for opts in scenarios:
                old_slots, old_obj = baseline.solveGHP(arrivals, flights,[2,4],times,**opts)
                new_slots, new_obj = solveGHP(arrivals, flights,[2,4],times,**opts)
                assert old_slots is not None and new_slots is not None
                assert np.isclose(old_obj,new_obj,atol=1e-6), (seed,opts,old_obj,new_obj)
        print('PASS: 24 baseline-vs-simplified synthetic MILP scenarios')
else:
    print('Original zip not available: skipping baseline comparison')
