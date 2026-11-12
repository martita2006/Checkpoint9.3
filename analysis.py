import numpy as np
import pandas as pd
from emissions import co2DelayRates
from costs import costs_for_assignment

def evaluate_assignment(name, arrivals, cta, flights, exempt, cost_data, rates=None):
    flights = list(flights)
    exempt = set(exempt)
    rates = rates if rates is not None else co2DelayRates(arrivals, exempt, flights)
    frame = pd.DataFrame(index=flights)
    frame['ARCID'] = arrivals.loc[flights, 'ARCID']
    frame['Airline'] = arrivals.loc[flights, 'Compañía'].astype(str)
    frame['Status'] = ['Exempt' if f in exempt else 'Controlled' for f in flights]
    frame['ETA_min'] = arrivals.loc[flights, 'ETA_min']
    frame['CTA_min'] = cta.loc[flights].astype(float)
    frame['Delay_min'] = (frame.CTA_min - frame.ETA_min).clip(lower=0)
    if ((frame.CTA_min - frame.ETA_min) < -1e-6).any():
        raise ValueError(f'{name}: some aircraft arrive before ETA')
    frame['AirDelay_min'] = np.where(frame.Status == 'Exempt', frame.Delay_min, 0.0)
    frame['GroundDelay_min'] = np.where(frame.Status == 'Controlled', frame.Delay_min, 0.0)
    frame['CO2_kg'] = [frame.loc[f, 'Delay_min'] * rates[f] for f in flights]
    costs = costs_for_assignment(arrivals, cta, flights, cost_data)
    frame['EconomicCost_EUR'] = [costs[f] for f in flights]
    delay = frame.Delay_min.to_numpy()
    kpis = {
        'Scenario': name, 'Flights': len(frame), 'Delayed flights': int((delay > 1e-7).sum()),
        'Total delay min': float(delay.sum()), 'Mean delay min': float(delay.mean()),
        'Max delay min': float(delay.max()), 'Delay std min': float(delay.std()),
        'Delay RSD %': float(100*delay.std()/delay.mean()) if delay.mean() else 0.0,
        'Air delay min': float(frame.AirDelay_min.sum()),
        'Max air delay min': float(frame.AirDelay_min.max()),
        'Ground delay min': float(frame.GroundDelay_min.sum()),
        'Max ground delay min': float(frame.GroundDelay_min.max()),
        'CO2 kg': float(frame.CO2_kg.sum()),
        'Economic cost EUR': float(frame.EconomicCost_EUR.sum()),
        'Cost / flight EUR': float(frame.EconomicCost_EUR.mean()),
        'Max individual cost EUR': float(frame.EconomicCost_EUR.max()),
    }
    return frame, kpis

def airline_comparison(scenarios, top_n=4):
    """Use same top 3-4 carriers from original affected-flight demand for all scenarios."""
    baseline = next(iter(scenarios.values()))
    top_airlines = baseline['Airline'].value_counts().head(top_n).index
    rows = []
    for name, frame in scenarios.items():
        for airline in top_airlines:
            group = frame.loc[frame['Airline'] == airline]
            delays = group.Delay_min.to_numpy(dtype=float)
            mean = delays.mean() if len(delays) else 0.0
            rows.append({'Scenario': name, 'Airline': airline, 'Flights': len(delays),
                         'Mean delay min': mean, 'Delay RSD %': 100*delays.std()/mean if mean else 0,
                         'Total delay min': delays.sum(), 'CO2 kg': group.CO2_kg.sum(),
                         'Economic cost EUR': group.EconomicCost_EUR.sum()})
    return pd.DataFrame(rows)

def validate_assignment(name, slots, flights, arrivals, max_air_delay, exempt):
    used = [int(row[1]) for row in slots if row[1] != 0]
    if sorted(used) != sorted(flights) or len(used) != len(set(used)):
        raise ValueError(f'{name}: missing/duplicate flight assignment')
    for time, fid, airline in slots:
        if fid == 0:
            continue
        delay = float(time) - float(arrivals.loc[fid, 'ETA_min'])
        if delay < -1e-6:
            raise ValueError(f'{name}: flight {fid} assigned before ETA')
        if fid in exempt and delay > max_air_delay + 1e-6:
            raise ValueError(f'{name}: exempt flight {fid} exceeds maximum air delay')
