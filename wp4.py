"""WP4: substituir per tren els vols de rutes nacionals curtes i tornar a fer GDP i GHP.

Suposicions (l'enunciat no les dona):
  - Rutes de rutas_tren.csv: tren <= 6 h i <= 2 transbords (temps fins a Fiumicino Aeroporto).
  - Passatgers = seients x load_factor (el mateix del WP3). Seients < 30 al CSV -> error de dades, es posa 150.
  - D2D amb trams de l'exemple d'ALG (Londres-Paris). Temps de vol estimat amb la distancia
    (30 min + distancia a 750 km/h), perque els ETD/ETA del CSV donen temps irreals per a vols curts.
  - CO2 per passatger de l'avio i del tren: EcoPassenger. Amb retard s'hi suma el CO2 del retard del GDP.
  - El tren no te retard i la xarxa ferroviaria te capacitat per a tots els passatgers.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import (filename, Hstart, Hend,max_air_delay, ground_limit,
                    co2_ground_penalty, load_factor)
from gdp import computeHNoReg, computeSlots, determineFlightStatus, assignSlotsGDP, delaysFromSlots
from emissions import co2DelayRates
from ghp import solveGHP
from costs import prepare_cost_data, flight_delay_cost
from analysis import evaluate_assignment, airline_comparison

RAIL_FILE = Path(__file__).resolve().parent / 'rutas_tren.csv'
ROUTES = {  # ciutat: (ICAO, distancia ortodromica a LIRF en km)
    'Napoli': ('LIRN', 199), 'Ancona': ('LIPY', 222), 'Firenze': ('LIRQ', 239), 'Bologna': ('LIPE', 314),
    'Bari': ('LIBD', 384), 'Genova': ('LIMJ', 401), 'Venezia': ('LIPZ', 412),
    'Lamezia Terme': ('LICA', 468), 'Milano Linate': ('LIML', 470),
    'Torino': ('LIMF', 529)}
AIR_EXTRA_MIN = 30 + 75 + 15   # acces a l'aeroport + facturacio/seguretat/embarcament + sortida
RAIL_EXTRA_MIN = 20 + 20 + 10  # acces a l'estacio + espera/embarcament + estacio -> terminal
AIR_EXTRA_CITY = 30 + 75 + 15 + 30   # com abans + trajecte de Fiumicino a Roma
RAIL_EXTRA_CITY = 20 + 20 + 20       # acces a l'estacio + espera + de Termini al desti

def run_scenario(arrivals,Hfile_opt,radius_opt):
    """GDP, GHP CO2 i GHP cost amb les mateixes hipotesis que el WP2/WP3."""
    t, demand, capacity, HNoReg, _ = computeHNoReg(arrivals['ETA_min'].values)
    status = determineFlightStatus(arrivals, Hfile_opt, HNoReg, radius_opt)
    Exempt, Controlled = status[4], status[5]
    affected = sorted(Exempt + Controlled)
    slot_times = computeSlots(HNoReg)[:, 0].astype(float)
    rates = co2DelayRates(arrivals, Exempt, affected)
    cost_data = prepare_cost_data(arrivals, filename, exempt_ids=Exempt)

    _, _, _, cta_gdp = assignSlotsGDP(computeSlots(HNoReg), Controlled, Exempt, arrivals)
    slots_co2, _ = solveGHP(arrivals, affected, Exempt, slot_times, max_air=max_air_delay,
                            limit=ground_limit, penalty=co2_ground_penalty, emission_rates=rates)
    slots_cost, _ = solveGHP(arrivals, affected, Exempt, slot_times, max_air=max_air_delay,
                             cost_function=lambda fid, d: flight_delay_cost(fid, d, cost_data))
    ctas = {'GDP': cta_gdp,
            'GHP CO2': delaysFromSlots(slots_co2, Controlled, arrivals)[2],
            'GHP cost': delaysFromSlots(slots_cost, Controlled, arrivals)[2]}
    frames, kpis = {}, {}
    for name, cta in ctas.items():
        frames[name], k = evaluate_assignment(name, arrivals, cta, affected, set(Exempt), cost_data, rates)
        kpis[name] = k
    return {'HNoReg': HNoReg, 'affected': len(affected), 'frames': frames,
            'kpis': pd.DataFrame(kpis).T, 'curves': (t, demand, capacity)}


def air_vs_rail(arrivals, subs, gdp_frame):
    """Per a cada ruta: CO2 total i D2D de l'avio (sense i amb el retard del GDP) i del tren,
    per al passatger en connexio (desti Fiumicino) i per al passatger amb desti Roma."""
    rail = pd.read_csv(RAIL_FILE, sep=';', decimal=',').dropna(subset=['Ciudad'])
    rail['Ciudad'] = rail['Ciudad'].str.strip()
    rail = rail.set_index(rail['Ciudad'].map(lambda c: ROUTES[c][0]))
    rows = []
    for fid in subs:
        r = rail.loc[arrivals.loc[fid, 'ADEP']]
        seats = arrivals.loc[fid, 'Seats']
        pax = (150 if pd.isna(seats) or seats < 30 else seats) * load_factor
        flight_min = 30 + 60 * ROUTES[r['Ciudad']][1] / 750
        delay = gdp_frame.loc[fid, 'Delay_min']
        rows.append({'Route': r['Ciudad'], 'Pax': pax,
                     'CO2 air': r['CO2_avion_EcoPassenger_kg_pax'] * pax,
                     'CO2 air + delay': r['CO2_avion_EcoPassenger_kg_pax'] * pax + gdp_frame.loc[fid, 'CO2_kg'],
                     'CO2 rail': r['CO2_tren_kg_pax'] * pax,
                     'D2D air': AIR_EXTRA_MIN + flight_min,
                     'D2D air + delay': AIR_EXTRA_MIN + flight_min + delay,
                     'D2D rail': RAIL_EXTRA_MIN + r['Tiempo a LIRF'],
                     'D2D air (Roma)': AIR_EXTRA_CITY + flight_min,
                     'D2D air + delay (Roma)': AIR_EXTRA_CITY + flight_min + delay,
                     'D2D rail (Roma)': RAIL_EXTRA_CITY + r['Tiempo a Roma']})
    df = pd.DataFrame(rows)
    d2d = ['D2D air', 'D2D air + delay', 'D2D rail',
           'D2D air (Roma)', 'D2D air + delay (Roma)', 'D2D rail (Roma)']
    table = df.groupby('Route').agg(Flights=('Pax', 'size'), Pax=('Pax', 'sum'),
                                    **{c: (c, 'sum') for c in ['CO2 air', 'CO2 air + delay', 'CO2 rail']},
                                    **{c: (c, 'mean') for c in d2d})
    table.loc['TOTAL'] = df[['Pax', 'CO2 air', 'CO2 air + delay', 'CO2 rail']].sum()
    table.loc['TOTAL', 'Flights'] = len(df)
    return table.sort_values('Flights')


def run_wp4(arrivals, Hfile_opt, radius_opt):
    base = run_scenario(arrivals, Hfile_opt, radius_opt)
    rail_icao = [icao for icao, _ in ROUTES.values()]
    in_window = (arrivals['ETA_min'] >= Hstart * 60) & (arrivals['ETA_min'] <= base['HNoReg'])
    subs = list(arrivals.index[in_window & arrivals['ADEP'].isin(rail_icao)])

    print('\nWP4: SUBSTITUCIO DE VOLS PER TREN')
    print(f'Vols substituits (ETA entre {Hstart:02d}:00 i {base["HNoReg"] // 60:02d}:{base["HNoReg"] % 60:02d}): '
          f'{len(subs)} -> {list(arrivals.loc[subs, "ARCID"])}')
    table = air_vs_rail(arrivals, subs, base['frames']['GDP'])
    print('\nAvio vs tren (CO2 total en kg, D2D mitja en min, retard del GDP):')
    print(table.round(0).to_string())

    new = run_scenario(arrivals.drop(index=subs), Hfile_opt, radius_opt)
    cols = [c for c in base['kpis'].columns if c != 'Scenario']
    print('\nRegulacio abans -> despres de treure els vols:')
    print(f'  HNoReg: {base["HNoReg"] // 60:02d}:{base["HNoReg"] % 60:02d} -> '
          f'{new["HNoReg"] // 60:02d}:{new["HNoReg"] % 60:02d}')
    print(f'  Vols afectats: {base["affected"]} -> {new["affected"]}')
    kpis = pd.concat({'Before': base['kpis'][cols], 'After': new['kpis'][cols]})
    kpis = kpis.swaplevel().loc[list(base['kpis'].index)]
    print(kpis.astype(float).round(0).T.to_string())
    for label, res in [('abans', base), ('despres', new)]:
        print(f'\nAerolinies principals ({label} de treure els vols):')
        print(airline_comparison(res['frames'], top_n=4).round(2).to_string(index=False))

    # Figura 1: CO2 i D2D per ruta
    t = table.drop(index='TOTAL')
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    t[['CO2 air', 'CO2 air + delay', 'CO2 rail']].plot(kind='bar', ax=ax1)
    ax1.set_ylabel('CO2 (kg)'); ax1.set_title('Total CO2 per route')
    t[['D2D air', 'D2D air + delay', 'D2D rail',
       'D2D air (Roma)', 'D2D air + delay (Roma)', 'D2D rail (Roma)']].plot(kind='bar', ax=ax2)
    ax2.set_ylabel('Door-to-door time (min)'); ax2.set_title('Mean D2D time per route')
    for ax in (ax1, ax2):
        ax.set_xlabel(''); ax.tick_params(axis='x', rotation=30); ax.grid(axis='y', alpha=0.3)
    fig1.tight_layout()

    # Figura 2: demanda i capacitat abans i despres
    fig2 = plt.figure(figsize=(10, 5))
    for res, style, label in [(base, '-', 'before'), (new, '--', 'after')]:
        time, demand, capacity = res['curves']
        plt.plot(time / 60, demand, 'g' + style, label=f'Demand {label}')
        plt.plot(time / 60, np.minimum(capacity, demand), 'r' + style, label=f'Capacity {label}')
        plt.axvline(res['HNoReg'] / 60, color='b', linestyle=style, label=f'HNoReg {label}')
    plt.xlim(Hstart - 1, base['HNoReg'] / 60 + 1)
    plt.xlabel('Time [h]'); plt.ylabel('Cumulative number of aircraft')
    plt.title('Aggregate demand and capacity before/after rail substitution')
    plt.grid(True); plt.legend(loc='upper left')
    return {'WP4: avio vs tren': fig1, 'WP4: demanda/capacitat': fig2}