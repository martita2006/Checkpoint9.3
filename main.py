import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import (filename, airport, AAR, PAAR, Hstart, Hend, Hfile,
                    radius as initial_radius,cancel_airline,
                    cancel_delay, max_air_delay, ground_limit, ground_penalty,
                    air_weight, results_folder, co2_ground_penalty)
from data import readArrivals
from gdp import (computeHNoReg, computeSlots, determineFlightStatus,
                 assignSlotsGDP, delayStatistics, unrecoverableDelay,
                 gdpKPIs, cancelFlights, fillEmptySlots, delaysFromSlots, printKPIs)
from emissions import airEmissions, groundEmissions, co2DelayRates
from ghp import solveGHP
from costs import prepare_cost_data, flight_delay_cost, cost_components
from analysis import evaluate_assignment, airline_comparison, validate_assignment
from plots import plotHistogram, choose_figures
from wp4 import run_wp4

np.set_printoptions(threshold=np.inf, suppress=True)

def main(plot_choice=None):
    plt.ioff()
    # Checkpoints 1 and 2: demand, aggregate curves and HNoReg
    arrivals = readArrivals(filename)
    plotHistogram(arrivals['ETA_min'] / 60, f'Arrival demand at {airport} (initial)')
    fig_initial = plt.gcf()

    t, demand, capacity, HNoReg, total_delay = computeHNoReg(arrivals['ETA_min'].values)
    affected = np.sum((arrivals['ETA_min'] >= Hstart * 60) & (arrivals['ETA_min'] <= HNoReg))
    print(f'HNoReg: {HNoReg // 60:02d}:{HNoReg % 60:02d} h')
    print(f'Theoretical total delay: {total_delay:.2f} min')
    print(f'Affected flights: {affected}')
    print(f'Theoretical average delay: {(total_delay / affected if affected else 0.0):.2f} min/flight' if affected else 'No affected flights: cannot compute average delay')

    plt.figure()
    fig_aggregate = plt.gcf()
    plt.plot(t / 60, demand, 'g-', linewidth=2, label='Aggregate demand')
    plt.plot(t / 60, np.minimum(capacity, demand), 'r:', linewidth=2, label='Aggregate capacity')
    for x, c, name in [(Hfile, 'gray', 'Hfile'), (Hstart, 'orange', 'Hstart'),
                       (Hend, 'purple', 'Hend'), (HNoReg / 60, 'blue', 'HNoReg')]:
        plt.axvline(x=x, color=c, linestyle='--', label=name)
    plt.xlabel('Time [h]')
    plt.ylabel('Cumulative number of aircraft')
    plt.title('Aggregate demand and capacity')
    plt.xlim(0, 24)
    plt.grid(True)
    plt.legend(loc='upper left')

    # Checkpoint 3: GDP with the initial Hfile and radius
    NotAffected, ExemptRadius, ExemptInternational, ExemptFlying, Exempt, Controlled = \
        determineFlightStatus(arrivals, Hfile, HNoReg, initial_radius)
    slots, GroundDelay, AirDelay, CTA = assignSlotsGDP(computeSlots(HNoReg), Controlled, Exempt, arrivals)
    arrivals['CTA_min'] = CTA

    print(f'\nNot affected flights: {len(NotAffected)}')
    print(f'Controlled flights: {len(Controlled)}')
    print(f'Exempt flights: {len(Exempt)} (radius {len(ExemptRadius)}, non-ECAC '
          f'{len(ExemptInternational)}, airborne {len(ExemptFlying)})')
    for name, delays in [('Ground', GroundDelay), ('Air', AirDelay)]:
        total, maximum, mean, rel_std = delayStatistics(delays)
        print(f'{name} delay: total {total:.2f} min, max {maximum:.2f} min, '
              f'mean {mean:.2f} min, relative std {rel_std:.1f} %')
    print(f'Total delay (ground + air): {delayStatistics(GroundDelay)[0] + delayStatistics(AirDelay)[0]:.2f} min')
    print('\nGroundDelay [ARCID, min] =')
    print(np.array([[arrivals.loc[id, 'ARCID'], delay] for id, delay in GroundDelay], dtype=object))
    print('\nslots [time, ARCID, airline] =')
    print(np.array([[s[0], arrivals.loc[s[1], 'ARCID'] if s[1] != 0 else 0, s[2]] for s in slots], dtype=object))

    plotHistogram(arrivals['CTA_min'] / 60, 'Arrivals after the GDP (regulated traffic)')
    fig_gdp = plt.gcf()

    unrec = unrecoverableDelay(arrivals, GroundDelay)
    total_unrec = sum(u for _, u in unrec)
    print(f'\nUnrecoverable ground delay: {total_unrec:.2f} min')
    print(f'Recoverable ground delay: {delayStatistics(GroundDelay)[0] - total_unrec:.2f} min')

    fuel_air, co2_air, nox_air, sox_air, h2o_air, co_air = airEmissions(
        arrivals, AirDelay
    )
    fuel_ground, co2_ground, nox_ground, sox_ground, h2o_ground, co_ground = groundEmissions(arrivals, GroundDelay)
    non_co2_air = nox_air + sox_air + h2o_air + co_air
    non_co2_ground = nox_ground + sox_ground + h2o_ground + co_ground
    print(f'\nAIR DELAY EMISSIONS:')
    print(f'Fuel: {fuel_air:.2f} kg')
    print(f'CO2: {co2_air:.2f} kg')
    print(f'NOx: {nox_air:.2f} kg')
    print(f'SOx: {sox_air:.2f} kg')
    print(f'H2O (Water Vapour): {h2o_air:.2f} kg')
    print(f'CO: {co_air:.2f} kg')
    print(f'GROUND DELAY EMISSIONS:')
    print(f'Fuel: {fuel_ground:.2f} kg')
    print(f'CO2: {co2_ground:.2f} kg')
    print(f'NOx: {nox_ground:.2f} kg')
    print(f'SOx: {sox_ground:.2f} kg')
    print(f'H2O (Water Vapour): {h2o_ground:.2f} kg')
    print(f'CO: {co_ground:.2f} kg')
    print(f'TOTALS:')
    print(f'Total CO2 due to delay (Air + Ground): {co2_air + co2_ground:.2f} kg')
    print(f'Total Non-CO2 due to air delay: {non_co2_air:.2f} kg')
    print(f'Total Non-CO2 due to ground delay: {non_co2_ground:.2f} kg')
    print(f'Total Non-CO2 overall: {non_co2_air + non_co2_ground:.2f} kg')

    # Checkpoint 4: effect of Hfile and radius, heatmaps and optimal values
    hfile_range = np.arange(4, 8.5, 0.5)
    radius_range = np.arange(400, 2200, 200)
    names = ['Air delay (min)', 'Ground delay (min)', 'Unrecoverable delay (min)',
             'CO2 air delay (kg)', 'CO2 ground delay (kg)', 'CO2 total (kg)']
    weights = [1, 0, 1, 0, 0, 1]
    maps = np.zeros((len(names), len(hfile_range), len(radius_range)))
    for i, hfile in enumerate(hfile_range):
        for j, radius_value in enumerate(radius_range):
            maps[:, i, j] = gdpKPIs(arrivals, hfile, radius_value, HNoReg)

    score = np.zeros((len(hfile_range), len(radius_range)))
    for k in range(len(names)):
        kpi_map = maps[k]
        weight = weights[k]
        name = names[k]
        if weight == 0:
            continue
        avg = np.ptp(kpi_map)
        if avg > 0:
            kpi_normalized = (kpi_map - kpi_map.min()) / avg
            score += weight * kpi_normalized

    i_opt, j_opt = np.unravel_index(np.argmin(score), score.shape)
    optimal_hfile = hfile_range[i_opt]
    optimal_radius = radius_range[j_opt]
    Hfile_opt, radius_opt = float(optimal_hfile), int(optimal_radius)

    print(f'Optimal values: Hfile = {optimal_hfile:.1f} h, radius = {optimal_radius} km')
    for k in range(len(names)):
        print(f'  {names[k]}: {maps[k, i_opt, j_opt]:.2f}')

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig_heatmaps = fig
    x_labels = radius_range
    y_labels = [f'{h:.1f}' for h in hfile_range]

    for k, (ax, name) in enumerate(zip(axes.flat, names)):
        kpi_map = maps[k]

        sns.heatmap(
            kpi_map,
            ax=ax,
            cmap='Blues',
            annot=True,
            fmt='.0f',
            annot_kws={'size': 7},
            xticklabels=x_labels,
            yticklabels=y_labels
        )

        optimal_box = plt.Rectangle((j_opt, i_opt), 1, 1, fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(optimal_box)

        ax.set_title(name)
        ax.set_xlabel('Radius (km)')
        ax.set_ylabel('Hfile (h)')

    plt.tight_layout()
    # Checkpoint 5: one airline cancels its most delayed flights, then substitution and compression
    NotAffected5, ExemptRadius5, ExemptInternational5, ExemptFlying5, Exempt5, Controlled5 = \
        determineFlightStatus(arrivals, Hfile_opt, HNoReg, radius_opt)
    slots5, GroundDelay5, AirDelay5, CTA5 = assignSlotsGDP(computeSlots(HNoReg), Controlled5, Exempt5, arrivals)
    slots_before = slots5.copy()
    ground_before = {id: delay for id, delay in GroundDelay5}

    # 1. Cancellation, 2. substitution (same airline), 3. compression (all airlines)
    slots5, cancelled = cancelFlights(slots5, GroundDelay5, arrivals, cancel_airline, cancel_delay)
    movable = set(Controlled5) - set(cancelled)
    slots5, n_substituted = fillEmptySlots(slots5, movable, arrivals, cancel_airline)
    slots5, n_compressed = fillEmptySlots(slots5, movable, arrivals)
    GroundDelayNew, AirDelayNew, CTA_new = delaysFromSlots(slots5, Controlled5, arrivals)
    ground_after = {id: delay for id, delay in GroundDelayNew}

    print(f'\nCHECKPOINT 5: Hfile = {Hfile_opt} h, radius = {radius_opt} km')
    print(f'Hypothesis: {cancel_airline} cancels its controlled flights with more than {cancel_delay} min of ground delay')
    print(f'{cancel_airline} controlled flights: {sum(arrivals.loc[id, "Compañía"] == cancel_airline for id in Controlled5)}'
          f' of {len(Controlled5)}')
    print(f'Cancelled flights: {len(cancelled)} -> {list(arrivals.loc[cancelled, "ARCID"])}')
    print(f'Seats cancelled: {arrivals.loc[cancelled, "Available seats"].sum():.0f}')
    print(f'Flights moved by substitution: {n_substituted}, by compression: {n_compressed}')

    # Ground delay before and after, for each group of flights
    airline_ids = [id for id in ground_before if arrivals.loc[id, 'Compañía'] == cancel_airline and id not in cancelled]
    other_ids = [id for id in ground_before if arrivals.loc[id, 'Compañía'] != cancel_airline]
    groups = [('Cancelled flights', cancelled), (f'{cancel_airline} flights not cancelled', airline_ids),
              ('Other airlines', other_ids), ('All controlled flights', list(ground_before))]
    for name, ids in groups:
        total_b, max_b, mean_b, _ = delayStatistics([[id, ground_before[id]] for id in ids])
        total_a, max_a, mean_a, _ = delayStatistics([[id, ground_after[id]] for id in ids if id in ground_after])
        print(f'{name} ({len(ids)}): ground delay total {total_b:.0f} -> {total_a:.0f} min, '
              f'mean {mean_b:.1f} -> {mean_a:.1f} min, max {max_b:.0f} -> {max_a:.0f} min')

    delays_b = np.array(list(ground_before.values()), dtype=float)
    delays_a = np.array(list(ground_after.values()), dtype=float)
    print(f'Flights with ground delay: {np.sum(delays_b > 0)} -> {np.sum(delays_a > 0)}')
    print(f'OTP (less than 15 min of ground delay): {np.sum(delays_b < 15)} -> {np.sum(delays_a < 15)}')
    print(f'Air delay: {delayStatistics(AirDelay5)[0]:.0f} -> {delayStatistics(AirDelayNew)[0]:.0f} min')
    print(f'CO2 ground delay: {groundEmissions(arrivals, GroundDelay5)[1]:.0f} -> '
          f'{groundEmissions(arrivals, GroundDelayNew)[1]:.0f} kg')
    regulated = (slots5[:, 0].astype(float) >= Hstart * 60) & (slots5[:, 0].astype(float) <= HNoReg)
    print(f'Empty slots between Hstart and HNoReg after compression: {np.sum(slots5[regulated, 1] == 0)}')

    print('\nSlots that changed [time, before, after]:')
    for k in range(len(slots5)):
        if slots5[k, 1] != slots_before[k, 1]:
            before = arrivals.loc[slots_before[k, 1], 'ARCID'] if slots_before[k, 1] != 0 else '-'
            after = arrivals.loc[slots5[k, 1], 'ARCID'] if slots5[k, 1] != 0 else '-'
            print(f'  {slots5[k, 0]:7.2f}  {before:>8s} -> {after}')

    # New assignment in a graphical way: ground delay of each flight before and after
    ids = sorted(ground_before)
    airline_after = [id for id in airline_ids if id in ground_after]
    other_after = [id for id in other_ids if id in ground_after]
    plt.figure(figsize=(12, 5))
    fig_cancellation = plt.gcf()
    plt.scatter(ids, [ground_before[id] for id in ids], color='lightgray', s=30, label='Before cancellation')
    plt.scatter(airline_after, [ground_after[id] for id in airline_after], color='red', s=15,
                label=f'{cancel_airline} after')
    plt.scatter(other_after, [ground_after[id] for id in other_after], color='tab:blue', s=15,
                label='Other airlines after')
    plt.scatter(cancelled, [ground_before[id] for id in cancelled], color='black', marker='x', s=60,
                label='Cancelled')
    plt.xlabel('Flight ID (ETA order)')
    plt.ylabel('Ground delay (min)')
    plt.title('Ground delay before and after the cancellation')
    plt.grid(True)
    plt.legend(loc='upper left')

    plotHistogram(CTA_new.drop(cancelled) / 60, 'Arrivals after cancellation and compression')
    fig_compression = plt.gcf()

    # Checkpoint 7 (WP3, task 1): Ground Holding Problem solved with linear programming
    # Same regulation, same affected flights and same slots as the GDP (Hfile and radius chosen in Checkpoint 4)
    # Reuse the same flight classification as Checkpoint 5.
    NotAffected7, Exempt7, Controlled7 = NotAffected5, Exempt5, Controlled5
    affected7 = sorted(Exempt7 + Controlled7)
    slot_times7 = computeSlots(HNoReg)[:, 0].astype(float)
    slotsGDP7, GroundDelayGDP7, AirDelayGDP7, CTA_GDP7 = assignSlotsGDP(computeSlots(HNoReg), Controlled7, Exempt7, arrivals)

    print(f'\nCHECKPOINT 7: GHP with Hfile = {Hfile_opt} h, radius = {radius_opt} km')
    print(f'Affected flights: {len(affected7)} ({len(Exempt7)} exempt, {len(Controlled7)} controlled), '
          f'slots: {len(slot_times7)}')
    print(f'DIAGNOSTIC: arrivals={len(arrivals)}, ETA range='
          f'{arrivals["ETA_min"].min():.1f}..{arrivals["ETA_min"].max():.1f} min, '
          f'Hstart={Hstart*60:.1f}, HNoReg={HNoReg:.1f} min, '
          f'not affected={len(NotAffected7)}')
    if len(affected7) == 0:
        raise ValueError(
            'Checkpoint 7 cannot proceed: no flights in the regulation window. '
            'Review the diagnostic line above and verify CSV, airport, ETA, Hstart and HNoReg.'
        )

    # a) Validation: unit cost (rf = 1) and only the basic constraints -> total cost must equal the GDP total delay
    slotsV, costV = solveGHP(arrivals, affected7, Exempt7, slot_times7)
    GroundDelayV, AirDelayV, CTA_V = delaysFromSlots(slotsV, Controlled7, arrivals)
    gdp_total = delayStatistics(GroundDelayGDP7)[0] + delayStatistics(AirDelayGDP7)[0]
    print(f'\nValidation (rf = 1): GHP total cost = {costV:.2f} min, GDP total delay = {gdp_total:.2f} min')
    printKPIs('GDP', GroundDelayGDP7, AirDelayGDP7)
    printKPIs('GHP validation (rf = 1, no extra constraints)', GroundDelayV, AirDelayV)

    # b) Feasibility constraints: maximum air delay and penalty for very long ground delays
    slotsC, costC = solveGHP(arrivals, affected7, Exempt7, slot_times7,
                             max_air=max_air_delay, limit=ground_limit, penalty=ground_penalty,
                             air_weight=air_weight)
    if slotsC is None:
        raise ValueError(f'No GHP solution with max_air_delay = {max_air_delay} min: increase it')
    GroundDelayC, AirDelayC, CTA_C = delaysFromSlots(slotsC, Controlled7, arrivals)
    printKPIs(f'GHP with constraints (max air delay {max_air_delay} min, ground delay over {ground_limit} min '
              f'costs x{ground_penalty}, air delay costs x{air_weight})', GroundDelayC, AirDelayC)
    print(f'  Objective function value: {costC:.2f}')
    print(f'  Flights with more than {ground_limit} min of ground delay: '
          f'GDP {sum(d > ground_limit for _, d in GroundDelayGDP7)}, GHP {sum(d > ground_limit for _, d in GroundDelayC)}')

    solution = pd.DataFrame({'ARCID': arrivals.loc[affected7, 'ARCID'],
                             'Airline': arrivals.loc[affected7, 'Compañía'],
                             'Status': ['Exempt' if id in Exempt7 else 'Controlled' for id in affected7],
                             'ETA_min': arrivals.loc[affected7, 'ETA_min'],
                             'Arrival_GHP_min': CTA_C[affected7]})
    solution['Delay_GHP_min'] = (solution['Arrival_GHP_min'] - solution['ETA_min']).round(2)
    solution['Arrival_GHP'] = [f'{int(m // 60):02d}:{int(m % 60):02d}' for m in solution['Arrival_GHP_min']]
    print('\nGHP solution (aircraft -> arrival time):')
    print(solution[['ARCID', 'Status', 'Arrival_GHP', 'Delay_GHP_min']].to_string())

    os.makedirs(results_folder, exist_ok=True)
    solution.to_csv(os.path.join(results_folder, 'ghp_task1_solution.csv'), sep=';', index=False)
    print(f'\nGHP solution saved in: {results_folder}')

    # WP3 TASK 2: GHP que minimiza las emisiones del retraso (CO2).
    # Se reutilizan los mismos slots, vuelos y maximo retraso aereo de la Task 1.
    co2_rates = co2DelayRates(arrivals, Exempt7, affected7)
    slotsE, co2_opt = solveGHP(
        arrivals, affected7, Exempt7, slot_times7,
        max_air=max_air_delay, limit=ground_limit, penalty=co2_ground_penalty, emission_rates=co2_rates
    )
    if slotsE is None:
        raise ValueError('Task 2 GHP emissions has no feasible solution. Check max_air_delay.')
    GroundDelayE, AirDelayE, CTA_E = delaysFromSlots(slotsE, Controlled7, arrivals)
    actual_co2 = airEmissions(arrivals, AirDelayE)[1] + groundEmissions(arrivals, GroundDelayE)[1]
    gdp_co2 = airEmissions(arrivals, AirDelayGDP7)[1] + groundEmissions(arrivals, GroundDelayGDP7)[1]
    task1_co2 = airEmissions(arrivals, AirDelayC)[1] + groundEmissions(arrivals, GroundDelayC)[1]
    # delaysFromSlots rounds to 0.01 min, so its KPI may differ slightly
    # from the MILP objective (which uses full slot precision).
    exact_co2 = sum((float(CTA_E.loc[fid]) - float(arrivals.loc[fid, 'ETA_min'])) * co2_rates[fid] for fid in affected7)
    # The objective also includes the penalty for ground delays over ground_limit, so it is >= the real CO2
    assert co2_opt >= exact_co2 - 1e-4, 'CO2 mismatch: objective vs exact flight delays'
    assert all(float(d) <= max_air_delay + 1e-7 for _, d in AirDelayE), 'Max air delay exceeded'
    assert len(GroundDelayE) + len(AirDelayE) == len(affected7), 'Missing GHP flights'
    print('\nWP3 TASK 2: GHP minimizing the CO2 of the delay')
    printKPIs('GHP emissions', GroundDelayE, AirDelayE)
    print(f'  GDP CO2: {gdp_co2:.2f} kg')
    print(f'  Task 1 GHP CO2: {task1_co2:.2f} kg')
    print(f'  Task 2 optimized CO2: {actual_co2:.2f} kg')
    print(f'  Flights with more than {ground_limit} min of ground delay: '
          f'GDP {sum(d > ground_limit for _, d in GroundDelayGDP7)}, GHP CO2 {sum(d > ground_limit for _, d in GroundDelayE)}')
    task2 = pd.DataFrame({
        'ARCID': arrivals.loc[affected7, 'ARCID'],
        'Airline': arrivals.loc[affected7, 'Compañía'],
        'Status': ['Exempt' if fid in Exempt7 else 'Controlled' for fid in affected7],
        'ETA_min': arrivals.loc[affected7, 'ETA_min'],
        'Arrival_GHP_CO2_min': CTA_E.loc[affected7],
        'CO2_kg_per_min': [co2_rates[fid] for fid in affected7],
    })
    task2['Delay_min'] = (task2['Arrival_GHP_CO2_min'] - task2['ETA_min']).round(2)
    task2['CO2_kg'] = task2['Delay_min'] * task2['CO2_kg_per_min']
    task2.to_csv(os.path.join(results_folder, 'ghp_task2_emissions.csv'), sep=';', index=False)
    print(f'  Saved: {os.path.join(results_folder, "ghp_task2_emissions.csv")}')

    # Task 2: effect of the penalty on long ground delays (justifies co2_ground_penalty)
    print('\nTask 2 penalty for ground delays over', ground_limit, 'min')
    for pen in [1, 2, 5, 10]:
        slots_p, _ = solveGHP(arrivals, affected7, Exempt7, slot_times7, max_air=max_air_delay,
                              limit=ground_limit, penalty=pen, emission_rates=co2_rates)
        ground_p, air_p, _ = delaysFromSlots(slots_p, Controlled7, arrivals)
        co2_p = sum(d * co2_rates[fid] for fid, d in list(ground_p) + list(air_p))
        print(f'  penalty x{pen:<3d} CO2 {co2_p:10.0f} kg, max ground delay {delayStatistics(ground_p)[1]:6.1f} min, '
              f'flights over {ground_limit} min: {sum(d > ground_limit for _, d in ground_p)}')

    # WP3 TASK 3: economic optimization using precomputed nonlinear delay costs.
    economic_data = prepare_cost_data(arrivals, filename, exempt_ids=Exempt7)
    economic_objective = lambda fid, delay: flight_delay_cost(fid, delay, economic_data)
    slotsCost, economic_opt = solveGHP(
        arrivals, affected7, Exempt7, slot_times7,
        max_air=max_air_delay, cost_function=economic_objective
    )
    if slotsCost is None:
        raise ValueError('Task 3 economic GHP infeasible: review model constraints.')
    GroundDelayCost, AirDelayCost, CTA_Cost = delaysFromSlots(slotsCost, Controlled7, arrivals)
    print('\nWP3 TASK 3: GHP minimizing the economic cost of the delay')
    printKPIs('GHP economic costs', GroundDelayCost, AirDelayCost)
    print(f'  Economic objective: {economic_opt:,.2f} EUR_2014 (EUROCONTROL curves + turnaround assumptions)')
    print(f'  Aircraft with identified following rotation: {economic_data.turnaround_min.notna().sum()}')

    # Task 3 sensitivity (slide 14): change the uncertain parameters and solve again
    print('\nTask 3 sensitivity analysis (one parameter changed each time)')
    sensitivity = []
    cases = [('Base (connecting share 0.3, reactionary 1.0)', None, 1.0),
             ('Connecting share 0', 0.0, 1.0), ('Connecting share 0.5', 0.5, 1.0),
             ('Reactionary 0', None, 0.0), ('Reactionary 0.5', None, 0.5)]
    for label, share, lam in cases:
        data_s = economic_data if share is None else prepare_cost_data(arrivals, filename, exempt_ids=Exempt7,
                                                                        connecting_share=share)
        cost_s = lambda fid, delay: cost_components(fid, delay, data_s, reactionary_multiplier=lam)['total_eur2014']
        slots_s, obj_s = solveGHP(arrivals, affected7, Exempt7, slot_times7, max_air=max_air_delay, cost_function=cost_s)
        ground_s, air_s, cta_s = delaysFromSlots(slots_s, Controlled7, arrivals)
        ity = [d for fid, d in list(ground_s) + list(air_s) if arrivals.loc[fid, 'Compañía'] == 'ITY']
        sensitivity.append({'Case': label, 'Cost EUR_2014': obj_s,
                            'Air delay min': delayStatistics(air_s)[0], 'Ground delay min': delayStatistics(ground_s)[0],
                            'Max ground delay min': delayStatistics(ground_s)[1],
                            'ITY mean delay min': float(np.mean(ity)) if ity else 0.0})
    sensitivity = pd.DataFrame(sensitivity).set_index('Case')
    print(sensitivity.round(1).to_string())
    sensitivity.to_csv(os.path.join(results_folder, 'wp3_task3_sensitivity.csv'), sep=';')

    # WP3 TASKS 4-5: SAME coefficients, flight set and KPI definitions for GDP/GHP.
    scen = {
        'GDP': (slotsGDP7, CTA_GDP7),
        'GHP_Task1': (slotsC, CTA_C),
        'GHP_CO2': (slotsE, CTA_E),
        'GHP_Cost': (slotsCost, CTA_Cost),
    }
    frames, summaries = {}, []
    for label, (assigned_slots, assigned_cta) in scen.items():
        # GDP is a comparison baseline and can exceed the Task 1 air-delay cap.
        validate_assignment(label, assigned_slots, affected7, arrivals,
                            max_air_delay if label != 'GDP' else float('inf'), set(Exempt7))
        table, summary = evaluate_assignment(label, arrivals, assigned_cta,
                                              affected7, set(Exempt7), economic_data, co2_rates)
        frames[label] = table
        summaries.append(summary)
        table.to_csv(os.path.join(results_folder, f'wp3_{label.lower()}_flights.csv'), sep=';', index_label='Flight_ID')
    kpi_table = pd.DataFrame(summaries).set_index('Scenario')
    kpi_table.to_csv(os.path.join(results_folder, 'wp3_task4_task5_comparison.csv'), sep=';')
    airlines = airline_comparison(frames, top_n=4)
    airlines.to_csv(os.path.join(results_folder, 'wp3_airline_fairness.csv'), sep=';', index=False)
    print('\nWP3 TASKS 4-5: GDP vs GHP KPIs')
    print(kpi_table.round(2).to_string())
    print('\nTop four airlines - fairness:')
    print(airlines.round(2).to_string(index=False))
    print('\nNOTE: economic results are EUR_2014, with proxy aircraft and turnaround choices; see REFERENCES_WP3_COSTS.md')

    fig_compare, ax = plt.subplots(figsize=(9, 5))
    kpi_table[['CO2 kg']].plot(kind='bar', ax=ax, legend=False)
    ax.set_ylabel('CO2 emissions from delay (kg)')
    ax.set_title('WP3 GDP vs GHP: emissions')
    ax.tick_params(axis='x', rotation=15)
    fig_compare.tight_layout()
    fig_costs, ax = plt.subplots(figsize=(9, 5))
    kpi_table[['Economic cost EUR']].plot(kind='bar', ax=ax, legend=False)
    ax.set_ylabel('Tactical delay cost (EUR, 2014 prices)')
    ax.set_title('WP3 GDP vs GHP: EUROCONTROL-based tactical cost')
    ax.tick_params(axis='x', rotation=15)
    fig_costs.tight_layout()
    fig_airlines, ax = plt.subplots(figsize=(10, 5))
    airlines.pivot(index='Airline', columns='Scenario', values='Mean delay min').plot(kind='bar', ax=ax)
    ax.set_ylabel('Mean delay (min)')
    ax.set_title('WP3 fairness: mean delay for top four airlines')
    fig_airlines.tight_layout()

    # WP4 (Checkpoint 9): substituir vols per tren i tornar a fer GDP i GHP amb la nova demanda
    wp4_figures = run_wp4(arrivals, Hfile_opt, radius_opt)
    figures = {
        'Checkpoint 1-2: demanda inicial': fig_initial,
        'Checkpoint 1-2: demanda/capacidad acumulada': fig_aggregate,
        'Checkpoint 3: llegadas después del GDP': fig_gdp,
        'Checkpoint 4: heatmaps de KPIs': fig_heatmaps,
        'Checkpoint 5: delay antes y después': fig_cancellation,
        'Checkpoint 5: llegadas tras compresión': fig_compression,
        'WP3: emisiones GDP vs GHP': fig_compare,
        'WP3: costes GDP vs GHP': fig_costs,
        'WP3: aerolíneas y equidad': fig_airlines,
        **wp4_figures,
    }
    choose_figures(figures, results_folder, interactive=(plot_choice is None), choice=plot_choice)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Proyecto LIRF modular, checkpoints 1-5 y 7')
    parser.add_argument('--plots', default=None, help='Selecciona figuras: 1,3,5 / all / none')
    args = parser.parse_args()
    main(plot_choice=args.plots)