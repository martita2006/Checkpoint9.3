import numpy as np
from config import AAR, PAAR, Hstart, Hend, margin
from emissions import airEmissions, groundEmissions

def computeHNoReg(eta):
    t = np.arange(0, 1441)
    demand = np.searchsorted(np.sort(eta), t, side="right")
    capacity = demand.astype(float)
    d0 = np.searchsorted(np.sort(eta), Hstart*60, side="left")
    during = (t > Hstart * 60) & (t <= Hend * 60)
    after = t > Hend * 60
    capacity[during] = d0 + (t[during] - Hstart * 60) * PAAR / 60
    capacity[after] = d0 + (Hend - Hstart) * PAAR + (t[after] - Hend * 60) * AAR / 60
    recovered_times = t[(t >= Hend * 60) & (capacity >= demand)]
    if len(recovered_times) == 0:
        raise ValueError('Capacity does not catch up with demand within the day; extend the time horizon.')
    HNoReg = recovered_times[0]
    queue = np.maximum(demand - capacity, 0)
    total_delay = queue[(t >= Hstart * 60) & (t <= HNoReg)].sum()
    return t, demand, capacity, HNoReg, total_delay

def computeSlots(HNoReg):
    slots = []  #crea una lista inicialmente vacía
    time = Hstart * 60.0  #convierte las 08:00 en el minuto 480 del día
    while time <= HNoReg + 60:
        slots.append([round(time, 2), 0, 0])  #añade un slot con su hora, sin vuelo ni aerolínea asignados
        time += 60 / PAAR if time < Hend * 60 else 60 / AAR  #
    return np.array(slots, dtype=object)

def determineFlightStatus(arrivals, Hfile, HNoReg, radius):
    NotAffected, ExemptRadius, ExemptInternational, ExemptFlying, Controlled = [], [], [], [], []
    for flight_id, flight in arrivals.iterrows():
        if flight['ETA_min'] < Hstart * 60 or flight['ETA_min'] > HNoReg:
            NotAffected.append(flight_id)
        elif flight['Distance'] > radius:
            ExemptRadius.append(flight_id)
        elif flight['NonECAC']:
            ExemptInternational.append(flight_id)
        elif flight['ETD_min'] < Hfile * 60 + margin:
            ExemptFlying.append(flight_id)
        else:
            Controlled.append(flight_id)
    Exempt = ExemptRadius + ExemptInternational + ExemptFlying
    return NotAffected, ExemptRadius, ExemptInternational, ExemptFlying, Exempt, Controlled

def delayStatistics(delay_list):
    delay_minutes = np.array([flight[1] for flight in delay_list], dtype=float)
    if len(delay_minutes) == 0:
        return 0.0, 0.0, 0.0, 0.0
    total_delay_minutes = delay_minutes.sum()
    maximum_delay_minutes = delay_minutes.max()
    mean_delay_minutes = delay_minutes.mean()
    standard_deviation = delay_minutes.std()
    relative_std_percent = (100 * standard_deviation / mean_delay_minutes) if mean_delay_minutes > 0 else 0.0
    return total_delay_minutes, maximum_delay_minutes, mean_delay_minutes, relative_std_percent

def assignSlotsGDP(slots, Controlled, Exempt, arrivals):
    #slots[time,id,company]
    slot_time = slots[:, 0].astype(float)
    used_slots = np.zeros(len(slots), dtype=bool)
    CTA = arrivals['ETA_min'].copy()
    GroundDelay = []
    AirDelay = []
    for id in sorted(Exempt) + sorted(Controlled):
        eta = arrivals.loc[id, 'ETA_min']
       #Cerca el primer slot not used_slots & hora>=eta.
        first_free_slot = np.where(~used_slots & (slot_time >= eta))[0]
        #Si no queda cap slot lliure posterior a l'ETA del vol, mostra un avís per pantalla i salta al següent vol
        if len(first_free_slot) == 0:
            print(f"Warning: no slot left for flight {arrivals.loc[id, 'ARCID']}")
            continue
        k = first_free_slot[0]
        used_slots[k] = True
        slots[k, 1] = id
        slots[k, 2] = arrivals.loc[id, 'Compañía']
        CTA[id] = slot_time[k]
        delay = round(float(slot_time[k] - eta), 2)
        if id in Controlled:
            GroundDelay.append([id, delay])
        else:
            AirDelay.append([id, delay])
    return slots, np.array(GroundDelay, dtype=object), np.array(AirDelay, dtype=object), CTA

def unrecoverableDelay(arrivals, GroundDelay):
    unrecoverable = []
    for id, gd in GroundDelay:
        etd = arrivals.loc[id, 'ETD_min']
        if etd >= Hstart * 60:
            unrec = 0.0
        elif etd + gd <= Hstart * 60:
            unrec = gd
        else:
            unrec = Hstart * 60 - etd
        unrecoverable.append([id, round(float(unrec), 2)])
    return np.array(unrecoverable, dtype=object)

def gdpKPIs(arrivals, Hfile, radius, HNoReg):
    status = determineFlightStatus(arrivals, Hfile, HNoReg, radius)
    #STATUS: 5 -> ControlledFlight 4->Exempt in assignSlotsGDP
    slots, GroundDelay, AirDelay, CTA = assignSlotsGDP(computeSlots(HNoReg), status[5], status[4], arrivals)
    min_airdelay = sum(delay for _, delay in AirDelay)
    min_groundelay = sum(delay for _, delay in GroundDelay)
    min_unrecoverableDelay = sum(unrec for _, unrec in unrecoverableDelay(arrivals, GroundDelay))
    co2_air = airEmissions(arrivals, AirDelay)[1]
    co2_ground = groundEmissions(arrivals, GroundDelay)[1]
    return [min_airdelay, min_groundelay, min_unrecoverableDelay, co2_air, co2_ground, co2_air + co2_ground]

def cancelFlights(slots, GroundDelay, arrivals, airline, max_delay):
    #Vols de l'aerolinia amb més retard que max_delay -> cancel·lats
    cancelled = []
    for id, delay in GroundDelay:
        if arrivals.loc[id, 'Compañía'] == airline and delay > max_delay:
            cancelled.append(id)
    for k in range(len(slots)):
        if slots[k, 1] in cancelled:
            slots[k, 1] = 0
    return slots, cancelled

def fillEmptySlots(slots, movable, arrivals, airline=None):
    # airline given -> substitution: only its own empty slots and its own flights
    # airline None  -> compression: any empty slot, any controlled flight
    moved = 0
    for k in range(len(slots)):
        if slots[k, 1] != 0:
            continue
        if airline is not None and slots[k, 2] != airline:
            continue
        for j in range(k + 1, len(slots)):
            id = slots[j, 1]
            if id in movable and arrivals.loc[id, 'ETA_min'] <= slots[k, 0]:
                if airline is None or arrivals.loc[id, 'Compañía'] == airline:
                    slots[k, 1] = id
                    slots[k, 2] = arrivals.loc[id, 'Compañía']
                    slots[j, 1] = 0
                    moved += 1
                    break
    return slots, moved

def delaysFromSlots(slots, Controlled, arrivals):
    #Recalculem retards i CTA a partir de la nova taula de slots
    GroundDelay = []
    AirDelay = []
    CTA = arrivals['ETA_min'].copy()
    for time, id, airline in slots:
        if id == 0:
            continue
        CTA[id] = time
        delay = round(float(time - arrivals.loc[id, 'ETA_min']), 2)
        if id in Controlled:
            GroundDelay.append([id, delay])
        else:
            AirDelay.append([id, delay])
    return np.array(GroundDelay, dtype=object), np.array(AirDelay, dtype=object), CTA

def printKPIs(name, GroundDelay, AirDelay):
    all_delays = np.array([row[1] for row in list(GroundDelay) + list(AirDelay)], dtype=float)
    print(f'\n{name}')
    for kind, delays in [('Ground', GroundDelay), ('Air', AirDelay)]:
        total, maximum, mean, rel_std = delayStatistics(delays)
        print(f'  {kind} delay: total {total:.2f} min, max {maximum:.2f} min, '
              f'mean {mean:.2f} min, relative std {rel_std:.1f} %')
    print(f'  Total delay: {all_delays.sum():.2f} min, mean {(all_delays.mean() if len(all_delays) else 0.0):.2f} min/flight')
    print(f'  Delayed flights: {np.sum(all_delays > 0)}')
