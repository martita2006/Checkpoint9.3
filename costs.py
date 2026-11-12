"""WP3 option B: airline tactical delay-cost curves from EUROCONTROL / Westminster.

Source: Cook & Tanner (2015), European airline delay cost reference values,
version 4.1, Tables 22 (at-gate primary), 25 (arrival management primary),
26 (at-gate full), 29 (arrival management full). ALL VALUES IN EUR_2014.
https://www.eurocontrol.int/sites/default/files/publication/files/european-airline-delay-cost-reference-values-final-report-4-1.pdf

Data after 2014 and non-covered ATYP: proxy assumptions; no 2026 EUR conversion.
"""
import numpy as np
import pandas as pd

DELAY_KNOTS_MIN = np.array([0, 5, 15, 30, 60, 90, 120, 180, 240, 300], dtype=float)
TABLES_EUR2014 = {'AIR_FULL': {'A319': [220, 890, 2510, 9130, 22760, 41470, 56660, 76660, 102220],
              'A320': [250, 1000, 2820, 10370, 25940, 47270, 64460, 86940, 115440],
              'A321': [280, 1130, 3260, 12210, 30880, 56400, 76670, 103060, 136260],
              'A332': [470, 1840, 5250, 19880, 49730, 102150, 146340, 184100, 230690],
              'AT43': [70, 290, 830, 3030, 7590, 14130, 19830, 28060, 39730],
              'AT72': [90, 370, 1090, 4130, 10490, 19490, 26980, 37480, 51870],
              'B733': [210, 850, 2400, 8710, 21690, 39580, 54090, 73210, 97720],
              'B734': [250, 980, 2730, 9910, 24670, 44930, 61300, 82730, 109970],
              'B735': [180, 740, 2090, 7680, 19200, 35140, 48090, 65310, 87580],
              'B738': [250, 1020, 2910, 10790, 27160, 49610, 67530, 90940, 120570],
              'B744': [710, 2760, 7780, 29000, 72060, 147460, 211020, 264690, 330020],
              'B752': [290, 1180, 3420, 12880, 32640, 59670, 81010, 108730, 143480],
              'B763': [480, 1830, 5070, 18510, 45560, 92770, 133090, 167800, 210660],
              'DH8D': [110, 450, 1280, 4690, 11720, 21570, 29840, 41270, 56710],
              'E190': [180, 690, 1880, 6600, 16150, 29350, 40430, 55240, 74620]},
 'AIR_PRIMARY': {'A319': [230, 850, 2140, 5850, 10700, 16420, 30450, 47790, 68240],
                 'A320': [260, 950, 2390, 6610, 12120, 18640, 34660, 54480, 77880],
                 'A321': [280, 1070, 2730, 7670, 14190, 21940, 41050, 64780, 92870],
                 'A332': [440, 1630, 4130, 11510, 21210, 32720, 61030, 96120, 137610],
                 'AT43': [80, 290, 710, 1910, 3480, 5330, 9860, 15430, 22000],
                 'AT72': [100, 360, 920, 2560, 4720, 7290, 13620, 21460, 30740],
                 'B733': [220, 820, 2050, 5590, 10200, 15640, 28980, 45440, 64850],
                 'B734': [260, 940, 2330, 6370, 11630, 17840, 33050, 51820, 73950],
                 'B735': [190, 710, 1780, 4900, 8960, 13760, 25540, 40100, 57270],
                 'B738': [260, 970, 2460, 6830, 12570, 19380, 36130, 56880, 81410],
                 'B744': [670, 2470, 6200, 17070, 31280, 48090, 89340, 140350, 200580],
                 'B752': [300, 1120, 2870, 8080, 14960, 23140, 43310, 68370, 98020],
                 'B763': [450, 1640, 4060, 11010, 20030, 30670, 56670, 88720, 126470],
                 'DH8D': [120, 440, 1090, 2980, 5440, 8340, 15440, 24210, 34550],
                 'E190': [190, 670, 1630, 4330, 7800, 11870, 21780, 33930, 48210]},
 'GATE_FULL': {'A319': [70, 440, 1600, 7320, 20040, 37850, 51240, 69420, 93180],
               'A320': [80, 500, 1820, 8350, 22920, 43250, 58420, 78890, 105380],
               'A321': [100, 580, 2160, 10010, 27580, 51990, 70060, 94250, 125240],
               'A332': [180, 990, 3550, 16480, 44620, 95330, 136120, 170480, 213660],
               'AT43': [30, 180, 610, 2610, 6960, 13290, 18550, 26360, 37610],
               'AT72': [40, 240, 820, 3600, 9690, 18430, 25380, 35350, 49210],
               'B733': [70, 430, 1550, 7020, 19160, 36220, 49040, 66480, 89310],
               'B734': [80, 480, 1740, 7930, 21690, 40960, 55340, 74780, 100040],
               'B735': [70, 390, 1400, 6280, 17110, 32350, 43900, 59720, 80590],
               'B738': [90, 540, 1940, 8860, 24270, 45750, 61740, 83220, 110920],
               'B744': [240, 1370, 5000, 23430, 63710, 136330, 194330, 242440, 302200],
               'B752': [100, 620, 2290, 10620, 29250, 55150, 74240, 99700, 132200],
               'B763': [170, 900, 3200, 14780, 39960, 85300, 121880, 152860, 191990],
               'DH8D': [40, 250, 890, 3900, 10530, 19990, 27480, 38120, 52780],
               'E190': [60, 320, 1150, 5140, 13970, 26440, 36060, 49420, 67340]},
 'GATE_PRIMARY': {'A319': [80, 400, 1230, 4040, 7980, 12800, 25030, 40550, 59200],
                  'A320': [90, 450, 1390, 4590, 9100, 14620, 28620, 46430, 67820],
                  'A321': [100, 520, 1630, 5470, 10890, 17530, 34440, 55970, 81850],
                  'A332': [150, 780, 2430, 8110, 16100, 25900, 50810, 82500, 120580],
                  'AT43': [40, 180, 490, 1490, 2850, 4490, 8580, 13730, 19880],
                  'AT72': [50, 230, 650, 2030, 3920, 6230, 12020, 19330, 28080],
                  'B733': [80, 400, 1200, 3900, 7670, 12280, 23930, 38710, 56440],
                  'B734': [90, 440, 1340, 4390, 8650, 13870, 27090, 43870, 64020],
                  'B735': [80, 360, 1090, 3500, 6870, 10970, 21350, 34510, 50280],
                  'B738': [100, 490, 1490, 4900, 9680, 15520, 30340, 49160, 71760],
                  'B744': [200, 1080, 3420, 11500, 22930, 36960, 72650, 118100, 172760],
                  'B752': [110, 560, 1740, 5820, 11570, 18620, 36540, 59340, 86740],
                  'B763': [140, 710, 2190, 7280, 14430, 23200, 45460, 73780, 107800],
                  'DH8D': [50, 240, 700, 2190, 4250, 6760, 13080, 21060, 30620],
                  'E190': [70, 300, 900, 2870, 5620, 8960, 17410, 28110, 40930]}}
ATYP_TO_REFERENCE = {'A20N': 'A320',
 'A21N': 'A321',
 'A318': 'A319',
 'A319': 'A319',
 'A320': 'A320',
 'A321': 'A321',
 'A332': 'A332',
 'A333': 'A332',
 'A339': 'A332',
 'A359': 'A332',
 'A35K': 'A332',
 'A388': 'B744',
 'AT43': 'AT43',
 'AT72': 'AT72',
 'AT76': 'AT72',
 'B38M': 'B738',
 'B733': 'B733',
 'B734': 'B734',
 'B735': 'B735',
 'B737': 'B738',
 'B738': 'B738',
 'B739': 'B738',
 'B744': 'B744',
 'B752': 'B752',
 'B763': 'B763',
 'B764': 'B763',
 'B772': 'B763',
 'B77W': 'B763',
 'B788': 'B763',
 'B789': 'B763',
 'B78X': 'B763',
 'BCS1': 'E190',
 'BCS3': 'A319',
 'CL35': 'AT43',
 'DH8D': 'DH8D',
 'E190': 'E190',
 'E195': 'E190',
 'E295': 'E190'}
# Only used for aircraft not directly covered; approximation, not a published mapping.
RECAT_TO_REFERENCE = {'A': 'B744', 'B': 'A332', 'C': 'A321',
                      'D': 'A320', 'E': 'AT72', 'F': 'AT43'}
# 45 min is a modelling assumption for required minimum turnaround, not
# the observed 2014 EUROCONTROL scheduled/actual turnaround averages.
TURNAROUND_MIN_REQUIRED = 45.0
# 0..1 adjustable sensitivity coefficient applied to the *published difference*
# between full and primary curves, NOT an added arbitrary EUR/min fee.
REACTIONARY_MULTIPLIER = 1.0


def _interpolate_eur(values, delay_min):
    """Piecewise linear interpolation; beyond 300 min, last-slope extension.

    Monotonicity ensured on the source curves; zero delay costs zero.
    The last segment is an explicit extrapolation assumption.
    """
    d = max(0.0, float(delay_min))
    y = np.array([0.0] + list(values), dtype=float)
    if d <= DELAY_KNOTS_MIN[-1]:
        return float(np.interp(d, DELAY_KNOTS_MIN, y))
    slope = (y[-1] - y[-2]) / (DELAY_KNOTS_MIN[-1] - DELAY_KNOTS_MIN[-2])
    return float(y[-1] + (d - DELAY_KNOTS_MIN[-1]) * slope)


def prepare_cost_data(arrivals, csv_path, exempt_ids=None, connecting_share=None):
    """Aircraft reference types and *observed* next-rotation turnaround by RM.

    Also computes a passenger weight per flight (seats x load factor, plus the
    connecting share for the hub airline). RM matching is restricted to
    next local departure later than arrival (day-crossing not reconstructed).
    """
    from data import parse_clock_minutes
    df = pd.read_csv(csv_path, sep=';')
    out = pd.DataFrame(index=arrivals.index)
    exempt = set(exempt_ids or [])
    out['is_exempt'] = [fid in exempt for fid in arrivals.index]
    out['ATYP'] = arrivals['ATYP'].astype('string').str.strip().str.upper()
    out['reference_aircraft'] = [ATYP_TO_REFERENCE.get(str(typ), RECAT_TO_REFERENCE.get(str(cat).strip().upper(), 'A320'))
                                 for typ, cat in zip(out['ATYP'], arrivals['Category'])]
    out['reference_is_proxy'] = [str(typ) != ref for typ, ref in zip(out['ATYP'], out['reference_aircraft'])]
    out['next_departure_min'] = np.nan
    rm_col = next((c for c in ['RM', 'Registration', 'Registration mark'] if c in df and c in arrivals), None)
    if rm_col and 'ADEP' in df and 'ETD' in df:
        dep = df.loc[df['ADEP'].astype(str).str.strip().str.upper().eq('LIRF')].copy()
        if not dep.empty:
            dep['departure_min'] = parse_clock_minutes(dep['ETD'], 'ETD_departures')
            dep['rm_key'] = dep[rm_col].astype(str).str.strip().str.upper()
            groups = {rm: sorted(g['departure_min'].astype(float)) for rm,g in dep.groupby('rm_key')
                      if rm not in ('', 'NAN', 'NONE')}
            for fid, flight in arrivals.iterrows():
                rm = str(flight[rm_col]).strip().upper()
                eta = float(flight['ETA_min'])
                times = groups.get(rm, []) if rm not in ('','NAN','NONE') else []
                candidates = [t for t in times if t > eta]
                if candidates:
                    out.loc[fid, 'next_departure_min'] = min(candidates)
    out['turnaround_min'] = out['next_departure_min'] - arrivals['ETA_min']
    # New (Checkpoint 8): passenger weight of each flight (slide 12 of WP3-4)
    # passengers = available seats x load factor; hub airline passengers count (1 + connecting share)
    from config import load_factor, hub_airline, hub_connecting_share
    share = hub_connecting_share if connecting_share is None else connecting_share
    seats = pd.to_numeric(arrivals['Available seats'], errors='coerce')
    seats = seats.fillna(seats.median())
    out['passengers'] = seats * load_factor
    is_hub = arrivals['Compañía'].astype(str).str.strip() == hub_airline
    weighted = out['passengers'] * np.where(is_hub, 1 + share, 1.0)
    out['pax_weight'] = weighted / weighted.mean()  # mean weight = 1: total cost level of the curves is kept
    print('EUROCONTROL aircraft proxy count:',int(out.reference_is_proxy.sum()), '/',len(out))
    print('Following rotation identified:',int(out.turnaround_min.notna().sum()), '/',len(out))
    return out


def cost_components(fid, delay_min, data, reactionary_multiplier=REACTIONARY_MULTIPLIER,
                    min_turnaround=TURNAROUND_MIN_REQUIRED):
    """Primary cost + conditional share of published reactionary cost; EUR_2014.

    Reactionary share ramps over delay exceeding turnaround slack. This scaling
    is a transparent, sensitivity-testable model assumption (not EUROCONTROL).
    The primary cost is multiplied by the passenger weight of the flight
    (mean weight 1), so flights with more passengers or connections cost more.
    """
    d = max(0.0, float(delay_min))
    ref = str(data.loc[fid, 'reference_aircraft'])
    phase = 'AIR' if bool(data.loc[fid, 'is_exempt']) else 'GATE'
    primary = _interpolate_eur(TABLES_EUR2014[phase + '_PRIMARY'][ref], d)
    full = _interpolate_eur(TABLES_EUR2014[phase + '_FULL'][ref], d)
    reactionary_increment = max(0.0, full - primary)  # published reactionary part, before the passenger weight
    if 'pax_weight' in data:
        primary *= float(data.loc[fid, 'pax_weight'])  # more passengers (or connections): higher passenger cost
    turnaround = float(data.loc[fid, 'turnaround_min'])
    if not np.isfinite(turnaround):
        reaction_fraction = 0.0  # unknown RM: no observed same-aircraft reactionary term
    else:
        slack = max(0.0, turnaround - float(min_turnaround))
        # Zero below slack; progressive thereafter; capped at the published full increment.
        reaction_fraction = min(1.0, max(0.0, d - slack) / max(1.0, float(min_turnaround)))
    reactionary = reactionary_increment * reaction_fraction * float(reactionary_multiplier)
    return {'primary_eur2014':primary,'reactionary_eur2014':reactionary,
            'reaction_fraction':reaction_fraction,'total_eur2014':primary+reactionary}


def flight_delay_cost(fid, delay_min, data):
    return cost_components(fid, delay_min, data)['total_eur2014']


def costs_for_assignment(arrivals, cta, flight_ids, data):
    return {fid: flight_delay_cost(fid,float(cta.loc[fid])-float(arrivals.loc[fid,'ETA_min']),data)
            for fid in flight_ids}
