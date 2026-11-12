from pathlib import Path

# Si el CSV está junto a main.py se detecta automáticamente.
CSV_NAME = 'LIRF_10AUG2025_2111.2.csv'
filename = str(Path(__file__).resolve().parent / CSV_NAME)
airport = 'LIRF'
AAR, PAAR = 49, 20
Hstart, Hend = 8, 13
Hfile, radius, margin = Hstart-2, 1000, 30
cancel_airline, cancel_delay = 'ITY', 120
max_air_delay, ground_limit, ground_penalty, air_weight = 45, 120, 2, 3
results_folder = str(Path(__file__).resolve().parent / 'results')
apu_fuel = {'A':500.0,'B':300.0,'C':220.0,'D':130.0,'E':60.0,'F':0.0}
# Task 2: CO2 penalty for ground delays over ground_limit (new).
#x2 ja té tot l’efecte, i x10 deixa 5 vols, els mateixos que el GDP.
co2_ground_penalty = 10
# Task 3: passengers per flight (new)
load_factor = 0.841  # IATA, European airlines 2024 (press release 30 Jan 2025)
hub_airline = 'ITY'  # ITA Airways has its hub at LIRF
hub_connecting_share = 0.30  # assumption: share of ITA passengers connecting at LIRF (tested 0, 0.3, 0.5)
