from config import apu_fuel

"""
Aircraft Emissions Modeling Library
==================================

Provides functions to estimate emissions per Available Seat Kilometer (ASK) for various aircraft types
based on flight distance and number of available seats.
"""


def compute_fuel_ask(distance_km: float, available_seats: int, force: bool = False) -> float:
    """
    Estimate the fuel consumption per Available Seat Kilometer (ASK).

    Equation 1: Small aircraft (50–172 seats, 100–5,000 km)
    Equation 2: Large aircraft (172–365 seats, 200–12,000 km)

    Parameters:
        distance_km (float): Flight distance in kilometers.
        available_seats (int): Number of available seats.
        force (bool): If True, bypasses range checks. Defaults to False.

    Returns:
        float: Estimated fuel consumption per ASK (rounded to 2 decimals) in gFuel/ASK.

    Raises:
        ValueError: On invalid inputs unless force=True.
    """

    def validate_inputs():
        if not (50 <= available_seats <= 365) and not force:
            raise ValueError("Seats available out of range (50 - 365)")
        if not (100 <= distance_km <= 12000) and not force:
            raise ValueError("Distance out of range (100 - 12,000 km)")
        if distance_km > 5000 and available_seats < 172 and not force:
            raise ValueError("Flights over 5,000 km require at least 172 seats")
        if available_seats >= 172 and distance_km < 200 and not force:
            raise ValueError("Flights under 200 km are invalid for aircraft with 172+ seats")

    def should_use_eq3():
        if not force:
            return 172 <= available_seats <= 365 and 200 <= distance_km <= 12000
        return available_seats >= 172 or abs(available_seats - 172) < abs(available_seats - 50)

    validate_inputs()
    use_eq3 = should_use_eq3()

    if use_eq3:
        # Equation 3
        fuel_ask = (
                0.7361 + 6651 / distance_km + 5.989e-4 * distance_km +
                6.152e-2 * available_seats - 1.014e-6 * distance_km * available_seats
        )
    else:
        # Equation 2
        fuel_ask = (
                34.67 + 6608 / distance_km - 1.196e-3 * distance_km -
                0.1354 * available_seats + 1.338e-5 * distance_km * available_seats
        )

    return round(fuel_ask, 2)


def compute_co2_ask(distance_km: float, available_seats: int, force: bool = False) -> float:
    """
    Estimate CO₂ emissions per ASK by scaling fuel consumption.

    Parameters:
        distance_km (float): Flight distance in kilometers.
        available_seats (int): Number of available seats.
        force (bool): If True, bypass validation in fuel model. Defaults to False.

    Returns:
        float: Estimated CO₂ emissions per ASK (rounded to 2 decimals) in gCO2/ASK.
    """
    co2_kg_per_kg_fuel = 3.16
    return round(compute_fuel_ask(distance_km, available_seats, force) * co2_kg_per_kg_fuel, 2)


def compute_sox_ask(distance_km: float, available_seats: int, force: bool = False) -> float:
    """
    Estimate SOₓ emissions per ASK by scaling fuel consumption.

    Parameters:
        distance_km (float): Flight distance in kilometers.
        available_seats (int): Number of available seats.
        force (bool): If True, bypass validation in fuel model. Defaults to False.

    Returns:
        float: Estimated SOₓ emissions per ASK (rounded to 2 decimals) in gSOx/ASK.
    """
    sox_kg_per_kg_fuel = 0.84 / 1000
    return round(compute_fuel_ask(distance_km, available_seats, force) * sox_kg_per_kg_fuel, 2)


def compute_water_vapour_ask(distance_km: float, available_seats: int, force: bool = False) -> float:
    """
    Estimate water vapour emissions per ASK by scaling fuel consumption.

    Parameters:
        distance_km (float): Flight distance in kilometers.
        available_seats (int): Number of available seats.
        force (bool): If True, bypass validation in fuel model. Defaults to False.

    Returns:
        float: Estimated water vapour emissions per ASK (rounded to 2 decimals) in gSOx/ASK.
    """
    water_vapour_kg_per_kg_fuel = 1.237
    return round(compute_fuel_ask(distance_km, available_seats, force) * water_vapour_kg_per_kg_fuel, 2)


def compute_nox_ask(distance_km: float, available_seats: int, force: bool = False) -> float:
    """
    Estimate NOₓ emissions per ASK.

    Equation 4: Small aircraft (50–172 seats, 100–5,000 km)
    Equation 5: Large aircraft (172–365 seats, 200–12,000 km)

    Parameters:
        distance_km (float): Flight distance in kilometers.
        available_seats (int): Number of available seats.
        force (bool): If True, bypass validation. Defaults to False.

    Returns:
        float: Estimated NOₓ emissions per ASK (rounded to 2 decimals) in gNOx/ASK.

    Raises:
        ValueError: On invalid inputs unless force=True.
    """

    def validate_inputs():
        if not (50 <= available_seats <= 365) and not force:
            raise ValueError("Seats available out of range (50 - 365)")
        if not (100 <= distance_km <= 12000) and not force:
            raise ValueError("Distance out of range (100 - 12,000 km)")
        if distance_km > 5000 and available_seats < 172 and not force:
            raise ValueError("Flights over 5,000 km require at least 172 seats")
        if available_seats >= 172 and distance_km < 200 and not force:
            raise ValueError("Flights under 200 km are invalid for aircraft with 172+ seats")

    def should_use_eq5():
        if not force:
            return 172 <= available_seats <= 365 and 200 <= distance_km <= 12000
        return available_seats >= 172 or abs(available_seats - 172) < abs(available_seats - 50)

    validate_inputs()
    use_eq5 = should_use_eq5()

    if use_eq5:
        # Equation 5
        nox_ask = (
                -1.427 + 152.1 / distance_km + 143.5 / available_seats +
                3.625e-6 * distance_km + 4.18e-3 * available_seats
        )
    else:
        # Equation 4
        nox_ask = (
                0.1512 + 63.34 / distance_km + 0.2954 / available_seats +
                2.214e-6 * distance_km + 6.217e-4 * available_seats
        )

    return round(nox_ask, 2)


def compute_co_ask(distance_km: float, available_seats: int, force: bool = False) -> float:
    """
    Estimate carbon monoxide (CO) emissions per ASK.

    Equation 6: Small aircraft (50–172 seats, 100–5,000 km)
    Equation 7: Large aircraft (172–365 seats, 200–12,000 km)

    Parameters:
        distance_km (float): Flight distance in kilometers.
        available_seats (int): Number of available seats.
        force (bool): If True, bypass validation. Defaults to False.

    Returns:
        float: Estimated CO emissions per ASK (rounded to 2 decimals) in gCO/ASK.

    Raises:
        ValueError: On invalid inputs unless force=True.
    """

    def validate_inputs():
        if not (172 <= available_seats <= 365) and not force:
            raise ValueError("Seats available out of range (172 - 365)")
        if not (200 <= distance_km <= 12000) and not force:
            raise ValueError("Distance out of range (200 - 12,000 km)")

    def should_use_eq7():
        if not force:
            return 172 <= available_seats <= 365 and 200 <= distance_km <= 12000
        return available_seats >= 172 or abs(available_seats - 172) < abs(available_seats - 50)

    validate_inputs()
    use_eq7 = should_use_eq7()

    if use_eq7:
        # Equation 7
        co_ask = (
                -0.5736 + 65.11 / distance_km + 51.85 / available_seats +
                2.489e-5 * distance_km + 1.411e-3 * available_seats -
                8.39e-8 * distance_km * available_seats
        )
    else:
        # Equation 6
        co_ask = (
                0.08338 + 96.54 / distance_km + 2.184 / available_seats +
                2.433e-6 * distance_km - 8.602e-4 * available_seats -
                6.053e-8 * distance_km * available_seats
        )

    return round(co_ask, 2)


def air_co2_rate(flight):
    # New (Checkpoint 8): CO2 (kg) per minute of air delay, with the same professor model used in airEmissions
    km_per_min = flight['Mean velocity (kts)'] * 1.852 / 60
    seat_km_per_min = flight['Available seats'] * km_per_min
    return compute_co2_ask(flight['Distance'], flight['Available seats'], force=True) * seat_km_per_min / 1000

def airEmissions(arrivals, AirDelay):
    fuel, co2, nox, sox, h2o, co = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for id, delay in AirDelay:
        flights = arrivals.loc[id]
        km_per_min = flights['Mean velocity (kts)'] * 1.852 / 60 #NM->Km
        seat_km = flights['Available seats'] * km_per_min * delay  # seat-km flown while holding
        # Càlcul de totes les emissions (convertint de g a kg quan pertoca)
        fuel += compute_fuel_ask(flights['Distance'], flights['Available seats'], force=True) * seat_km / 1000
        co2 += compute_co2_ask(flights['Distance'], flights['Available seats'], force=True) * seat_km / 1000
        nox += compute_nox_ask(flights['Distance'], flights['Available seats'], force=True) * seat_km / 1000
        sox += compute_sox_ask(flights['Distance'], flights['Available seats'], force=True) * seat_km / 1000
        h2o += compute_water_vapour_ask(flights['Distance'], flights['Available seats'], force=True) * seat_km / 1000
        co += compute_co_ask(flights['Distance'], flights['Available seats'], force=True) * seat_km / 1000
    return fuel, co2, nox, sox, h2o, co  # kg

def groundEmissions(arrivals, GroundDelay):
    fuel = 0.0
    for id, delay in GroundDelay:
        category = str(arrivals.loc[id, 'Category']).strip().upper()
        fuel += apu_fuel.get(category) * delay / 60
    co2 = fuel * 3.16
    h2o = fuel * 1.237
    sox = fuel * 0.00084
    nox = fuel * 0.0022
    co = fuel * 0.0105
    return fuel, co2, nox, sox, h2o, co

def co2DelayRates(arrivals, Exempt, flights):
    """CO2 kg per minute of delay: professor model in the air, APU at the gate."""
    import math
    rates = {}
    exempt = set(Exempt)
    for fid in flights:
        flight = arrivals.loc[fid]
        if fid in exempt:
            rate = air_co2_rate(flight)
        else:
            category = str(flight['Category']).strip().upper()
            if category not in apu_fuel:
                raise ValueError(f'Flight {fid}: unknown APU category {category!r}')
            rate = apu_fuel[category] * 3.16 / 60
        if not math.isfinite(rate) or rate < 0:
            raise ValueError(f'Flight {fid}: invalid CO2 rate: {rate}')
        rates[fid] = float(rate)
    return rates
