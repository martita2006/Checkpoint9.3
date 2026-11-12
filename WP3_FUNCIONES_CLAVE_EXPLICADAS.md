# WP3 — Guía de estudio del código simplificado

> Esta es una **pestaña/archivo independiente para estudiar**. No se ejecuta: el código funcional permanece en los módulos `.py`.

## Qué pide cada task y dónde está resuelta

| Task | Resultado que implementa el proyecto | Funciones/archivo |
|---|---|---|
| Task 1 | Optimizar retraso total; validación con coste unitario; límites al retraso aéreo y penalizaciones | `main.py` (WP3 Task 1), `ghp.py::solveGHP` |
| Task 2 | Optimizar emisiones por retraso y comparar con GDP / Task 1 | `emissions.py::co2DelayRates`, `airEmissions`, `groundEmissions`, `ghp.py::solveGHP`, `main.py` |
| Task 3 | Minimizar costes económicos con curvas EUROCONTROL y rotaciones | `costs.py::prepare_cost_data`, `_interpolate_eur`, `cost_components`, `ghp.py::solveGHP` |
| Task 4 | Comparar escenarios con KPI homogéneos | `analysis.py::evaluate_assignment`, `main.py` |
| Task 5 | Comparar la distribución de retrasos y costes entre compañías | `analysis.py::airline_comparison`, `main.py` |

**Alcance:** esta tabla refleja la implementación y los comentarios de las tasks en el ZIP aportado, no sustituye una revisión del enunciado original si pide más entregables.

## Fórmulas esenciales

1. Retraso: `d(f,s) = t_slot(s) − ETA(f)`, con `d ≥ 0`.
2. Decisión binaria: `x(f,s) ∈ {0,1}`; vale 1 si el vuelo f ocupa el slot s.
3. Un slot por vuelo: `Σ_s x(f,s) = 1`. Un vuelo como máximo por slot: `Σ_f x(f,s) ≤ 1`.
4. Objetivo Task 1: `min Σ_f,s c(f,s) x(f,s)` con `c=d·w_air` para exentos, `c=d` para controlados y penalización `(penalty−1)·max(0,d−limit)` para controlados.
5. Objetivo Task 2: `min Σ_f,s d(f,s)·r_CO2(f)·x(f,s)`; `r_air = fuel_air_kg/min · 3.16`; `r_ground = APU_fuel_kg/h · 3.16/60`.
6. Objetivo Task 3: `min Σ_f,s C(f,d(f,s))·x(f,s)` con `C=P+λ·q·max(0,F−P)`.
7. Reacción: `slack=max(0, turnaround−m)` y `q=min(1,max(0,d−slack)/max(1,m))` si se ha identificado rotación; si no, `q=0`.
8. KPI de dispersión: `RSD(%) = 100 · std(delays) / mean(delays)` si la media es positiva.

**Advertencia:** los valores económicos están en EUR de 2014; los 45 minutos de turnaround, las equivalencias ATYP y la regla q son hipótesis de escenario, no datos validados por EUROCONTROL.

---

## TASK 1/2/3 — Construcción y resolución del MILP
**Ubicación:** `ghp.py`, función `solveGHP()`, líneas 7–65.

**En general:** La función construye una variable binaria por combinación vuelo–slot, asigna su coste y resuelve el problema de asignación.

**Código real del proyecto simplificado:**

```python
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
```

**Comentario de cada línea útil:**

- **L7** `def solveGHP(arrivals, flights, Exempt, slot_times, max_air=None, limit=None,` — Define la función y sus argumentos; los argumentos controlan el escenario o los datos usados.
- **L8** `penalty=1, air_weight=1, emission_rates=None, cost_function=None):` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L11** `The mathematical model and feasible flight-slot combinations are unchanged.` — Continúa la expresión de Python iniciada en las líneas anteriores.
- **L13** `if emission_rates is not None and cost_function is not None:` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L14** `raise ValueError('Choose either emissions or economic cost objective.')` — Interrumpe el cálculo cuando no se puede construir una solución válida.
- **L15** `if not flights or not len(slot_times):` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L16** `raise ValueError('GHP needs affected flights and arrival slots.')` — Interrumpe el cálculo cuando no se puede construir una solución válida.
- **L18** `exempt = set(Exempt)` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L19** `pairs, costs = [], []` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L20** `for fid in flights:` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L21** `eta = arrivals.loc[fid, 'ETA_min']` — Lee la hora estimada de llegada del vuelo.
- **L22** `for slot_id, arrival_time in enumerate(slot_times):` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L23** `delay = arrival_time - eta` — Calcula minutos de retraso como llegada asignada menos ETA.
- **L24** `if delay < 0 or (fid in exempt and max_air is not None and delay > max_air):` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L25** `continue` — Descarta esta combinación y sigue con la siguiente.
- **L26** `if cost_function is not None:` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L27** `cost = float(cost_function(fid, delay))` — Calcula el coste de la asignación según el objetivo elegido.
- **L28** `elif emission_rates is not None:` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L29** `cost = delay * emission_rates[fid]` — Calcula el coste de la asignación según el objetivo elegido.
- **L30** `else:` — Usa la opción restante.
- **L31** `cost = delay * (air_weight if fid in exempt else 1)` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L32** `if fid not in exempt and limit is not None and delay > limit:` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L33** `cost += (penalty - 1) * (delay - limit)` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L34** `pairs.append((fid, slot_id))` — Guarda una opción concreta de vuelo y slot.
- **L35** `costs.append(cost)` — Añade el coeficiente de coste de esa combinación a la función objetivo.
- **L37** `if not np.isfinite(costs).all():` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L38** `raise ValueError('GHP objective has non-finite coefficients.')` — Interrumpe el cálculo cuando no se puede construir una solución válida.
- **L39** `if not pairs or set(flights) - {fid for fid, _ in pairs}:` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L40** `raise ValueError('Some flights have no feasible flight-slot assignment.')` — Interrumpe el cálculo cuando no se puede construir una solución válida.
- **L43** `flight_row = {fid: row for row, fid in enumerate(flights)}` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L44** `flights_matrix = lil_matrix((len(flights), len(pairs)))` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L45** `slots_matrix = lil_matrix((len(slot_times), len(pairs)))` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L46** `for col, (fid, slot_id) in enumerate(pairs):` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L47** `flights_matrix[flight_row[fid], col] = 1` — Asocia un identificador real de vuelo con su fila en la matriz de restricciones.
- **L48** `slots_matrix[slot_id, col] = 1` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L50** `constraints = [` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L51** `LinearConstraint(flights_matrix.tocsr(), 1, 1),` — Crea la restricción lineal que se usará en el modelo MILP.
- **L52** `LinearConstraint(slots_matrix.tocsr(), 0, 1),` — Crea la restricción lineal que se usará en el modelo MILP.
- **L54** `result = milp(c=np.asarray(costs), constraints=constraints,` — Resuelve la programación lineal entera con variables binarias.
- **L55** `integrality=np.ones(len(pairs)), bounds=Bounds(0, 1))` — Fija variables enteras (junto con límites 0–1 son binarias).
- **L56** `if not result.success:` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L57** `print('GHP: no feasible solution found ->', result.message)` — Continúa la expresión de Python iniciada en las líneas anteriores.
- **L58** `return None, None` — Devuelve el resultado calculado al módulo que llamó a esta función.
- **L60** `slots = np.array([[time, 0, 0] for time in slot_times], dtype=object)` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L61** `for selected, (fid, slot_id) in zip(result.x, pairs):` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L62** `if selected > 0.5:` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L63** `slots[slot_id, 1] = fid` — Rellena en la solución el vuelo y su compañía en el slot asignado.
- **L64** `slots[slot_id, 2] = arrivals.loc[fid, 'Compañía']` — Rellena en la solución el vuelo y su compañía en el slot asignado.
- **L65** `return slots, result.fun` — Devuelve el resultado calculado al módulo que llamó a esta función.

---

## TASK 2 — Coeficiente de CO₂ por minuto
**Ubicación:** `emissions.py`, función `co2DelayRates()`, líneas 166–183.

**En general:** Transforma combustible en kg de CO₂ por minuto. En el aire utiliza la tasa por tipo de aeronave; en tierra el consumo de APU.

**Código real del proyecto simplificado:**

```python
def co2DelayRates(arrivals, Exempt, flights):
    """Consistent marginal CO2 kg/min: EUROCONTROL arrival management or ground APU."""
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
```

**Comentario de cada línea útil:**

- **L166** `def co2DelayRates(arrivals, Exempt, flights):` — Define la función y sus argumentos; los argumentos controlan el escenario o los datos usados.
- **L168** `import math` — Importa las herramientas de cálculo que necesita la función.
- **L169** `rates = {}` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L170** `exempt = set(Exempt)` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L171** `for fid in flights:` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L172** `flight = arrivals.loc[fid]` — Selecciona filas concretas de una tabla mediante sus identificadores.
- **L173** `if fid in exempt:` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L174** `rate = air_co2_rate(flight)` — Calcula la tasa de emisiones de CO₂ en kg/min.
- **L175** `else:` — Usa la opción restante.
- **L176** `category = str(flight['Category']).strip().upper()` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L177** `if category not in apu_fuel:` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L178** `raise ValueError(f'Flight {fid}: unknown APU category {category!r}')` — Interrumpe el cálculo cuando no se puede construir una solución válida.
- **L179** `rate = apu_fuel[category] * 3.16 / 60` — Calcula la tasa de emisiones de CO₂ en kg/min.
- **L180** `if not math.isfinite(rate) or rate < 0:` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L181** `raise ValueError(f'Flight {fid}: invalid CO2 rate: {rate}')` — Interrumpe el cálculo cuando no se puede construir una solución válida.
- **L182** `rates[fid] = float(rate)` — Guarda la tasa correspondiente al identificador de vuelo.
- **L183** `return rates` — Devuelve el resultado calculado al módulo que llamó a esta función.

---

## TASK 2 — CO₂ durante la espera aérea
**Ubicación:** `emissions.py`, función `airEmissions()`, líneas 138–152.

**En general:** Suma combustible y emisiones de los minutos de retraso en vuelo.

**Código real del proyecto simplificado:**

```python
def airEmissions(arrivals, AirDelay):
    """Delay emissions in air; CO2 uses arrival fuel rates, other species are estimates."""
    fuel, co2, nox, sox, h2o, co = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for fid, delay in AirDelay:
        flight = arrivals.loc[fid]
        d = float(delay)
        fuel_for_delay = air_co2_rate(flight) / 3.16 * d
        fuel += fuel_for_delay
        co2 += fuel_for_delay * 3.16
        # Other pollutants: preliminary proportional estimates, not calibrated.
        nox += fuel_for_delay * 0.015
        sox += fuel_for_delay * 0.00084
        h2o += fuel_for_delay * 1.237
        co += fuel_for_delay * 0.0105
    return fuel, co2, nox, sox, h2o, co
```

**Comentario de cada línea útil:**

- **L138** `def airEmissions(arrivals, AirDelay):` — Define la función y sus argumentos; los argumentos controlan el escenario o los datos usados.
- **L140** `fuel, co2, nox, sox, h2o, co = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L141** `for fid, delay in AirDelay:` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L142** `flight = arrivals.loc[fid]` — Selecciona filas concretas de una tabla mediante sus identificadores.
- **L143** `d = float(delay)` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L144** `fuel_for_delay = air_co2_rate(flight) / 3.16 * d` — Calcula minutos de retraso como llegada asignada menos ETA.
- **L145** `fuel += fuel_for_delay` — Acumula el combustible consumido.
- **L146** `co2 += fuel_for_delay * 3.16` — Acumula CO₂ de este vuelo.
- **L148** `nox += fuel_for_delay * 0.015` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L149** `sox += fuel_for_delay * 0.00084` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L150** `h2o += fuel_for_delay * 1.237` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L151** `co += fuel_for_delay * 0.0105` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L152** `return fuel, co2, nox, sox, h2o, co` — Devuelve el resultado calculado al módulo que llamó a esta función.

---

## TASK 2 — CO₂ en tierra
**Ubicación:** `emissions.py`, función `groundEmissions()`, líneas 154–164.

**En general:** Suma los consumos APU por categoría y los convierte a emisiones.

**Código real del proyecto simplificado:**

```python
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
```

**Comentario de cada línea útil:**

- **L154** `def groundEmissions(arrivals, GroundDelay):` — Define la función y sus argumentos; los argumentos controlan el escenario o los datos usados.
- **L155** `fuel = 0.0` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L156** `for id, delay in GroundDelay:` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L157** `category = str(arrivals.loc[id, 'Category']).strip().upper()` — Selecciona filas concretas de una tabla mediante sus identificadores.
- **L158** `fuel += apu_fuel.get(category) * delay / 60` — Acumula el combustible consumido.
- **L159** `co2 = fuel * 3.16` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L160** `h2o = fuel * 1.237` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L161** `sox = fuel * 0.00084` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L162** `nox = fuel * 0.0022` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L163** `co = fuel * 0.0105` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L164** `return fuel, co2, nox, sox, h2o, co` — Devuelve el resultado calculado al módulo que llamó a esta función.

---

## TASK 3 — Interpolación EUROCONTROL
**Ubicación:** `costs.py`, función `_interpolate_eur()`, líneas 123–134.

**En general:** Interpola el coste total de un retraso entre los tiempos de referencia; después de 300 minutos prolonga la última pendiente.

**Código real del proyecto simplificado:**

```python
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
```

**Comentario de cada línea útil:**

- **L123** `def _interpolate_eur(values, delay_min):` — Define la función y sus argumentos; los argumentos controlan el escenario o los datos usados.
- **L126** `Monotonicity ensured on the source curves; zero delay costs zero.` — Continúa la expresión de Python iniciada en las líneas anteriores.
- **L127** `The last segment is an explicit extrapolation assumption.` — Continúa la expresión de Python iniciada en las líneas anteriores.
- **L129** `d = max(0.0, float(delay_min))` — Impone un límite inferior al cálculo para que no se considere negativo.
- **L130** `y = np.array([0.0] + list(values), dtype=float)` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L131** `if d <= DELAY_KNOTS_MIN[-1]:` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L132** `return float(np.interp(d, DELAY_KNOTS_MIN, y))` — Devuelve el resultado calculado al módulo que llamó a esta función.
- **L133** `slope = (y[-1] - y[-2]) / (DELAY_KNOTS_MIN[-1] - DELAY_KNOTS_MIN[-2])` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L134** `return float(y[-1] + (d - DELAY_KNOTS_MIN[-1]) * slope)` — Devuelve el resultado calculado al módulo que llamó a esta función.

---

## TASK 3 — Emparejar aeronaves y rotaciones
**Ubicación:** `costs.py`, función `prepare_cost_data()`, líneas 137–172.

**En general:** Obtiene aeronave equivalente y siguiente salida local por matrícula para calcular holgura de rotación.

**Código real del proyecto simplificado:**

```python
def prepare_cost_data(arrivals, csv_path, exempt_ids=None):
    """Aircraft reference types and *observed* next-rotation turnaround by RM.

    Does not fabricate passenger counts: reference curves already include
    representative passenger / connection costs. RM matching is restricted to
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
    print('EUROCONTROL aircraft proxy count:',int(out.reference_is_proxy.sum()), '/',len(out))
    print('Following rotation identified:',int(out.turnaround_min.notna().sum()), '/',len(out))
    return out
```

**Comentario de cada línea útil:**

- **L137** `def prepare_cost_data(arrivals, csv_path, exempt_ids=None):` — Define la función y sus argumentos; los argumentos controlan el escenario o los datos usados.
- **L140** `Does not fabricate passenger counts: reference curves already include` — Continúa la expresión de Python iniciada en las líneas anteriores.
- **L141** `representative passenger / connection costs. RM matching is restricted to` — Continúa la expresión de Python iniciada en las líneas anteriores.
- **L142** `next local departure later than arrival (day-crossing not reconstructed).` — Continúa la expresión de Python iniciada en las líneas anteriores.
- **L144** `from data import parse_clock_minutes` — Importa las herramientas de cálculo que necesita la función.
- **L145** `df = pd.read_csv(csv_path, sep=';')` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L146** `out = pd.DataFrame(index=arrivals.index)` — Crea una tabla para organizar cálculos y exportaciones.
- **L147** `exempt = set(exempt_ids or [])` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L148** `out['is_exempt'] = [fid in exempt for fid in arrivals.index]` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L149** `out['ATYP'] = arrivals['ATYP'].astype('string').str.strip().str.upper()` — Crea una columna auxiliar con datos de aeronave o rotación.
- **L150** `out['reference_aircraft'] = [ATYP_TO_REFERENCE.get(str(typ), RECAT_TO_REFERENCE.get(str(cat).strip().upper(), 'A320'))` — Crea una columna auxiliar con datos de aeronave o rotación.
- **L151** `for typ, cat in zip(out['ATYP'], arrivals['Category'])]` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L152** `out['reference_is_proxy'] = [str(typ) != ref for typ, ref in zip(out['ATYP'], out['reference_aircraft'])]` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L153** `out['next_departure_min'] = np.nan` — Crea una columna auxiliar con datos de aeronave o rotación.
- **L154** `rm_col = next((c for c in ['RM', 'Registration', 'Registration mark'] if c in df and c in arrivals), None)` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L155** `if rm_col and 'ADEP' in df and 'ETD' in df:` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L156** `dep = df.loc[df['ADEP'].astype(str).str.strip().str.upper().eq('LIRF')].copy()` — Selecciona filas concretas de una tabla mediante sus identificadores.
- **L157** `if not dep.empty:` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L158** `dep['departure_min'] = parse_clock_minutes(dep['ETD'], 'ETD_departures')` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L159** `dep['rm_key'] = dep[rm_col].astype(str).str.strip().str.upper()` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L160** `groups = {rm: sorted(g['departure_min'].astype(float)) for rm,g in dep.groupby('rm_key')` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L161** `if rm not in ('', 'NAN', 'NONE')}` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L162** `for fid, flight in arrivals.iterrows():` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L163** `rm = str(flight[rm_col]).strip().upper()` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L164** `eta = float(flight['ETA_min'])` — Lee la hora estimada de llegada del vuelo.
- **L165** `times = groups.get(rm, []) if rm not in ('','NAN','NONE') else []` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L166** `candidates = [t for t in times if t > eta]` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L167** `if candidates:` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L168** `out.loc[fid, 'next_departure_min'] = min(candidates)` — Selecciona filas concretas de una tabla mediante sus identificadores.
- **L169** `out['turnaround_min'] = out['next_departure_min'] - arrivals['ETA_min']` — Crea una columna auxiliar con datos de aeronave o rotación.
- **L170** `print('EUROCONTROL aircraft proxy count:',int(out.reference_is_proxy.sum()), '/',len(out))` — Suma valores por vuelo o por escenario.
- **L171** `print('Following rotation identified:',int(out.turnaround_min.notna().sum()), '/',len(out))` — Suma valores por vuelo o por escenario.
- **L172** `return out` — Devuelve el resultado calculado al módulo que llamó a esta función.

---

## TASK 3 — Coste primario y reaccionario
**Ubicación:** `costs.py`, función `cost_components()`, líneas 175–198.

**En general:** Aplica las curvas publicadas y pondera la diferencia full−primary por la holgura disponible.

**Código real del proyecto simplificado:**

```python
def cost_components(fid, delay_min, data, reactionary_multiplier=REACTIONARY_MULTIPLIER,
                    min_turnaround=TURNAROUND_MIN_REQUIRED):
    """Primary cost + conditional share of published reactionary cost; EUR_2014.

    Reactionary share ramps over delay exceeding turnaround slack. This scaling
    is a transparent, sensitivity-testable model assumption (not EUROCONTROL).
    No separately-added pax/connection/aircraft-rotation/long-delay charge:
    these mechanisms already contribute to the source curves.
    """
    d = max(0.0, float(delay_min))
    ref = str(data.loc[fid, 'reference_aircraft'])
    phase = 'AIR' if bool(data.loc[fid, 'is_exempt']) else 'GATE'
    primary = _interpolate_eur(TABLES_EUR2014[phase + '_PRIMARY'][ref], d)
    full = _interpolate_eur(TABLES_EUR2014[phase + '_FULL'][ref], d)
    turnaround = float(data.loc[fid, 'turnaround_min'])
    if not np.isfinite(turnaround):
        reaction_fraction = 0.0  # unknown RM: no observed same-aircraft reactionary term
    else:
        slack = max(0.0, turnaround - float(min_turnaround))
        # Zero below slack; progressive thereafter; capped at the published full increment.
        reaction_fraction = min(1.0, max(0.0, d - slack) / max(1.0, float(min_turnaround)))
    reactionary = max(0.0, full - primary) * reaction_fraction * float(reactionary_multiplier)
    return {'primary_eur2014':primary,'reactionary_eur2014':reactionary,
            'reaction_fraction':reaction_fraction,'total_eur2014':primary+reactionary}
```

**Comentario de cada línea útil:**

- **L175** `def cost_components(fid, delay_min, data, reactionary_multiplier=REACTIONARY_MULTIPLIER,` — Define la función y sus argumentos; los argumentos controlan el escenario o los datos usados.
- **L176** `min_turnaround=TURNAROUND_MIN_REQUIRED):` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L179** `Reactionary share ramps over delay exceeding turnaround slack. This scaling` — Continúa la expresión de Python iniciada en las líneas anteriores.
- **L180** `is a transparent, sensitivity-testable model assumption (not EUROCONTROL).` — Continúa la expresión de Python iniciada en las líneas anteriores.
- **L181** `No separately-added pax/connection/aircraft-rotation/long-delay charge:` — Continúa la expresión de Python iniciada en las líneas anteriores.
- **L182** `these mechanisms already contribute to the source curves.` — Continúa la expresión de Python iniciada en las líneas anteriores.
- **L184** `d = max(0.0, float(delay_min))` — Impone un límite inferior al cálculo para que no se considere negativo.
- **L185** `ref = str(data.loc[fid, 'reference_aircraft'])` — Selecciona filas concretas de una tabla mediante sus identificadores.
- **L186** `phase = 'AIR' if bool(data.loc[fid, 'is_exempt']) else 'GATE'` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L187** `primary = _interpolate_eur(TABLES_EUR2014[phase + '_PRIMARY'][ref], d)` — Coste primario EUROCONTROL correspondiente a la fase y aeronave.
- **L188** `full = _interpolate_eur(TABLES_EUR2014[phase + '_FULL'][ref], d)` — Coste completo EUROCONTROL para la misma fase y aeronave.
- **L189** `turnaround = float(data.loc[fid, 'turnaround_min'])` — Obtiene el intervalo observado entre llegada y siguiente salida.
- **L190** `if not np.isfinite(turnaround):` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L191** `reaction_fraction = 0.0  # unknown RM: no observed same-aircraft reactionary term` — Fracción entre cero y uno del coste reaccionario atribuida a este retraso.
- **L192** `else:` — Usa la opción restante.
- **L193** `slack = max(0.0, turnaround - float(min_turnaround))` — Holgura de rotación: margen hasta el siguiente vuelo menos el turnaround mínimo.
- **L195** `reaction_fraction = min(1.0, max(0.0, d - slack) / max(1.0, float(min_turnaround)))` — Fracción entre cero y uno del coste reaccionario atribuida a este retraso.
- **L196** `reactionary = max(0.0, full - primary) * reaction_fraction * float(reactionary_multiplier)` — Coste reaccionario añadido: diferencia full-primary por fracción y multiplicador.
- **L197** `return {'primary_eur2014':primary,'reactionary_eur2014':reactionary,` — Devuelve el resultado calculado al módulo que llamó a esta función.
- **L198** `'reaction_fraction':reaction_fraction,'total_eur2014':primary+reactionary}` — Continúa la expresión de Python iniciada en las líneas anteriores.

---

## TASK 4 — KPIs comparables
**Ubicación:** `analysis.py`, función `evaluate_assignment()`, líneas 6–39.

**En general:** Calcula con las mismas reglas el retraso, CO₂ y coste de cada escenario.

**Código real del proyecto simplificado:**

```python
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
```

**Comentario de cada línea útil:**

- **L6** `def evaluate_assignment(name, arrivals, cta, flights, exempt, cost_data, rates=None):` — Define la función y sus argumentos; los argumentos controlan el escenario o los datos usados.
- **L7** `flights = list(flights)` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L8** `exempt = set(exempt)` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L9** `rates = rates if rates is not None else co2DelayRates(arrivals, exempt, flights)` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L10** `frame = pd.DataFrame(index=flights)` — Crea una tabla para organizar cálculos y exportaciones.
- **L11** `frame['ARCID'] = arrivals.loc[flights, 'ARCID']` — Crea o rellena una columna con la magnitud indicada.
- **L12** `frame['Airline'] = arrivals.loc[flights, 'Compañía'].astype(str)` — Crea o rellena una columna con la magnitud indicada.
- **L13** `frame['Status'] = ['Exempt' if f in exempt else 'Controlled' for f in flights]` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L14** `frame['ETA_min'] = arrivals.loc[flights, 'ETA_min']` — Crea o rellena una columna con la magnitud indicada.
- **L15** `frame['CTA_min'] = cta.loc[flights].astype(float)` — Crea o rellena una columna con la magnitud indicada.
- **L16** `frame['Delay_min'] = (frame.CTA_min - frame.ETA_min).clip(lower=0)` — Crea o rellena una columna con la magnitud indicada.
- **L17** `if ((frame.CTA_min - frame.ETA_min) < -1e-6).any():` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L18** `raise ValueError(f'{name}: some aircraft arrive before ETA')` — Interrumpe el cálculo cuando no se puede construir una solución válida.
- **L19** `frame['AirDelay_min'] = np.where(frame.Status == 'Exempt', frame.Delay_min, 0.0)` — Crea o rellena una columna con la magnitud indicada.
- **L20** `frame['GroundDelay_min'] = np.where(frame.Status == 'Controlled', frame.Delay_min, 0.0)` — Crea o rellena una columna con la magnitud indicada.
- **L21** `frame['CO2_kg'] = [frame.loc[f, 'Delay_min'] * rates[f] for f in flights]` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L22** `costs = costs_for_assignment(arrivals, cta, flights, cost_data)` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L23** `frame['EconomicCost_EUR'] = [costs[f] for f in flights]` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L24** `delay = frame.Delay_min.to_numpy()` — Calcula minutos de retraso como llegada asignada menos ETA.
- **L25** `kpis = {` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L26** `'Scenario': name, 'Flights': len(frame), 'Delayed flights': int((delay > 1e-7).sum()),` — Suma valores por vuelo o por escenario.
- **L27** `'Total delay min': float(delay.sum()), 'Mean delay min': float(delay.mean()),` — Suma valores por vuelo o por escenario.
- **L28** `'Max delay min': float(delay.max()), 'Delay std min': float(delay.std()),` — Impone un límite inferior al cálculo para que no se considere negativo.
- **L29** `'Delay RSD %': float(100*delay.std()/delay.mean()) if delay.mean() else 0.0,` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L30** `'Air delay min': float(frame.AirDelay_min.sum()),` — Suma valores por vuelo o por escenario.
- **L31** `'Max air delay min': float(frame.AirDelay_min.max()),` — Impone un límite inferior al cálculo para que no se considere negativo.
- **L32** `'Ground delay min': float(frame.GroundDelay_min.sum()),` — Suma valores por vuelo o por escenario.
- **L33** `'Max ground delay min': float(frame.GroundDelay_min.max()),` — Impone un límite inferior al cálculo para que no se considere negativo.
- **L34** `'CO2 kg': float(frame.CO2_kg.sum()),` — Suma valores por vuelo o por escenario.
- **L35** `'Economic cost EUR': float(frame.EconomicCost_EUR.sum()),` — Suma valores por vuelo o por escenario.
- **L36** `'Cost / flight EUR': float(frame.EconomicCost_EUR.mean()),` — Calcula una media aritmética.
- **L37** `'Max individual cost EUR': float(frame.EconomicCost_EUR.max()),` — Impone un límite inferior al cálculo para que no se considere negativo.
- **L39** `return frame, kpis` — Devuelve el resultado calculado al módulo que llamó a esta función.

---

## TASK 5 — Equidad por aerolínea
**Ubicación:** `analysis.py`, función `airline_comparison()`, líneas 41–55.

**En general:** Compara retrasos, CO₂ y costes de las cuatro compañías con más vuelos afectados.

**Código real del proyecto simplificado:**

```python
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
```

**Comentario de cada línea útil:**

- **L41** `def airline_comparison(scenarios, top_n=4):` — Define la función y sus argumentos; los argumentos controlan el escenario o los datos usados.
- **L43** `baseline = next(iter(scenarios.values()))` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L44** `top_airlines = baseline['Airline'].value_counts().head(top_n).index` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L45** `rows = []` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L46** `for name, frame in scenarios.items():` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L47** `for airline in top_airlines:` — Recorre los elementos indicados; cada repetición agrega variables o resultados.
- **L48** `group = frame.loc[frame['Airline'] == airline]` — Crea o rellena una columna con la magnitud indicada.
- **L49** `delays = group.Delay_min.to_numpy(dtype=float)` — Guarda o actualiza el valor indicado a la izquierda a partir de la expresión de la derecha.
- **L50** `mean = delays.mean() if len(delays) else 0.0` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L51** `rows.append({'Scenario': name, 'Airline': airline, 'Flights': len(delays),` — Continúa la expresión de Python iniciada en las líneas anteriores.
- **L52** `'Mean delay min': mean, 'Delay RSD %': 100*delays.std()/mean if mean else 0,` — Evalúa una condición que selecciona la fórmula o impone un requisito.
- **L53** `'Total delay min': delays.sum(), 'CO2 kg': group.CO2_kg.sum(),` — Suma valores por vuelo o por escenario.
- **L54** `'Economic cost EUR': group.EconomicCost_EUR.sum()})` — Suma valores por vuelo o por escenario.
- **L55** `return pd.DataFrame(rows)` — Devuelve el resultado calculado al módulo que llamó a esta función.

---

## Dónde están las llamadas a las Tasks en `main.py`

El bloque de WP3 comienza en la marca `# Checkpoint 7 (WP3, task 1)`; después aparecen las marcas `WP3 TASK 2`, `WP3 TASK 3` y `WP3 TASKS 4-5`. Esas llamadas conectan las fórmulas anteriores con los datos y guardan los CSV de resultados.

## Qué se mantiene intacto

- Las curvas y los puntos publicados en `costs.py`.
- Los factores de emisiones y el tratamiento diferenciado de exentos/controlados.
- La formulación MILP y los objetivos (incluida la penalización en tierra).
- Los KPI de comparación, los archivos de resultados y los gráficos.

## Limitaciones de validación

- Falta el CSV original de tráfico LIRF. No se ha ejecutado un resultado end-to-end sobre el día real.
- Hay pruebas de regresión sintéticas y unitarias. No garantizan identidad de la asignación cuando hay óptimos empatados; sí comparan su valor objetivo.
- El fichero `REFERENCES_WP3_COSTS.md` documenta la procedencia y las hipótesis de los datos de costes.
