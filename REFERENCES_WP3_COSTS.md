# Task 3 cost model: EUROCONTROL option B (2014 EUR)

## Main source
Cook & Tanner (2015), *European airline delay cost reference values: updated and extended values*, version 4.1, University of Westminster for EUROCONTROL PRU. https://www.eurocontrol.int/sites/default/files/publication/files/european-airline-delay-cost-reference-values-final-report-4-1.pdf

These tables are costs **for a whole delay of the specified number of minutes**, not EUR per minute. The top of each curve is 300 min; extrapolation above is a modelling assumption. All values are **2014 EUR**, without inflation adjustment.

- Table **22**: At-gate BASE primary tactical cost, excluding reactionary costs.
- Table **25**: Arrival management BASE primary tactical cost, excluding reactionary costs.
- Table **26**: At-gate BASE full tactical cost, including reactionary costs.
- Table **29**: Arrival management BASE full tactical cost, including reactionary costs.
- Knots: 0 (added as zero), **5, 15, 30, 60, 90, 120, 180, 240, 300 minutes** (published).

## Cost function
For flight f, delay d, phase p (ground at-gate for controlled; arrival management for exempt), and representative aircraft a:

P(f,d) = interpolated published primary tactical delay cost.
R(f,d) = max(0, interpolated full cost - P(f,d)).
S(f) = max(0, observed same-registration next departure ETD - ETA - m), m=45 min assumption.
q(f,d) = clip((d-S(f))/m, 0, 1) for observed rotation, else 0.
**C(f,d) = P(f,d) + lambda * q(f,d) * R(f,d)**, lambda=1.0 baseline.

Thus passenger cost, connections, APU, and delay-duration dependence are embedded in the published curves; the old 0.1 €/pax/min, 200 €/connection, 20 €/min reactionary and 0.015 quadratic coefficient are **removed**, not added again. Connection share/load factor removed from operational cost path; the underlying published reference study uses its own typical passenger profiles, not your flight manifests.

## Assumptions to defend (NOT sourced constants)
- **m=45 min** minimum required turnaround, scenario assumption; not an observed minimum. EUROCONTROL sources document reactionary delay but do not prove 45 min.
- **lambda=1.0** baseline of potential published reactionary increment, bounded by 1; sensitivity suggested 0, 0.5, 1.0.
- **q** ramp scale `m`; modelling assumption to link actual RM turnaround to reactionary increment. This is *not* an empirically calibrated probability.
- **ATYP_TO_REFERENCE** mappings for types not explicitly in reference tables, e.g. A20N→A320, A359→A332, B789→B763. These are proxies and may have major error, especially wide-body/ultra-large aircraft. `reference_is_proxy` lets you quantify coverage.
- GroundDelay approximated as **AT-GATE**; AirDelay approximated as **ARRIVAL MANAGEMENT**. Actual delays may occur in other phases.
- `np.interp` between knots, linear extension beyond 300 minutes. No published values above 300 in these reference tables.
- **2014 EUR** unadjusted; comparing 2025 operational day against reference historic prices is only a model benchmarking exercise, not 2025/2026 airline expenditures.
- For flights without matched next rotation, **q=0** rather than guessing a same-aircraft reactionary cost; network passenger/crew reactionary delay is then omitted.

## Recent research context
- Gasparin et al. (2025), *A User-Driven Prioritisation Process implementation and optimisation for ATFM hotspot resolution*, Transportation Research Part C, https://doi.org/10.1016/j.trc.2024.104894 — costs per flight and prioritisation.
- Li et al. (2025), *A Review of Research on Flight Delay Propagation*, https://doi.org/10.1155/atr/4851103 — propagation mechanisms.
- EUROCONTROL (2021), *Data Snapshot #20 - Recovering flights and increasing delays*, https://www.eurocontrol.int/publication/eurocontrol-data-snapshot-20-recovering-flights-and-increasing-delays — rotation/propagation context.

## Sensitivity recommended by WP3 slides
Test m ∈ {30,45,60}, lambda ∈ {0,0.5,1} and alternative aircraft proxies, then recompute the Task 3 optimal assignment and Tasks 4/5 KPI. These are **proposed tests**, not empirical credible intervals.

## Comparability
GDP, GHP Task1, GHP CO2 and GHP Cost are evaluated with the **same** cost function after modification. The previous monetary values from the invented four-component model are not comparable to the new EUR_2014 values. Do not reuse them in the presentation.

## Checkpoint 8 update: passengers per flight
Slide 12 of the WP3-4 presentation asks to use passengers and connecting passengers. The primary cost P(f,d) is multiplied by a passenger weight w(f):
- passengers(f) = available seats x load factor, load factor = 0.841 (IATA, European airlines 2024, press release 30 January 2025: https://www.iata.org/en/pressroom/2025-releases/2025-01-30-01/).
- ITA Airways (ITY, hub carrier at LIRF) passengers count (1 + s), s = 0.30 connecting share. This value is an assumption (no public source found); it is tested with s = 0 and s = 0.5.
- w(f) = passengers(f)(1 + s_f) / mean over all arrivals, so the mean weight is 1 and the overall cost level of the EUROCONTROL curves is kept.
Final cost: C(f,d) = w(f) P(f,d) + lambda q(f,d) R(f,d). The sensitivity results (s = 0, 0.3, 0.5 and lambda = 0, 0.5, 1) are saved in results/wp3_task3_sensitivity.csv.
