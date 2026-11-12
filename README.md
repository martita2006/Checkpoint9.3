# LIRF – MGTA modular project

Run `main.py`. Files: `config.py` (scenario), `data.py` (CSV parsing),
`gdp.py` (GDP), `ghp.py` (MILP), `emissions.py` (environmental model),
`costs.py` (economic model), `analysis.py` (KPIs/fairness),
`plots.py` (figure menu).

Install dependencies: `pip install -r requirements.txt`.
Set the CSV path in `config.py`; the original input must contain the columns
expected by `data.py` (including previously enriched fields for emissions).
Run `python main.py --plots none` without displaying figures,
or `python main.py --plots 1,3,7` to select them.
Outputs are saved to `results/`.

Economic coefficients and aircraft seating capacities are scenario assumptions;
validate these and emissions factors before reporting as real-world figures.


## WP3 Option B updated economic model
See `REFERENCES_WP3_COSTS.md`. Run `python test_costs_option_b.py` for local unit validation. Old results are preserved under `legacy_results_UNMODIFIED/` and MUST NOT be cited as new Eurocontrol model results. The raw LIRF traffic CSV is not present in this archive; update `config.py` CSV_NAME as necessary, and rerun `python main.py --plots none` once the CSV is available.

## Simplificación y explicación WP3
Las funciones núcleo simplificadas están en `ghp.py` y `costs.py`.
Consultar **WP3_FUNCIONES_CLAVE_EXPLICADAS.md** para una pestaña independiente con fórmulas, ubicación, código y comentarios por línea.
La regresión de optimización sintética está en `test_wp3_regression.py`.
