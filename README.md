# Fleet Fuel Efficiency Prototype

![tests](https://github.com/yourusername/fuel-efficiency-prototype/actions/workflows/tests.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-blue.svg)
![python](https://img.shields.io/badge/python-3.11%2B-blue.svg)

A GitHub-prototypable ML pipeline that predicts expected fuel consumption for
construction machines based on their operating conditions, then flags
machines burning **more fuel than expected** — a leading indicator of
maintenance issues (clogged filters, worn injectors, misaligned tracks)
long before they cause a breakdown.

This is a **learning/portfolio prototype**: it runs entirely on synthetic
telemetry generated from a physics-based fuel model (EPA NONROAD-style:
`fuel_rate = rated_hp * load_factor * BSFC`), so it needs no proprietary
data or live machine access to build and test end-to-end.

## Why this matters

Fuel is one of the largest controllable operating costs on a jobsite. Two
identical machines under identical workload should burn roughly the same
fuel — when one doesn't, that gap (the "residual") is actionable
information, not noise.

## Architecture

```
generate_data.py  -->  telemetry.csv  -->  build_features.py  -->  train.py  -->  fuel_model.joblib
                                                                                        |
                                                                                        v
                                                                    predict.py / dashboard/app.py
                                                                    (fleet ranking + drill-down)
```

- **Group-aware validation**: the model is evaluated on machines it never
  saw during training (not just random rows), because the real use case is
  predicting fuel for a newly onboarded machine.
- **Two machines are deliberately degraded** in the simulator (injector
  wear, clogged filter) so the residual-ranking logic has something real
  to catch — check that your pipeline finds them before trusting it on
  real data.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate synthetic fleet telemetry
python -m src.simulation.generate_data --out data/telemetry.csv

# 2. Train the model (baseline Linear Regression + Random Forest)
python -m src.models.train --data data/telemetry.csv --out models_artifacts/

# 3. See which machines are burning more fuel than expected
python -m src.models.predict --data data/telemetry.csv --model models_artifacts/fuel_model.joblib

# 4. Interactive dashboard
streamlit run src/dashboard/app.py

# Run tests
pytest -v
```

## Results

Trained on 5 synthetic machines (5,000 machine-hours), validated on 2 machines
held out entirely from training (group-aware split, not a random row split):

| Model | MAE (gal/hr) | RMSE (gal/hr) | R² |
|---|---|---|---|
| Linear Regression (baseline) | 0.74 | 0.90 | 0.89 |
| Random Forest | **0.35** | **0.47** | **0.97** |

The fleet efficiency ranking correctly surfaced the held-out machine with
injected fuel-system degradation as the top offender (+11.6% residual vs.
expected burn). **Limitation worth knowing:** a degraded machine that *was*
part of the training set gets "explained away" rather than flagged, because
the model learns its specific behavior as normal — catching degradation
that *develops over a machine's own lifetime* needs a per-machine
early-life-vs-recent comparison, which is a natural next iteration.

## Moving from synthetic to real data

Keep the exact schema in `data/telemetry.csv`
(`timestamp, machine_id, rated_hp, mode, load_factor, rpm, hyd_pressure_bar,
ambient_temp_c, engine_hours, fuel_rate_gal_hr`) and swap the generator for
a real telemetry export you're authorized to use (e.g. a J1939 CAN log
parsed with `python-can`, or an OEM telematics API export). No other code
should need to change.

**Do not** feed this pipeline proprietary employer telematics data or
internal fuel models — use only data you own or are explicitly authorized
to use for a personal/portfolio project.

## Next steps (see the full project roadmap in conversation)

- Add SHAP-based explainability for individual predictions
- Add an idle-time reduction rule engine on top of the `idle_pct_last_24h` feature
- Feed the fuel model into the truck–excavator dispatch simulation
