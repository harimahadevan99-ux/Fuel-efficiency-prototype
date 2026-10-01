"""
Synthetic construction-machine telemetry generator.

Ground-truth physics model (EPA NONROAD-style):
    fuel_rate_gal_per_hr = rated_hp * load_factor * BSFC / UNIT_FACTOR

Load factor depends on operating mode (idle / travel / dig / overload).
A small amount of gaussian noise is added to every signal to mimic sensor
noise, and two machines are deliberately given a "degraded" fuel curve
(as if they had a clogged filter / worn injector) so that later notebooks
have something real to detect via residual analysis.

Run:
    python -m src.simulation.generate_data --out data/telemetry.csv
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import numpy as np
import pandas as pd

MODES = ["idle", "travel", "dig", "overload"]
MODE_PROBS = [0.15, 0.25, 0.50, 0.10]
BASE_LOAD_FACTOR = {"idle": 0.12, "travel": 0.30, "dig": 0.55, "overload": 0.75}
BSFC = 0.37          # lb fuel per hp-hr, typical modern diesel
UNIT_FACTOR = 6.0    # lb/gal-ish lumped constant to land in realistic gal/hr range


@dataclass
class MachineSpec:
    machine_id: str
    rated_hp: float
    degradation_pct: float = 0.0   # 0.10 = burns 10% more fuel than it should


FLEET = [
    MachineSpec("EXC-101", rated_hp=250, degradation_pct=0.00),
    MachineSpec("EXC-102", rated_hp=250, degradation_pct=0.00),
    MachineSpec("EXC-103", rated_hp=300, degradation_pct=0.18),  # injector wear
    MachineSpec("EXC-104", rated_hp=180, degradation_pct=0.00),
    MachineSpec("EXC-105", rated_hp=180, degradation_pct=0.12),  # clogged filter
]


def simulate_machine(spec: MachineSpec, n_hours: int, rng: np.random.Generator) -> pd.DataFrame:
    mode = rng.choice(MODES, size=n_hours, p=MODE_PROBS)
    load_factor = np.array(
        [BASE_LOAD_FACTOR[m] + rng.normal(0, 0.03) for m in mode]
    ).clip(0.05, 0.95)

    rpm = 800 + load_factor * 1400 + rng.normal(0, 20, n_hours)
    hyd_pressure_bar = 50 + load_factor * 250 + rng.normal(0, 10, n_hours)
    ambient_temp_c = rng.normal(22, 8, n_hours)
    engine_hours = np.arange(n_hours) + rng.integers(500, 5000)

    true_fuel_rate = spec.rated_hp * load_factor * BSFC / UNIT_FACTOR
    degraded_fuel_rate = true_fuel_rate * (1 + spec.degradation_pct)
    fuel_rate_gal_hr = degraded_fuel_rate + rng.normal(0, 0.15, n_hours)
    fuel_rate_gal_hr = fuel_rate_gal_hr.clip(min=0.2)

    timestamps = pd.date_range("2026-01-01", periods=n_hours, freq="h")

    return pd.DataFrame({
        "timestamp": timestamps,
        "machine_id": spec.machine_id,
        "rated_hp": spec.rated_hp,
        "mode": mode,
        "load_factor": load_factor,
        "rpm": rpm,
        "hyd_pressure_bar": hyd_pressure_bar,
        "ambient_temp_c": ambient_temp_c,
        "engine_hours": engine_hours,
        "fuel_rate_gal_hr": fuel_rate_gal_hr,
    })


def generate_fleet_data(n_hours_per_machine: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [simulate_machine(spec, n_hours_per_machine, rng) for spec in FLEET]
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/telemetry.csv")
    parser.add_argument("--hours", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = generate_fleet_data(n_hours_per_machine=args.hours, seed=args.seed)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows for {df.machine_id.nunique()} machines to {args.out}")


if __name__ == "__main__":
    main()
