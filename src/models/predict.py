"""
Score machines by how much MORE fuel they burn than the model expects for
their operating conditions. This residual is the actually-actionable output:
a persistently high residual points at a maintenance issue (clogged filter,
worn injector, misaligned track) rather than just "this machine works hard".

Run:
    python -m src.models.predict --data data/telemetry.csv --model models_artifacts/fuel_model.joblib
"""
from __future__ import annotations
import argparse

import joblib
import pandas as pd

from src.features.build_features import FEATURE_COLUMNS, TARGET_COLUMN, build_features


def score_fleet(df: pd.DataFrame, model) -> pd.DataFrame:
    df_feat = build_features(df)
    df_feat["predicted_fuel_rate_gal_hr"] = model.predict(df_feat[FEATURE_COLUMNS])
    df_feat["residual_gal_hr"] = df_feat[TARGET_COLUMN] - df_feat["predicted_fuel_rate_gal_hr"]
    df_feat["residual_pct"] = 100 * df_feat["residual_gal_hr"] / df_feat["predicted_fuel_rate_gal_hr"]

    summary = (
        df_feat.groupby("machine_id")
        .agg(
            avg_actual_fuel=(TARGET_COLUMN, "mean"),
            avg_predicted_fuel=("predicted_fuel_rate_gal_hr", "mean"),
            avg_residual_pct=("residual_pct", "mean"),
            idle_pct=("mode", lambda s: (s == "idle").mean() * 100),
            hours_logged=(TARGET_COLUMN, "count"),
        )
        .round(2)
        .sort_values("avg_residual_pct", ascending=False)
    )
    return df_feat, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/telemetry.csv")
    parser.add_argument("--model", default="models_artifacts/fuel_model.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    model = joblib.load(args.model)
    _, summary = score_fleet(df, model)

    print("Fleet Fuel Efficiency Ranking (worst offenders first)\n")
    print(summary.to_string())
    print(
        "\nInterpretation: avg_residual_pct near 0 = burning fuel as expected "
        "for its workload.\nMachines well above 0 are strong candidates for a "
        "maintenance inspection (filters, injectors, tracks)."
    )


if __name__ == "__main__":
    main()
