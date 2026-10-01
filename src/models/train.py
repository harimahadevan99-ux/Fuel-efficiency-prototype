"""
Train a fuel-consumption regression model.

Important design choice: we split by machine_id (GroupShuffleSplit), not
randomly by row. Random row splits let the model "cheat" by memorizing a
machine's specific fuel curve; a group split forces it to prove it can
generalize to a machine it has never seen -- which is the actual real-world
use case (predicting fuel for a newly onboarded machine).

Run:
    python -m src.models.train --data data/telemetry.csv --out models_artifacts/
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

from src.features.build_features import FEATURE_COLUMNS, get_feature_target_split


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "MAE_gal_hr": round(mean_absolute_error(y_test, preds), 4),
        "RMSE_gal_hr": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
        "R2": round(r2_score(y_test, preds), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/telemetry.csv")
    parser.add_argument("--out", default="models_artifacts/")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=["timestamp"])
    X, y, groups, df_feat = get_feature_target_split(df)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    held_out_machines = sorted(groups.iloc[test_idx].unique())

    baseline = LinearRegression().fit(X_train, y_train)
    baseline_metrics = evaluate(baseline, X_test, y_test)

    model = RandomForestRegressor(
        n_estimators=300, max_depth=10, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    ).fit(X_train, y_train)
    model_metrics = evaluate(model, X_test, y_test)

    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4)))
    importances = dict(sorted(importances.items(), key=lambda kv: -kv[1]))

    joblib.dump(model, out_dir / "fuel_model.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "held_out_machines": held_out_machines,
            "baseline_linear_regression": baseline_metrics,
            "random_forest": model_metrics,
            "feature_importance": importances,
        }, f, indent=2)

    print("Held-out machines (never seen in training):", held_out_machines)
    print("\nBaseline (Linear Regression):", baseline_metrics)
    print("Random Forest              :", model_metrics)
    print("\nFeature importance:")
    for feat, imp in importances.items():
        print(f"  {feat:<22} {imp}")
    print(f"\nSaved model -> {out_dir/'fuel_model.joblib'}")
    print(f"Saved metrics -> {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
