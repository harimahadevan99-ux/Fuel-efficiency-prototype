"""
Feature engineering for the fuel-consumption model.

Turns raw per-hour telemetry into model-ready features:
  - one-hot encoded operating mode
  - rolling averages (captures short-term trend, not just instant reading)
  - rpm volatility (proxy for jerky / aggressive operation)
  - idle_pct_last_24h (a feature AND a useful KPI on its own)
"""
from __future__ import annotations
import pandas as pd

MODE_DUMMIES = ["mode_dig", "mode_idle", "mode_overload", "mode_travel"]

FEATURE_COLUMNS = [
    "rated_hp", "load_factor", "rpm", "hyd_pressure_bar", "ambient_temp_c",
    "rpm_rolling_std_5", "idle_pct_last_24h",
] + MODE_DUMMIES

TARGET_COLUMN = "fuel_rate_gal_hr"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["machine_id", "timestamp"]).copy()

    mode_dummies = pd.get_dummies(df["mode"], prefix="mode")
    for col in MODE_DUMMIES:
        if col not in mode_dummies.columns:
            mode_dummies[col] = 0
    df = pd.concat([df, mode_dummies[MODE_DUMMIES]], axis=1)

    df["rpm_rolling_std_5"] = (
        df.groupby("machine_id")["rpm"]
        .transform(lambda s: s.rolling(5, min_periods=1).std())
        .fillna(0)
    )

    df["is_idle"] = (df["mode"] == "idle").astype(int)
    df["idle_pct_last_24h"] = (
        df.groupby("machine_id")["is_idle"]
        .transform(lambda s: s.rolling(24, min_periods=1).mean())
    )

    return df


def get_feature_target_split(df: pd.DataFrame):
    df_feat = build_features(df)
    X = df_feat[FEATURE_COLUMNS]
    y = df_feat[TARGET_COLUMN]
    groups = df_feat["machine_id"]
    return X, y, groups, df_feat
