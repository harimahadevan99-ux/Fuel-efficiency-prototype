import numpy as np
import pandas as pd

from src.simulation.generate_data import generate_fleet_data, FLEET
from src.features.build_features import build_features, get_feature_target_split, FEATURE_COLUMNS


def test_generate_fleet_data_shape():
    df = generate_fleet_data(n_hours_per_machine=50, seed=1)
    assert len(df) == 50 * len(FLEET)
    assert df["machine_id"].nunique() == len(FLEET)
    assert (df["fuel_rate_gal_hr"] > 0).all()


def test_degraded_machine_burns_more_fuel_on_average():
    df = generate_fleet_data(n_hours_per_machine=500, seed=2)
    avg_fuel = df.groupby("machine_id")["fuel_rate_gal_hr"].mean()
    avg_load = df.groupby("machine_id")["load_factor"].mean()
    # normalize by load so we're comparing efficiency, not just workload
    efficiency = avg_fuel / avg_load
    # EXC-103 has 18% injected degradation -> should be least efficient
    assert efficiency.idxmax() == "EXC-103"


def test_build_features_no_nans_and_expected_columns():
    df = generate_fleet_data(n_hours_per_machine=30, seed=3)
    df_feat = build_features(df)
    assert not df_feat[FEATURE_COLUMNS].isna().any().any()
    for col in FEATURE_COLUMNS:
        assert col in df_feat.columns


def test_feature_target_split_shapes_match():
    df = generate_fleet_data(n_hours_per_machine=30, seed=4)
    X, y, groups, df_feat = get_feature_target_split(df)
    assert len(X) == len(y) == len(groups) == len(df)
