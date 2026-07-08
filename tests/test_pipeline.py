"""
Basic sanity checks for the data generator and feature engineering step.
Run with: pytest
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_data import generate
from train_model import engineer_features, hour_bucket


def test_generate_shape_and_columns():
    df = generate(n_posts=200, seed=1)
    assert len(df) == 200
    expected_cols = {
        "post_id", "platform", "post_type", "day_of_week", "hour_of_day",
        "is_weekend", "hashtag_count", "caption_length", "follower_count",
        "likes", "comments", "shares", "engagement_rate",
    }
    assert expected_cols.issubset(df.columns)


def test_generate_is_reproducible():
    df1 = generate(n_posts=100, seed=7)
    df2 = generate(n_posts=100, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_engagement_rate_is_non_negative():
    df = generate(n_posts=500, seed=3)
    assert (df["engagement_rate"] >= 0).all()


def test_hour_bucket_covers_all_hours():
    buckets = {hour_bucket(h) for h in range(24)}
    assert buckets == {"early_morning", "morning", "afternoon", "evening", "late_night"}


def test_engineer_features_produces_dummies_without_raw_categoricals():
    df = generate(n_posts=100, seed=1)
    df_fe, feature_cols = engineer_features(df)
    for raw_col in ["platform", "post_type", "day_of_week", "hour_bucket"]:
        assert raw_col not in feature_cols
    assert "engagement_rate" not in feature_cols
    assert len(feature_cols) > 0
