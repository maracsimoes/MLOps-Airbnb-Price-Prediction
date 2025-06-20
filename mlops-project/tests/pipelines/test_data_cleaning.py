import pandas as pd
import numpy as np
import pytest

from src.mlops_project.pipelines.data_cleaning.nodes import (
    change_data_types,
    clean_accommodates,
    keep_reasonable_bedroom_counts,
    keep_reasonable_prices,
    keep_reasonable_min_nights,
    keep_reasonable_max_nights,
    replace_zeros_with_nan,
    filter_by_max,
    filter_by_range,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "host_is_superhost": ["t", "f", "True", "False", "other"],
        "host_since": ["2020-01-01", "invalid_date", None, "2021-06-15", "2022-12-31"],
        "bedrooms": ["1", "2", "3", "invalid", None],
        "accommodates": [0, 2, 3, 0, 1],
        "price": [0, 50, 600, 300, 10],
        "minimum_nights": [1, 50, 5, 0, 3],
        "maximum_nights": [100, 2000, 500, 100, 50]
    })

def test_change_data_types(sample_df):
    df = change_data_types(sample_df)
    assert pd.api.types.is_bool_dtype(df["host_is_superhost"])
    assert pd.api.types.is_datetime64_any_dtype(df["host_since"])
    assert pd.api.types.is_integer_dtype(df["bedrooms"])

def test_replace_zeros_with_nan(sample_df):
    df = replace_zeros_with_nan(sample_df, "accommodates")
    assert df["accommodates"].isna().sum() == 2

def test_filter_by_max(sample_df):
    df = filter_by_max(sample_df, "maximum_nights", 1150)
    assert (df["maximum_nights"] > 1150).sum() == 0

def test_filter_by_range(sample_df):
    df = filter_by_range(sample_df, "price", 0, 500)
    assert df["price"].min() > 0
    assert df["price"].max() <= 500

def test_clean_accommodates(sample_df):
    df = clean_accommodates(sample_df)
    assert df["accommodates"].isna().sum() == 2

def test_keep_reasonable_bedroom_counts(sample_df):
    df = change_data_types(sample_df)
    df = keep_reasonable_bedroom_counts(df, max_bedrooms=2)
    assert df["bedrooms"].max() <= 2 or df["bedrooms"].isna().all()

def test_keep_reasonable_prices(sample_df):
    df = keep_reasonable_prices(sample_df, min_price=0, max_price=500)
    assert df["price"].min() > 0
    assert df["price"].max() <= 500

def test_keep_reasonable_min_nights(sample_df):
    df = keep_reasonable_min_nights(sample_df, max_nights=40)
    assert (df["minimum_nights"] > 40).sum() == 0

def test_keep_reasonable_max_nights(sample_df):
    df = keep_reasonable_max_nights(sample_df, max_nights=1150)
    assert (df["maximum_nights"] > 1150).sum() == 0
