"""
Nodes for the `data_cleaning` pipeline
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BOOL_COLS: List[str] = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "instant_bookable",
]

DATE_COLS: List[str] = ["host_since"]


def change_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast raw columns to proper dtypes.

    • 't'/'f'/'True'/'False' → boolean  
    • 'host_since'          → datetime64[ns]  
    • 'bedrooms'            → pandas nullable Int64
    """
    out = df.copy()

    # Booleans
    for col in set(BOOL_COLS) & set(out.columns):
        out[col] = (
            out[col]
            .astype(str)
            .str.lower()
            .map({"t": True, "f": False, "true": True, "false": False})
            .astype("boolean")
        )

    # Dates
    for col in set(DATE_COLS) & set(out.columns):
        out[col] = pd.to_datetime(out[col], errors="coerce")

    # Nullable integer
    if "bedrooms" in out.columns:
        out["bedrooms"] = pd.to_numeric(out["bedrooms"], errors="coerce").astype("Int64")

    logger.debug("change_data_types(): dtypes after cast\n%s", out.dtypes)
    return out

def replace_zeros_with_nan(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Return a copy where 0s in *column* are replaced by NaN.
    """
    out = df.copy()
    n_before = out[column].eq(0).sum()
    out[column] = out[column].replace(0, np.nan)
    logger.debug("replace_zeros_with_nan(): replaced %d zeros in '%s'", n_before, column)
    return out


def filter_by_max(
    df: pd.DataFrame, column: str, max_value: float | int
) -> pd.DataFrame:
    """
    Keep rows where *column* ≤ *max_value* (NaNs are kept).

    Returns a new DataFrame.
    """
    mask = df[column].le(max_value) | df[column].isna()
    logger.debug(
        "filter_by_max(): removed %d rows where %s > %s",
        (~mask).sum(), column, max_value
    )
    return df[mask].copy()


def filter_by_range(
    df: pd.DataFrame,
    column: str,
    min_exclusive: float | int,
    max_inclusive: float | int,
) -> pd.DataFrame:
    """
    Keep rows where *column* is > *min_exclusive* and ≤ *max_inclusive*.

    NaNs are dropped.
    """
    mask = (df[column] > min_exclusive) & (df[column] <= max_inclusive)
    logger.debug(
        "filter_by_range(): removed %d rows outside (%s, %s] in '%s'",
        (~mask).sum(), min_exclusive, max_inclusive, column
    )
    return df[mask].copy()

def clean_accommodates(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 0 → NaN in 'accommodates'."""
    return replace_zeros_with_nan(df, column="accommodates")


def keep_reasonable_bedroom_counts(df: pd.DataFrame, max_bedrooms: int = 10) -> pd.DataFrame:
    """Drop listings with > *max_bedrooms* bedrooms (keep NaNs)."""
    return filter_by_max(df, column="bedrooms", max_value=max_bedrooms)


def keep_reasonable_prices(df: pd.DataFrame,
                           min_price: float = 0,
                           max_price: float = 500) -> pd.DataFrame:
    """Keep listings where 0 < price ≤ 500 (by default)."""
    return filter_by_range(df, "price", min_price, max_price)


def keep_reasonable_min_nights(df: pd.DataFrame, max_nights: int = 40) -> pd.DataFrame:
    """Drop rows with minimum_nights > max_nights."""
    return filter_by_max(df, "minimum_nights", max_nights)


def keep_reasonable_max_nights(df: pd.DataFrame, max_nights: int = 1150) -> pd.DataFrame:
    """Drop rows with maximum_nights > max_nights."""
    return filter_by_max(df, "maximum_nights", max_nights)


__all__ = [
    "change_data_types",
    "remove_irrelevant_columns",
    "clean_accommodates",
    "keep_reasonable_bedroom_counts",
    "keep_reasonable_prices",
    "keep_reasonable_min_nights",
    "keep_reasonable_max_nights",
]
