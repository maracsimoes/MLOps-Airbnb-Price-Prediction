"""
Nodes for the `preprocessing_batch` pipeline
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

def clean_accommodates(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 0 → NaN in 'accommodates'."""
    return replace_zeros_with_nan(df, column="accommodates")

__all__ = [
    "change_data_types",
    "remove_irrelevant_columns",
    "clean_accommodates"
]


