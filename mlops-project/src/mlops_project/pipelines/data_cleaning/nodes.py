"""
This is a boilerplate pipeline 'data_cleaning'
generated using Kedro 0.19.13
"""
import pandas as pd
import numpy as np
import logging
import ast
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler

def change_data_types(df: pd.DataFrame) -> pd.DataFrame:
    bool_cols = ["host_is_superhost", "host_has_profile_pic", "host_identity_verified", "instant_bookable"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({'t': True, 'f': False}).astype('boolean')
    df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
    df['bedrooms'] = df['bedrooms'].astype('Int64')
    return df

def remove_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = ['city', 'host_location', 'district', 'name']
    return df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

