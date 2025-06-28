import pandas as pd
from typing import Any, Dict, Tuple
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import pickle

def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Index]:
    target_column = "price"
    test_fraction = 0.2
    random_state = 42

    y = data[target_column]
    X = data.drop(columns=[target_column, "index"], errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_fraction,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test, X_train.columns

def identify_data_types_node(df: pd.DataFrame):
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical = df.select_dtypes(include=['object','category']).columns.tolist()
    boolean = df.select_dtypes(include=['bool']).columns.tolist()
    datetime = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    return numerical, categorical, boolean, datetime


def handle_missing_values_node(X_train, X_val, y_train, y_val,
                             numerical_features, categorical_features, boolean_features):
    medians = {}
    modes = {}

    # Preenche missing numéricos com mediana do train
    for col in numerical_features:
        median_val = X_train[col].median()
        medians[col] = median_val
        X_train[col] = X_train[col].fillna(median_val)
        if col in X_val.columns:
            X_val[col] = X_val[col].fillna(median_val)
    # Preenche missing em booleanos e categóricos com moda do train
    for col in list(boolean_features) + list(categorical_features):
        mode_val = X_train[col].mode(dropna=True)
        if not mode_val.empty:
            mode_val = mode_val[0]
        else:
            mode_val = None
        modes[col] = mode_val
        X_train[col] = X_train[col].fillna(mode_val)
        if col in X_val.columns:
            X_val[col] = X_val[col].fillna(mode_val)

    median_y = y_train.median().item()  # se for DataFrame de uma coluna só

    if y_train.isnull().values.any():
        y_train = y_train.fillna(median_y)

    if y_val.isnull().values.any():
        y_val = y_val.fillna(median_y)

    
    missing_cols = X_train.columns[X_train.isnull().any()]
    if not missing_cols.empty:
        print("Columns with missing values in X_train:", missing_cols.tolist())
        print(X_train[missing_cols].isnull().sum())

    assert not X_train.isnull().any().any()
    assert not X_val.isnull().any().any()
        
    return X_train, X_val, y_train, y_val, medians, modes


# Categorical variable encoding
def encode_categorical_node(X_train, X_val):
    with open(os.path.join(os.getcwd(), 'data', '04_feature', 'le_encoder.pkl'), 'rb') as f:
        le_dict = pickle.load(f)
    for col, le in le_dict.items():
        if col in X_train.columns:
            X_train[col] = le.transform(X_train[col].astype(str))
        if col in X_val.columns:
            X_val[col] = le.transform(X_val[col].astype(str))
    return X_train, X_val


def scale_numeric_node(X_train, X_val):
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    return X_train, X_val, scaler

def split_encode_scale_input(df):
    numerical, categorical, boolean, datetime = identify_data_types_node(df)
    numerical.remove("price") 
    X_train, X_val, y_train, y_val, columns = split_data(df)
    #X_train = handle_outliers_node(X_train, numerical)
    X_train, X_val, y_train, y_val, medians, modes = handle_missing_values_node(
        X_train, X_val, y_train, y_val, numerical, categorical, boolean
    )
    X_train, X_val = encode_categorical_node(X_train, X_val)
    X_train, X_val, scaler = scale_numeric_node(X_train, X_val)
    y_train = y_train.to_frame(name="price")
    y_val = y_val.to_frame(name="price")
    return X_train, X_val, y_train, y_val, scaler, columns, medians, modes
