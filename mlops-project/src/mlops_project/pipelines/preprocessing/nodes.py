from kedro.pipeline import node, Pipeline
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler



def split_data_node(df, target_col='price'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val


def identify_data_types_node(df: pd.DataFrame):
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical = df.select_dtypes(include=['object']).columns.tolist()
    boolean = df.select_dtypes(include=['bool']).columns.tolist()
    datetime = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()
    return numerical, categorical, boolean, datetime


def handle_missing_values_node(X_train, X_val, y_train, y_val,
                             numerical_features, categorical_features, boolean_features):
    # Preenche missing numéricos com mediana do train
    for col in numerical_features:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        if col in X_val.columns:
            X_val[col] = X_val[col].fillna(median_val)

    # Preenche missing em booleanos e categóricos com moda do train
    for col in boolean_features + categorical_features:
        mode_val = X_train[col].mode(dropna=True)
        if not mode_val.empty:
            mode_val = mode_val[0]
            X_train[col] = X_train[col].fillna(mode_val)
            if col in X_val.columns:
                X_val[col] = X_val[col].fillna(mode_val)

    median_y = y_train.median() 
    if y_train.isnull().any():
        y_train = y_train.fillna(median_y)

    if y_val.isnull().any():
        y_val = y_val.fillna(median_y)


    #print("X_train missing values before assert:")
    #print(X_train.isnull().sum())
    #print("X_val missing values before assert:")
    #print(X_val.isnull().sum())

    assert not X_train.isnull().any().any()
    assert not X_val.isnull().any().any()
        
    return X_train, X_val, y_train, y_val



def handle_outliers_node(df, numerical_features, factor=1.5):
    for col in numerical_features:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

# Categorical variable encoding
def encode_categorical_node(X_train, X_val):
    le_dict = {}
    for col in X_train.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_val[col] = X_val[col].astype(str)
        X_val[col] = X_val[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1).fillna(-1)
        le_dict[col] = le
    return X_train, X_val

def scale_numeric_node(X_train, X_val):
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    return X_train, X_val