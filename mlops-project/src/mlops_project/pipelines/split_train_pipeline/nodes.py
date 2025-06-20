import pandas as pd
from typing import Any, Dict, Tuple
from sklearn.model_selection import train_test_split

def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Index]:
    target_column = "target"
    test_fraction = 0.2
    random_state = 42

    y = data[target_column]
    X = data.drop(columns=[target_column, "index"], errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        stratify=y,
        test_size=test_fraction,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test, X_train.columns
