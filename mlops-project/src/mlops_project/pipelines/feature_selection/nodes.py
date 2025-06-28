import logging
from typing import Any, Dict
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

def feature_selection(X_train: pd.DataFrame, y_train: pd.DataFrame, parameters) -> list:
    log = logging.getLogger(__name__)
    log.info(f"Starting feature selection with {len(X_train.columns)} features")

    if parameters.get("method") == "rfe":
        y_train = np.ravel(y_train)

        try:
            with open(os.path.join(os.getcwd(), 'data', '06_models', 'fitted_rfe.pkl'), 'rb') as f:
                classifier = pickle.load(f)
                log.info("Loaded pre-trained from pickle.")
        except Exception as e:
            log.warning(f"Failed to load model from pickle: {e}. Creating new RandomForest.")
            classifier = RandomForestClassifier(**parameters.get('baseline_model_params', {}))

        # Remove datetime columns from X_train
        datetime_cols = X_train.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns
        if len(datetime_cols) > 0:
            print("Dropping datetime columns from X_train:", list(datetime_cols))

        X_train = X_train.drop(columns=datetime_cols)
        y_train = np.ravel(y_train)
        y_train = pd.Series(y_train, index=X_train.index)

        if len(X_train) > 10000:
            X_train_sample = X_train.sample(10000, random_state=42)
            y_train_sample = y_train.loc[X_train_sample.index]
        else:
            X_train_sample = X_train
            y_train_sample = y_train

        rfe = RFE(classifier)
        rfe = rfe.fit(X_train_sample, y_train_sample)
        selected_features = X_train_sample.columns[rfe.get_support()].tolist()

        log.info(f"Selected {len(selected_features)} features after RFE.")
        return selected_features, classifier

    log.warning("No feature selection method specified or method not supported.")
    return X_train.columns.tolist(), []
