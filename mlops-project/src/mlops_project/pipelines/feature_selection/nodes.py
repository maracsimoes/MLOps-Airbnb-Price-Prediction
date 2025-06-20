import logging
from typing import Any, Dict
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

def feature_selection(X_train: pd.DataFrame, y_train: pd.DataFrame, parameters: Dict[str, Any]) -> list:
    log = logging.getLogger(__name__)
    log.info(f"Starting feature selection with {len(X_train.columns)} features")

    if parameters.get("feature_selection") == "rfe":
        y_train = np.ravel(y_train)

        try:
            with open(os.path.join(os.getcwd(), 'data', '06_models', 'champion_model.pkl'), 'rb') as f:
                classifier = pickle.load(f)
                log.info("Loaded champion model from pickle.")
        except Exception as e:
            log.warning(f"Failed to load model from pickle: {e}. Creating new RandomForest.")
            classifier = RandomForestClassifier(**parameters.get('baseline_model_params', {}))

        rfe = RFE(classifier)
        rfe = rfe.fit(X_train, y_train)

        selected_features = X_train.columns[rfe.get_support()].tolist()

        log.info(f"Selected {len(selected_features)} features after RFE.")
        return selected_features

    log.warning("No feature selection method specified or method not supported.")
    return X_train.columns.tolist()  
