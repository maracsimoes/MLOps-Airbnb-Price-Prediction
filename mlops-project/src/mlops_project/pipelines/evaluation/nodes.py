import logging
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import shap

def evaluate_model_node(model, X_val, y_val):
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    logging.info(f"MAE: {mae:.2f}")
    logging.info(f"RMSE: {rmse:.2f}")

    try:
        if not isinstance(X_val, pd.DataFrame):
            X_val = pd.DataFrame(X_val)

        X_val = X_val.apply(pd.to_numeric, errors='coerce').fillna(0).replace([np.inf, -np.inf], 0).astype(np.float64)

        rng = np.random.default_rng(seed=42)  # Gerador de números aleatórios explícito

        explainer = shap.Explainer(model.predict, X_val, rng=rng)
        shap_values = explainer(X_val)

        plt.figure()
        shap.summary_plot(shap_values, X_val, show=False)
        plt.tight_layout()
        shap_fig = plt.gcf()
        plt.savefig("shap_summary.png")

        mlflow.log_figure(shap_fig, "shap_summary.png")
        plt.close()

    except Exception as e:
        logging.warning(f"SHAP failed: {e}")
        plt.figure()
        shap_fig = plt.gcf()
        plt.close()

    return mae, rmse, shap_fig
