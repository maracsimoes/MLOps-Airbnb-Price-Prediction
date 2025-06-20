import pandas as pd
import nannyml as nml
from evidently import Report
from evidently.metric_preset import RegressionPreset

def evaluate_drift(X_test, y_test, y_pred):
    # --- regression metrics ----------
    mae  = (y_test - y_pred).abs().mean()
    rmse = ((y_test - y_pred)**2).mean()**0.5

    # --- nannyml: target drift -------
    ref_df  = pd.DataFrame({"price": y_test})        # last train chunk
    analy_df = pd.DataFrame({"price": y_pred})
    calc = nml.TargetDriftCalculator(
        y_pred="price", y_true=None,
        chunk_size=100, method='wasserstein')         # numeric drift
    calc.fit(ref_df)
    drift_df = calc.calculate(analy_df).to_df()

    calc.plot().write_html("data/08_reporting/price_target_drift.html")

    # --- evidently full report -------
    rep = Report(metrics=[RegressionPreset()])
    rep.run(current_data=analy_df, reference_data=ref_df)
    rep.save_html("data/08_reporting/price_regression_report.html")

    return {"mae": float(mae), "rmse": float(rmse)}