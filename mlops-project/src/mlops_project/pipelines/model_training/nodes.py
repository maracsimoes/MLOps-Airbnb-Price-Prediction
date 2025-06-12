import logging
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_model_node(X_train, y_train):
    logging.info("Training LinearRegression...")
    mlflow.start_run()
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    train_preds = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_preds)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    
    logging.info(f"Train MAE: {train_mae:.2f}")
    logging.info(f"Train RMSE: {train_rmse:.2f}")
    
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.sklearn.log_model(model, name="linear_regression_model", input_example=X_train.iloc[:5])
    
    mlflow.end_run()
    return model


