# Airbnb Price Prediction MLOps Pipeline

## 1. Introduction

Airbnb transformed short-term rentals, making price setting complex due to many factors. This project builds an end-to-end MLOps pipeline to predict Airbnb listing prices using rich, real-world data from Inside-Airbnb.

Our goal is to deliver a maintainable, reproducible, and monitorable system applying MLOps best practices — including data validation, model training, version control with MLflow, explainability (SHAP), and data drift detection.

This pipeline shows how to turn a predictive model into a reliable, scalable, and auditable service fit for real-world deployment.

## 2. Data Collection & Preprocessing

### 2.1. Data Sources

The dataset was obtained from Maven Analytics' Data Playground, specifically the Airbnb "Listings" dataset. We focused on Paris listings to reduce complexity and data volume, aligning with our goal to implement MLOps methodologies rather than scaling Big Data systems.

### 2.2. Data Characteristics

- Format: CSV with tabular records (33 columns, ~47,000 rows after filtering).  
- Features: numeric (price, accommodation, review scores), categorical (room type, property type, superhost status), textual (amenities), geolocation (latitude, longitude).

### 2.3. Data Cleaning & Preprocessing

- Removed features with missing values, redundant location attributes, irrelevant host and review features (except overall review score).  
- Removed <0.1% rows with unrealistic values. Imputed missing values with median.  
- Addressed outliers by removal or capping using interquartile ranges.  
- Encoded categorical variables: amenities grouped from 618 to 14 categories, property_type reduced from 66 to 5 main groups. 
- Engineered features: `host_listings_count`, `host_total_listings_count`, `days_since_host`.

## 3. Methodology & Tools

### 3.1. Model Training & Evaluation

- Pipeline structured with **Kedro** for modularity and reproducibility.  
- Data validated using **Great Expectations** before training (schema, uniqueness, non-null constraints).  
- Dataset splits: Reference vs. Analysis (for drift detection), then Analysis split into Train/Test (80/20).  
- Models trained: Linear Regression (baseline), Random Forest, Gradient Boosting Regressor (best performing).  
- Metrics: MAE and RMSE (primary).  
- Model tracking and logging with **MLflow** (features, parameters, metrics, artifacts).  
- Final evaluation: Gradient Boosting Regressor with MAE=31.02, RMSE=48.84.  
- Explainability via **SHAP** highlighted key features: accommodates, host_total_listings_count, amenities_length.

### 3.2. Deployment Simulation

Pipeline execution steps:

1. Data ingestion  
2. Data splitting  
3. Preprocessing (training)  
4. Feature selection  
5. Model training  
6. Preprocessing (batch for inference)  
7. Model prediction

All steps can be run sequentially using the default Kedro pipeline.

Simulated architecture features:

- **Model Serving:** Batch predictions using trained model and preprocessing.  
- **Monitoring:** Data drift detection via pipeline comparing new data distributions to training data. Metrics like MAE/RMSE tracked, **NannyML** for target drift detection, and **Evidently AI** for detailed drift reports.  

### 3.3. Technologies, Risks & Mitigations

- **Kedro:** Great for modular pipelines; might require migration to Spark for big data scale.  
- **Great Expectations:** Manual and verbose; automated profiling recommended for future.  
- **Hopsworks:** Adds reproducibility but requires secure credential management.  
- **MLflow:** Local tracking used here; remote server recommended for production and collaboration.

### 3.4. Packages & Versions

| Package            | Version          | Purpose                                   |
|--------------------|------------------|-------------------------------------------|
| ipython            | >=8.10           | Interactive shell for analysis             |
| jupyterlab         | >=3.0            | Notebook environment                       |
| kedro              | ~=0.19.13        | Pipeline structuring and execution         |
| notebook           | No specified     | Running .ipynb notebooks                    |
| kedro-telemetry    | >=0.3.1          | Usage tracking for Kedro                    |
| pytest             | ~=7.2            | Unit testing                               |
| pytest-cov         | ~=3.0            | Test coverage measurement                  |
| pytest-mock        | >=1.7.1,<2.0     | Mocking support in tests                    |
| ruff               | ~=0.1.8          | Linting and code style enforcement         |
| pandas             | 2.2.2            | Data manipulation                          |
| numpy              | No specified     | Numerical computing                        |
| scikit-learn       | No specified     | ML models and preprocessing                |
| great_expectations | 0.18.14          | Data validation and quality checks         |

## 4. Conclusion

This project aimed to build not just a predictive model but a full machine learning system suitable for real-world production. Using real Airbnb Paris data, we tackled common challenges like missing values and complex features. The focus was on reproducibility, modularity, and traceability, employing MLOps tools such as Kedro, MLflow, Great Expectations, SHAP, and Evidently.

Currently, the pipeline is executed manually in Kedro with plans to adopt orchestration tools like Airflow or Prefect for scheduling, retraining, and monitoring automation.

Our Gradient Boosting Regressor model achieved reasonable performance, serving as a solid base for future improvement and scaling.
