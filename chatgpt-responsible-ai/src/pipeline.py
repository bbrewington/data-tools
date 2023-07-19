import pandas as pd
import numpy as np
from google.cloud import bigquery
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import shap

# Initialize BigQuery client
client = bigquery.Client()

# Define SQL query to retrieve data from BigQuery
query = """
    SELECT *
    FROM your_project.your_dataset.your_table
"""

# Retrieve data from BigQuery and convert to Pandas dataframe
df = client.query(query).to_dataframe()

# Define features to be used in the model
numeric_features = ['feature1', 'feature2', ...]
categorical_features = ['feature3', 'feature4', ...]
target = 'target_variable'

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define XGBoost model
model = xgb.XGBClassifier()

# Define hyperparameter tuning grid
param_grid = {
    'learning_rate': [0.1, 0.01],
    'max_depth': [3, 5],
    'n_estimators': [100, 500],
}

# Define Scikit-learn pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Define GridSearchCV to tune hyperparameters
grid_search = GridSearchCV(pipeline, param_grid=param_grid, cv=5)

# Fit model and tune hyperparameters
X = df.drop(columns=[target])
y = df[target]
grid_search.fit(X, y)

# Output evaluation metrics
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Generate SHAP values for each instance
explainer = shap.Explainer(grid_search.best_estimator_)
shap_values = explainer(X)

# Plot the SHAP values for each feature
shap.summary_plot(shap_values, X, plot_type='bar')