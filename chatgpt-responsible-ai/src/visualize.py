# Import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from yellowbrick.classifier import ConfusionMatrix, ROCAUC, ClassificationReport, ClassPredictionError, DiscriminationThreshold
from yellowbrick.features import Rank2D
from preprocessing import preprocess_data

# Load trained model from .pkl file
model = joblib.load('path/to/trained/model.pkl')

# Load data from parquet file
df = pd.read_parquet('path/to/data.parquet')

# Preprocess data
X, y, numeric_features, categorical_features = preprocess_data(df)

# Instantiate visualizers
cm = ConfusionMatrix(model)
roc_auc = ROCAUC(model)
cr = ClassificationReport(model)
cpe = ClassPredictionError(model)
dt = DiscriminationThreshold(model)
r2d = Rank2D(features=numeric_features)

# Visualize data
cm.fit(X, y)
cm.show()

roc_auc.fit(X, y)
roc_auc.show()

cr.fit(X, y)
cr.show()

cpe.fit(X, y)
cpe.show()

dt.fit(X, y)
dt.show()

r2d.fit_transform(X)
r2d.show()