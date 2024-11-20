import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocess_data(df):
    # Define numeric and categorical features
    numeric_features = ['feature1', 'feature2', ...]
    categorical_features = ['feature3', 'feature4', ...]
    
    # Impute missing values in numeric features using mean imputation
    numeric_transformer = SimpleImputer(strategy='mean')
    transformed_numeric = numeric_transformer.fit_transform(df[numeric_features])
    
    # One-hot encode categorical features
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    transformed_categorical = categorical_transformer.fit_transform(df[categorical_features])
    
    # Combine numeric and categorical features
    transformed_features = pd.concat([pd.DataFrame(transformed_numeric), pd.DataFrame(transformed_categorical.toarray())], axis=1)
    
    # Get target variable
    target = df['target_variable']
    
    return transformed_features, target, numeric_features, categorical_features