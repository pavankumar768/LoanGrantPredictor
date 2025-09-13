import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import streamlit as st

def handle_missing_values(data):
    """
    Handle missing values in the dataset
    """
    data_clean = data.copy()
    
    # Numeric columns - fill with median
    numeric_columns = data_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if data_clean[col].isnull().sum() > 0:
            data_clean[col].fillna(data_clean[col].median(), inplace=True)
    
    # Categorical columns - fill with mode
    categorical_columns = data_clean.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if data_clean[col].isnull().sum() > 0:
            mode_value = data_clean[col].mode()
            if len(mode_value) > 0:
                data_clean[col].fillna(mode_value[0], inplace=True)
            else:
                # If no mode, fill with 'Unknown'
                data_clean[col].fillna('Unknown', inplace=True)
    
    return data_clean

def preprocess_data(data):
    """
    Preprocess the loan dataset for machine learning
    """
    # Make a copy of the data
    df = data.copy()
    
    # Define target variable
    target_col = 'Loan_Status'
    if target_col not in df.columns:
        # If no target column, create a dummy one for preprocessing setup
        df[target_col] = 'Y'
        st.warning("No 'Loan_Status' column found. Creating dummy target for preprocessing.")
    
    # Encode target variable
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[target_col])
    
    # Drop target and ID columns
    feature_cols = df.columns.drop([target_col])
    if 'Loan_ID' in feature_cols:
        feature_cols = feature_cols.drop(['Loan_ID'])
    
    X = df[feature_cols]
    
    # Identify column types
    categorical_features = []
    numerical_features = []
    
    for col in X.columns:
        if X[col].dtype == 'object':
            categorical_features.append(col)
        else:
            numerical_features.append(col)
    
    # Create preprocessing pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names
    feature_names = []
    
    # Add numerical feature names
    feature_names.extend(numerical_features)
    
    # Add categorical feature names
    if len(categorical_features) > 0:
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        feature_names.extend(cat_feature_names)
    
    return X_processed, y, preprocessor, feature_names

def prepare_single_prediction(input_data, preprocessor):
    """
    Prepare a single data point for prediction
    """
    # Convert input data to DataFrame
    df = pd.DataFrame([input_data])
    
    # Handle missing conditional fields
    if 'Property_Value' not in df.columns:
        df['Property_Value'] = 0
    if 'Course_Type' not in df.columns:
        df['Course_Type'] = 'Other'
    if 'Academic_Performance' not in df.columns:
        df['Academic_Performance'] = 3.0
    
    # Transform the data
    X_processed = preprocessor.transform(df)
    
    return X_processed
