# Bank Loan Eligibility Predictor

## Overview

A Streamlit-based machine learning application that predicts bank loan eligibility using multiple ML algorithms. The system provides a complete workflow from data upload and model training to loan predictions with explanations. Users can upload loan datasets, train and compare different models (Logistic Regression, Random Forest, XGBoost), and make individual loan eligibility predictions with interpretability features.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with multi-page navigation
- **UI Components**: Sidebar navigation, progress bars, interactive plots using Plotly
- **Session State Management**: Persistent storage of trained models, preprocessors, and results across page navigation
- **Page Structure**: Four main sections - Data Upload & Training, Model Comparison, Loan Prediction, and About

### Backend Architecture
- **Model Training Pipeline**: Modular approach with separate utilities for data processing, model training, evaluation, and prediction
- **Data Processing**: Automated handling of missing values, feature encoding, and preprocessing pipeline using scikit-learn
- **Model Management**: Support for multiple ML algorithms with optional hyperparameter tuning using GridSearchCV
- **Prediction Engine**: Single prediction capability with optional SHAP-based explanations

### Data Processing Strategy
- **Missing Value Handling**: Median imputation for numeric features, mode imputation for categorical features
- **Feature Engineering**: StandardScaler for numeric features, OneHotEncoder for categorical features
- **Pipeline Architecture**: ColumnTransformer-based preprocessing pipeline for consistent data transformation
- **Target Encoding**: LabelEncoder for binary loan status classification

### Model Training Approach
- **Multi-Algorithm Support**: Logistic Regression, Random Forest, and XGBoost classifiers
- **Hyperparameter Optimization**: Optional GridSearchCV with configurable cross-validation folds
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, F1-score, and ROC-AUC
- **Performance Tracking**: Cross-validation scores and detailed classification reports

### Prediction and Interpretability
- **Single Prediction Interface**: Individual loan application processing with probability scores
- **Model Explainability**: SHAP integration for prediction explanations with feature importance fallback
- **Error Handling**: Graceful degradation when SHAP is unavailable or explanation generation fails

## External Dependencies

### Core ML and Data Science Libraries
- **scikit-learn**: Complete machine learning pipeline including preprocessing, model training, and evaluation
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations and array operations
- **XGBoost**: Gradient boosting algorithm implementation

### Visualization and UI
- **Streamlit**: Web application framework for the user interface
- **Plotly**: Interactive plotting library for model comparison and data visualization
- **matplotlib/seaborn**: Additional plotting capabilities for model evaluation

### Model Interpretability
- **SHAP**: Optional dependency for advanced model explanation and feature importance analysis
- **Feature Importance Fallback**: Built-in fallback mechanism when SHAP is not available

### Data Persistence
- **pickle**: Model serialization for saving and loading trained models
- **Session State**: Streamlit's built-in session management for maintaining application state

### Development and Deployment
- **Standard Python Libraries**: os for file operations, time for performance tracking
- **Cross-platform Compatibility**: Designed to work across different operating systems and Python environments