import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import time
import streamlit as st

def train_models(X, y, perform_tuning=True, cv_folds=5):
    """
    Train multiple models and return results
    """
    models = {}
    model_results = {}
    
    # Define models
    model_configs = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            } if perform_tuning else {}
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10]
            } if perform_tuning else {}
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            } if perform_tuning else {}
        }
    }
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, config) in enumerate(model_configs.items()):
        status_text.text(f"Training {name}...")
        
        start_time = time.time()
        
        if perform_tuning and config['params']:
            # Perform hyperparameter tuning
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=cv_folds,
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X, y)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # Use default parameters
            best_model = config['model']
            best_model.fit(X, y)
            best_params = {}
        
        # Cross-validation scores
        cv_scores = cross_val_score(best_model, X, y, cv=cv_folds, scoring='accuracy')
        
        training_time = time.time() - start_time
        
        # Store results
        models[name] = best_model
        model_results[name] = {
            'cv_scores': cv_scores,
            'best_params': best_params,
            'training_time': training_time
        }
        
        # Update progress
        progress_bar.progress((i + 1) / len(model_configs))
    
    status_text.text("Model training completed!")
    
    return models, model_results

def perform_hyperparameter_tuning(model, param_grid, X, y, cv_folds=5):
    """
    Perform hyperparameter tuning for a specific model
    """
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
