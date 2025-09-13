import numpy as np
import pandas as pd
from utils.data_processing import prepare_single_prediction
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available - using feature importance fallback")
import streamlit as st

def make_prediction(input_data, model, preprocessor, feature_names):
    """
    Make a prediction for a single loan application
    """
    try:
        # Prepare the input data
        X_processed = prepare_single_prediction(input_data, preprocessor)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0]
        
        # Generate explanation
        explanation = None
        try:
            explanation = explain_prediction(X_processed, model, feature_names)
        except Exception as e:
            st.warning(f"Could not generate explanation: {str(e)}")
        
        return prediction, probability, explanation
    
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def explain_prediction(X, model, feature_names):
    """
    Generate explanations for the prediction using SHAP or feature importance fallback
    """
    if not SHAP_AVAILABLE:
        # Fallback to feature importance if SHAP is not available
        return get_feature_importance_explanation(X, model, feature_names)
    
    try:
        # Create SHAP explainer
        if hasattr(model, 'predict_proba'):
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            
            # For binary classification, use positive class SHAP values
            if len(shap_values.values.shape) > 1 and shap_values.values.shape[1] == 2:
                shap_vals = shap_values.values[0, :, 1]  # Positive class
            else:
                shap_vals = shap_values.values[0]
        else:
            # For models without predict_proba, use TreeExplainer if possible
            if hasattr(model, 'feature_importances_'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                shap_vals = shap_values[0]
            else:
                return get_feature_importance_explanation(X, model, feature_names)
        
        # Prepare explanation data
        explanation = {
            'shap_values': shap_vals,
            'feature_names': feature_names[:len(shap_vals)],
            'feature_values': X[0][:len(shap_vals)]
        }
        
        return explanation
    
    except Exception as e:
        print(f"SHAP explanation failed: {str(e)}")
        return get_feature_importance_explanation(X, model, feature_names)

def get_feature_importance(model, feature_names):
    """
    Get feature importance from the model
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    elif hasattr(model, 'coef_'):
        # For linear models, use absolute coefficients
        importance = np.abs(model.coef_[0])
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance_df
    
    else:
        return None

def get_feature_importance_explanation(X, model, feature_names):
    """
    Generate explanation using feature importance as fallback when SHAP is not available
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            return None
        
        # Create pseudo-SHAP values using feature importance
        # This is a simplified approach - multiply importance by feature values
        feature_values = X[0][:len(importance)]
        pseudo_shap = importance * feature_values
        
        explanation = {
            'shap_values': pseudo_shap,
            'feature_names': feature_names[:len(pseudo_shap)],
            'feature_values': feature_values
        }
        
        return explanation
    
    except Exception as e:
        print(f"Feature importance explanation failed: {str(e)}")
        return None

def generate_recommendations(input_data, prediction, probability):
    """
    Generate personalized recommendations based on the prediction
    """
    recommendations = []
    
    if prediction == 0:  # Loan rejected
        if probability[1] < 0.3:
            recommendations.extend([
                "Consider improving your credit history before reapplying",
                "Increase your monthly income or add a co-applicant with stable income",
                "Reduce the requested loan amount",
                "Consider a longer loan term to reduce monthly payment burden"
            ])
        elif probability[1] < 0.5:
            recommendations.extend([
                "Your application is borderline - consider minor improvements",
                "Provide additional documentation to support your financial stability",
                "Consider waiting a few months to improve your financial profile"
            ])
    else:  # Loan approved
        recommendations.extend([
            "Congratulations! Your loan is likely to be approved",
            "Shop around for the best interest rates from different lenders",
            "Ensure you have all required documentation ready",
            "Review all loan terms and conditions carefully"
        ])
    
    # Add specific recommendations based on input data
    if input_data.get('Credit_History', 1.0) == 0.0:
        recommendations.append("Focus on building a positive credit history")
    
    if input_data.get('ApplicantIncome', 0) < 3000:
        recommendations.append("Consider increasing your income before applying")
    
    debt_to_income = input_data.get('LoanAmount', 0) / max(input_data.get('ApplicantIncome', 1), 1)
    if debt_to_income > 5:
        recommendations.append("Your loan amount is high relative to income - consider reducing it")
    
    return recommendations
