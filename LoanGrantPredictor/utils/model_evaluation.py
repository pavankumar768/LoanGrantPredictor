import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def evaluate_models(models, X, y):
    """
    Evaluate multiple models and return comprehensive metrics
    """
    evaluation_results = {}
    
    for name, model in models.items():
        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y, y_pred, average='weighted', zero_division=0),
        }
        
        # ROC-AUC (only for binary classification)
        if y_pred_proba is not None and len(np.unique(y)) == 2:
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        evaluation_results[name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm,
            'classification_report': classification_report(y, y_pred)
        }
    
    return evaluation_results

def plot_model_comparison(evaluation_results):
    """
    Create comprehensive model comparison plots
    """
    # Extract metrics for comparison
    models = list(evaluation_results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy Comparison', 'Precision vs Recall', 
                       'F1-Score Comparison', 'ROC-AUC Comparison'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Prepare data
    metrics_data = {}
    for metric in metrics_names:
        metrics_data[metric] = []
        for model in models:
            if metric in evaluation_results[model]['metrics']:
                metrics_data[metric].append(evaluation_results[model]['metrics'][metric])
            else:
                metrics_data[metric].append(0)
    
    # Accuracy comparison
    fig.add_trace(
        go.Bar(x=models, y=metrics_data['accuracy'], name='Accuracy'),
        row=1, col=1
    )
    
    # Precision vs Recall scatter
    fig.add_trace(
        go.Scatter(
            x=metrics_data['precision'], 
            y=metrics_data['recall'],
            mode='markers+text',
            text=models,
            textposition="top center",
            name='Precision vs Recall'
        ),
        row=1, col=2
    )
    
    # F1-Score comparison
    fig.add_trace(
        go.Bar(x=models, y=metrics_data['f1_score'], name='F1-Score'),
        row=2, col=1
    )
    
    # ROC-AUC comparison
    if any(score > 0 for score in metrics_data['roc_auc']):
        fig.add_trace(
            go.Bar(x=models, y=metrics_data['roc_auc'], name='ROC-AUC'),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=False, title_text="Model Performance Comparison")
    
    return fig

def plot_confusion_matrices(evaluation_results):
    """
    Plot confusion matrices for all models
    """
    models = list(evaluation_results.keys())
    n_models = len(models)
    
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    if n_models == 1:
        axes = [axes]
    
    for i, (name, results) in enumerate(evaluation_results.items()):
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{name}\nConfusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    return fig

def plot_roc_curves(evaluation_results, y_true):
    """
    Plot ROC curves for all models
    """
    fig = go.Figure()
    
    for name, results in evaluation_results.items():
        if results['probabilities'] is not None:
            fpr, tpr, _ = roc_curve(y_true, results['probabilities'])
            auc_score = roc_auc_score(y_true, results['probabilities'])
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{name} (AUC = {auc_score:.3f})'
            ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(dash='dash', color='gray'),
        name='Random Classifier'
    ))
    
    fig.update_layout(
        title='ROC Curves Comparison',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600, height=500
    )
    
    return fig

def create_metrics_dataframe(evaluation_results):
    """
    Create a comprehensive metrics DataFrame for display
    """
    metrics_data = []
    
    for model_name, results in evaluation_results.items():
        metrics = results['metrics']
        row = {'Model': model_name}
        row.update(metrics)
        metrics_data.append(row)
    
    df = pd.DataFrame(metrics_data)
    
    # Round numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].round(4)
    
    return df
