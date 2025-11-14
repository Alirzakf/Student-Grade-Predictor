"""
Visualization module for Student Grade Predictor
Creates plots for data analysis and model interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


def plot_correlation_heatmap(df, figsize=(10, 8)):
    """
    Create a correlation heatmap for all numerical features.
    
    Args:
        df (pd.DataFrame): Dataset with numerical and categorical columns
        figsize (tuple): Figure size
    
    Returns:
        plt.Figure: Matplotlib figure object
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    plt.figure(figsize=figsize)
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', cbar_kws={'label': 'Correlation'})
    plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return plt


def plot_feature_importance(importances, feature_names, model_name='Random Forest'):
    """
    Create a bar chart of feature importances.
    
    Args:
        importances (array): Array of feature importance values
        feature_names (list): List of feature names
        model_name (str): Name of the model
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Ensure feature_names is a list
    if not isinstance(feature_names, (list, np.ndarray)):
        feature_names = list(feature_names) if hasattr(feature_names, '__iter__') else [str(feature_names)]
    else:
        feature_names = list(feature_names)
    
    # Ensure importances is a list
    importances_list = list(importances) if not isinstance(importances, list) else importances
    
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances_list
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_importance['Feature'],
        x=df_importance['Importance'],
        orientation='h',
        marker=dict(color='steelblue')
    ))
    
    fig.update_layout(
        title=f'{model_name} - Feature Importance',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        height=400,
        showlegend=False
    )
    
    return fig


def plot_linear_coefficients(coefficients, feature_names):
    """
    Create a bar chart of linear regression coefficients.
    
    Args:
        coefficients (array): Array of coefficients
        feature_names (list): List of feature names
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Ensure feature_names is a list
    if not isinstance(feature_names, (list, np.ndarray)):
        feature_names = list(feature_names) if hasattr(feature_names, '__iter__') else [str(feature_names)]
    else:
        feature_names = list(feature_names)
    
    # Handle multi-dimensional coefficients (for multi-class)
    if len(coefficients.shape) > 1:
        # Take mean across classes for visualization
        coefficients = coefficients.mean(axis=0)
    else:
        coefficients = list(coefficients) if not isinstance(coefficients, list) else coefficients
    
    df_coef = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values('Coefficient', ascending=True)
    
    colors = ['red' if x < 0 else 'green' for x in df_coef['Coefficient']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_coef['Feature'],
        x=df_coef['Coefficient'],
        orientation='h',
        marker=dict(color=colors)
    ))
    
    fig.update_layout(
        title='Linear Regression - Feature Coefficients',
        xaxis_title='Coefficient Value',
        yaxis_title='Features',
        height=400,
        showlegend=False
    )
    
    return fig


def plot_predictions_vs_actual(y_true, y_pred, model_name='Model'):
    """
    Create a scatter plot comparing predictions vs actual values.
    
    Args:
        y_true (array): Actual values
        y_pred (array): Predicted values
        model_name (str): Name of the model
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    fig = go.Figure()
    
    # Scatter plot
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode='markers',
        marker=dict(size=8, color='steelblue', opacity=0.6),
        name='Predictions'
    ))
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title=f'{model_name} - Predictions vs Actual',
        xaxis_title='Actual Grade',
        yaxis_title='Predicted Grade',
        height=500,
        showlegend=True
    )
    
    return fig


def plot_residuals(y_true, y_pred, model_name='Model'):
    """
    Create a residuals plot.
    
    Args:
        y_true (array): Actual values
        y_pred (array): Predicted values
        model_name (str): Name of the model
    
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    residuals = y_true - y_pred
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals,
        mode='markers',
        marker=dict(size=8, color='steelblue', opacity=0.6)
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash='dash', line_color='red')
    
    fig.update_layout(
        title=f'{model_name} - Residuals Plot',
        xaxis_title='Predicted Grade',
        yaxis_title='Residuals',
        height=500,
        showlegend=False
    )
    
    return fig


if __name__ == '__main__':
    print("Visualization module loaded successfully!")
