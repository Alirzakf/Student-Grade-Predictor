"""
Data preprocessing module for Student Grade Predictor
Uses OULAD (Open University Learning Analytics Dataset)
Handles missing values, scaling, encoding, and train/test split
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

# Import OULAD downloader
try:
    from .download_oulad import load_or_generate_oulad
except ImportError:
    from download_oulad import load_or_generate_oulad


# OULAD Feature Configuration
CATEGORICAL_FEATURES = ['age_band', 'gender', 'disability', 'region']
NUMERIC_FEATURES = ['studied_credits', 'num_of_prev_attempts', 'clicks_total', 
                    'assessment_score_avg', 'days_since_registration']
TARGET_VARIABLE = 'final_result'  # Can be 'Pass', 'Fail', 'Withdrawn', 'Distinction'


def load_oulad_data():
    """
    Load OULAD dataset (real or generated sample).
    
    Returns:
        pd.DataFrame: OULAD dataset with features
    """
    print("Loading OULAD dataset...")
    df = load_or_generate_oulad()
    
    print(f"\n📊 Dataset Information:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    return df


def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values in OULAD dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): Strategy for handling missing values ('mean', 'median', 'drop')
    
    Returns:
        pd.DataFrame: DataFrame with missing values handled
    """
    if df.isnull().sum().sum() == 0:
        print("✅ No missing values found.")
        return df
    
    print(f"🔧 Handling missing values using '{strategy}' strategy...")
    
    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if strategy == 'mean':
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].mean())
    elif strategy == 'median':
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    if strategy == 'drop':
        df = df.dropna()
    
    print(f"✅ Missing values handled. Remaining: {df.isnull().sum().sum()}")
    return df


def encode_categorical_features(df, feature_names=None, fit_encoders=False, encoders=None):
    """
    Encode categorical features using LabelEncoder.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature_names (list): Names of features to encode
        fit_encoders (bool): Whether to fit new encoders
        encoders (dict): Pre-fitted encoders
    
    Returns:
        pd.DataFrame, dict: Encoded dataframe and encoders dictionary
    """
    if feature_names is None:
        feature_names = CATEGORICAL_FEATURES
    
    if encoders is None:
        encoders = {}
    
    df_encoded = df.copy()
    
    for feature in feature_names:
        if feature in df_encoded.columns:
            if fit_encoders or feature not in encoders:
                encoders[feature] = LabelEncoder()
                df_encoded[feature] = encoders[feature].fit_transform(df_encoded[feature].astype(str))
            else:
                df_encoded[feature] = encoders[feature].transform(df_encoded[feature].astype(str))
    
    print(f"✅ Encoded {len(encoders)} categorical features")
    return df_encoded, encoders


def preprocess_oulad_data(test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline for OULAD data:
    Load, clean, encode, scale, and split data.
    
    Args:
        test_size (float): Fraction of data for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        dict: Dictionary containing:
            - X_train, X_test, y_train, y_test (scaled)
            - scaler: Fitted StandardScaler
            - encoders: Fitted LabelEncoders
            - feature_names: List of all feature names
            - numeric_features: List of numeric feature names
            - categorical_features: List of categorical feature names
            - original_df: Original dataframe
            - label_encoder: Encoder for target variable
    """
    print("\n" + "="*60)
    print("OULAD DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load OULAD data
    df = load_oulad_data()
    original_df = df.copy()
    
    # Handle missing values
    df = handle_missing_values(df, strategy='mean')
    
    # Check for target variable
    if TARGET_VARIABLE not in df.columns:
        print(f"⚠️  Target variable '{TARGET_VARIABLE}' not found in dataset")
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError(f"Target variable '{TARGET_VARIABLE}' not in dataset")
    
    # Select only OULAD features and target
    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    available_cols = [col for col in feature_cols if col in df.columns]
    df = df[available_cols + [TARGET_VARIABLE]].copy()
    
    # Separate features and target
    X = df.drop(TARGET_VARIABLE, axis=1)
    y = df[TARGET_VARIABLE]
    
    # Encode categorical features in X
    X, encoders = encode_categorical_features(X, feature_names=CATEGORICAL_FEATURES, 
                                              fit_encoders=True, encoders=None)
    
    # Encode target variable if it's categorical
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))
    
    print(f"\n✅ Encoded target variable. Classes: {label_encoder.classes_.tolist()}")
    
    # Store feature information
    all_feature_names = X.columns.tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Scale numeric features only (don't convert already-encoded categorical to float)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Fit scaler on numeric features in training data
    numeric_cols = [col for col in all_feature_names if col in NUMERIC_FEATURES]
    if numeric_cols:
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    print(f"\n✅ Features scaled: {numeric_cols}")
    print(f"   Categorical features (encoded): {CATEGORICAL_FEATURES}")
    
    return {
        'X_train': X_train_scaled.astype(float).values,
        'X_test': X_test_scaled.astype(float).values,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'encoders': encoders,
        'label_encoder': label_encoder,
        'feature_names': all_feature_names,
        'numeric_features': numeric_cols,
        'categorical_features': CATEGORICAL_FEATURES,
        'original_df': original_df,
        'X_train_df': X_train,
        'X_test_df': X_test,
        'y_labels': label_encoder.classes_.tolist()
    }


# Keep old function for backwards compatibility
def preprocess_data(data_path='data/students.csv', test_size=0.2, random_state=42):
    """
    Legacy function - calls OULAD preprocessing.
    For backwards compatibility.
    """
    return preprocess_oulad_data(test_size=test_size, random_state=random_state)


if __name__ == '__main__':
    # Test preprocessing with OULAD data
    data = preprocess_oulad_data()
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"X_test shape: {data['X_test'].shape}")
    print(f"Features: {data['feature_names']}")
    print(f"Target classes: {data['y_labels']}")

