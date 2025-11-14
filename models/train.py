"""
Training module for Student Grade Predictor
Uses OULAD dataset for student outcome prediction
Trains Logistic Regression and Random Forest Classifier models
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score
)
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocess import preprocess_oulad_data


class StudentGradePredictor:
    """
    Main class for training and evaluating student outcome prediction models
    using OULAD data.
    """
    
    def __init__(self, models_dir='models'):
        """
        Initialize the predictor.
        
        Args:
            models_dir (str): Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.models = {
            'logistic_regression': None,
            'random_forest': None
        }
        
        self.scalers = {}
        self.encoders = {}
        self.label_encoder = None
        self.feature_names = []
        self.y_labels = []
        self.results = {}
    
    def train_models(self):
        """
        Train both Logistic Regression and Random Forest Classifier models
        using OULAD data.
        
        Returns:
            dict: Training and evaluation results
        """
        print("\n" + "="*60)
        print("TRAINING STUDENT OUTCOME PREDICTION MODELS (OULAD)")
        print("="*60)
        
        # Preprocess OULAD data
        data = preprocess_oulad_data()
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
        
        self.scalers['standard'] = data['scaler']
        self.encoders = data['encoders']
        self.label_encoder = data['label_encoder']
        self.feature_names = data['feature_names']
        self.y_labels = data['y_labels']
        
        results = {}
        
        # Train Logistic Regression
        print("\n" + "-"*60)
        print("Training Logistic Regression Classifier...")
        print("-"*60)
        lr_model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
        lr_model.fit(X_train, y_train)
        self.models['logistic_regression'] = lr_model
        
        lr_pred_train = lr_model.predict(X_train)
        lr_pred_test = lr_model.predict(X_test)
        
        results['logistic_regression'] = self._evaluate_classifier(
            'Logistic Regression', y_train, y_test, lr_pred_train, lr_pred_test
        )
        
        # Train Random Forest Classifier
        print("\n" + "-"*60)
        print("Training Random Forest Classifier...")
        print("-"*60)
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=15,
            min_samples_split=5
        )
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model
        
        rf_pred_train = rf_model.predict(X_train)
        rf_pred_test = rf_model.predict(X_test)
        
        results['random_forest'] = self._evaluate_classifier(
            'Random Forest', y_train, y_test, rf_pred_train, rf_pred_test
        )
        
        self.results = results
        
        # Save models
        self._save_models()
        
        return results, data
    
    def _evaluate_classifier(self, model_name, y_train, y_test, y_pred_train, y_pred_test):
        """
        Evaluate classifier model performance using classification metrics.
        
        Args:
            model_name (str): Name of the model
            y_train, y_test: Training and test targets
            y_pred_train, y_pred_test: Training and test predictions
        
        Returns:
            dict: Evaluation metrics
        """
        # Training metrics
        train_acc = accuracy_score(y_train, y_pred_train)
        train_prec = precision_score(y_train, y_pred_train, average='weighted', zero_division=0)
        train_rec = recall_score(y_train, y_pred_train, average='weighted', zero_division=0)
        train_f1 = f1_score(y_train, y_pred_train, average='weighted', zero_division=0)
        
        # Test metrics
        test_acc = accuracy_score(y_test, y_pred_test)
        test_prec = precision_score(y_test, y_pred_test, average='weighted', zero_division=0)
        test_rec = recall_score(y_test, y_pred_test, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        
        print(f"\n{model_name} Results:")
        print(f"  Train Accuracy:  {train_acc:.4f}")
        print(f"  Train Precision: {train_prec:.4f}")
        print(f"  Train Recall:    {train_rec:.4f}")
        print(f"  Train F1:        {train_f1:.4f}")
        print(f"  Test Accuracy:   {test_acc:.4f}")
        print(f"  Test Precision:  {test_prec:.4f}")
        print(f"  Test Recall:     {test_rec:.4f}")
        print(f"  Test F1:         {test_f1:.4f}")
        
        # Classification report
        print(f"\nClassification Report for {model_name}:")
        print(classification_report(y_test, y_pred_test, 
                                   target_names=self.label_encoder.classes_.tolist(),
                                   zero_division=0))
        
        return {
            'train_accuracy': train_acc,
            'train_precision': train_prec,
            'train_recall': train_rec,
            'train_f1': train_f1,
            'test_accuracy': test_acc,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_f1': test_f1,
            'y_pred_test': y_pred_test,
            'y_test': y_test
        }
    
    def _save_models(self):
        """Save trained models to disk."""
        for model_name, model in self.models.items():
            if model is not None:
                filepath = os.path.join(self.models_dir, f'{model_name}.pkl')
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
                print(f"✅ Saved {model_name} to {filepath}")
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers['standard'], f)
        print(f"✅ Saved scaler to {scaler_path}")
        
        # Save encoders
        encoders_path = os.path.join(self.models_dir, 'encoders.pkl')
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.encoders, f)
        print(f"✅ Saved encoders to {encoders_path}")
        
        # Save label encoder
        label_encoder_path = os.path.join(self.models_dir, 'label_encoder.pkl')
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"✅ Saved label encoder to {label_encoder_path}")
        
        # Save feature names
        feature_names_path = os.path.join(self.models_dir, 'feature_names.pkl')
        with open(feature_names_path, 'wb') as f:
            pickle.dump({
                'feature_names': self.feature_names,
                'y_labels': self.y_labels
            }, f)
        print(f"✅ Saved feature names to {feature_names_path}")
    
    def get_model_comparison(self):
        """
        Get a comparison table of all classification models.
        
        Returns:
            pd.DataFrame: Model comparison table
        """
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Train Accuracy': f"{metrics['train_accuracy']:.4f}",
                'Test Accuracy': f"{metrics['test_accuracy']:.4f}",
                'Test Precision': f"{metrics['test_precision']:.4f}",
                'Test Recall': f"{metrics['test_recall']:.4f}",
                'Test F1': f"{metrics['test_f1']:.4f}",
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*60)
        print("MODEL COMPARISON (OULAD Classification)")
        print("="*60)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def get_feature_importance(self):
        """
        Get feature importance from Random Forest Classifier.
        
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.models['random_forest'] is None:
            return None
        
        importances = self.models['random_forest'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def get_logistic_coefficients(self):
        """
        Get coefficients from Logistic Regression.
        
        Returns:
            pd.DataFrame: Coefficients dataframe
        """
        if self.models['logistic_regression'] is None:
            return None
        
        coefficients = self.models['logistic_regression'].coef_
        
        # For multiclass, average across classes
        mean_coef = np.abs(coefficients).mean(axis=0)
        
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': mean_coef
        }).sort_values('Coefficient', ascending=False)
        
        return coef_df


def main():
    """Main training script for OULAD models."""
    predictor = StudentGradePredictor(models_dir='models')
    results, data = predictor.train_models()
    
    # Print comparison table
    comparison_df = predictor.get_model_comparison()
    
    # Print feature importance
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE (Random Forest)")
    print("="*60)
    importance_df = predictor.get_feature_importance()
    print(importance_df.to_string(index=False))
    
    # Print coefficients
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION COEFFICIENTS")
    print("="*60)
    coef_df = predictor.get_logistic_coefficients()
    print(coef_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Models saved to models/ directory")
    print(f"Target classes: {predictor.y_labels}")


if __name__ == '__main__':
    main()
