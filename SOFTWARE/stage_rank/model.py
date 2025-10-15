"""
ML models for Doctrina pipeline.
Uses XGBoost regressor for stage ranking and XGBoost classifier for binary classification.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, Any, Tuple
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, classification_report
from .io_utils import print_info, print_success, show_progress_bar


class StageRankModel:
    """
    XGBoost regression model for candidate stage ranking.
    """
    
    def __init__(self, **model_params):
        """
        Initialize the model with XGBoost parameters.
        
        Args:
            **model_params: XGBoost regressor parameters
        """
        # Default parameters optimized for stage ranking
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse'
        }
        
        # Update with provided parameters
        default_params.update(model_params)
        
        self.model_params = default_params
        self.model = xgb.XGBRegressor(**default_params)
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Fit the XGBoost model.
        
        Args:
            X: Feature matrix
            y: Target values
            validation_split: Fraction for validation set
            
        Returns:
            Training history and metrics
        """
        print_info("Training XGBoost model...", "🚀")
        
        # Show training progress
        print_info(f"Training samples: {len(X):,}", "📊")
        print_info(f"Features: {X.shape[1]:,}", "🔧")
        
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )
            
            # Train with progress bar
            print_info("Training progress:", "🚀")
            n_estimators = self.model_params.get('n_estimators', 100)
            
            # Create a custom callback to show progress
            class ProgressCallback:
                def __init__(self, n_estimators):
                    self.n_estimators = n_estimators
                    self.current = 0
                
                def __call__(self, env):
                    self.current = env.iteration + 1
                    show_progress_bar(
                        self.current, 
                        self.n_estimators, 
                        f"   Training trees"
                    )
                    if self.current == self.n_estimators:
                        print()  # New line after completion
            
            progress_callback = ProgressCallback(n_estimators)
            
            # Train with early stopping and progress callback
            try:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False,
                    callbacks=[progress_callback]
                )
            except TypeError:
                # Fallback for older XGBoost versions - use verbose mode for real progress
                print_info(f"Training XGBoost Regression (500 trees, ~2-3 min). Every 50 trees shown:", "📊")
                self.model.set_params(verbosity=0)  # Quiet mode, use our verbose parameter
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=50  # Print every 50 iterations: [0], [50], [100]...[450]
                )
                print_info("Training complete!", "✅")
                print()  # New line
            
            # Get validation predictions for metrics
            val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
            val_mae = np.mean(np.abs(y_val - val_pred))
            
            print_info(f"Validation RMSE: {val_rmse:.4f}", "📊")
            print_info(f"Validation MAE: {val_mae:.4f}", "📊")
            
            history = {
                'validation_rmse': val_rmse,
                'validation_mae': val_mae,
                'training_samples': len(X_train),
                'validation_samples': len(X_val)
            }
        else:
            # Train on full dataset
            self.model.fit(X, y)
            
            train_pred = self.model.predict(X)
            train_rmse = np.sqrt(np.mean((y - train_pred) ** 2))
            train_mae = np.mean(np.abs(y - train_pred))
            
            print(f"   📊 Training RMSE: {train_rmse:.4f}")
            print(f"   📊 Training MAE: {train_mae:.4f}")
            
            history = {
                'training_rmse': train_rmse,
                'training_mae': train_mae,
                'training_samples': len(X)
            }
        
        self.is_fitted = True
        print_success("Model training completed")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted stage scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        predictions = self.model.predict(X)
        
        # Ensure predictions are in valid range [0, 1]
        predictions = np.clip(predictions, 0.0, 1.0)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities (for regression, returns confidence scores).
        
        Args:
            X: Feature matrix
            
        Returns:
            Confidence scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # For regression, we can use leaf values as confidence measure
        predictions = self.predict(X)
        
        # Simple confidence based on prediction certainty
        # Higher confidence for predictions near 0 or 1, lower for middle values
        confidence = 1.0 - 2.0 * np.abs(predictions - 0.5)
        confidence = np.clip(confidence, 0.1, 1.0)  # Min confidence of 0.1
        
        return confidence
    
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        importance = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save(self, path: str):
        """Save the fitted model."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"   💾 Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'StageRankModel':
        """Load a fitted model."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"   📂 Model loaded from {path}")
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and parameters."""
        info = {
            'model_type': 'XGBoost Regressor',
            'is_fitted': self.is_fitted,
            'parameters': self.model_params
        }
        
        if self.is_fitted:
            info['n_features'] = self.model.n_features_in_
            info['best_iteration'] = getattr(self.model, 'best_iteration', None)
        
        return info


class BinaryClassificationModel:
    """
    XGBoost binary classification model for candidate classification.
    """
    
    def __init__(self, **model_params):
        """
        Initialize the model with XGBoost parameters.
        
        Args:
            **model_params: XGBoost classifier parameters
        """
        # Default parameters optimized for binary classification
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        
        # Update with provided parameters
        default_params.update(model_params)
        
        self.model_params = default_params
        self.model = xgb.XGBClassifier(**default_params)
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Fit the XGBoost binary classifier.
        
        Args:
            X: Feature matrix
            y: Target values (0 or 1)
            validation_split: Fraction for validation set
            
        Returns:
            Training history and metrics
        """
        print_info("Training XGBoost binary classifier...", "🚀")
        
        # Show training progress
        print_info(f"Training samples: {len(X):,}", "📊")
        print_info(f"Features: {X.shape[1]:,}", "🔧")
        print_info(f"Class distribution: {np.bincount(y.astype(int))}", "⚖️")
        
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42, stratify=y
            )
            
            # Train with progress bar
            print_info("Training progress:", "🚀")
            n_estimators = self.model_params.get('n_estimators', 100)
            
            # Create a custom callback to show progress
            class ProgressCallback:
                def __init__(self, n_estimators):
                    self.n_estimators = n_estimators
                    self.current = 0
                
                def __call__(self, env):
                    self.current = env.iteration + 1
                    show_progress_bar(
                        self.current, 
                        self.n_estimators, 
                        f"   Training trees"
                    )
                    if self.current == self.n_estimators:
                        print()  # New line after completion
            
            progress_callback = ProgressCallback(n_estimators)
            
            # Train with early stopping and progress callback
            try:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    early_stopping_rounds=10,
                    verbose=False,
                    callbacks=[progress_callback]
                )
            except TypeError:
                # Fallback for older XGBoost versions - use verbose mode for real progress
                print_info(f"Training XGBoost Binary (500 trees, ~2-3 min). Every 50 trees shown:", "📊")
                self.model.set_params(verbosity=0)  # Quiet mode, use our verbose parameter
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=50  # Print every 50 iterations: [0], [50], [100]...[450]
                )
                print_info("Training complete!", "✅")
                print()  # New line
            
            # Get validation predictions for metrics
            val_pred = self.model.predict(X_val)
            val_pred_proba = self.model.predict_proba(X_val)[:, 1]
            
            val_f1 = f1_score(y_val, val_pred)
            val_auc = roc_auc_score(y_val, val_pred_proba)
            val_precision = precision_score(y_val, val_pred)
            val_recall = recall_score(y_val, val_pred)
            val_accuracy = accuracy_score(y_val, val_pred)
            
            print_info(f"Validation F1: {val_f1:.4f}", "📊")
            print_info(f"Validation AUC: {val_auc:.4f}", "📊")
            print_info(f"Validation Accuracy: {val_accuracy:.4f}", "📊")
            
            history = {
                'validation_f1': val_f1,
                'validation_auc': val_auc,
                'validation_precision': val_precision,
                'validation_recall': val_recall,
                'validation_accuracy': val_accuracy,
                'training_samples': len(X_train),
                'validation_samples': len(X_val)
            }
        else:
            # Train on full dataset
            self.model.fit(X, y)
            
            train_pred = self.model.predict(X)
            train_pred_proba = self.model.predict_proba(X)[:, 1]
            
            train_f1 = f1_score(y, train_pred)
            train_auc = roc_auc_score(y, train_pred_proba)
            train_accuracy = accuracy_score(y, train_pred)
            
            print(f"   📊 Training F1: {train_f1:.4f}")
            print(f"   📊 Training AUC: {train_auc:.4f}")
            print(f"   📊 Training Accuracy: {train_accuracy:.4f}")
            
            history = {
                'training_f1': train_f1,
                'training_auc': train_auc,
                'training_accuracy': train_accuracy,
                'training_samples': len(X)
            }
        
        self.is_fitted = True
        print_success("Binary classifier training completed")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted classes (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities for each class
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        importance = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save(self, path: str):
        """Save the fitted model."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"   💾 Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BinaryClassificationModel':
        """Load a fitted model."""
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print(f"   📂 Model loaded from {path}")
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and parameters."""
        info = {
            'model_type': 'XGBoost Binary Classifier',
            'is_fitted': self.is_fitted,
            'parameters': self.model_params
        }
        
        if self.is_fitted:
            info['n_features'] = self.model.n_features_in_
            info['best_iteration'] = getattr(self.model, 'best_iteration', None)
        
        return info
