"""
Doctrina Hyperparameter Tuning Module
Clean, modular implementation for automated hyperparameter optimization.

This module provides:
- Grid search for systematic parameter exploration
- Random search for efficient parameter space sampling  
- Easy integration with existing training pipeline
- Clear reporting and visualization of results
"""

import time
import itertools
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from sklearn.model_selection import train_test_split

from .model import StageRankModel, BinaryClassificationModel
from .metrics import compute_metrics, compute_binary_metrics


@dataclass
class TuningResult:
    """Container for hyperparameter tuning results"""
    params: Dict[str, Any]
    train_score: float
    val_score: float
    training_time: float
    score_metric: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HyperparameterTuner:
    """
    Clean, modular hyperparameter tuner for Doctrina models.
    
    Features:
    - Grid search and random search
    - Cross-validation support
    - Automatic metric selection based on mode
    - Progress tracking and early stopping
    - Result caching and reporting
    """
    
    def __init__(self, mode: str = 'regression', cv_folds: int = 3, random_seed: int = 42):
        """
        Initialize hyperparameter tuner.
        
        Args:
            mode: 'regression' or 'binary'
            cv_folds: Number of cross-validation folds
            random_seed: Random seed for reproducibility
        """
        self.mode = mode
        self.cv_folds = cv_folds
        self.random_seed = random_seed
        self.results: List[TuningResult] = []
        
        # Define scoring metrics based on mode
        if mode == 'regression':
            self.primary_metric = 'r2_score'
            self.metric_direction = 'maximize'  # Higher R² is better
        elif mode == 'binary':
            self.primary_metric = 'f1_score'
            self.metric_direction = 'maximize'  # Higher F1 is better
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def get_default_param_grid(self) -> Dict[str, List[Any]]:
        """Get sensible default parameter grid for the model type"""
        if self.mode == 'regression':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        else:  # binary
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
    
    def get_aggressive_param_grid(self) -> Dict[str, List[Any]]:
        """Get more comprehensive parameter grid for thorough search"""
        if self.mode == 'regression':
            return {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [3, 6, 9, 12],
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.5, 1.0, 2.0]
            }
        else:  # binary
            return {
                'n_estimators': [50, 100, 200, 500],
                'max_depth': [3, 6, 9, 12],
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.5, 1.0, 2.0]
            }
    
    def _evaluate_params(self, params: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> TuningResult:
        """Evaluate a single parameter combination using cross-validation"""
        start_time = time.time()
        
        # Perform cross-validation
        fold_scores_train = []
        fold_scores_val = []
        
        for fold in range(self.cv_folds):
            # Split data for this fold
            if self.mode == 'binary':
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_seed + fold, stratify=y
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_seed + fold
                )
            
            # Train model with current parameters
            if self.mode == 'regression':
                model = StageRankModel(**params)
            else:
                model = BinaryClassificationModel(**params)
            
            model.fit(X_train, y_train, validation_split=0.0)  # No internal validation
            
            # Evaluate on train and validation sets
            if self.mode == 'regression':
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                
                train_metrics = compute_metrics(y_train, train_pred, verbose=False)
                val_metrics = compute_metrics(y_val, val_pred, verbose=False)
                
            else:  # binary
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                train_proba = model.predict_proba(X_train)[:, 1]
                val_proba = model.predict_proba(X_val)[:, 1]
                
                train_metrics = compute_binary_metrics(y_train, train_pred, train_proba, verbose=False)
                val_metrics = compute_binary_metrics(y_val, val_pred, val_proba, verbose=False)
            
            fold_scores_train.append(train_metrics[self.primary_metric])
            fold_scores_val.append(val_metrics[self.primary_metric])
        
        # Average scores across folds
        avg_train_score = np.mean(fold_scores_train)
        avg_val_score = np.mean(fold_scores_val)
        training_time = time.time() - start_time
        
        return TuningResult(
            params=params.copy(),
            train_score=avg_train_score,
            val_score=avg_val_score,
            training_time=training_time,
            score_metric=self.primary_metric
        )
    
    def grid_search(self, X: np.ndarray, y: np.ndarray, 
                   param_grid: Optional[Dict[str, List[Any]]] = None,
                   max_combinations: Optional[int] = None,
                   verbose: bool = True) -> List[TuningResult]:
        """
        Perform grid search over parameter combinations.
        
        Args:
            X: Feature matrix
            y: Target vector
            param_grid: Parameter grid to search (uses default if None)
            max_combinations: Maximum number of combinations to try
            verbose: Print progress
            
        Returns:
            List of tuning results sorted by validation score
        """
        if param_grid is None:
            param_grid = self.get_default_param_grid()
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(itertools.product(*param_values))
        
        # Limit combinations if requested
        if max_combinations and len(all_combinations) > max_combinations:
            random.seed(self.random_seed)
            all_combinations = random.sample(all_combinations, max_combinations)
        
        if verbose:
            print(f"🔍 Grid Search: Testing {len(all_combinations)} parameter combinations")
            print(f"📊 Mode: {self.mode}, Metric: {self.primary_metric}, CV Folds: {self.cv_folds}")
        
        results = []
        for i, combination in enumerate(all_combinations):
            params = dict(zip(param_names, combination))
            
            if verbose and (i + 1) % max(1, len(all_combinations) // 10) == 0:
                print(f"   Progress: {i + 1}/{len(all_combinations)} ({(i + 1)/len(all_combinations)*100:.1f}%)")
            
            try:
                result = self._evaluate_params(params, X, y)
                results.append(result)
                
                if verbose:
                    print(f"   {params} → {self.primary_metric}: {result.val_score:.4f} (train: {result.train_score:.4f})")
                    
            except Exception as e:
                if verbose:
                    print(f"   ❌ Failed: {params} - {e}")
                continue
        
        # Sort by validation score
        if self.metric_direction == 'maximize':
            results.sort(key=lambda x: x.val_score, reverse=True)
        else:
            results.sort(key=lambda x: x.val_score)
        
        self.results.extend(results)
        return results
    
    def random_search(self, X: np.ndarray, y: np.ndarray,
                     param_distributions: Optional[Dict[str, List[Any]]] = None,
                     n_iter: int = 50,
                     verbose: bool = True) -> List[TuningResult]:
        """
        Perform random search over parameter space.
        
        Args:
            X: Feature matrix
            y: Target vector
            param_distributions: Parameter distributions to sample from
            n_iter: Number of parameter combinations to try
            verbose: Print progress
            
        Returns:
            List of tuning results sorted by validation score
        """
        if param_distributions is None:
            param_distributions = self.get_aggressive_param_grid()
        
        if verbose:
            print(f"🎲 Random Search: Testing {n_iter} random parameter combinations")
            print(f"📊 Mode: {self.mode}, Metric: {self.primary_metric}, CV Folds: {self.cv_folds}")
        
        results = []
        param_names = list(param_distributions.keys())
        
        for i in range(n_iter):
            # Sample random parameters
            params = {}
            for name in param_names:
                params[name] = random.choice(param_distributions[name])
            
            if verbose and (i + 1) % max(1, n_iter // 10) == 0:
                print(f"   Progress: {i + 1}/{n_iter} ({(i + 1)/n_iter*100:.1f}%)")
            
            try:
                result = self._evaluate_params(params, X, y)
                results.append(result)
                
                if verbose:
                    print(f"   {params} → {self.primary_metric}: {result.val_score:.4f}")
                    
            except Exception as e:
                if verbose:
                    print(f"   ❌ Failed: {params} - {e}")
                continue
        
        # Sort by validation score
        if self.metric_direction == 'maximize':
            results.sort(key=lambda x: x.val_score, reverse=True)
        else:
            results.sort(key=lambda x: x.val_score)
        
        self.results.extend(results)
        return results
    
    def get_best_params(self, top_k: int = 1) -> List[Dict[str, Any]]:
        """Get the best parameter combinations from all searches"""
        if not self.results:
            return []
        
        # Sort all results
        if self.metric_direction == 'maximize':
            sorted_results = sorted(self.results, key=lambda x: x.val_score, reverse=True)
        else:
            sorted_results = sorted(self.results, key=lambda x: x.val_score)
        
        return [result.params for result in sorted_results[:top_k]]
    
    def print_summary(self, top_k: int = 5) -> None:
        """Print summary of tuning results"""
        if not self.results:
            print("🚫 No tuning results available")
            return
        
        # Sort results
        if self.metric_direction == 'maximize':
            sorted_results = sorted(self.results, key=lambda x: x.val_score, reverse=True)
        else:
            sorted_results = sorted(self.results, key=lambda x: x.val_score)
        
        print(f"\n📊 Hyperparameter Tuning Summary")
        print(f"=" * 50)
        print(f"Mode: {self.mode}")
        print(f"Metric: {self.primary_metric} ({'higher is better' if self.metric_direction == 'maximize' else 'lower is better'})")
        print(f"Total combinations tested: {len(self.results)}")
        print(f"Cross-validation folds: {self.cv_folds}")
        
        print(f"\n🏆 Top {min(top_k, len(sorted_results))} Results:")
        for i, result in enumerate(sorted_results[:top_k]):
            print(f"\n#{i+1}: {self.primary_metric} = {result.val_score:.4f} (train: {result.train_score:.4f})")
            print(f"   Training time: {result.training_time:.2f}s")
            print(f"   Parameters: {result.params}")
        
        # Show improvement over default
        if len(sorted_results) > 1:
            best_score = sorted_results[0].val_score
            default_params = self.get_default_param_grid()
            # Find result closest to default params (approximate)
            default_score = None
            for result in sorted_results:
                if all(result.params.get(k) in v for k, v in default_params.items() if k in result.params):
                    default_score = result.val_score
                    break
            
            if default_score:
                improvement = ((best_score - default_score) / abs(default_score)) * 100
                print(f"\n📈 Improvement over default: {improvement:+.1f}%")
    
    def save_results(self, output_path: str) -> None:
        """Save tuning results to JSON file"""
        results_data = {
            'mode': self.mode,
            'primary_metric': self.primary_metric,
            'cv_folds': self.cv_folds,
            'random_seed': self.random_seed,
            'total_combinations': len(self.results),
            'results': [result.to_dict() for result in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Tuning results saved to: {output_path}")


def quick_tune(X: np.ndarray, y: np.ndarray, mode: str = 'regression', 
               method: str = 'grid', verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function for quick hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Target vector
        mode: 'regression' or 'binary'
        method: 'grid' or 'random'
        verbose: Print progress
        
    Returns:
        Best parameters found
    """
    tuner = HyperparameterTuner(mode=mode, cv_folds=3)
    
    if method == 'grid':
        results = tuner.grid_search(X, y, verbose=verbose, max_combinations=20)
    elif method == 'random':
        results = tuner.random_search(X, y, n_iter=20, verbose=verbose)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if verbose:
        tuner.print_summary(top_k=3)
    
    best_params = tuner.get_best_params(top_k=1)
    return best_params[0] if best_params else {} 