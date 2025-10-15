"""
Metrics and evaluation utilities for Doctrina stage ranking pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix
from scipy.stats import pearsonr, spearmanr


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   verbose: bool = True) -> Dict[str, float]:
    """
    Compute comprehensive metrics for stage ranking predictions.
    
    Args:
        y_true: True stage scores
        y_pred: Predicted stage scores
        verbose: Whether to print metrics
        
    Returns:
        Dictionary of computed metrics
    """
    # Basic regression metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Correlation metrics (important for ranking)
    pearson_corr, pearson_p = pearsonr(y_true, y_pred)
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    
    # Stage-specific metrics
    stage_accuracy = compute_stage_accuracy(y_true, y_pred)
    ranking_quality = compute_ranking_quality(y_true, y_pred)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'pearson_correlation': pearson_corr,
        'spearman_correlation': spearman_corr,
        'stage_accuracy': stage_accuracy,
        'ranking_quality': ranking_quality,
        'mean_prediction': np.mean(y_pred),
        'std_prediction': np.std(y_pred),
        'prediction_range': np.max(y_pred) - np.min(y_pred),
        'min_pred': np.min(y_pred),
        'max_pred': np.max(y_pred),
        'min_target': np.min(y_true),
        'max_target': np.max(y_true)
    }
    
    if verbose:
        print("\n📊 Model Performance Metrics")
        print("=" * 35)
        print(f"🎯 RMSE:              {rmse:.4f}")
        print(f"🎯 MAE:               {mae:.4f}")
        print(f"🎯 R² Score:          {r2:.4f}")
        print(f"📈 Pearson Corr:      {pearson_corr:.4f}")
        print(f"📈 Spearman Corr:     {spearman_corr:.4f}")
        print(f"🏆 Stage Accuracy:    {stage_accuracy:.1%}")
        print(f"🏆 Ranking Quality:   {ranking_quality:.4f}")
        print(f"📊 Prediction Range:  [{np.min(y_pred):.3f}, {np.max(y_pred):.3f}]")
    
    return metrics


def compute_stage_accuracy(y_true: np.ndarray, y_pred: np.ndarray, 
                          tolerance: float = 0.1) -> float:
    """
    Compute stage accuracy - percentage of predictions within tolerance.
    
    Args:
        y_true: True stage scores
        y_pred: Predicted stage scores
        tolerance: Acceptable prediction error
        
    Returns:
        Accuracy percentage (0-1)
    """
    errors = np.abs(y_true - y_pred)
    within_tolerance = errors <= tolerance
    accuracy = np.mean(within_tolerance)
    
    return accuracy


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None,
                          verbose: bool = True) -> Dict[str, float]:
    """
    Compute comprehensive metrics for binary classification.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for positive class
        verbose: Whether to print metrics
        
    Returns:
        Dictionary of computed metrics
    """
    # Basic classification metrics
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    # AUC-ROC (if probabilities provided)
    auc_roc = None
    if y_pred_proba is not None:
        auc_roc = roc_auc_score(y_true, y_pred_proba)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'auc_roc': auc_roc or 0.0,
        'confusion_matrix': {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        },
        'class_distribution': {
            'class_0': int(np.sum(y_true == 0)),
            'class_1': int(np.sum(y_true == 1))
        }
    }
    
    if verbose:
        print("\n📊 Binary Classification Metrics")
        print("=" * 40)
        print(f"🎯 F1 Score:          {f1:.4f}")
        print(f"🎯 Precision:         {precision:.4f}")
        print(f"🎯 Recall:            {recall:.4f}")
        print(f"🎯 Accuracy:          {accuracy:.4f}")
        if auc_roc:
            print(f"📈 AUC-ROC:           {auc_roc:.4f}")
        print(f"📊 Confusion Matrix:")
        print(f"     TN: {tn:4d}  FP: {fp:4d}")
        print(f"     FN: {fn:4d}  TP: {tp:4d}")
    
    return metrics


def compute_ranking_quality(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute ranking quality using normalized discounted cumulative gain.
    
    Args:
        y_true: True stage scores
        y_pred: Predicted stage scores
        
    Returns:
        NDCG score (0-1, higher is better)
    """
    # Sort by predictions (descending)
    sorted_indices = np.argsort(y_pred)[::-1]
    sorted_true = y_true[sorted_indices]
    
    # Compute DCG
    dcg = 0.0
    for i, relevance in enumerate(sorted_true):
        dcg += relevance / np.log2(i + 2)  # +2 because log2(1) = 0
    
    # Compute ideal DCG
    ideal_sorted = np.sort(y_true)[::-1]
    idcg = 0.0
    for i, relevance in enumerate(ideal_sorted):
        idcg += relevance / np.log2(i + 2)
    
    # Compute NDCG
    if idcg == 0:
        return 0.0
    
    ndcg = dcg / idcg
    return ndcg


def analyze_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                       stage_names: list = None) -> pd.DataFrame:
    """
    Analyze predictions by stage ranges.
    
    Args:
        y_true: True stage scores
        y_pred: Predicted stage scores
        stage_names: Optional list of stage names for interpretation
        
    Returns:
        DataFrame with analysis by stage ranges
    """
    # Define stage ranges
    if stage_names is None:
        stage_ranges = [
            (0.0, 0.2, "Early"),
            (0.2, 0.4, "Contacted"),
            (0.4, 0.6, "Qualified"),
            (0.6, 0.8, "Advanced"),
            (0.8, 1.0, "Closing")
        ]
    else:
        # Create ranges based on stage names
        n_stages = len(stage_names)
        stage_ranges = []
        for i, name in enumerate(stage_names):
            start = i / n_stages
            end = (i + 1) / n_stages
            stage_ranges.append((start, end, name))
    
    analysis_data = []
    
    for start, end, name in stage_ranges:
        # Find samples in this stage range
        in_range = (y_true >= start) & (y_true < end)
        if not np.any(in_range):
            continue
        
        true_vals = y_true[in_range]
        pred_vals = y_pred[in_range]
        
        # Compute metrics for this stage
        stage_metrics = {
            'stage': name,
            'count': len(true_vals),
            'true_mean': np.mean(true_vals),
            'pred_mean': np.mean(pred_vals),
            'rmse': np.sqrt(np.mean((true_vals - pred_vals) ** 2)),
            'mae': np.mean(np.abs(true_vals - pred_vals)),
            'accuracy_10%': np.mean(np.abs(true_vals - pred_vals) <= 0.1)
        }
        
        analysis_data.append(stage_metrics)
    
    df = pd.DataFrame(analysis_data)
    return df


def create_performance_summary(metrics: Dict[str, Any], 
                             model_info: Dict[str, Any],
                             training_time: float = None,
                             config: Dict[str, Any] = None,
                             column_info: Dict[str, Any] = None,
                             mode: str = 'regression') -> Dict[str, Any]:
    """
    Create a comprehensive performance summary for research documentation.
    
    Args:
        metrics: Performance metrics (can be flat dict or nested with train/val/test)
        model_info: Model information
        training_time: Training time in seconds
        config: Training configuration
        column_info: Column usage information
        
    Returns:
        Complete performance summary
    """
    summary = {
        'experiment_info': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'training_time_seconds': training_time or 'Unknown',
            'pipeline_version': '1.0.0',
            'mode': mode
        },
        
        'dataset_info': {
            'full_dataset_samples': model_info.get('full_dataset_samples', 'Unknown'),
            'training_samples': model_info.get('training_samples', 'Unknown'),
            'validation_samples': model_info.get('validation_samples', 'Unknown'),
            'test_samples': model_info.get('test_samples', 'Unknown'),
            'n_features_generated': model_info.get('n_features', 'Unknown'),
            'target_column': config.get('target_column', 'Unknown') if config else 'Unknown'
        },
        
        'column_configuration': column_info or {},
        
        'model_configuration': {
            'model_type': model_info.get('model_type', 'XGBoost'),
            'text_method': config.get('text_method', 'Unknown') if config else 'Unknown',
            'max_tfidf_features': config.get('max_tfidf_features', 'Unknown') if config else 'Unknown',
            'validation_split': config.get('train_test_split', 'Unknown') if config else 'Unknown',
            'random_seed': config.get('random_seed', 'Unknown') if config else 'Unknown'
        },
        
        'hyperparameters': {
            'n_estimators': config.get('n_estimators', 'Unknown') if config else 'Unknown',
            'max_depth': config.get('max_depth', 'Unknown') if config else 'Unknown',
            'learning_rate': config.get('learning_rate', 'Unknown') if config else 'Unknown',
            'subsample': config.get('subsample', 'Unknown') if config else 'Unknown',
            'colsample_bytree': config.get('colsample_bytree', 'Unknown') if config else 'Unknown',
            'regularization': {
                'reg_alpha': config.get('reg_alpha', 'Unknown') if config else 'Unknown',
                'reg_lambda': config.get('reg_lambda', 'Unknown') if config else 'Unknown'
            }
        },
        
        'performance_metrics': {}
    }
    
    # Handle both old flat format and new nested format
    if 'training' in metrics and 'validation' in metrics and 'test' in metrics:
        # New nested format with train/val/test splits
        train_metrics = metrics['training']
        val_metrics = metrics['validation']
        test_metrics = metrics['test']
        
        if mode == 'regression':
            summary['performance_metrics'] = {
                'training_set': {
                    'rmse': train_metrics.get('rmse', 0),
                    'mae': train_metrics.get('mae', 0),
                    'r2_score': train_metrics.get('r2_score', 0),
                    'spearman_correlation': train_metrics.get('spearman_correlation', 0),
                    'stage_accuracy': train_metrics.get('stage_accuracy', 0)
                },
                'validation_set': {
                    'rmse': val_metrics.get('rmse', 0),
                    'mae': val_metrics.get('mae', 0),
                    'r2_score': val_metrics.get('r2_score', 0),
                    'spearman_correlation': val_metrics.get('spearman_correlation', 0),
                    'stage_accuracy': val_metrics.get('stage_accuracy', 0)
                },
                'test_set': {
                    'rmse': test_metrics.get('rmse', 0),
                    'mae': test_metrics.get('mae', 0),
                    'r2_score': test_metrics.get('r2_score', 0),
                    'spearman_correlation': test_metrics.get('spearman_correlation', 0),
                    'stage_accuracy': test_metrics.get('stage_accuracy', 0)
                }
            }
        elif mode == 'binary':
            summary['performance_metrics'] = {
                'training_set': {
                    'f1_score': train_metrics.get('f1_score', 0),
                    'auc_roc': train_metrics.get('auc_roc', 0),
                    'accuracy': train_metrics.get('accuracy', 0),
                    'precision': train_metrics.get('precision', 0),
                    'recall': train_metrics.get('recall', 0),
                    'confusion_matrix': train_metrics.get('confusion_matrix', {})
                },
                'validation_set': {
                    'f1_score': val_metrics.get('f1_score', 0),
                    'auc_roc': val_metrics.get('auc_roc', 0),
                    'accuracy': val_metrics.get('accuracy', 0),
                    'precision': val_metrics.get('precision', 0),
                    'recall': val_metrics.get('recall', 0),
                    'confusion_matrix': val_metrics.get('confusion_matrix', {})
                },
                'test_set': {
                    'f1_score': test_metrics.get('f1_score', 0),
                    'auc_roc': test_metrics.get('auc_roc', 0),
                    'accuracy': test_metrics.get('accuracy', 0),
                    'precision': test_metrics.get('precision', 0),
                    'recall': test_metrics.get('recall', 0),
                    'confusion_matrix': test_metrics.get('confusion_matrix', {})
                }
            }
    else:
        # Old flat format (backward compatibility)
        if mode == 'regression':
            summary['performance_metrics'] = {
                'regression': {
                    'rmse': metrics.get('rmse', 0),
                    'mae': metrics.get('mae', 0),
                    'r2_score': metrics.get('r2_score', 0)
                },
                'ranking': {
                    'pearson_correlation': metrics.get('pearson_correlation', 0),
                    'spearman_correlation': metrics.get('spearman_correlation', 0),
                    'ranking_quality_ndcg': metrics.get('ranking_quality', 0)
                },
                'classification': {
                    'stage_accuracy_10pct': metrics.get('stage_accuracy', 0)
                }
            }
        elif mode == 'binary':
            summary['performance_metrics'] = {
                'binary_classification': {
                    'f1_score': metrics.get('f1_score', 0),
                    'auc_roc': metrics.get('auc_roc', 0),
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'confusion_matrix': metrics.get('confusion_matrix', {}),
                    'class_distribution': metrics.get('class_distribution', {})
                }
            }
    
    # Performance interpretation for paper (use test set metrics for unbiased assessment)
    if mode == 'regression':
        if 'test' in metrics:
            # Use test set metrics for assessment
            test_m = metrics['test']
            rmse = test_m.get('rmse', 0)
            r2 = test_m.get('r2_score', 0)
            stage_acc = test_m.get('stage_accuracy', 0)
            spearman = test_m.get('spearman_correlation', 0)
        else:
            # Fallback to flat format
            rmse = metrics.get('rmse', 0)
            r2 = metrics.get('r2_score', 0)
            stage_acc = metrics.get('stage_accuracy', 0)
            spearman = metrics.get('spearman_correlation', 0)
        
        if rmse < 0.1 and r2 > 0.8 and spearman > 0.85:
            performance_level = "Excellent"
            interpretation = "High accuracy ordinal regression with strong ranking correlation"
        elif rmse < 0.15 and r2 > 0.6 and spearman > 0.7:
            performance_level = "Good"
            interpretation = "Satisfactory ordinal regression performance"
        elif rmse < 0.2 and r2 > 0.4 and spearman > 0.5:
            performance_level = "Fair"
            interpretation = "Moderate ordinal regression performance"
        else:
            performance_level = "Needs Improvement"
            interpretation = "Limited ordinal regression performance"
        
        strengths = []
        improvements = []
        
        if r2 > 0.8:
            strengths.append("Strong variance explanation (R² > 0.8)")
        if spearman > 0.85:
            strengths.append("Excellent ranking preservation (Spearman > 0.85)")
        if stage_acc > 0.9:
            strengths.append("High stage classification accuracy (>90%)")
        
        if rmse > 0.15:
            improvements.append("High prediction error (RMSE > 0.15)")
        if r2 < 0.6:
            improvements.append("Limited variance explanation (R² < 0.6)")
        if spearman < 0.7:
            improvements.append("Weak ranking correlation (Spearman < 0.7)")
    
    elif mode == 'binary':
        if 'test' in metrics:
            # Use test set metrics for assessment
            test_m = metrics['test']
            f1 = test_m.get('f1_score', 0)
            auc = test_m.get('auc_roc', 0)
            accuracy = test_m.get('accuracy', 0)
            precision = test_m.get('precision', 0)
            recall = test_m.get('recall', 0)
        else:
            # Fallback to flat format
            f1 = metrics.get('f1_score', 0)
            auc = metrics.get('auc_roc', 0)
            accuracy = metrics.get('accuracy', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
        
        if f1 > 0.85 and auc > 0.9 and accuracy > 0.9:
            performance_level = "Excellent"
            interpretation = "High performance binary classification with excellent discrimination"
        elif f1 > 0.75 and auc > 0.8 and accuracy > 0.8:
            performance_level = "Good"
            interpretation = "Satisfactory binary classification performance"
        elif f1 > 0.6 and auc > 0.7 and accuracy > 0.7:
            performance_level = "Fair"
            interpretation = "Moderate binary classification performance"
        else:
            performance_level = "Needs Improvement"
            interpretation = "Limited binary classification performance"
        
        strengths = []
        improvements = []
        
        if f1 > 0.85:
            strengths.append("Excellent F1 Score (> 0.85)")
        if auc > 0.9:
            strengths.append("Outstanding discrimination (AUC > 0.9)")
        if accuracy > 0.9:
            strengths.append("High overall accuracy (> 90%)")
        if precision > 0.85 and recall > 0.85:
            strengths.append("Well-balanced precision and recall")
        
        if f1 < 0.7:
            improvements.append("Low F1 Score (< 0.7)")
        if auc < 0.8:
            improvements.append("Limited discrimination ability (AUC < 0.8)")
        if accuracy < 0.8:
            improvements.append("Low overall accuracy (< 80%)")
        if abs(precision - recall) > 0.2:
            improvements.append("Imbalanced precision-recall trade-off")
    
    summary['performance_assessment'] = {
        'overall_level': performance_level,
        'interpretation': interpretation,
        'key_strengths': strengths,
        'areas_for_improvement': improvements
    }
    
    return summary
