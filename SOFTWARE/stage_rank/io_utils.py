"""
I/O utilities for Doctrina stage ranking pipeline.
"""

import pandas as pd
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Color support
try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    # Fallback if colorama not available
    class _DummyColor:
        def __getattr__(self, name):
            return ""
    
    Fore = Back = Style = _DummyColor()
    COLORS_AVAILABLE = False


def colored_print(text: str, color: str = None, bold: bool = False):
    """Print text with optional color and bold formatting."""
    if not COLORS_AVAILABLE:
        print(text)
        return
    
    color_map = {
        'red': Fore.RED,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'blue': Fore.BLUE,
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'white': Fore.WHITE,
    }
    
    prefix = ""
    if bold:
        prefix += Style.BRIGHT
    if color and color in color_map:
        prefix += color_map[color]
    
    print(f"{prefix}{text}{Style.RESET_ALL}")


def show_progress_bar(current: int, total: int, description: str = "", width: int = 40):
    """Display a sci-fi style progress bar."""
    if total == 0:
        return
    
    percent = current / total
    filled = int(width * percent)
    
    # Sci-fi progress chars
    progress_chars = ["▱", "▰"]
    loading_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    # Create animated loading spinner
    spinner = loading_chars[current % len(loading_chars)]
    
    # Build the bar with gradient effect
    bar_filled = "▰" * filled
    bar_empty = "▱" * (width - filled)
    
    if COLORS_AVAILABLE:
        # Gradient from cyan to green for filled portion
        if filled > 0:
            bar_colored = f"{Fore.CYAN}{bar_filled[:filled//2]}{Fore.GREEN}{bar_filled[filled//2:]}{Fore.WHITE}{bar_empty}{Style.RESET_ALL}"
        else:
            bar_colored = f"{Fore.WHITE}{bar_empty}{Style.RESET_ALL}"
        
        if percent >= 1.0:
            status = f"{Fore.GREEN}✓ COMPLETE{Style.RESET_ALL}"
        else:
            status = f"{Fore.CYAN}{spinner}{Style.RESET_ALL}"
        
        print(f"\r{description} [{bar_colored}] {percent*100:.1f}% {status} ({current}/{total})", end="", flush=True)
    else:
        bar = bar_filled + bar_empty
        status = "✓" if percent >= 1.0 else spinner
        print(f"\r{description} [{bar}] {percent*100:.1f}% {status} ({current}/{total})", end="", flush=True)


def print_header(text: str, emoji: str = "🎯"):
    """Print a colored header."""
    colored_print(f"\n{emoji} {text}", 'cyan', bold=True)
    colored_print("=" * (len(text) + 3), 'cyan')


def print_success(text: str, emoji: str = "✅"):
    """Print success message."""
    colored_print(f"{emoji} {text}", 'green')


def print_warning(text: str, emoji: str = "⚠️"):
    """Print warning message."""
    colored_print(f"{emoji} {text}", 'yellow')


def print_error(text: str, emoji: str = "❌"):
    """Print error message."""
    colored_print(f"{emoji} {text}", 'red')


def print_info(text: str, emoji: str = "📊"):
    """Print info message."""
    colored_print(f"{emoji} {text}", 'blue')


def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file and return DataFrame."""
    try:
        # Force item_category to be string to avoid mixed type issues
        df = pd.read_csv(filepath, dtype={'item_category': str})
        
        # Ensure item_category is string if it exists
        if 'item_category' in df.columns:
            df['item_category'] = df['item_category'].astype(str)
        
        print(f"📄 Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        raise Exception(f"Failed to load CSV: {e}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from plain text file.
    
    Format:
        # Comments start with #
        key = value
        section_name =
            item1:value1
            item2:value2
    """
    config = {}
    
    try:
        with open(config_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        raise Exception(f"Config file not found: {config_path}")
    
    lines = content.strip().split('\n')
    current_section = None
    section_data = {}
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            i += 1
            continue
        
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Remove inline comments from value
            if '#' in value:
                value = value.split('#')[0].strip()
            
            if not value:  # Section start
                # Save previous section if exists
                if current_section and section_data:
                    config[current_section] = section_data
                
                current_section = key
                section_data = {}
                
                # Read section items
                i += 1
                while i < len(lines):
                    section_line = lines[i]
                    if not section_line.strip() or section_line.strip().startswith('#'):
                        i += 1
                        continue
                    
                    if section_line.startswith('    ') or section_line.startswith('\t'):
                        # This is a section item
                        item_line = section_line.strip()
                        if ':' in item_line:
                            item_key, item_value = item_line.split(':', 1)
                            item_key = item_key.strip()
                            item_value = item_value.strip()
                            
                            # Remove inline comments from item value
                            if '#' in item_value:
                                item_value = item_value.split('#')[0].strip()
                            
                            section_data[item_key] = _parse_value(item_value)
                        i += 1
                    else:
                        # End of section
                        break
                
                # Save the section
                if section_data:
                    config[current_section] = section_data
                current_section = None
                section_data = {}
                
                # Don't increment i here since we need to process the current line
                continue
                
            else:  # Regular key-value
                config[key] = _parse_value(value)
        
        i += 1
    
    print(f"⚙️  Loaded config from {config_path}")
    return config


def _parse_value(value: str) -> Any:
    """Parse string value to appropriate Python type."""
    value = value.strip()
    
    # Try boolean
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    
    # Try int
    try:
        if '.' not in value:
            return int(value)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string
    return value


def infer_column_types(df: pd.DataFrame, explicit_types: Dict[str, str] = None) -> Dict[str, str]:
    """
    Infer column types based on data characteristics.
    
    Args:
        df: DataFrame to analyze
        explicit_types: Dict of explicitly specified column types
    
    Returns:
        Dict mapping column names to types: 'text', 'categorical', 'numeric'
    """
    explicit_types = explicit_types or {}
    column_types = {}
    
    for col in df.columns:
        if col in explicit_types:
            column_types[col] = explicit_types[col]
            continue
        
        # Get non-null values
        values = df[col].dropna()
        if len(values) == 0:
            column_types[col] = 'text'  # Default for empty columns
            continue
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(values):
            column_types[col] = 'numeric'
            continue
        
        # For string columns, analyze characteristics
        str_values = values.astype(str)
        unique_ratio = len(str_values.unique()) / len(str_values)
        avg_length = str_values.str.len().mean()
        
        # Heuristics for text vs categorical
        if unique_ratio > 0.8 or avg_length > 50:
            column_types[col] = 'text'
        else:
            column_types[col] = 'categorical'
    
    return column_types


def parse_binary_labels(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse binary label mapping from config.
    
    Format in config:
        binary_labels =
            0:[value1, value2, value3]
            1:[value4, value5]
    
    Returns:
        Dict with 'mapping' (value -> binary_label) and 'label_names' (0/1 -> name)
    """
    if 'binary_labels' not in config:
        return None
    
    binary_config = config['binary_labels']
    mapping = {}
    label_names = {}
    
    for label_str, values in binary_config.items():
        try:
            label = int(label_str)
        except ValueError:
            print_warning(f"Invalid binary label: {label_str} (must be 0 or 1)")
            continue
        
        if label not in [0, 1]:
            print_warning(f"Binary label must be 0 or 1, got: {label}")
            continue
        
        # Parse values list (should be a string like "[val1, val2, val3]")
        if isinstance(values, str):
            # Remove brackets and split by comma
            values_str = values.strip('[]')
            value_list = [v.strip() for v in values_str.split(',') if v.strip()]
        elif isinstance(values, list):
            value_list = values
        else:
            print_warning(f"Invalid format for binary label {label}: {values}")
            continue
        
        # Convert values to appropriate types
        for value in value_list:
            # Try to convert to number if possible
            try:
                if '.' in str(value):
                    parsed_value = float(value)
                else:
                    parsed_value = int(value)
            except (ValueError, TypeError):
                parsed_value = str(value)
            
            mapping[parsed_value] = label
            
        # Set label name
        label_names[label] = f"Class_{label}"
    
    return {
        'mapping': mapping,
        'label_names': label_names
    }


def save_results(results: Dict[str, Any], output_path: str):
    """Save comprehensive training results to file for research documentation."""
    import json
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("DOCTRINA ML PIPELINE - TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {results.get('experiment_info', {}).get('timestamp', 'Unknown')}\n")
        f.write(f"Pipeline Version: {results.get('experiment_info', {}).get('pipeline_version', 'Unknown')}\n\n")
        
        # Experiment Overview
        f.write("EXPERIMENT OVERVIEW\n")
        f.write("-" * 20 + "\n")
        exp_info = results.get('experiment_info', {})
        f.write(f"Training Time: {exp_info.get('training_time_seconds', 'Unknown')} seconds\n")
        
        dataset_info = results.get('dataset_info', {})
        f.write(f"Full Dataset Size: {dataset_info.get('full_dataset_samples', 'Unknown')} samples\n")
        f.write(f"Training Set Size: {dataset_info.get('training_samples', 'Unknown')} samples\n")
        f.write(f"Test Set Size: {dataset_info.get('test_samples', 'Unknown')} samples\n")
        f.write(f"Target Column: {dataset_info.get('target_column', 'Unknown')}\n")
        f.write(f"Target Range: {dataset_info.get('target_range', 'Unknown')}\n")
        f.write(f"Generated Features: {dataset_info.get('n_features_generated', 'Unknown')}\n")
        # Check if we have validation samples to determine the note
        validation_samples = dataset_info.get('validation_samples', 0)
        if validation_samples and validation_samples != 'Unknown':
            f.write(f"Validation Set Size: {validation_samples} samples\n")
            f.write(f"✅ NOTE: Metrics computed on separate Train/Validation/Test sets (no data leakage)\n\n")
        else:
            f.write(f"⚠️  NOTE: All metrics computed on TEST SET ONLY (no data leakage)\n\n")
        # Column Configuration
        f.write("COLUMN CONFIGURATION\n")
        f.write("-" * 20 + "\n")
        col_info = results.get('column_configuration', {})
        f.write(f"Total Dataset Columns: {col_info.get('total_columns_in_dataset', 'Unknown')}\n")
        
        excluded = col_info.get('excluded_columns', {})
        f.write(f"Excluded Columns ({excluded.get('total_excluded', 0)}):\n")
        f.write(f"  - Target: {excluded.get('target_column', 'Unknown')}\n")
        manual_excluded = excluded.get('manually_excluded', [])
        if manual_excluded:
            f.write(f"  - Manual: {', '.join(manual_excluded)}\n")
        f.write("\n")
        
        features = col_info.get('feature_columns', {})
        f.write(f"Feature Columns ({features.get('total_feature_columns', 0)}):\n")
        
        text_cols = features.get('text_columns', {})
        f.write(f"  - Text ({text_cols.get('count', 0)}): {', '.join(text_cols.get('names', []))}\n")
        
        cat_cols = features.get('categorical_columns', {})
        f.write(f"  - Categorical ({cat_cols.get('count', 0)}): {', '.join(cat_cols.get('names', []))}\n")
        
        num_cols = features.get('numeric_columns', {})
        f.write(f"  - Numeric ({num_cols.get('count', 0)}): {', '.join(num_cols.get('names', []))}\n\n")
        
        # Model Configuration
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 20 + "\n")
        model_config = results.get('model_configuration', {})
        f.write(f"Model Type: {model_config.get('model_type', 'Unknown')}\n")
        f.write(f"Text Method: {model_config.get('text_method', 'Unknown')}\n")
        f.write(f"Max TF-IDF Features: {model_config.get('max_tfidf_features', 'Unknown')}\n")
        f.write(f"Validation Split: {model_config.get('validation_split', 'Unknown')}\n")
        f.write(f"Random Seed: {model_config.get('random_seed', 'Unknown')}\n\n")
        
        # Hyperparameters
        f.write("HYPERPARAMETERS\n")
        f.write("-" * 15 + "\n")
        hyperparams = results.get('hyperparameters', {})
        f.write(f"N Estimators: {hyperparams.get('n_estimators', 'Unknown')}\n")
        f.write(f"Max Depth: {hyperparams.get('max_depth', 'Unknown')}\n")
        f.write(f"Learning Rate: {hyperparams.get('learning_rate', 'Unknown')}\n")
        f.write(f"Subsample: {hyperparams.get('subsample', 'Unknown')}\n")
        f.write(f"Column Sample: {hyperparams.get('colsample_bytree', 'Unknown')}\n")
        
        reg = hyperparams.get('regularization', {})
        f.write(f"L1 Regularization: {reg.get('reg_alpha', 'Unknown')}\n")
        f.write(f"L2 Regularization: {reg.get('reg_lambda', 'Unknown')}\n\n")
        
        # Performance Metrics
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 19 + "\n")
        perf = results.get('performance_metrics', {})
        
        # Mode-specific metrics display
        mode = results.get('experiment_info', {}).get('mode', 'regression')
        
        # Check if we have the new train/val/test structure
        if 'training_set' in perf and 'validation_set' in perf and 'test_set' in perf:
            # New nested structure - show all three sets
            if mode == 'regression':
                f.write("Training Set Metrics:\n")
                train_m = perf.get('training_set', {})
                f.write(f"  - RMSE: {train_m.get('rmse', 0):.4f}\n")
                f.write(f"  - MAE: {train_m.get('mae', 0):.4f}\n")
                f.write(f"  - R² Score: {train_m.get('r2_score', 0):.4f}\n")
                f.write(f"  - Spearman Correlation: {train_m.get('spearman_correlation', 0):.4f}\n")
                f.write(f"  - Stage Accuracy: {train_m.get('stage_accuracy', 0):.1%}\n\n")
                
                f.write("Validation Set Metrics:\n")
                val_m = perf.get('validation_set', {})
                f.write(f"  - RMSE: {val_m.get('rmse', 0):.4f}\n")
                f.write(f"  - MAE: {val_m.get('mae', 0):.4f}\n")
                f.write(f"  - R² Score: {val_m.get('r2_score', 0):.4f}\n")
                f.write(f"  - Spearman Correlation: {val_m.get('spearman_correlation', 0):.4f}\n")
                f.write(f"  - Stage Accuracy: {val_m.get('stage_accuracy', 0):.1%}\n\n")
                
                f.write("Test Set Metrics (Unbiased Final Assessment):\n")
                test_m = perf.get('test_set', {})
                f.write(f"  - RMSE: {test_m.get('rmse', 0):.4f}\n")
                f.write(f"  - MAE: {test_m.get('mae', 0):.4f}\n")
                f.write(f"  - R² Score: {test_m.get('r2_score', 0):.4f}\n")
                f.write(f"  - Spearman Correlation: {test_m.get('spearman_correlation', 0):.4f}\n")
                f.write(f"  - Stage Accuracy: {test_m.get('stage_accuracy', 0):.1%}\n")
                
            elif mode == 'binary':
                f.write("Training Set Metrics:\n")
                train_m = perf.get('training_set', {})
                f.write(f"  - F1 Score: {train_m.get('f1_score', 0):.4f}\n")
                f.write(f"  - AUC-ROC: {train_m.get('auc_roc', 0):.4f}\n")
                f.write(f"  - Accuracy: {train_m.get('accuracy', 0):.4f}\n")
                f.write(f"  - Precision: {train_m.get('precision', 0):.4f}\n")
                f.write(f"  - Recall: {train_m.get('recall', 0):.4f}\n\n")
                
                f.write("Validation Set Metrics:\n")
                val_m = perf.get('validation_set', {})
                f.write(f"  - F1 Score: {val_m.get('f1_score', 0):.4f}\n")
                f.write(f"  - AUC-ROC: {val_m.get('auc_roc', 0):.4f}\n")
                f.write(f"  - Accuracy: {val_m.get('accuracy', 0):.4f}\n")
                f.write(f"  - Precision: {val_m.get('precision', 0):.4f}\n")
                f.write(f"  - Recall: {val_m.get('recall', 0):.4f}\n\n")
                
                f.write("Test Set Metrics (Unbiased Final Assessment):\n")
                test_m = perf.get('test_set', {})
                f.write(f"  - F1 Score: {test_m.get('f1_score', 0):.4f}\n")
                f.write(f"  - AUC-ROC: {test_m.get('auc_roc', 0):.4f}\n")
                f.write(f"  - Accuracy: {test_m.get('accuracy', 0):.4f}\n")
                f.write(f"  - Precision: {test_m.get('precision', 0):.4f}\n")
                f.write(f"  - Recall: {test_m.get('recall', 0):.4f}\n")
        else:
            # Old flat structure - backward compatibility
            if mode == 'regression':
                reg_metrics = perf.get('regression', {})
                f.write(f"Regression Metrics:\n")
                f.write(f"  - RMSE: {reg_metrics.get('rmse', 0):.4f}\n")
                f.write(f"  - MAE: {reg_metrics.get('mae', 0):.4f}\n")
                f.write(f"  - R² Score: {reg_metrics.get('r2_score', 0):.4f}\n")
                
                rank_metrics = perf.get('ranking', {})
                f.write(f"Ranking Metrics:\n")
                f.write(f"  - Pearson Correlation: {rank_metrics.get('pearson_correlation', 0):.4f}\n")
                f.write(f"  - Spearman Correlation: {rank_metrics.get('spearman_correlation', 0):.4f}\n")
                f.write(f"  - NDCG Quality: {rank_metrics.get('ranking_quality_ndcg', 0):.4f}\n")
                
                class_metrics = perf.get('classification', {})
                f.write(f"Classification Metrics:\n")
                f.write(f"  - Stage Accuracy (±10%): {class_metrics.get('stage_accuracy_10pct', 0):.1%}\n")
            
            elif mode == 'binary':
                binary_metrics = perf.get('binary_classification', {})
                f.write(f"Binary Classification Metrics:\n")
                f.write(f"  - F1 Score: {binary_metrics.get('f1_score', 0):.4f}\n")
                f.write(f"  - AUC-ROC: {binary_metrics.get('auc_roc', 0):.4f}\n")
                f.write(f"  - Accuracy: {binary_metrics.get('accuracy', 0):.4f}\n")
                f.write(f"  - Precision: {binary_metrics.get('precision', 0):.4f}\n")
                f.write(f"  - Recall: {binary_metrics.get('recall', 0):.4f}\n")
                
                conf_matrix = binary_metrics.get('confusion_matrix', {})
                f.write(f"Confusion Matrix:\n")
                f.write(f"  - True Negatives: {conf_matrix.get('tn', 0)}\n")
                f.write(f"  - False Positives: {conf_matrix.get('fp', 0)}\n")
                f.write(f"  - False Negatives: {conf_matrix.get('fn', 0)}\n")
                f.write(f"  - True Positives: {conf_matrix.get('tp', 0)}\n")
            
            f.write("\n")
            
            # Performance Assessment
            f.write("PERFORMANCE ASSESSMENT\n")
            f.write("-" * 22 + "\n")
            assessment = results.get('performance_assessment', {})
            f.write(f"Overall Level: {assessment.get('overall_level', 'Unknown')}\n")
            f.write(f"Interpretation: {assessment.get('interpretation', 'Unknown')}\n\n")
            
            strengths = assessment.get('key_strengths', [])
            if strengths:
                f.write("Key Strengths:\n")
                for strength in strengths:
                    f.write(f"  ✓ {strength}\n")
                f.write("\n")
            
            improvements = assessment.get('areas_for_improvement', [])
            if improvements:
                f.write("Areas for Improvement:\n")
                for improvement in improvements:
                    f.write(f"  • {improvement}\n")
                f.write("\n")
            
            # Raw JSON for programmatic access
            f.write("RAW DATA (JSON)\n")
            f.write("-" * 15 + "\n")
            f.write(json.dumps(results, indent=2, default=str))
    
    print_success(f"Comprehensive training summary saved to {output_path}", "💾")
