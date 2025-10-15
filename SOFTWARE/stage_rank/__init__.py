"""
Stage Ranking ML Pipeline
Lightweight ML pipeline for candidate ranking using stage progression scores.
"""

__version__ = "1.0.0"

from .io_utils import load_csv, load_config, infer_column_types, save_results, print_header, print_info, print_success, print_warning, show_progress_bar, parse_binary_labels
from .featurize import FeatureEngineer, map_target_values
from .model import StageRankModel, BinaryClassificationModel
from .metrics import compute_metrics, compute_binary_metrics, create_performance_summary

import time
import numpy as np
from pathlib import Path

__all__ = ['load_csv', 'load_config', 'FeatureEngineer', 'StageRankModel', 
           'compute_metrics', 'train_model']


def train_model(input_file: str, config: dict, output_dir: str, mode: str = 'regression'):
    """
    Main training function for the Doctrina pipeline.
    
    Args:
        input_file: Path to input CSV file
        config: Configuration dictionary
        output_dir: Directory to save model and results
        mode: Training mode - 'regression' or 'binary'
    """
    start_time = time.time()
    
    # Overall progress tracker
    total_phases = 6
    current_phase = 0
    
    def update_overall_progress(phase_name: str):
        nonlocal current_phase
        current_phase += 1
        show_progress_bar(current_phase, total_phases, f"🎯 Pipeline Progress")
        print(f" - {phase_name}")
    
    # Phase 1: Load data
    update_overall_progress("Loading data")
    df = load_csv(input_file)
    
    # Phase 2: Column type inference
    update_overall_progress("Analyzing data structure")
    target_column = config.get('target_column', 'stage')
    target_mapping = config.get('map_target', {})
    explicit_types = config.get('column_types', {})
    
    # Infer column types for all columns
    column_types = infer_column_types(df, explicit_types)
    
    # Get columns to exclude (in addition to target column)
    exclude_columns = config.get('exclude_columns', [])
    # Handle both string and list formats
    if isinstance(exclude_columns, str):
        exclude_columns = [col.strip() for col in exclude_columns.split(',') if col.strip()]
    elif exclude_columns is None:
        exclude_columns = []
    all_excluded = set([target_column] + exclude_columns)
    
    # Separate columns into excluded vs included
    excluded_cols = {col: ctype for col, ctype in column_types.items() if col in all_excluded}
    included_cols = {col: ctype for col, ctype in column_types.items() if col not in all_excluded}
    
    print_info("Column Analysis:", "🔍")
    print(f"   📊 Total columns in dataset: {len(column_types)}")
    print(f"   ❌ Excluded from training: {len(excluded_cols)}")
    print(f"   ✅ Used for training: {len(included_cols)}")
    print()
    
    if excluded_cols:
        print_warning("Excluded Columns (not used for training):")
        for col, ctype in excluded_cols.items():
            reason = "TARGET" if col == target_column else "MANUAL"
            print(f"   ❌ {col}: {ctype} ({reason})")
        print()
    
    print_success("Training Columns (features):")
    for col, ctype in included_cols.items():
        marker = "✓" if col not in explicit_types else "⚙️"
        print(f"   {marker} {col}: {ctype}")
    print()
    
    # Phase 3: Target mapping
    update_overall_progress("Mapping target values")
    
    if mode == 'regression':
        y = map_target_values(df, target_column, target_mapping)
        print_info(f"Target range: {y.min():.3f} to {y.max():.3f}", "🎯")
    elif mode == 'binary':
        # Parse binary labels from config
        binary_config = parse_binary_labels(config)
        if binary_config is None:
            raise ValueError("Binary mode requires 'binary_labels' section in config")
        
        # Map target values to binary labels
        target_values = df[target_column]
        y = target_values.map(binary_config['mapping'])
        
        # Check for unmapped values
        unmapped_mask = y.isna()
        if unmapped_mask.any():
            unmapped_values = target_values[unmapped_mask].unique()
            raise ValueError(f"Unmapped target values in binary classification: {unmapped_values}")
        
        y = y.astype(int)
        class_distribution = y.value_counts().sort_index()
        print_info(f"Binary class distribution: {dict(class_distribution)}", "⚖️")
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'regression' or 'binary'")
    
    # Phase 4: Feature engineering
    update_overall_progress("Building features")
    text_method = config.get('text_method', 'tfidf')
    sbert_model = config.get('sbert_model', 'all-MiniLM-L6-v2')
    max_tfidf_features = config.get('max_tfidf_features', 1000)
    
    feature_engineer = FeatureEngineer(
        text_method=text_method,
        model_name=sbert_model,
        max_features=max_tfidf_features
    )
    X = feature_engineer.fit_transform(df, column_types, target_column, exclude_columns)
    
    # Extract training parameters from config
    model_params = {}
    for key, value in config.items():
        if key in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 
                   'colsample_bytree', 'min_child_weight', 'reg_alpha', 'reg_lambda']:
            model_params[key] = value
    
    # Phase 5: Model training with train/test split
    update_overall_progress("Training ML model")
    
    if mode == 'regression':
        model = StageRankModel(**model_params)
    elif mode == 'binary':
        model = BinaryClassificationModel(**model_params)
    
    validation_split = config.get('train_test_split', 0.2)
    
    # Ensure we have numpy arrays for consistent indexing
    X = np.array(X)
    y = np.array(y)
    
    # Standard train/test split
    from sklearn.model_selection import train_test_split
    if mode == 'binary':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
    
    print_info(f"Random split: {len(X_train):,} train, {len(X_test):,} test", "📊")
    
    # Train model (this will use its own internal validation split from training data)
    training_history = model.fit(X_train, y_train, validation_split=0.2)
    
    # Phase 6: Final evaluation ON TEST SET ONLY
    update_overall_progress("Computing test set metrics")
    
    if mode == 'regression':
        y_pred_test = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred_test)
    elif mode == 'binary':
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        metrics = compute_binary_metrics(y_test, y_pred_test, y_pred_proba_test)
    
    # Prepare comprehensive column information for summary
    column_info = {
        'total_columns_in_dataset': len(column_types),
        'excluded_columns': {
            'target_column': target_column,
            'manually_excluded': exclude_columns,
            'total_excluded': len(all_excluded)
        },
        'feature_columns': {
            'text_columns': {
                'count': len(feature_engineer.text_columns),
                'names': feature_engineer.text_columns
            },
            'categorical_columns': {
                'count': len(feature_engineer.categorical_columns), 
                'names': feature_engineer.categorical_columns
            },
            'numeric_columns': {
                'count': len(feature_engineer.numeric_columns),
                'names': feature_engineer.numeric_columns
            },
            'total_feature_columns': len(feature_engineer.text_columns) + len(feature_engineer.categorical_columns) + len(feature_engineer.numeric_columns)
        },
        'feature_engineering': {
            'text_method': config.get('text_method', 'tfidf'),
            'max_tfidf_features': config.get('max_tfidf_features', 1000),
            'final_ml_features': X.shape[1]
        }
    }
    
    # Save everything
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model and feature engineer
    model.save(str(output_path / 'model.pkl'))
    feature_engineer.save(str(output_path / 'feature_engineer.pkl'))
    
    # Create and save comprehensive summary with CORRECTED dataset info
    training_time = time.time() - start_time
    model_info = model.get_model_info()
    model_info.update(training_history)
    
    # Override training_samples to reflect the full dataset size for accurate reporting
    model_info['full_dataset_samples'] = len(df)
    model_info['training_samples'] = len(X_train)
    model_info['test_samples'] = len(X_test)
    
    summary = create_performance_summary(metrics, model_info, training_time, config, column_info, mode)
    
    # Save results
    save_results(summary, str(output_path / 'training_summary.txt'))
    
    print(f"\n🎉 Training completed in {training_time:.1f} seconds!")
    print(f"📁 All files saved to: {output_dir}")
    print_warning("⚠️  Note: Metrics are computed on TEST SET ONLY to prevent data leakage")
    
    return summary
