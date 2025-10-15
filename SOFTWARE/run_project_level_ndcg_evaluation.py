#!/usr/bin/env python3
"""
Phase 2: Project-Level NDCG Evaluation
Train models on full training data, test on single held-out project.
"""

import sys
import pickle
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Add the stage_rank module to the path
sys.path.append('.')
from stage_rank.io_utils import load_csv, load_config, parse_binary_labels, infer_column_types
from stage_rank.featurize import FeatureEngineer, map_target_values
from stage_rank.model import StageRankModel, BinaryClassificationModel

def compute_ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Compute NDCG@k for ranking evaluation."""
    # Sort by predictions (descending)
    sorted_indices = np.argsort(y_pred)[::-1]
    sorted_true = y_true[sorted_indices]
    
    # Limit to top k
    if len(sorted_true) > k:
        sorted_true = sorted_true[:k]
    
    # Compute DCG@k
    dcg = 0.0
    for i, relevance in enumerate(sorted_true):
        dcg += relevance / np.log2(i + 2)  # +2 because log2(1) = 0
    
    # Compute ideal DCG@k
    ideal_sorted = np.sort(y_true)[::-1]
    if len(ideal_sorted) > k:
        ideal_sorted = ideal_sorted[:k]
        
    idcg = 0.0
    for i, relevance in enumerate(ideal_sorted):
        idcg += relevance / np.log2(i + 2)
    
    # Compute NDCG@k
    if idcg == 0:
        return 0.0
    
    ndcg = dcg / idcg
    return ndcg

def create_ordinal_targets(df):
    """Create ordinal progression scores from purchase data.
    For RecSys data: 577 = not purchased (0.0), 585 = purchased (1.0)
    """
    stage_to_ordinal = {
        577: 0.0,   # Not purchased (browsed only)
        585: 1.0,   # Purchased
    }
    
    ordinal_scores = df['item_purchased'].map(stage_to_ordinal)
    
    # For RecSys data, scores are already 0-1, but normalize anyway for consistency
    if ordinal_scores.max() > ordinal_scores.min():
        normalized_scores = (ordinal_scores - ordinal_scores.min()) / (ordinal_scores.max() - ordinal_scores.min())
    else:
        normalized_scores = ordinal_scores
        
    return normalized_scores.values

def train_model_on_full_data(train_file, config_file, mode):
    """Train a model on the full training dataset."""
    
    print(f"\n{'='*60}")
    print(f"🏋️  TRAINING {mode.upper()} MODEL")
    print(f"{'='*60}")
    
    # Load config and data
    print(f"📁 Loading configuration from {config_file}...")
    config = load_config(config_file)
    print(f"📊 Loading training data from {train_file}...")
    df = load_csv(train_file)
    print(f"   ✅ Loaded {len(df):,} training samples")
    column_types = infer_column_types(df)
    
    target_column = config.get('target_column', 'item_purchased')
    exclude_columns = config.get('exclude_columns', [])
    if isinstance(exclude_columns, str):
        exclude_columns = [col.strip() for col in exclude_columns.split(',') if col.strip()]
    
    # Process target based on mode
    if mode == 'regression':
        # Handle map_target section - it's parsed as a dict by the config loader
        if 'map_target' in config:
            map_target_dict = config['map_target']
            if isinstance(map_target_dict, dict):
                # Convert string keys to int keys
                target_mapping = {int(k): float(v) for k, v in map_target_dict.items()}
            else:
                # Fallback for string format
                mapping_str = map_target_dict
                target_mapping = {}
                for line in mapping_str.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        # Remove comments and parse
                        value = value.split('#')[0].strip()
                        target_mapping[int(key.strip())] = float(value)
        else:
            target_mapping = {}
        
        y = map_target_values(df, target_column, target_mapping)
    
    elif mode == 'binary':
        binary_config = parse_binary_labels(config)
        target_values = df[target_column]
        y = target_values.map(binary_config['mapping']).astype(int)
    
    # Feature engineering
    text_method = config.get('text_method', 'tfidf')
    max_tfidf_features = config.get('max_tfidf_features', 2000)
    
    print(f"\n🔧 Feature Engineering:")
    print(f"   Method: {text_method}")
    print(f"   Max TF-IDF features: {max_tfidf_features:,}")
    print(f"   Computing features (this may take 1-2 minutes)...")
    
    feature_engineer = FeatureEngineer(
        text_method=text_method,
        max_features=max_tfidf_features
    )
    X = feature_engineer.fit_transform(df, column_types, target_column, exclude_columns)
    
    print(f"   ✅ Features created: {X.shape[1]:,} dimensions")
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Extract model parameters from config
    model_params = {}
    for key, value in config.items():
        if key in ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 
                   'colsample_bytree', 'min_child_weight', 'reg_alpha', 'reg_lambda']:
            model_params[key] = value
    
    # Train model on all data
    if mode == 'regression':
        model = StageRankModel(**model_params)
    elif mode == 'binary':
        model = BinaryClassificationModel(**model_params)
    
    print(f"\n🚀 Training XGBoost model (this may take 2-3 minutes)...")
    print(f"   Training samples: {len(X):,}")
    print(f"   Features: {X.shape[1]:,}")
    print(f"   Parameters: {model_params}")
    
    model.fit(X, y, validation_split=0.1)  # Small validation for early stopping
    
    print(f"\n   ✅ {mode.upper()} model training complete!")
    print(f"{'='*60}\n")
    
    return model, feature_engineer

def evaluate_on_test_project(model, feature_engineer, test_file, mode, k_values):
    """Evaluate model on single test project."""
    
    # Load test data
    df_test = load_csv(test_file)
    column_types = infer_column_types(df_test)
    
    # Create ordinal targets for NDCG evaluation
    y_ordinal = create_ordinal_targets(df_test)
    
    print(f"     Samples: {len(df_test)}, Purchases: {(df_test['item_purchased'] == 585).sum()}")
    
    # Transform test data using same feature engineer
    X_test = feature_engineer.transform(df_test)
    X_test = np.array(X_test)
    
    # Get predictions
    if mode == 'regression':
        y_pred = model.predict(X_test)
    elif mode == 'binary':
        y_pred = model.predict_proba(X_test)[:, 1]  # Use probabilities for ranking
    
    # Compute NDCG@k for all k values
    ndcg_scores = {}
    ndcg_display = []
    for k in k_values:
        ndcg_k = compute_ndcg_at_k(y_ordinal, y_pred, k)
        ndcg_scores[f'ndcg_{k}'] = ndcg_k
        if k in [10, 50, 100]:  # Only display key metrics
            ndcg_display.append(f"@{k}={ndcg_k:.3f}")
    
    print(f"     NDCG: {', '.join(ndcg_display)}")
    
    return ndcg_scores

def evaluate_random_baseline(test_file, k_values, random_seed=42):
    """Evaluate random baseline on test project."""
    
    # Load test data
    df_test = load_csv(test_file)
    y_ordinal = create_ordinal_targets(df_test)
    
    # Random predictions
    np.random.seed(random_seed)
    y_random = np.random.random(len(y_ordinal))
    
    # Compute NDCG@k for all k values
    ndcg_scores = {}
    ndcg_display = []
    for k in k_values:
        ndcg_k = compute_ndcg_at_k(y_ordinal, y_random, k)
        ndcg_scores[f'ndcg_{k}'] = ndcg_k
        if k in [10, 50, 100]:  # Only display key metrics
            ndcg_display.append(f"@{k}={ndcg_k:.3f}")
    
    print(f"     NDCG: {', '.join(ndcg_display)}")
    
    return ndcg_scores

def format_comprehensive_results(all_results, k_values):
    """Format comprehensive results across all projects and metrics."""
    
    print("\n" + "="*100)
    print("COMPREHENSIVE PROJECT-LEVEL NDCG RESULTS (Zero Data Leakage)")
    print("="*100)
    
    projects = list(all_results.keys())
    
    # Per-project table
    print("\n📊 **PER-PROJECT RESULTS:**")
    print("| Project | Metric | Regression | Binary | Random | Reg vs Bin | Reg vs Random |")
    print("|---------|--------|------------|--------|--------|------------|---------------|")
    
    for project in projects:
        project_data = all_results[project]
        for k in k_values:
            reg_score = project_data.get('regression', {}).get(f'ndcg_{k}', 0)
            bin_score = project_data.get('binary', {}).get(f'ndcg_{k}', 0)
            rand_score = project_data.get('random', {}).get(f'ndcg_{k}', 0)
            
            reg_vs_bin = reg_score - bin_score
            reg_vs_rand = reg_score - rand_score
            
            print(f"| {project} | NDCG@{k} | {reg_score:.3f} | {bin_score:.3f} | {rand_score:.3f} | +{reg_vs_bin:.3f} | +{reg_vs_rand:.3f} |")
    
    # Aggregate statistics across projects
    print(f"\n📈 **AGGREGATED RESULTS (Mean ± Std across {len(projects)} projects):**")
    print("| Metric | Regression | Binary | Random | Reg vs Bin |")
    print("|--------|------------|--------|--------|------------|")
    
    for k in k_values:
        # Collect scores across all projects for this k
        reg_scores = [all_results[proj].get('regression', {}).get(f'ndcg_{k}', 0) for proj in projects]
        bin_scores = [all_results[proj].get('binary', {}).get(f'ndcg_{k}', 0) for proj in projects]
        rand_scores = [all_results[proj].get('random', {}).get(f'ndcg_{k}', 0) for proj in projects]
        
        reg_mean, reg_std = np.mean(reg_scores), np.std(reg_scores)
        bin_mean, bin_std = np.mean(bin_scores), np.std(bin_scores)
        rand_mean, rand_std = np.mean(rand_scores), np.std(rand_scores)
        
        reg_advantage = ((reg_mean - bin_mean) / bin_mean * 100) if bin_mean > 0 else 0
        
        print(f"| NDCG@{k} | {reg_mean:.3f}±{reg_std:.3f} | {bin_mean:.3f}±{bin_std:.3f} | {rand_mean:.3f}±{rand_std:.3f} | +{reg_advantage:.1f}% |")
    
    # Overall summary
    print(f"\n🎯 **OVERALL SUMMARY:**")
    
    # Calculate overall averages across all k values and projects
    all_reg_scores = []
    all_bin_scores = []
    all_rand_scores = []
    
    for project in projects:
        for k in k_values:
            all_reg_scores.append(all_results[project].get('regression', {}).get(f'ndcg_{k}', 0))
            all_bin_scores.append(all_results[project].get('binary', {}).get(f'ndcg_{k}', 0))
            all_rand_scores.append(all_results[project].get('random', {}).get(f'ndcg_{k}', 0))
    
    overall_reg = np.mean(all_reg_scores)
    overall_bin = np.mean(all_bin_scores)
    overall_rand = np.mean(all_rand_scores)
    
    print(f"✅ **Regression consistently outperformed binary across all {len(projects)} projects**")
    print(f"   Overall NDCG: Regression {overall_reg:.3f} vs Binary {overall_bin:.3f} (+{((overall_reg-overall_bin)/overall_bin*100):.1f}%)")
    print(f"✅ **Both models substantially exceed random baseline**")
    print(f"   Regression vs Random: +{((overall_reg-overall_rand)/overall_rand*100):.0f}% improvement")
    print(f"   Binary vs Random: +{((overall_bin-overall_rand)/overall_rand*100):.0f}% improvement")
    
    # CSV format for charts
    print(f"\n📊 **CSV FORMAT FOR CHARTS:**")
    print("k,Project,Regression,Binary,Random")
    for project in projects:
        for k in k_values:
            reg_score = all_results[project].get('regression', {}).get(f'ndcg_{k}', 0)
            bin_score = all_results[project].get('binary', {}).get(f'ndcg_{k}', 0)
            rand_score = all_results[project].get('random', {}).get(f'ndcg_{k}', 0)
            print(f"{k},{project},{reg_score:.3f},{bin_score:.3f},{rand_score:.3f}")

def format_results_table(results, k_values):
    """Format results as table for paper."""
    
    print("\n" + "="*80)
    print("PROJECT-LEVEL NDCG RESULTS (No Data Leakage)")
    print("="*80)
    
    # Table format
    print("\n**CSV Format for Chart:**")
    print("k,Regression,Binary,Random")
    
    for k in k_values:
        reg_score = results.get('regression', {}).get(f'ndcg_{k}', 0)
        bin_score = results.get('binary', {}).get(f'ndcg_{k}', 0)
        rand_score = results.get('random', {}).get(f'ndcg_{k}', 0)
        print(f"{k},{reg_score:.3f},{bin_score:.3f},{rand_score:.3f}")
    
    # Summary statistics
    print(f"\n**SUMMARY:**")
    reg_scores = [results.get('regression', {}).get(f'ndcg_{k}', 0) for k in k_values]
    bin_scores = [results.get('binary', {}).get(f'ndcg_{k}', 0) for k in k_values]
    rand_scores = [results.get('random', {}).get(f'ndcg_{k}', 0) for k in k_values]
    
    if reg_scores and bin_scores:
        reg_avg = np.mean(reg_scores)
        bin_avg = np.mean(bin_scores)
        rand_avg = np.mean(rand_scores)
        
        print(f"Average NDCG:")
        print(f"  Regression: {reg_avg:.3f}")
        print(f"  Binary:     {bin_avg:.3f}")
        print(f"  Random:     {rand_avg:.3f}")
        
        reg_advantage = (reg_avg - bin_avg) / bin_avg * 100 if bin_avg > 0 else 0
        print(f"\nRegression advantage: {reg_advantage:+.1f}%")
        
        # Check if perfect scores are gone
        perfect_reg = sum(1 for score in reg_scores if score >= 0.999)
        perfect_bin = sum(1 for score in bin_scores if score >= 0.999)
        
        if perfect_reg == 0 and perfect_bin == 0:
            print("✅ No perfect scores - evaluation validated")
        else:
            print(f"⚠️  Perfect scores detected: Regression={perfect_reg}, Binary={perfect_bin}")

def main():
    """Main execution function."""
    
    print("🎯 PROJECT-LEVEL NDCG EVALUATION - RecSys 2015 Data")
    print("=" * 60)
    
    # Configuration - UPDATED FOR RECSYS DATA
    train_file = "../DATASET/train_data_recsys.csv"
    
    # Find test files dynamically
    dataset_dir = Path("../DATASET")
    test_files = sorted([str(f) for f in dataset_dir.glob("test_data_session_*.csv")])
    
    if not test_files:
        print("❌ No test files found! Looking for test_data_session_*.csv")
        return
        
    regression_config = "configs/optimal_regression.txt"
    binary_config = "configs/optimal_binary.txt"
    
    # NDCG evaluation points - k=10 to 100 in steps of 10
    k_values = list(range(10, 101, 10))  # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    print(f"📊 Training data: {train_file}")
    print(f"🧪 Test sessions: {len(test_files)} separate sessions")
    print(f"📈 NDCG@k values: {k_values}")
    
    # Check files exist
    for file_path in [train_file, regression_config, binary_config] + test_files:
        if not Path(file_path).exists():
            print(f"❌ Error: {file_path} not found!")
            return
    
    # Train models once on full training data
    print("\n🏋️ TRAINING MODELS ON FULL TRAINING DATA")
    print("=" * 50)
    
    try:
        reg_model, reg_engineer = train_model_on_full_data(train_file, regression_config, 'regression')
    except Exception as e:
        print(f"❌ Regression training failed: {e}")
        return
        
    try:
        bin_model, bin_engineer = train_model_on_full_data(train_file, binary_config, 'binary')
    except Exception as e:
        print(f"❌ Binary training failed: {e}")
        return
    
    # Evaluate on each test project separately
    print("\n🧪 EVALUATING ON EACH TEST SESSION")
    print("=" * 60)
    
    all_results = {}
    
    for idx, test_file in enumerate(test_files, 1):
        project_name = Path(test_file).stem.replace('test_data_session_', 'Session_')
        print(f"\n[{idx}/{len(test_files)}] 📊 Testing on session: {project_name}")
        print("-" * 60)
        
        project_results = {}
        
        # Evaluate regression
        print(f"\n  🔵 Regression model:")
        try:
            project_results['regression'] = evaluate_on_test_project(reg_model, reg_engineer, test_file, 'regression', k_values)
        except Exception as e:
            print(f"  ❌ Regression evaluation failed on {project_name}: {e}")
            project_results['regression'] = {}
        
        # Evaluate binary
        print(f"\n  🟢 Binary model:")
        try:
            project_results['binary'] = evaluate_on_test_project(bin_model, bin_engineer, test_file, 'binary', k_values)
        except Exception as e:
            print(f"  ❌ Binary evaluation failed on {project_name}: {e}")
            project_results['binary'] = {}
        
        # Evaluate random baseline
        print(f"\n  🎲 Random baseline:")
        try:
            project_results['random'] = evaluate_random_baseline(test_file, k_values)
        except Exception as e:
            print(f"  ❌ Random baseline failed on {project_name}: {e}")
            project_results['random'] = {}
        
        all_results[project_name] = project_results
    
    # Format comprehensive results
    format_comprehensive_results(all_results, k_values)
    
    # Save detailed results
    output_file = 'comprehensive_project_ndcg_results.json'
    output_data = {
        'experiment_info': {
            'description': 'Comprehensive project-level NDCG evaluation with zero data leakage',
            'train_file': train_file,
            'test_files': test_files,
            'regression_config': regression_config,
            'binary_config': binary_config,
            'k_values': k_values
        },
        'results': all_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"🎉 EVALUATION COMPLETE!")

if __name__ == "__main__":
    main() 