#!/usr/bin/env python3
"""
Add Linear Baseline to Existing NDCG Results
============================================

This script runs a logistic regression baseline on the same TF-IDF features
used by XGBoost models and computes NDCG scores for the three test projects.

Usage:
    python add_linear_baseline.py

Outputs:
    - Prints NDCG scores for linear baseline
    - Updates existing results with baseline comparison
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score
import json
from pathlib import Path

# Import your existing modules
from stage_rank import load_config, load_csv, infer_column_types, map_target_values, parse_binary_labels
from stage_rank.featurize import FeatureEngineer

def compute_ndcg_at_k(y_true, y_scores, k_values):
    """Compute NDCG@K for multiple K values."""
    ndcg_scores = {}
    
    for k in k_values:
        if len(y_true) >= k:
            ndcg = ndcg_score([y_true], [y_scores], k=k)
            ndcg_scores[k] = ndcg
        else:
            ndcg_scores[k] = ndcg_score([y_true], [y_scores])
    
    return ndcg_scores

def run_linear_baseline():
    """Run logistic regression baseline on test projects."""
    
    print("🔍 Adding Linear Baseline to NDCG Evaluation")
    print("=" * 50)
    
    # Configuration
    train_file = "../DATASET/train_data_multi_project.csv"
    config_file = "configs/optimal_binary.txt"  # Use binary config for classification
    
    test_files = [
        "../DATASET/test_data_project_KS23-AVCEO.csv",
        "../DATASET/test_data_project_KS23-HYPAPR.csv", 
        "../DATASET/test_data_project_KS24-SYVPC.csv"
    ]
    
    k_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    # Load config and training data
    print("📊 Loading training data and config...")
    config = load_config(config_file)
    train_df = load_csv(train_file)
    column_types = infer_column_types(train_df)
    
    target_column = config.get('target_column', 'candidate_Stage')
    exclude_columns = config.get('exclude_columns', [])
    if isinstance(exclude_columns, str):
        exclude_columns = [col.strip() for col in exclude_columns.split(',') if col.strip()]
    
    # Process training targets (binary classification)
    binary_config = parse_binary_labels(config)
    y_train = train_df[target_column].map(binary_config['mapping']).astype(int)
    
    # Feature engineering on training data
    print("🔧 Building TF-IDF features...")
    feature_engineer = FeatureEngineer(
        text_method='tfidf',
        max_features=2000
    )
    X_train = feature_engineer.fit_transform(train_df, column_types, target_column, exclude_columns)
    
    # Train logistic regression
    print("📈 Training logistic regression baseline...")
    linear_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    linear_model.fit(X_train, y_train)
    
    print(f"   ✅ Model trained on {len(X_train):,} samples with {X_train.shape[1]:,} features")
    
    # Evaluate on each test project
    all_ndcg_results = []
    project_names = []
    
    for test_file in test_files:
        project_name = Path(test_file).stem.split('_')[-1]  # Extract project ID
        project_names.append(project_name)
        
        print(f"\n🎯 Evaluating on {project_name}...")
        
        # Load test data
        test_df = load_csv(test_file)
        y_test = test_df[target_column].map(binary_config['mapping']).astype(int)
        
        # Transform test features
        X_test = feature_engineer.transform(test_df)
        
        # Get predictions (probabilities)
        y_pred_proba = linear_model.predict_proba(X_test)[:, 1]
        
        # Compute NDCG@K
        ndcg_results = compute_ndcg_at_k(y_test, y_pred_proba, k_values)
        all_ndcg_results.append(ndcg_results)
        
        # Print results for this project
        print(f"   📊 {project_name} NDCG@K results:")
        for k in [10, 50, 100]:
            print(f"      NDCG@{k}: {ndcg_results[k]:.3f}")
    
    # Compute mean NDCG across projects
    print(f"\n📋 Linear Baseline Summary (n={len(test_files)} projects)")
    print("-" * 50)
    
    mean_ndcg = {}
    for k in k_values:
        scores = [result[k] for result in all_ndcg_results]
        mean_ndcg[k] = np.mean(scores)
        std_ndcg = np.std(scores)
        print(f"NDCG@{k:3d}: {mean_ndcg[k]:.3f} ± {std_ndcg:.3f}")
    
    overall_mean = np.mean(list(mean_ndcg.values()))
    print(f"\nOverall Mean NDCG: {overall_mean:.3f}")
    
    # Create table row for paper
    print(f"\n📄 Table Row for Paper:")
    print("Linear Baseline &", end=" ")
    for k in k_values:
        print(f"{mean_ndcg[k]:.3f}", end=" & " if k < 100 else f" & {overall_mean:.3f} \\\\")
    print()
    
    # Save detailed results
    results = {
        'method': 'logistic_regression_baseline',
        'projects': project_names,
        'individual_results': all_ndcg_results,
        'mean_ndcg': mean_ndcg,
        'overall_mean': overall_mean,
        'model_params': {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42
        },
        'feature_info': {
            'total_features': X_train.shape[1],
            'text_method': 'tfidf',
            'max_features': 2000
        }
    }
    
    output_file = "linear_baseline_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to {output_file}")
    print("\n✅ Linear baseline evaluation complete!")
    
    return results

if __name__ == "__main__":
    results = run_linear_baseline() 