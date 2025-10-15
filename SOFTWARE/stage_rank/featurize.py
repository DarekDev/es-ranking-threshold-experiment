"""
Feature engineering for Doctrina stage ranking pipeline.
Handles text embeddings (SBERT), categorical encoding, and numeric preprocessing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from .io_utils import print_info, print_warning, print_success, colored_print, show_progress_bar

# Optional SBERT import
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    SentenceTransformer = None


class FeatureEngineer:
    """
    Feature engineering pipeline for candidate ranking.
    
    - Text columns: TF-IDF (default) or SBERT embeddings (optional)
    - Categorical columns: One-hot encoding
    - Numeric columns: Standard scaling
    """
    
    def __init__(self, text_method: str = 'tfidf', model_name: str = 'all-MiniLM-L6-v2', 
                 max_features: int = 1000):
        """
        Initialize feature engineer.
        
        Args:
            text_method: 'tfidf' or 'sbert' for text feature extraction
            model_name: SentenceTransformer model name (when using SBERT)
            max_features: Maximum TF-IDF features (when using TF-IDF)
        """
        self.text_method = text_method
        self.model_name = model_name
        self.max_features = max_features
        self.sentence_transformer = None
        self.tfidf_vectorizers = {}
        self.preprocessor = None
        self.text_columns = []
        self.categorical_columns = []
        self.numeric_columns = []
        self.is_fitted = False
        
    def fit(self, df: pd.DataFrame, column_types: Dict[str, str], target_column: str, 
            exclude_columns: List[str] = None):
        """
        Fit the feature engineering pipeline.
        
        Args:
            df: Training DataFrame
            column_types: Dict mapping column names to types
            target_column: Name of target column to exclude
            exclude_columns: Additional columns to exclude from features
        """
        print_info("Building Feature Engineering Pipeline", "🔧")
        
        # Columns to exclude from features
        exclude_columns = exclude_columns or []
        all_excluded = set([target_column] + exclude_columns)
        
        # Separate columns by type (excluding target and excluded columns)
        self.text_columns = [col for col, ctype in column_types.items() 
                           if ctype == 'text' and col not in all_excluded]
        self.categorical_columns = [col for col, ctype in column_types.items() 
                                  if ctype == 'categorical' and col not in all_excluded]
        self.numeric_columns = [col for col, ctype in column_types.items() 
                              if ctype == 'numeric' and col not in all_excluded]
        
        print_info("Feature Columns by Type:", "📋")
        print(f"   📝 Text ({len(self.text_columns)}): {self.text_columns}")
        print(f"   🏷️  Categorical ({len(self.categorical_columns)}): {self.categorical_columns}")
        print(f"   🔢 Numeric ({len(self.numeric_columns)}): {self.numeric_columns}")
        print(f"   📊 Total feature columns: {len(self.text_columns) + len(self.categorical_columns) + len(self.numeric_columns)}")
        print()
        
        # Initialize text feature extractor
        if self.text_columns:
            if self.text_method == 'sbert':
                if not SBERT_AVAILABLE:
                    raise ImportError("SBERT requested but sentence-transformers not installed. "
                                    "Install with: pip install sentence-transformers")
                print(f"   🤖 Loading SentenceTransformer: {self.model_name}")
                self.sentence_transformer = SentenceTransformer(self.model_name)
            else:  # TF-IDF
                print_info(f"Initializing TF-IDF vectorizers (max_features={self.max_features})", "📊")
                for i, col in enumerate(self.text_columns):
                    show_progress_bar(i + 1, len(self.text_columns), f"   Fitting {col}")
                    self.tfidf_vectorizers[col] = TfidfVectorizer(
                        max_features=self.max_features,
                        stop_words='english',
                        lowercase=True,
                        ngram_range=(1, 2)  # Unigrams and bigrams
                    )
                    # Fit the vectorizer
                    texts = df[col].fillna('').astype(str).tolist()
                    self.tfidf_vectorizers[col].fit(texts)
                if self.text_columns:
                    print()  # New line after progress
        
        # Set up preprocessor for categorical and numeric columns
        transformers = []
        
        if self.categorical_columns:
            # Convert all categorical columns to strings to avoid type comparison issues
            for col in self.categorical_columns:
                df[col] = df[col].astype(str)
            
            transformers.append(
                ('categorical', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                 self.categorical_columns)
            )
        
        if self.numeric_columns:
            transformers.append(
                ('numeric', StandardScaler(), self.numeric_columns)
            )
        
        if transformers:
            self.preprocessor = ColumnTransformer(transformers, remainder='drop')
            # Fit on non-text columns
            non_text_df = df[self.categorical_columns + self.numeric_columns]
            if not non_text_df.empty:
                self.preprocessor.fit(non_text_df)
        
        self.is_fitted = True
        print("   ✅ Feature engineering pipeline fitted")
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform DataFrame to feature matrix.
        
        Args:
            df: DataFrame to transform
            
        Returns:
            Feature matrix as numpy array
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        features = []
        
        # Process text columns
        if self.text_columns:
            if self.text_method == 'sbert':
                print("   🔤 Generating SBERT embeddings...")
                for col in self.text_columns:
                    texts = df[col].fillna('').astype(str).tolist()
                    embeddings = self.sentence_transformer.encode(texts)
                    features.append(embeddings)
            else:  # TF-IDF
                print_info("Generating TF-IDF features...", "📊")
                for i, col in enumerate(self.text_columns):
                    show_progress_bar(i + 1, len(self.text_columns), f"   Processing {col}")
                    texts = df[col].fillna('').astype(str).tolist()
                    tfidf_features = self.tfidf_vectorizers[col].transform(texts).toarray()
                    features.append(tfidf_features)
                if self.text_columns:
                    print()  # New line after progress
        
        # Process categorical and numeric columns
        if self.preprocessor:
            non_text_df = df[self.categorical_columns + self.numeric_columns].copy()
            
            # Convert categorical columns to strings to avoid type comparison issues
            for col in self.categorical_columns:
                if col in non_text_df.columns:
                    non_text_df[col] = non_text_df[col].astype(str)
            
            if not non_text_df.empty:
                processed_features = self.preprocessor.transform(non_text_df)
                features.append(processed_features)
        
        # Combine all features
        if features:
            combined_features = np.hstack(features)
        else:
            # If no features, create a dummy feature
            combined_features = np.ones((len(df), 1))
        
        print_success(f"Feature Engineering Complete!")
        print(f"   📊 Generated {combined_features.shape[1]:,} ML features from {len(self.text_columns) + len(self.categorical_columns) + len(self.numeric_columns)} input columns")
        print(f"   🎯 Ready for training on {combined_features.shape[0]:,} samples")
        return combined_features
    
    def fit_transform(self, df: pd.DataFrame, column_types: Dict[str, str], 
                     target_column: str, exclude_columns: List[str] = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df, column_types, target_column, exclude_columns)
        return self.transform(df)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features."""
        feature_names = []
        
        # Text feature names
        for col in self.text_columns:
            if self.text_method == 'sbert':
                embedding_dim = 384  # Default for all-MiniLM-L6-v2
                feature_names.extend([f"{col}_embed_{i}" for i in range(embedding_dim)])
            else:  # TF-IDF
                if col in self.tfidf_vectorizers:
                    tfidf_names = self.tfidf_vectorizers[col].get_feature_names_out()
                    feature_names.extend([f"{col}_tfidf_{name}" for name in tfidf_names])
        
        # Categorical and numeric feature names
        if self.preprocessor:
            try:
                other_names = self.preprocessor.get_feature_names_out()
                feature_names.extend(other_names)
            except:
                # Fallback if get_feature_names_out not available
                for col in self.categorical_columns:
                    feature_names.append(f"{col}_encoded")
                for col in self.numeric_columns:
                    feature_names.append(f"{col}_scaled")
        
        return feature_names
    
    def save(self, path: str):
        """Save the fitted feature engineer."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"   💾 Feature engineer saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FeatureEngineer':
        """Load a fitted feature engineer."""
        with open(path, 'rb') as f:
            feature_engineer = pickle.load(f)
        print(f"   📂 Feature engineer loaded from {path}")
        return feature_engineer


def map_target_values(df: pd.DataFrame, target_column: str, 
                     target_mapping: Dict[str, float]) -> pd.Series:
    """
    Map target values according to provided mapping.
    
    Args:
        df: DataFrame containing target column
        target_column: Name of target column
        target_mapping: Dict mapping target values to numeric scores
        
    Returns:
        Series with mapped target values
    """
    print_info(f"Mapping target values in column '{target_column}'", "🎯")
    
    target_values = df[target_column].copy()
    
    # Debug: Show data types and sample values
    print_info(f"Target column data type: {target_values.dtype}")
    unique_vals = sorted(target_values.unique())
    print_info(f"Unique target values in data: {unique_vals}")
    print_info(f"Config mapping keys: {list(target_mapping.keys())}")
    
    # Convert mapping keys to match data type
    # Try both string and numeric mappings
    converted_mapping = {}
    for key, value in target_mapping.items():
        # Add both string and numeric versions
        converted_mapping[str(key)] = value
        try:
            converted_mapping[int(key)] = value
        except (ValueError, TypeError):
            pass
        try:
            converted_mapping[float(key)] = value
        except (ValueError, TypeError):
            pass
    
    print_info(f"Converted mapping keys: {list(converted_mapping.keys())}")
    
    # Apply mapping
    mapped_values = target_values.map(converted_mapping)
    
    # Check for unmapped values
    unmapped = target_values[mapped_values.isna()]
    if len(unmapped) > 0:
        unique_unmapped = sorted(unmapped.unique())
        print_warning(f"{len(unmapped)} rows have unmapped target values: {unique_unmapped}")
        print_info(f"Available mappings: {list(target_mapping.keys())}")
        # Fill with 0.0 as default
        mapped_values = mapped_values.fillna(0.0)
    
    print_success(f"Target mapping complete. Range: [{mapped_values.min():.2f}, {mapped_values.max():.2f}]")
    return mapped_values
