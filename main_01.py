#!/usr/bin/env python3
"""
Research-Grade ALS Diagnosis Pipeline with SHAP Interpretability - FIXED VERSION
===============================================================================

Implementation following the exact methodology from:
"Exploiting Machine Learning And Gene Expression Analysis in Amyotrophic Lateral Sclerosis Diagnosis"

Enhanced with:
- SHAP explainable AI integration
- Research-compliant cross-validation
- Comprehensive statistical analysis
- Publication-ready visualizations
- Robust error handling and validation
- FIXED KeyError issues

Author: Research Team
Version: 2.1 (Fixed)
Date: 2024
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Core scientific libraries
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import logging
import yaml
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime

# Machine Learning Core
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, 
    GridSearchCV, cross_validate, ParameterGrid
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve,
    make_scorer
)
from sklearn.base import clone

# Advanced ML libraries
import xgboost as xgb
from xgboost import XGBClassifier

# Statistical analysis
from scipy.stats import pearsonr, ttest_ind, chi2_contingency, mannwhitneyu
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# SHAP for interpretability
import shap

# Progress tracking
from tqdm import tqdm

# Visualization setup
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 10

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/research_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("✅ Research-Grade Libraries Loaded Successfully!")
print(f"🐍 Python: {sys.version}")
print(f"📊 NumPy: {np.__version__}")
print(f"🐼 Pandas: {pd.__version__}")
print(f"🤖 Scikit-learn: {sklearn.__version__}")
print(f"🔍 SHAP: {shap.__version__}")

# =============================================================================
# RESEARCH CONFIGURATION MANAGEMENT
# =============================================================================

class ResearchConfig:
    """Research configuration management with validation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict:
        """Load configuration with defaults"""
        default_config = {
            'project': {
                'name': 'ALS_Research_Pipeline',
                'version': '2.1',
                'random_seed': 42
            },
            'data': {
                'test_size': 0.1,
                'validation_split': 0.2,
                'min_samples_per_class': 10
            },
            'preprocessing': {
                'normalization': 'StandardScaler',
                'remove_low_variance': True,
                'variance_threshold': 0.01,
                'handle_missing': 'drop'
            },
            'feature_selection': {
                'statistical_filter': {
                    'method': 'combined',
                    'p_threshold': 0.1,
                    'max_features': 100
                },
                'ridge_ranking': {
                    'alpha_range': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'cv_folds': 5
                },
                'sequential_selection': {
                    'min_features': 5,
                    'max_features': 30,
                    'cv_folds': 4,
                    'scoring': 'accuracy'
                }
            },
            'modeling': {
                'algorithms': [
                    'LogisticRegression', 'SVM', 'RandomForest', 
                    'XGBoost', 'AdaBoost', 'KNN', 'DecisionTree'
                ],
                'cv_strategy': 'StratifiedKFold',
                'cv_folds': 4,
                'hyperparameter_tuning': True,
                'n_jobs': -1
            },
            'shap': {
                'background_samples': 50,
                'explanation_samples': 100,
                'plot_types': ['summary', 'bar', 'waterfall', 'force'],
                'save_plots': True
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                # Merge with defaults
                return self._deep_merge(default_config, loaded_config)
            except Exception as e:
                logger.warning(f"Error loading config: {e}, using defaults")
                return default_config
        else:
            logger.info("No config file found, using defaults")
            return default_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def _validate_config(self):
        """Validate configuration parameters"""
        assert 0 < self.config['data']['test_size'] < 1, "Invalid test_size"
        assert self.config['feature_selection']['sequential_selection']['min_features'] > 0
        assert len(self.config['modeling']['algorithms']) > 0, "No algorithms specified"
        logger.info("✅ Configuration validated successfully")
    
    def get(self, key_path: str, default=None):
        """Get nested configuration value"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

# =============================================================================
# RESEARCH-GRADE DATA LOADER
# =============================================================================

class ResearchDataLoader:
    """Research-grade data loader with comprehensive validation"""
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.data_dir = Path("data/processed")
        self.X = None
        self.y = None
        self.metadata = None
        self.label_encoder = LabelEncoder()
        self.data_stats = {}
        
    def load_and_validate_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Load data with research-grade validation"""
        logger.info("🔬 Research-Grade Data Loading & Validation")
        logger.info("=" * 55)
        
        # Check file existence
        expression_file = self.data_dir / "combined_expression_data.csv"
        metadata_file = self.data_dir / "sample_metadata.csv"
        
        if not expression_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(
                f"Required data files not found:\n"
                f"  - {expression_file}\n"
                f"  - {metadata_file}\n"
                f"Please run the data preprocessing pipeline first."
            )
        
        # Load data
        logger.info("📂 Loading processed data files...")
        expression_data = pd.read_csv(expression_file, index_col=0)
        metadata = pd.read_csv(metadata_file)
        
        logger.info(f"✅ Data loaded:")
        logger.info(f"  Expression: {expression_data.shape}")
        logger.info(f"  Metadata: {metadata.shape}")
        
        # Comprehensive data validation
        self._validate_data_integrity(expression_data, metadata)
        
        # Prepare features and labels
        X, y = self._prepare_features_and_labels(expression_data, metadata)
        
        # Final validation
        self._final_data_validation(X, y, metadata)
        
        # Store data statistics
        self._compute_data_statistics(X, y, metadata)
        
        self.X, self.y, self.metadata = X, y, metadata
        return X, y, metadata
    
    def _validate_data_integrity(self, expression_data: pd.DataFrame, metadata: pd.DataFrame):
        """Comprehensive data integrity validation"""
        logger.info("🔍 Data integrity validation...")
        
        # Check for empty data
        if expression_data.empty or metadata.empty:
            raise ValueError("Empty datasets detected!")
        
        # Check sample alignment
        expr_samples = set(expression_data.columns)
        meta_samples = set(metadata['sample_id'])
        
        common_samples = expr_samples & meta_samples
        missing_in_expr = meta_samples - expr_samples
        missing_in_meta = expr_samples - meta_samples
        
        if len(missing_in_expr) > 0:
            logger.warning(f"⚠️ {len(missing_in_expr)} samples in metadata missing from expression")
        if len(missing_in_meta) > 0:
            logger.warning(f"⚠️ {len(missing_in_meta)} samples in expression missing from metadata")
        
        if len(common_samples) == 0:
            raise ValueError("No common samples between expression data and metadata!")
        
        logger.info(f"✅ Sample alignment: {len(common_samples)} common samples")
        
        # Check for required columns
        required_columns = ['sample_id', 'group']
        missing_columns = [col for col in required_columns if col not in metadata.columns]
        if missing_columns:
            raise ValueError(f"Missing required metadata columns: {missing_columns}")
        
        # Check group labels
        valid_groups = ['ALS', 'Control']
        unique_groups = metadata['group'].unique()
        invalid_groups = [g for g in unique_groups if g not in valid_groups]
        
        if len(invalid_groups) > 0:
            logger.warning(f"⚠️ Invalid group labels found: {invalid_groups}")
        
        logger.info("✅ Data integrity validation passed")
    
    def _prepare_features_and_labels(self, expression_data: pd.DataFrame, 
                                   metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix and labels with research standards"""
        logger.info("🎯 Preparing features and labels...")
        
        # Filter to common samples
        common_samples = list(set(expression_data.columns) & set(metadata['sample_id']))
        
        # Align data
        expression_aligned = expression_data[common_samples]
        metadata_aligned = metadata[metadata['sample_id'].isin(common_samples)].copy()
        metadata_aligned = metadata_aligned.set_index('sample_id').loc[common_samples].reset_index()
        
        # Filter to valid groups only
        valid_mask = metadata_aligned['group'].isin(['ALS', 'Control'])
        metadata_filtered = metadata_aligned[valid_mask].copy()
        valid_samples = metadata_filtered['sample_id'].tolist()
        expression_filtered = expression_aligned[valid_samples]
        
        # Transpose expression data (samples as rows, genes as columns)
        X = expression_filtered.T.copy()
        
        # Encode labels consistently
        self.label_encoder.fit(['ALS', 'Control'])  # Explicit class order
        y = pd.Series(
            self.label_encoder.transform(metadata_filtered['group']),
            index=X.index,
            name='ALS_label'
        )
        
        # Verify alignment
        assert len(X) == len(y), f"Length mismatch: X={len(X)}, y={len(y)}"
        assert all(X.index == metadata_filtered['sample_id']), "Index alignment error"
        
        logger.info(f"✅ Features and labels prepared:")
        logger.info(f"  Samples: {len(X)}")
        logger.info(f"  Features (genes): {X.shape[1]:,}")
        logger.info(f"  Label distribution: {dict(zip(['ALS', 'Control'], np.bincount(y)))}")
        
        return X, y
    
    def _final_data_validation(self, X: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame):
        """Final comprehensive data validation"""
        logger.info("🔬 Final data validation...")
        
        # Check for missing values
        missing_features = X.isnull().sum()
        total_missing = missing_features.sum()
        
        if total_missing > 0:
            logger.warning(f"⚠️ Missing values detected: {total_missing:,}")
            # Handle missing values
            if self.config.get('preprocessing.handle_missing') == 'drop':
                X_clean = X.dropna(axis=1)
                logger.info(f"Dropped {X.shape[1] - X_clean.shape[1]} features with missing values")
                X = X_clean
            else:
                X = X.fillna(X.mean())
                logger.info("Missing values imputed with feature means")
        
        # Check class balance
        class_counts = np.bincount(y)
        min_class_size = min(class_counts)
        
        if min_class_size < self.config.get('data.min_samples_per_class', 10):
            logger.warning(f"⚠️ Small class size: {min_class_size} samples")
        
        # Check feature variance
        if self.config.get('preprocessing.remove_low_variance'):
            var_threshold = self.config.get('preprocessing.variance_threshold', 0.01)
            feature_vars = X.var()
            low_var_features = feature_vars[feature_vars < var_threshold]
            
            if len(low_var_features) > 0:
                X = X.drop(columns=low_var_features.index)
                logger.info(f"Removed {len(low_var_features)} low-variance features")
        
        # Check for infinite values
        inf_mask = np.isinf(X).any(axis=1)
        if inf_mask.any():
            logger.warning(f"⚠️ Infinite values detected in {inf_mask.sum()} samples")
            X = X[~inf_mask]
            y = y[~inf_mask]
        
        logger.info("✅ Final validation completed")
    
    def _compute_data_statistics(self, X: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame):
        """Compute comprehensive data statistics for reporting"""
        logger.info("📊 Computing data statistics...")
        
        self.data_stats = {
            'total_samples': len(X),
            'total_features': X.shape[1],
            'class_distribution': dict(zip(['ALS', 'Control'], np.bincount(y))),
            'feature_range': {
                'min': float(X.min().min()),
                'max': float(X.max().max()),
                'mean': float(X.mean().mean()),
                'std': float(X.std().mean())
            },
            'missing_values': int(X.isnull().sum().sum()),
            'dataset_sources': metadata['dataset'].value_counts().to_dict() if 'dataset' in metadata.columns else {}
        }
        
        logger.info("✅ Data statistics computed")
    
    def get_label_mapping(self) -> Dict[int, str]:
        """Get label encoder mapping"""
        return dict(zip([0, 1], self.label_encoder.classes_))

# =============================================================================
# RESEARCH-GRADE FEATURE SELECTION PIPELINE - FIXED
# =============================================================================

class ResearchFeatureSelector:
    """
    Research-grade feature selection following exact paper methodology:
    1. Statistical pre-filtering (MMPC equivalent)
    2. Ridge coefficient ranking
    3. Sequential Forward Feature Selection (SFFS)
    """
    
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.selection_results = {}
        self.optimal_features = None
        self.optimal_model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ResearchFeatureSelector':
        """Execute complete feature selection pipeline"""
        logger.info("🔬 Research-Grade Feature Selection Pipeline")
        logger.info("=" * 50)
        
        # Set random seed for reproducibility
        np.random.seed(self.config.get('project.random_seed', 42))
        
        # Normalize features
        X_scaled = self._normalize_features(X)
        
        # Step 1: Statistical pre-filtering
        prefiltered_features = self._statistical_prefiltering(X_scaled, y)
        X_prefiltered = X_scaled[prefiltered_features]
        
        # Step 2: Ridge coefficient ranking
        ranked_features = self._ridge_coefficient_ranking(X_prefiltered, y)
        
        # Step 3: Sequential Forward Feature Selection
        optimal_config = self._sequential_forward_selection(X_prefiltered, y, ranked_features)
        
        # Store results
        self.optimal_features = optimal_config['features']
        self.optimal_model = optimal_config['model']
        
        logger.info("\n" + "=" * 50)
        logger.info("✅ FEATURE SELECTION COMPLETED!")
        logger.info("=" * 50)
        logger.info(f"🎯 Optimal Configuration:")
        logger.info(f"  Algorithm: {optimal_config['algorithm']}")
        logger.info(f"  Features: {len(optimal_config['features'])}")
        # FIXED: Use correct key names
        logger.info(f"  CV Accuracy: {optimal_config['cv_accuracy_mean']:.4f} ± {optimal_config['cv_accuracy_std']:.4f}")
        logger.info(f"  CV AUC: {optimal_config['cv_auc_mean']:.4f}")
        logger.info(f"  CV F1: {optimal_config['cv_f1_mean']:.4f}")
        logger.info(f"  CV Precision: {optimal_config['cv_precision_mean']:.4f}")
        logger.info(f"  CV Recall: {optimal_config['cv_recall_mean']:.4f}")
        
        return self
    
    def _normalize_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using StandardScaler"""
        logger.info("📏 Normalizing features...")
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        
        logger.info(f"✅ Features normalized: mean≈0, std≈1")
        return X_scaled
    
    def _statistical_prefiltering(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Statistical pre-filtering equivalent to MMPC algorithm
        Uses multiple statistical tests to identify relevant features
        """
        logger.info("🧪 Step 1: Statistical Pre-filtering (MMPC equivalent)")
        logger.info("-" * 45)
        
        feature_stats = []
        p_threshold = self.config.get('feature_selection.statistical_filter.p_threshold', 0.1)
        max_features = self.config.get('feature_selection.statistical_filter.max_features', 100)
        
        logger.info(f"Computing statistical tests for {len(X.columns):,} features...")
        
        for feature in tqdm(X.columns, desc="Statistical analysis"):
            try:
                feature_values = X[feature].values
                
                # Test 1: Pearson correlation
                corr_coef, p_corr = pearsonr(feature_values, y)
                
                # Test 2: T-test between groups
                als_values = feature_values[y == 1]
                ctrl_values = feature_values[y == 0]
                
                if len(als_values) > 1 and len(ctrl_values) > 1:
                    t_stat, p_ttest = ttest_ind(als_values, ctrl_values)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(als_values) - 1) * np.var(als_values, ddof=1) +
                                        (len(ctrl_values) - 1) * np.var(ctrl_values, ddof=1)) /
                                       (len(als_values) + len(ctrl_values) - 2))
                    cohens_d = (np.mean(als_values) - np.mean(ctrl_values)) / pooled_std
                else:
                    t_stat, p_ttest, cohens_d = 0, 1, 0
                
                # Test 3: Mann-Whitney U test (non-parametric)
                try:
                    u_stat, p_mannwhitney = mannwhitneyu(als_values, ctrl_values, alternative='two-sided')
                except:
                    u_stat, p_mannwhitney = 0, 1
                
                # Combined significance score
                combined_p = min(p_corr, p_ttest, p_mannwhitney)
                relevance_score = abs(corr_coef) * abs(cohens_d) * (1 - combined_p)
                
                feature_stats.append({
                    'feature': feature,
                    'correlation': abs(corr_coef),
                    'p_correlation': p_corr,
                    'p_ttest': p_ttest,
                    'p_mannwhitney': p_mannwhitney,
                    'cohens_d': abs(cohens_d),
                    'combined_p': combined_p,
                    'relevance_score': relevance_score
                })
                
            except Exception as e:
                # Handle problematic features
                logger.warning(f"Error processing feature {feature}: {e}")
                continue
        
        # Convert to DataFrame and filter
        stats_df = pd.DataFrame(feature_stats)
        
        # Apply multiple filtering criteria
        significant_features = set()
        
        # Criterion 1: Significant correlation
        sig_corr = stats_df[stats_df['p_correlation'] < p_threshold]
        significant_features.update(sig_corr.head(max_features)['feature'].tolist())
        
        # Criterion 2: Significant t-test
        sig_ttest = stats_df[stats_df['p_ttest'] < p_threshold]
        significant_features.update(sig_ttest.head(max_features)['feature'].tolist())
        
        # Criterion 3: Top by relevance score
        top_relevant = stats_df.nlargest(max_features, 'relevance_score')
        significant_features.update(top_relevant['feature'].tolist())
        
        # Ensure minimum features
        min_features = 50  # Ensure reasonable number for downstream analysis
        if len(significant_features) < min_features:
            top_features = stats_df.nlargest(min_features, 'relevance_score')
            significant_features.update(top_features['feature'].tolist())
        
        selected_features = list(significant_features)
        
        # Store results
        self.selection_results['statistical_prefiltering'] = {
            'selected_features': selected_features,
            'stats_df': stats_df,
            'selection_criteria': {
                'p_threshold': p_threshold,
                'max_features': max_features,
                'n_selected': len(selected_features)
            }
        }
        
        logger.info(f"✅ Statistical pre-filtering completed:")
        logger.info(f"  Selected features: {len(selected_features)}")
        logger.info(f"  Significant correlations: {len(sig_corr)}")
        logger.info(f"  Significant t-tests: {len(sig_ttest)}")
        
        return selected_features
    
    def _ridge_coefficient_ranking(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Ridge coefficient ranking with cross-validation
        Following exact paper methodology
        """
        logger.info("\n🎯 Step 2: Ridge Coefficient Ranking")
        logger.info("-" * 35)
        
        # Ridge hyperparameter optimization
        alpha_range = self.config.get('feature_selection.ridge_ranking.alpha_range')
        cv_folds = self.config.get('feature_selection.ridge_ranking.cv_folds')
        
        ridge = RidgeClassifier(random_state=self.config.get('project.random_seed'))
        param_grid = {'alpha': alpha_range}
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                           random_state=self.config.get('project.random_seed'))
        
        logger.info("Optimizing Ridge hyperparameters...")
        grid_search = GridSearchCV(
            ridge, param_grid, cv=cv, scoring='accuracy',
            n_jobs=self.config.get('modeling.n_jobs', -1)
        )
        
        grid_search.fit(X, y)
        best_ridge = grid_search.best_estimator_
        
        logger.info(f"✅ Best Ridge alpha: {best_ridge.alpha}")
        logger.info(f"✅ Best CV accuracy: {grid_search.best_score_:.4f}")
        
        # Get feature importance from coefficients
        coefficients = np.abs(best_ridge.coef_[0] if len(best_ridge.coef_.shape) > 1 else best_ridge.coef_)
        
        # Create ranking DataFrame
        ranking_df = pd.DataFrame({
            'feature': X.columns,
            'coefficient': coefficients
        }).sort_values('coefficient', ascending=False)
        
        # Store results
        self.selection_results['ridge_ranking'] = {
            'ranking_df': ranking_df,
            'best_ridge': best_ridge,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_
        }
        
        logger.info(f"✅ Ridge ranking completed")
        
        return ranking_df['feature'].tolist()
    
    def _sequential_forward_selection(self, X: pd.DataFrame, y: pd.Series, 
                                    ranked_features: List[str]) -> Dict[str, Any]:
        """
        Sequential Forward Feature Selection (SFFS) - FIXED VERSION
        Following exact paper methodology with comprehensive algorithm testing
        """
        logger.info("\n🚀 Step 3: Sequential Forward Feature Selection")
        logger.info("-" * 50)
        
        # Algorithm configurations following the paper
        algorithms = {
            'SVM': {
                'model': SVC(kernel='rbf', probability=True, 
                           random_state=self.config.get('project.random_seed')),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 0.1, 0.01, 0.001]
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(n_estimators=100, 
                                              random_state=self.config.get('project.random_seed')),
                'params': {
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'XGBoost': {
                'model': XGBClassifier(random_state=self.config.get('project.random_seed'), 
                                     eval_metric='logloss', verbosity=0),
                'params': {
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.1, 0.2, 0.3],
                    'n_estimators': [100, 200]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=self.config.get('project.random_seed'), 
                                          max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'AdaBoost': {
                'model': AdaBoostClassifier(random_state=self.config.get('project.random_seed')),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.5, 1.0, 1.5]
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            },
            'DecisionTree': {
                'model': DecisionTreeClassifier(random_state=self.config.get('project.random_seed')),
                'params': {
                    'max_depth': [5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }
        }
        
        # Filter to requested algorithms
        requested_algorithms = self.config.get('modeling.algorithms', list(algorithms.keys()))
        algorithms = {k: v for k, v in algorithms.items() if k in requested_algorithms}
        
        # SFFS parameters
        min_features = self.config.get('feature_selection.sequential_selection.min_features', 5)
        max_features = min(len(ranked_features), 
                          self.config.get('feature_selection.sequential_selection.max_features', 30))
        cv_folds = self.config.get('feature_selection.sequential_selection.cv_folds', 4)
        
        # Cross-validation strategy
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                           random_state=self.config.get('project.random_seed'))
        
        all_results = {}
        best_overall_score = 0
        best_overall_config = None
        
        logger.info(f"Testing {len(algorithms)} algorithms with {min_features}-{max_features} features...")
        
        for alg_name, alg_config in algorithms.items():
            logger.info(f"\n  🧪 Testing {alg_name}...")
            alg_results = []
            
            # Test different numbers of features
            feature_range = range(min_features, max_features + 1)
            
            for n_features in tqdm(feature_range, desc=f"{alg_name}", leave=False):
                try:
                    # Select top n features
                    selected_features = ranked_features[:n_features]
                    X_selected = X[selected_features]
                    
                    # Skip if insufficient class representation
                    if len(np.unique(y)) < 2:
                        continue
                    
                    # Grid search with cross-validation
                    grid_search = GridSearchCV(
                        alg_config['model'],
                        alg_config['params'],
                        cv=cv,
                        scoring='accuracy',
                        n_jobs=self.config.get('modeling.n_jobs', -1),
                        error_score='raise'
                    )
                    
                    grid_search.fit(X_selected, y)
                    best_model = grid_search.best_estimator_
                    
                    # Comprehensive cross-validation evaluation
                    cv_scores = cross_validate(
                        best_model, X_selected, y, cv=cv,
                        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                        n_jobs=self.config.get('modeling.n_jobs', -1)
                    )
                    
                    # Store results with FIXED key names
                    result = {
                        'n_features': n_features,
                        'features': selected_features.copy(),
                        'cv_accuracy_mean': cv_scores['test_accuracy'].mean(),
                        'cv_accuracy_std': cv_scores['test_accuracy'].std(),
                        'cv_precision_mean': cv_scores['test_precision'].mean(),
                        'cv_precision_std': cv_scores['test_precision'].std(),
                        'cv_recall_mean': cv_scores['test_recall'].mean(),
                        'cv_recall_std': cv_scores['test_recall'].std(),
                        'cv_f1_mean': cv_scores['test_f1'].mean(),
                        'cv_f1_std': cv_scores['test_f1'].std(),
                        'cv_auc_mean': cv_scores['test_roc_auc'].mean(),
                        'cv_auc_std': cv_scores['test_roc_auc'].std(),
                        'best_params': grid_search.best_params_,
                        'model': clone(best_model)
                    }
                    
                    alg_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"    Failed with {n_features} features: {str(e)}")
                    continue
            
            if alg_results:
                all_results[alg_name] = alg_results
                
                # Find best result for this algorithm
                best_result = max(alg_results, key=lambda x: x['cv_accuracy_mean'])
                logger.info(f"    Best: {best_result['n_features']} features, "
                          f"Accuracy: {best_result['cv_accuracy_mean']:.4f} ± {best_result['cv_accuracy_std']:.4f}, "
                          f"AUC: {best_result['cv_auc_mean']:.4f}, "
                          f"F1: {best_result['cv_f1_mean']:.4f}")
                
                # Check if this is the overall best
                if best_result['cv_accuracy_mean'] > best_overall_score:
                    best_overall_score = best_result['cv_accuracy_mean']
                    best_overall_config = {
                        'algorithm': alg_name,
                        **best_result
                    }
        
        # Store comprehensive results
        self.selection_results['sequential_selection'] = {
            'all_results': all_results,
            'best_config': best_overall_config
        }
        
        if best_overall_config is None:
            raise ValueError("No valid feature selection results obtained!")
        
        logger.info(f"\n✅ Sequential Forward Selection completed!")
        logger.info(f"    Best algorithm: {best_overall_config['algorithm']}")
        logger.info(f"    Best features: {best_overall_config['n_features']}")
        logger.info(f"    Best CV accuracy: {best_overall_config['cv_accuracy_mean']:.4f}")
        logger.info(f"    Best CV F1: {best_overall_config['cv_f1_mean']:.4f}")
        logger.info(f"    Best CV Precision: {best_overall_config['cv_precision_mean']:.4f}")
        logger.info(f"    Best CV Recall: {best_overall_config['cv_recall_mean']:.4f}")
        
        return best_overall_config

# =============================================================================
# ENHANCED SHAP ANALYZER FOR RESEARCH
# =============================================================================

class ResearchSHAPAnalyzer:
    """
    Research-grade SHAP analyzer with publication-ready visualizations
    """
    
    def __init__(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame,
                 y_train: pd.Series, y_test: pd.Series, feature_names: List[str],
                 label_mapping: Dict[int, str], config: ResearchConfig):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.label_mapping = label_mapping
        self.config = config
        
        self.explainer = None
        self.shap_values = None
        self.expected_value = None
        self.X_sample = None
        self.y_sample = None
        
        # Results storage
        self.results_dir = Path("results/shap_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def create_explainer(self) -> None:
        """Create appropriate SHAP explainer based on model type"""
        logger.info("🔧 Creating SHAP explainer...")
        
        background_samples = self.config.get('shap.background_samples', 50)
        
        # Select background data
        if len(self.X_train) > background_samples:
            background_idx = np.random.choice(len(self.X_train), background_samples, replace=False)
            background_data = self.X_train.iloc[background_idx]
        else:
            background_data = self.X_train
        
        model_name = type(self.model).__name__
        logger.info(f"Model type: {model_name}")
        
        try:
            # Try model-specific explainers first
            if any(tree_type in model_name for tree_type in ['Tree', 'Forest', 'XGB', 'Gradient']):
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                    logger.info("✅ Using TreeExplainer")
                    return
                except Exception as e:
                    logger.warning(f"TreeExplainer failed: {e}")
            
            if 'Linear' in model_name or 'Logistic' in model_name:
                try:
                    self.explainer = shap.LinearExplainer(self.model, background_data.values)
                    logger.info("✅ Using LinearExplainer")
                    return
                except Exception as e:
                    logger.warning(f"LinearExplainer failed: {e}")
            
            # Fallback to model-agnostic explainer
            def model_predict(x):
                x_df = pd.DataFrame(x, columns=self.feature_names)
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(x_df)
                    return proba[:, 1] if proba.shape[1] > 1 else proba.ravel()
                else:
                    return self.model.predict(x_df)
            
            self.explainer = shap.KernelExplainer(model_predict, background_data.values)
            logger.info("✅ Using KernelExplainer (model-agnostic)")
            
        except Exception as e:
            logger.error(f"❌ Failed to create SHAP explainer: {e}")
            raise
    
    def calculate_shap_values(self) -> None:
        """Calculate SHAP values with progress tracking"""
        logger.info("🧮 Calculating SHAP values...")
        
        max_samples = self.config.get('shap.explanation_samples', 100)
        
        # Select sample for explanation
        if len(self.X_test) > max_samples:
            sample_indices = np.random.choice(len(self.X_test), max_samples, replace=False)
            self.X_sample = self.X_test.iloc[sample_indices]
            self.y_sample = self.y_test.iloc[sample_indices]
        else:
            self.X_sample = self.X_test
            self.y_sample = self.y_test
        
        logger.info(f"Computing SHAP values for {len(self.X_sample)} samples...")
        
        try:
            # Calculate SHAP values
            if isinstance(self.explainer, shap.TreeExplainer):
                self.shap_values = self.explainer.shap_values(self.X_sample.values)
                self.expected_value = self.explainer.expected_value
            else:
                self.shap_values = self.explainer.shap_values(self.X_sample.values)
                self.expected_value = self.explainer.expected_value
            
            # Handle different output formats
            if isinstance(self.shap_values, list):
                # Binary classification - take positive class
                self.shap_values = self.shap_values[1] if len(self.shap_values) > 1 else self.shap_values[0]
                
            if isinstance(self.expected_value, list):
                self.expected_value = self.expected_value[1] if len(self.expected_value) > 1 else self.expected_value[0]
            
            logger.info(f"✅ SHAP values calculated successfully!")
            logger.info(f"   Shape: {self.shap_values.shape}")
            logger.info(f"   Expected value: {self.expected_value:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Error calculating SHAP values: {e}")
            raise
    
    def create_publication_plots(self) -> Dict[str, Any]:
        """Create publication-ready SHAP visualizations"""
        logger.info("📊 Creating publication-ready SHAP plots...")
        
        plot_results = {}
        
        # 1. Summary plot (beeswarm)
        logger.info("Creating summary plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_sample,
            feature_names=self.feature_names,
            show=False,
            max_display=min(20, len(self.feature_names))
        )
        plt.title('SHAP Summary Plot - Gene Impact on ALS Prediction', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=12)
        plt.tight_layout()
        
        if self.config.get('shap.save_plots', True):
            plt.savefig(self.results_dir / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Feature importance bar plot
        logger.info("Creating feature importance plot...")
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'gene': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(12, max(8, len(self.feature_names) * 0.4)))
        colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
        bars = plt.barh(range(len(importance_df)), importance_df['importance'], color=colors)
        
        plt.yticks(range(len(importance_df)), importance_df['gene'])
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.title('Gene Importance for ALS Diagnosis\n(Mean Absolute SHAP Values)', 
                 fontsize=16, fontweight='bold')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, importance_df['importance'])):
            plt.text(value + max(importance_df['importance']) * 0.01, i, 
                    f'{value:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        if self.config.get('shap.save_plots', True):
            plt.savefig(self.results_dir / 'feature_importance_bar.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        plot_results['feature_importance'] = importance_df
        
        # 3. Waterfall plots for representative samples
        logger.info("Creating waterfall plots...")
        self._create_waterfall_plots()
        
        # 4. Dependence plots for top features
        logger.info("Creating dependence plots...")
        self._create_dependence_plots(importance_df.tail(5)['gene'].tolist())
        
        return plot_results
    
    def _create_waterfall_plots(self, n_samples: int = 3):
        """Create waterfall plots for individual predictions"""
        try:
            for i in range(min(n_samples, len(self.shap_values))):
                # Get prediction information
                if hasattr(self.model, 'predict_proba'):
                    pred_proba = self.model.predict_proba(self.X_sample.iloc[i:i+1])[0, 1]
                else:
                    pred_proba = self.model.predict(self.X_sample.iloc[i:i+1])[0]
                
                actual_label = self.label_mapping.get(self.y_sample.iloc[i], 'Unknown')
                predicted_label = 'ALS' if pred_proba > 0.5 else 'Control'
                
                # Create waterfall plot
                plt.figure(figsize=(12, 8))
                
                # Create explanation object for waterfall
                explanation = shap.Explanation(
                    values=self.shap_values[i],
                    base_values=self.expected_value,
                    data=self.X_sample.iloc[i].values,
                    feature_names=self.feature_names
                )
                
                shap.waterfall_plot(explanation, show=False, max_display=15)
                plt.title(f'Individual Prediction Explanation - Sample {i+1}\n'
                         f'Actual: {actual_label}, Predicted: {predicted_label} '
                         f'(Prob: {pred_proba:.3f})', 
                         fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                if self.config.get('shap.save_plots', True):
                    plt.savefig(self.results_dir / f'waterfall_sample_{i+1}.png', 
                               dpi=300, bbox_inches='tight')
                plt.show()
                
        except Exception as e:
            logger.warning(f"Could not create waterfall plots: {e}")
    
    def _create_dependence_plots(self, top_features: List[str]):
        """Create partial dependence plots for top features"""
        try:
            for feature in top_features[:3]:  # Limit to top 3 features
                if feature in self.feature_names:
                    feature_idx = self.feature_names.index(feature)
                    
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(
                        feature_idx,
                        self.shap_values,
                        self.X_sample,
                        feature_names=self.feature_names,
                        show=False
                    )
                    plt.title(f'SHAP Dependence Plot - {feature}', 
                             fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    
                    if self.config.get('shap.save_plots', True):
                        safe_filename = feature.replace('/', '_').replace('\\', '_')
                        plt.savefig(self.results_dir / f'dependence_{safe_filename}.png', 
                                   dpi=300, bbox_inches='tight')
                    plt.show()
                    
        except Exception as e:
            logger.warning(f"Could not create dependence plots: {e}")
    
    def generate_clinical_insights(self) -> pd.DataFrame:
        """Generate comprehensive clinical insights from SHAP analysis"""
        logger.info("🧬 Generating clinical insights...")
        
        # Calculate comprehensive statistics
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        mean_shap = self.shap_values.mean(axis=0)
        std_shap = self.shap_values.std(axis=0)
        
        # Consistency metric (stability across samples)
        consistency = mean_abs_shap / (std_shap + 1e-8)
        
        # Directional analysis
        pos_contributions = (self.shap_values > 0).sum(axis=0)
        neg_contributions = (self.shap_values < 0).sum(axis=0)
        total_samples = len(self.shap_values)
        
        # Statistical significance of SHAP values
        shap_pvalues = []
        for i in range(len(self.feature_names)):
            try:
                _, p_val = ttest_ind(self.shap_values[:, i], np.zeros(len(self.shap_values)))
                shap_pvalues.append(p_val)
            except:
                shap_pvalues.append(1.0)
        
        # Multiple testing correction
        rejected, corrected_pvalues, _, _ = multipletests(shap_pvalues, method='fdr_bh')
        
        # Create comprehensive insights DataFrame
        insights_df = pd.DataFrame({
            'gene': self.feature_names,
            'mean_abs_shap': mean_abs_shap,
            'mean_shap': mean_shap,
            'std_shap': std_shap,
            'consistency_score': consistency,
            'positive_ratio': pos_contributions / total_samples,
            'negative_ratio': neg_contributions / total_samples,
            'shap_pvalue': shap_pvalues,
            'shap_pvalue_corrected': corrected_pvalues,
            'significant_after_correction': rejected,
            'primary_effect': ['Increases ALS Risk' if x > 0 else 'Decreases ALS Risk' 
                             for x in mean_shap],
            'effect_magnitude': pd.cut(mean_abs_shap, 
                                     bins=[0, np.percentile(mean_abs_shap, 33),
                                           np.percentile(mean_abs_shap, 67), np.inf],
                                     labels=['Low', 'Medium', 'High']),
            'reliability': pd.cut(consistency,
                                bins=[0, 1, 2, np.inf],
                                labels=['Variable', 'Moderate', 'Highly Reliable'])
        }).sort_values('mean_abs_shap', ascending=False)
        
        # Clinical classification
        insights_df['clinical_category'] = insights_df.apply(self._classify_clinical_relevance, axis=1)
        
        # Save insights
        insights_df.to_csv(self.results_dir / 'clinical_insights.csv', index=False)
        
        # Print summary
        logger.info("\n🎯 TOP 10 ALS BIOMARKERS (by SHAP importance):")
        logger.info("-" * 65)
        for i, (_, row) in enumerate(insights_df.head(10).iterrows(), 1):
            significance = "***" if row['significant_after_correction'] else ""
            logger.info(f"{i:2d}. {row['gene']:20s} - Importance: {row['mean_abs_shap']:.4f} {significance}")
            logger.info(f"    Effect: {row['primary_effect']}")
            logger.info(f"    Reliability: {row['reliability']}")
            logger.info(f"    Clinical Category: {row['clinical_category']}")
            logger.info("")
        
        return insights_df
    
    def _classify_clinical_relevance(self, row: pd.Series) -> str:
        """Classify genes by clinical relevance based on multiple criteria"""
        # High importance + high reliability + significant
        if (row['mean_abs_shap'] > np.percentile(row['mean_abs_shap'], 80) and
            row['consistency_score'] > 2.0 and
            row['significant_after_correction']):
            return 'Primary Biomarker Candidate'
        
        # High importance + moderate reliability + significant
        elif (row['mean_abs_shap'] > np.percentile(row['mean_abs_shap'], 70) and
              row['consistency_score'] > 1.5 and
              row['significant_after_correction']):
            return 'Secondary Biomarker Candidate'
        
        # Consistent direction (>80% samples show same effect)
        elif (row['positive_ratio'] > 0.8 or row['negative_ratio'] > 0.8):
            return 'Consistent Direction Marker'
        
        # High importance but variable
        elif row['mean_abs_shap'] > np.percentile(row['mean_abs_shap'], 60):
            return 'Variable Effect Marker'
        
        else:
            return 'Low Priority Marker'

# =============================================================================
# RESEARCH PIPELINE ORCHESTRATOR - ENHANCED
# =============================================================================

class ResearchPipeline:
    """
    Main research pipeline orchestrator with enhanced clinical metrics
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ResearchConfig(config_path)
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup results directory
        self.results_dir = Path(f"results/research_run_{self.timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(self.config.get('project.random_seed'))
        
        logger.info(f"🧬 Research Pipeline Initialized")
        logger.info(f"📁 Results will be saved to: {self.results_dir}")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete research pipeline"""
        logger.info("🚀 STARTING RESEARCH-GRADE ALS ANALYSIS PIPELINE")
        logger.info("=" * 65)
        logger.info(f"Pipeline version: {self.config.get('project.version')}")
        logger.info(f"Random seed: {self.config.get('project.random_seed')}")
        
        try:
            # Step 1: Data Loading & Validation
            logger.info("\n📊 STEP 1: DATA LOADING & VALIDATION")
            data_loader = ResearchDataLoader(self.config)
            X, y, metadata = data_loader.load_and_validate_data()
            
            self.results['data_stats'] = data_loader.data_stats
            self.results['label_mapping'] = data_loader.get_label_mapping()
            
            # Step 2: Feature Selection
            logger.info("\n🔬 STEP 2: RESEARCH-GRADE FEATURE SELECTION")
            feature_selector = ResearchFeatureSelector(self.config)
            feature_selector.fit(X, y)
            
            optimal_config = feature_selector.selection_results['sequential_selection']['best_config']
            selected_features = optimal_config['features']
            best_model = optimal_config['model']
            
            self.results['feature_selection'] = feature_selector.selection_results
            self.results['optimal_features'] = selected_features
            
            # Step 3: Final Model Training & Evaluation
            logger.info("\n🎯 STEP 3: FINAL MODEL TRAINING & EVALUATION")
            final_results = self._train_and_evaluate_final_model(
                X, y, selected_features, best_model, metadata
            )
            self.results.update(final_results)
            
            # Step 4: SHAP Analysis
            logger.info("\n🔍 STEP 4: SHAP INTERPRETABILITY ANALYSIS")
            shap_results = self._perform_shap_analysis(
                final_results['final_model'],
                final_results['X_train'], final_results['X_test'],
                final_results['y_train'], final_results['y_test'],
                selected_features, data_loader.get_label_mapping()
            )
            self.results['shap_analysis'] = shap_results
            
            # Step 5: Generate Research Report
            logger.info("\n📝 STEP 5: GENERATING RESEARCH REPORT")
            self._generate_research_report()
            
            # Step 6: Save Complete Results
            logger.info("\n💾 STEP 6: SAVING COMPLETE RESULTS")
            self._save_complete_results()
            
            logger.info("\n" + "=" * 65)
            logger.info("✅ RESEARCH PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 65)
            
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _train_and_evaluate_final_model(self, X: pd.DataFrame, y: pd.Series,
                                       selected_features: List[str], model: Any,
                                       metadata: pd.DataFrame) -> Dict[str, Any]:
        """Train and evaluate the final model with comprehensive metrics"""
        
        # Prepare final dataset
        X_selected = X[selected_features]
        
        # Research-compliant train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y,
            test_size=self.config.get('data.test_size'),
            random_state=self.config.get('project.random_seed'),
            stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=selected_features,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=selected_features,
            index=X_test.index
        )
        
        # Train final model
        final_model = clone(model)
        final_model.fit(X_train_scaled, y_train)
        
        # Comprehensive evaluation
        train_metrics = self._calculate_comprehensive_metrics(
            final_model, X_train_scaled, y_train, "Training"
        )
        test_metrics = self._calculate_comprehensive_metrics(
            final_model, X_test_scaled, y_test, "Testing"
        )
        
        # Cross-validation on full dataset for additional validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, 
                           random_state=self.config.get('project.random_seed'))
        cv_scores = cross_validate(
            clone(model), scaler.fit_transform(X_selected), y,
            cv=cv, scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        )
        
        cv_metrics = {
            f'cv_{metric}_mean': scores.mean()
            for metric, scores in cv_scores.items()
            if metric.startswith('test_')
        }
        cv_metrics.update({
            f'cv_{metric}_std': scores.std()
            for metric, scores in cv_scores.items()
            if metric.startswith('test_')
        })
        
        logger.info("📊 FINAL MODEL PERFORMANCE:")
        logger.info(f"Training Accuracy: {train_metrics['accuracy']:.4f}")
        logger.info(f"Testing Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Testing F1: {test_metrics['f1_score']:.4f}")
        logger.info(f"Testing Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Testing Recall: {test_metrics['recall']:.4f}")
        logger.info(f"Testing Specificity: {test_metrics['specificity']:.4f}")
        if 'auc' in test_metrics:
            logger.info(f"Testing AUC: {test_metrics['auc']:.4f}")
        logger.info(f"CV Accuracy: {cv_metrics['cv_test_accuracy_mean']:.4f} ± {cv_metrics['cv_test_accuracy_std']:.4f}")
        
        return {
            'final_model': final_model,
            'scaler': scaler,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'cv_metrics': cv_metrics
        }
    
    def _calculate_comprehensive_metrics(self, model: Any, X: pd.DataFrame,
                                       y: pd.Series, dataset_name: str) -> Dict[str, float]:
        """Calculate comprehensive performance metrics including clinical metrics"""
        
        y_pred = model.predict(X)
        
        # Standard classification metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'specificity': recall_score(y, y_pred, pos_label=0, zero_division=0)
        }
        
        # Add AUC if model supports probability prediction
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)[:, 1]
            metrics['auc'] = roc_auc_score(y, y_proba)
        
        # Clinical interpretation metrics
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        
        # Sensitivity (True Positive Rate) - ability to correctly identify ALS patients
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Specificity (True Negative Rate) - ability to correctly identify healthy controls
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Positive Predictive Value (Precision) - probability that positive prediction is correct
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Negative Predictive Value - probability that negative prediction is correct
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        # Clinical utility metrics
        metrics['positive_likelihood_ratio'] = metrics['sensitivity'] / (1 - metrics['specificity']) if metrics['specificity'] != 1 else float('inf')
        metrics['negative_likelihood_ratio'] = (1 - metrics['sensitivity']) / metrics['specificity'] if metrics['specificity'] != 0 else float('inf')
        
        return metrics
    
    def _perform_shap_analysis(self, model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame,
                              y_train: pd.Series, y_test: pd.Series, feature_names: List[str],
                              label_mapping: Dict[int, str]) -> Dict[str, Any]:
        """Perform comprehensive SHAP analysis"""
        
        shap_analyzer = ResearchSHAPAnalyzer(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
            label_mapping=label_mapping,
            config=self.config
        )
        
        # Create explainer and calculate SHAP values
        shap_analyzer.create_explainer()
        shap_analyzer.calculate_shap_values()
        
        # Generate plots and insights
        plot_results = shap_analyzer.create_publication_plots()
        clinical_insights = shap_analyzer.generate_clinical_insights()
        
        return {
            'shap_values': shap_analyzer.shap_values,
            'expected_value': shap_analyzer.expected_value,
            'clinical_insights': clinical_insights,
            'plot_results': plot_results
        }
    
    def _generate_research_report(self):
        """Generate comprehensive research report with clinical metrics"""
        logger.info("📝 Generating research report...")
        
        report_path = self.results_dir / "research_report.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# ALS Diagnosis Research Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Pipeline Version:** {self.config.get('project.version')}\n\n")
            
            # Data Summary
            f.write("## Data Summary\n\n")
            data_stats = self.results['data_stats']
            f.write(f"- Total Samples: {data_stats['total_samples']}\n")
            f.write(f"- Total Features: {data_stats['total_features']:,}\n")
            f.write(f"- Class Distribution: {data_stats['class_distribution']}\n")
            f.write(f"- Feature Range: {data_stats['feature_range']['min']:.3f} to {data_stats['feature_range']['max']:.3f}\n\n")
            
            # Feature Selection Results
            f.write("## Feature Selection Results\n\n")
            optimal_config = self.results['feature_selection']['sequential_selection']['best_config']
            f.write(f"- Optimal Algorithm: {optimal_config['algorithm']}\n")
            f.write(f"- Selected Features: {len(optimal_config['features'])}\n")
            f.write(f"- Cross-Validation Accuracy: {optimal_config['cv_accuracy_mean']:.4f} ± {optimal_config['cv_accuracy_std']:.4f}\n")
            f.write(f"- Cross-Validation AUC: {optimal_config['cv_auc_mean']:.4f}\n")
            f.write(f"- Cross-Validation F1: {optimal_config['cv_f1_mean']:.4f}\n")
            f.write(f"- Cross-Validation Precision: {optimal_config['cv_precision_mean']:.4f}\n")
            f.write(f"- Cross-Validation Recall: {optimal_config['cv_recall_mean']:.4f}\n\n")
            
            # Model Performance - Clinical Metrics
            f.write("## Final Model Performance\n\n")
            test_metrics = self.results['test_metrics']
            cv_metrics = self.results['cv_metrics']
            
            f.write("### Test Set Performance (Clinical Metrics)\n")
            f.write("| Metric | Value | Clinical Interpretation |\n")
            f.write("|--------|-------|------------------------|\n")
            f.write(f"| Accuracy | {test_metrics['accuracy']:.4f} | Overall diagnostic accuracy |\n")
            f.write(f"| Sensitivity (Recall) | {test_metrics['sensitivity']:.4f} | Ability to identify ALS patients |\n")
            f.write(f"| Specificity | {test_metrics['specificity']:.4f} | Ability to identify healthy controls |\n")
            f.write(f"| Precision (PPV) | {test_metrics['precision']:.4f} | Probability positive prediction is correct |\n")
            f.write(f"| F1-Score | {test_metrics['f1_score']:.4f} | Balanced precision-recall measure |\n")
            if 'npv' in test_metrics:
                f.write(f"| NPV | {test_metrics['npv']:.4f} | Probability negative prediction is correct |\n")
            if 'auc' in test_metrics:
                f.write(f"| AUC | {test_metrics['auc']:.4f} | Area under ROC curve |\n")
            
            f.write("\n### Cross-Validation Performance\n")
            f.write(f"- CV Accuracy: {cv_metrics['cv_test_accuracy_mean']:.4f} ± {cv_metrics['cv_test_accuracy_std']:.4f}\n")
            f.write(f"- CV Precision: {cv_metrics['cv_test_precision_mean']:.4f} ± {cv_metrics.get('cv_test_precision_std', 0):.4f}\n")
            f.write(f"- CV Recall: {cv_metrics['cv_test_recall_mean']:.4f} ± {cv_metrics.get('cv_test_recall_std', 0):.4f}\n")
            f.write(f"- CV F1: {cv_metrics['cv_test_f1_mean']:.4f} ± {cv_metrics.get('cv_test_f1_std', 0):.4f}\n")
            f.write(f"- CV AUC: {cv_metrics['cv_test_roc_auc_mean']:.4f} ± {cv_metrics['cv_test_roc_auc_std']:.4f}\n\n")
            
            # Literature Comparison
            f.write("## Literature Comparison\n\n")
            paper_results = {'accuracy': 0.8830, 'auc': 0.9111, 'features': 20}
            f.write(f"| Metric | Literature (Nguyen et al.) | Our Results | Improvement |\n")
            f.write(f"|--------|---------------------------|-------------|-------------|\n")
            f.write(f"| Accuracy | {paper_results['accuracy']:.4f} | {test_metrics['accuracy']:.4f} | {test_metrics['accuracy'] - paper_results['accuracy']:+.4f} |\n")
            if 'auc' in test_metrics:
                f.write(f"| AUC | {paper_results['auc']:.4f} | {test_metrics['auc']:.4f} | {test_metrics['auc'] - paper_results['auc']:+.4f} |\n")
            f.write(f"| Features | {paper_results['features']} | {len(optimal_config['features'])} | {len(optimal_config['features']) - paper_results['features']:+d} |\n\n")
            
            # Clinical Significance
            f.write("## Clinical Significance\n\n")
            f.write("### Key Clinical Metrics:\n")
            f.write(f"- **Sensitivity ({test_metrics['sensitivity']:.1%})**: Out of 100 ALS patients, {test_metrics['sensitivity']*100:.0f} would be correctly identified\n")
            f.write(f"- **Specificity ({test_metrics['specificity']:.1%})**: Out of 100 healthy individuals, {test_metrics['specificity']*100:.0f} would be correctly identified\n")
            if 'ppv' in test_metrics:
                f.write(f"- **PPV ({test_metrics['ppv']:.1%})**: When test is positive, there's a {test_metrics['ppv']*100:.0f}% chance of having ALS\n")
            if 'npv' in test_metrics:
                f.write(f"- **NPV ({test_metrics['npv']:.1%})**: When test is negative, there's a {test_metrics['npv']*100:.0f}% chance of being healthy\n")
            
            # Top Biomarkers
            f.write("\n## Top ALS Biomarkers (SHAP Analysis)\n\n")
            clinical_insights = self.results['shap_analysis']['clinical_insights']
            top_biomarkers = clinical_insights.head(10)
            
            f.write("| Rank | Gene | SHAP Importance | Effect | Reliability | Clinical Category |\n")
            f.write("|------|------|-----------------|--------|-------------|-------------------|\n")
            for i, (_, row) in enumerate(top_biomarkers.iterrows(), 1):
                f.write(f"| {i} | {row['gene']} | {row['mean_abs_shap']:.4f} | {row['primary_effect']} | {row['reliability']} | {row['clinical_category']} |\n")
            
            f.write("\n## Selected Gene Panel\n\n")
            f.write("### Final Gene Panel for ALS Diagnosis:\n")
            for i, feature in enumerate(optimal_config['features'], 1):
                f.write(f"{i}. **{feature}**\n")
            
            f.write(f"\n### Statistical Validation:\n")
            f.write(f"- Features selected through rigorous statistical filtering\n")
            f.write(f"- Ridge regression coefficient ranking\n")
            f.write(f"- Sequential forward feature selection with {optimal_config['algorithm']}\n")
            f.write(f"- {self.config.get('feature_selection.sequential_selection.cv_folds')}-fold cross-validation\n")
            
            f.write("\n---\n")
            f.write("*Report generated by Research-Grade ALS Analysis Pipeline*\n")
            f.write("*Following methodology from: Nguyen et al. (2024) - Exploiting Machine Learning and Gene Expression Analysis in ALS Diagnosis*")
        
        logger.info(f"✅ Research report saved to: {report_path}")
    
    def _save_complete_results(self):
        """Save all results with proper organization"""
        logger.info("💾 Saving complete results...")
        
        # Save main results as pickle
        results_file = self.results_dir / "complete_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save key DataFrames as CSV
        clinical_insights = self.results['shap_analysis']['clinical_insights']
        clinical_insights.to_csv(self.results_dir / "biomarker_analysis.csv", index=False)
        
        # Save feature selection details
        fs_results = self.results['feature_selection']
        if 'statistical_prefiltering' in fs_results:
            stats_df = fs_results['statistical_prefiltering']['stats_df']
            stats_df.to_csv(self.results_dir / "feature_statistics.csv", index=False)
        
        # Save model performance summary with clinical metrics
        performance_summary = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC', 'Sensitivity', 'PPV', 'NPV'],
            'Training': [
                self.results['train_metrics'].get('accuracy', 0),
                self.results['train_metrics'].get('precision', 0),
                self.results['train_metrics'].get('recall', 0),
                self.results['train_metrics'].get('f1_score', 0),
                self.results['train_metrics'].get('specificity', 0),
                self.results['train_metrics'].get('auc', 0),
                self.results['train_metrics'].get('sensitivity', 0),
                self.results['train_metrics'].get('ppv', 0),
                self.results['train_metrics'].get('npv', 0)
            ],
            'Testing': [
                self.results['test_metrics'].get('accuracy', 0),
                self.results['test_metrics'].get('precision', 0),
                self.results['test_metrics'].get('recall', 0),
                self.results['test_metrics'].get('f1_score', 0),
                self.results['test_metrics'].get('specificity', 0),
                self.results['test_metrics'].get('auc', 0),
                self.results['test_metrics'].get('sensitivity', 0),
                self.results['test_metrics'].get('ppv', 0),
                self.results['test_metrics'].get('npv', 0)
            ]
        })
        performance_summary.to_csv(self.results_dir / "performance_summary.csv", index=False)
        
        # Save selected features with importance
        optimal_config = self.results['feature_selection']['sequential_selection']['best_config']
        feature_panel = pd.DataFrame({
            'rank': range(1, len(optimal_config['features']) + 1),
            'gene': optimal_config['features'],
            'algorithm': optimal_config['algorithm'],
            'cv_accuracy': optimal_config['cv_accuracy_mean'],
            'cv_f1': optimal_config['cv_f1_mean']
        })
        feature_panel.to_csv(self.results_dir / "selected_gene_panel.csv", index=False)
        
        # Save configuration for reproducibility
        config_file = self.results_dir / "pipeline_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config.config, f, default_flow_style=False)
        
        logger.info(f"✅ All results saved to: {self.results_dir}")
        logger.info("📁 Generated files:")
        for file_path in self.results_dir.glob("*"):
            logger.info(f"  - {file_path.name}")

# =============================================================================
# DEMO DATA GENERATOR FOR TESTING
# =============================================================================

def generate_research_demo_data(n_samples: int = 500, n_genes: int = 2000, 
                               n_predictive: int = 25) -> None:
    """
    Generate high-quality synthetic data for pipeline testing
    """
    logger.info("🧪 Generating research-grade synthetic ALS data...")
    
    np.random.seed(42)
    
    # Generate realistic gene expression data
    # Base expression levels (log-transformed microarray-like data)
    base_expression = np.random.normal(8, 2, (n_samples, n_genes))
    
    # Add some correlation structure
    for i in range(0, n_genes, 50):
        end_idx = min(i + 50, n_genes)
        corr_noise = np.random.normal(0, 0.5, (n_samples, 1))
        base_expression[:, i:end_idx] += corr_noise
    
    # Generate realistic sample IDs and metadata
    sample_ids = [f"DEMO_Sample_{i:04d}" for i in range(n_samples)]
    
    # Generate balanced classes with slight imbalance (realistic)
    als_samples = int(n_samples * 0.38)  # ~38% ALS (realistic prevalence in studies)
    y = np.array([1] * als_samples + [0] * (n_samples - als_samples))
    np.random.shuffle(y)
    
    # Create predictive features
    predictive_genes = np.random.choice(n_genes, n_predictive, replace=False)
    
    for i, gene_idx in enumerate(predictive_genes):
        # Different types of effects for realism
        effect_type = i % 4
        
        if effect_type == 0:  # Upregulated in ALS
            effect_size = np.random.uniform(0.8, 2.5)
            base_expression[y == 1, gene_idx] += effect_size + np.random.normal(0, 0.3, sum(y == 1))
            
        elif effect_type == 1:  # Downregulated in ALS
            effect_size = np.random.uniform(0.8, 2.5)
            base_expression[y == 1, gene_idx] -= effect_size + np.random.normal(0, 0.3, sum(y == 1))
            
        elif effect_type == 2:  # Increased variance in ALS
            base_expression[y == 1, gene_idx] += np.random.normal(0, 1.5, sum(y == 1))
            
        else:  # Bimodal distribution in ALS
            als_mask = y == 1
            n_als = sum(als_mask)
            bimodal_shift = np.where(np.random.random(n_als) > 0.5, 1.5, -1.5)
            base_expression[als_mask, gene_idx] += bimodal_shift
    
    # Add realistic noise
    noise = np.random.normal(0, 0.1, (n_samples, n_genes))
    expression_data = base_expression + noise
    
    # Create DataFrames
    gene_names = [f"DEMO_Gene_{i:04d}" for i in range(n_genes)]
    
    expression_df = pd.DataFrame(
        expression_data.T,  # Genes as rows, samples as columns
        index=gene_names,
        columns=sample_ids
    )
    
    metadata_df = pd.DataFrame({
        'sample_id': sample_ids,
        'group': ['ALS' if label == 1 else 'Control' for label in y],
        'title': [f"Demo sample {i} - {'ALS' if label == 1 else 'Control'}" 
                 for i, label in enumerate(y)],
        'source': 'Synthetic Demo Data',
        'dataset': 'DEMO_RESEARCH'
    })
    
    # Save data
    data_dir = Path("data/processed")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    expression_df.to_csv(data_dir / "combined_expression_data.csv")
    metadata_df.to_csv(data_dir / "sample_metadata.csv", index=False)
    
    # Save data generation report
    with open(data_dir / "demo_data_info.txt", 'w') as f:
        f.write("DEMO DATA GENERATION REPORT\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Total Samples: {n_samples}\n")
        f.write(f"Total Genes: {n_genes}\n")
        f.write(f"Predictive Genes: {n_predictive}\n")
        f.write(f"ALS Samples: {sum(y)} ({sum(y)/len(y)*100:.1f}%)\n")
        f.write(f"Control Samples: {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)\n")
        f.write(f"Predictive Gene Indices: {sorted(predictive_genes)}\n")
        f.write(f"Random Seed: 42\n")
    
    logger.info(f"✅ Research demo data generated:")
    logger.info(f"  - Samples: {n_samples} ({sum(y)} ALS, {sum(y == 0)} Control)")
    logger.info(f"  - Genes: {n_genes} ({n_predictive} predictive)")
    logger.info(f"  - Files saved to: {data_dir}")

# =============================================================================
# MAIN EXECUTION - FIXED
# =============================================================================

def main():
    """Main execution function with enhanced clinical reporting"""
    print("🧬 RESEARCH-GRADE ALS DIAGNOSIS PIPELINE")
    print("=" * 50)
    print("Enhanced with SHAP interpretability and clinical insights")
    print("Following exact methodology from published research")
    print("FIXED VERSION 2.1 - Enhanced Clinical Metrics")
    print("=" * 50)
    
    try:
        # Initialize and run pipeline
        pipeline = ResearchPipeline()
        results = pipeline.run_complete_pipeline()
        
        # Print final summary
        print("\n" + "🎊" * 50)
        print("RESEARCH PIPELINE COMPLETED SUCCESSFULLY!")
        print("🎊" * 50)
        
        test_metrics = results['test_metrics']
        optimal_config = results['feature_selection']['sequential_selection']['best_config']
        
        print(f"📊 FINAL RESULTS SUMMARY:")
        print(f"  ✅ Algorithm: {optimal_config['algorithm']}")
        print(f"  ✅ Features Selected: {len(optimal_config['features'])}")
        print(f"  ✅ Test Accuracy: {test_metrics['accuracy']*100:.2f}%")
        print(f"  ✅ Test Sensitivity: {test_metrics['sensitivity']*100:.2f}%")
        print(f"  ✅ Test Specificity: {test_metrics['specificity']*100:.2f}%")
        print(f"  ✅ Test F1-Score: {test_metrics['f1_score']*100:.2f}%")
        print(f"  ✅ Test Precision: {test_metrics['precision']*100:.2f}%")
        print(f"  ✅ Test Recall: {test_metrics['recall']*100:.2f}%")
        if 'auc' in test_metrics:
            print(f"  ✅ Test AUC: {test_metrics['auc']*100:.2f}%")
        print(f"  ✅ Results Directory: {pipeline.results_dir}")
        
        # Clinical interpretation
        print(f"\n🏥 CLINICAL INTERPRETATION:")
        print(f"  • Out of 100 ALS patients, {test_metrics['sensitivity']*100:.0f} would be correctly identified")
        print(f"  • Out of 100 healthy controls, {test_metrics['specificity']*100:.0f} would be correctly identified")
        print(f"  • When test is positive, {test_metrics.get('ppv', 0)*100:.0f}% chance of having ALS")
        
        # Top biomarkers
        clinical_insights = results['shap_analysis']['clinical_insights']
        print(f"\n🧬 TOP 5 ALS BIOMARKERS:")
        for i, (_, row) in enumerate(clinical_insights.head(5).iterrows(), 1):
            print(f"  {i}. {row['gene']} - {row['primary_effect']} ({row['clinical_category']})")
        
        print(f"\n📁 All results saved to: {pipeline.results_dir}")
        print("🔬 Ready for publication and clinical validation!")
        print("\n📊 Key Clinical Metrics Available:")
        print("  • Sensitivity, Specificity, PPV, NPV")
        print("  • F1-Score, Precision, Recall")
        print("  • AUC-ROC for diagnostic performance")
        print("  • Cross-validation statistics")
        
        return results
        
    except FileNotFoundError:
        print("\n⚠️ Data files not found. Generating demo data...")
        generate_research_demo_data()
        print("✅ Demo data generated. Restarting pipeline...")
        
        # Retry with demo data
        pipeline = ResearchPipeline()
        return pipeline.run_complete_pipeline()
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Research-Grade ALS Analysis Pipeline - FIXED')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--demo', action='store_true', help='Generate and use demo data')
    parser.add_argument('--demo-samples', type=int, default=500, help='Number of demo samples')
    parser.add_argument('--demo-genes', type=int, default=2000, help='Number of demo genes')
    
    args = parser.parse_args()
    
    if args.demo:
        print("🧪 Generating demo data...")
        generate_research_demo_data(args.demo_samples, args.demo_genes)
        print("✅ Demo data generated!")
    
    # Run main pipeline
    results = main()
    requested_features = results['feature_selection']['sequential_selection']['best_config']['features']
    print(f"\nSelected features for final model: {requested_features}")
    print("🔬 Ready for clinical validation and publication!")