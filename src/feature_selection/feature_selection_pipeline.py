#!/usr/bin/env python3
"""
Complete Feature Selection Pipeline combining MMPC, Ridge Ranking, and SFFS
This file implements the training pipeline from the ALS research paper
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import logging
from typing import List, Tuple, Dict, Any
import pickle
from pathlib import Path
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class FeatureSelectionPipeline:
    """
    Complete feature selection pipeline implementing:
    1. MMPC (Max-Min Parents and Children) feature selection
    2. Ridge Classifier coefficient ranking
    3. Sequential Forward Feature Selection (SFFS)
    
    Based on the methodology from the ALS research paper
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize results storage
        self.mmpc_features_ = None
        self.ridge_ranking_ = None
        self.sffs_results_ = None
        self.best_config_ = None
        self.pipeline_results_ = {}
        
        # Initialize scalers
        self.scaler = StandardScaler()
        
    def _default_config(self) -> Dict:
        """Default configuration matching the paper"""
        return {
            'mmpc': {
                'significance_threshold': 0.1,
                'max_features': 24  # As per paper
            },
            'ridge': {
                'alpha_range': [0.01, 0.1, 1.0, 10.0, 100.0],
                'cv_folds': 5
            },
            'sffs': {
                'max_features': 24,
                'cv_folds': 4,
                'scoring': 'accuracy'
            },
            'algorithms': {
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
                'AdaBoost': AdaBoostClassifier(random_state=42, n_estimators=100),
                'KNN': KNeighborsClassifier(),
                'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
                'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
                'SVM': SVC(random_state=42, kernel='rbf', probability=True),
                'DecisionTree': DecisionTreeClassifier(random_state=42)
            }
        }
    
    def mmpc_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        MMPC (Max-Min Parents and Children) feature selection
        Simplified implementation based on correlation analysis
        """
        self.logger.info("Running MMPC feature selection...")
        
        n_features = X.shape[1]
        feature_names = X.columns.tolist()
        
        # Calculate marginal correlations with target
        correlations = []
        for feature in feature_names:
            try:
                corr, p_value = pearsonr(X[feature], y)
                correlations.append({
                    'feature': feature,
                    'correlation': abs(corr),
                    'p_value': p_value
                })
            except:
                # Handle any numerical issues
                correlations.append({
                    'feature': feature,
                    'correlation': 0.0,
                    'p_value': 1.0
                })
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x['correlation'], reverse=True)
        
        # Select features meeting significance threshold
        selected_features = []
        for item in correlations:
            if (item['p_value'] < self.config['mmpc']['significance_threshold'] and 
                len(selected_features) < self.config['mmpc']['max_features']):
                selected_features.append(item['feature'])
        
        # Ensure we have at least some features
        if len(selected_features) < 10:
            selected_features = [item['feature'] for item in correlations[:24]]
        
        self.mmpc_features_ = selected_features
        self.logger.info(f"MMPC selected {len(selected_features)} features")
        
        return selected_features
    
    def ridge_coefficient_ranking(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Rank features using Ridge Classifier coefficients
        """
        self.logger.info("Ranking features using Ridge Classifier...")
        
        # Grid search for best Ridge parameters
        ridge = RidgeClassifier()
        param_grid = {'alpha': self.config['ridge']['alpha_range']}
        
        cv = StratifiedKFold(n_splits=self.config['ridge']['cv_folds'], 
                           shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(ridge, param_grid, cv=cv, scoring='accuracy')
        grid_search.fit(X, y)
        
        best_ridge = grid_search.best_estimator_
        
        # Get absolute coefficients as feature importance
        if hasattr(best_ridge, 'coef_'):
            coefficients = np.abs(best_ridge.coef_[0])
        else:
            coefficients = np.abs(best_ridge.coef_)
        
        # Create ranking DataFrame
        feature_ranking = pd.DataFrame({
            'feature': X.columns,
            'coefficient': coefficients
        }).sort_values('coefficient', ascending=False)
        
        self.ridge_ranking_ = feature_ranking
        ranked_features = feature_ranking['feature'].tolist()
        
        self.logger.info("Ridge ranking completed")
        return ranked_features
    
    def sequential_forward_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                           ranked_features: List[str]) -> Dict:
        """
        Sequential Forward Feature Selection with multiple algorithms
        """
        self.logger.info("Running Sequential Forward Feature Selection...")
        
        algorithms = self.config['algorithms']
        cv = StratifiedKFold(n_splits=self.config['sffs']['cv_folds'], 
                           shuffle=True, random_state=42)
        
        results = {}
        
        for alg_name, algorithm in algorithms.items():
            self.logger.info(f"Testing {alg_name}...")
            alg_results = []
            
            # Test different numbers of features (1 to max_features)
            max_features = min(len(ranked_features), self.config['sffs']['max_features'])
            
            for n_features in range(1, max_features + 1):
                # Select top n features
                selected_features = ranked_features[:n_features]
                X_selected = X[selected_features]
                
                try:
                    # Perform cross-validation
                    cv_scores = cross_val_score(algorithm, X_selected, y, cv=cv, 
                                              scoring='accuracy', n_jobs=-1)
                    
                    mean_score = cv_scores.mean()
                    std_score = cv_scores.std()
                    
                    # Calculate additional metrics on full dataset
                    algorithm.fit(X_selected, y)
                    y_pred = algorithm.predict(X_selected)
                    
                    if hasattr(algorithm, 'predict_proba'):
                        y_pred_proba = algorithm.predict_proba(X_selected)[:, 1]
                        auc_score = roc_auc_score(y, y_pred_proba)
                    else:
                        auc_score = None
                    
                    precision = precision_score(y, y_pred)
                    recall = recall_score(y, y_pred)
                    f1 = f1_score(y, y_pred)
                    
                    alg_results.append({
                        'n_features': n_features,
                        'features': selected_features.copy(),
                        'cv_mean': mean_score,
                        'cv_std': std_score,
                        'auc': auc_score,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'sensitivity': recall,  # Same as recall
                        'specificity': None  # Will calculate later if needed
                    })
                    
                    if n_features <= 5 or n_features % 5 == 0:
                        self.logger.info(f"  {n_features:2d} features: {mean_score:.4f} (+/- {std_score:.4f})")
                
                except Exception as e:
                    self.logger.warning(f"  Failed for {n_features} features: {str(e)}")
                    continue
            
            results[alg_name] = alg_results
        
        self.sffs_results_ = results
        return results
    
    def find_best_configuration(self) -> Tuple[str, int, List[str], float]:
        """
        Find the best algorithm and feature combination
        """
        if not self.sffs_results_:
            raise ValueError("Must run SFFS first")
        
        best_score = 0
        best_config = None
        
        for alg_name, alg_results in self.sffs_results_.items():
            for result in alg_results:
                if result['cv_mean'] > best_score:
                    best_score = result['cv_mean']
                    best_config = (alg_name, result['n_features'], 
                                 result['features'], best_score, result)
        
        self.best_config_ = best_config
        return best_config
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelectionPipeline':
        """
        Run the complete feature selection pipeline
        """
        self.logger.info("="*60)
        self.logger.info("STARTING FEATURE SELECTION PIPELINE")
        self.logger.info("="*60)
        self.logger.info(f"Input data shape: {X.shape}")
        self.logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Normalize data
        self.logger.info("Normalizing features...")
        X_normalized = pd.DataFrame(
            self.scaler.fit_transform(X),
            index=X.index,
            columns=X.columns
        )
        
        # Step 1: MMPC Feature Selection
        self.logger.info("\nSTEP 1: MMPC Feature Selection")
        mmpc_features = self.mmpc_feature_selection(X_normalized, y)
        X_mmpc = X_normalized[mmpc_features]
        
        # Step 2: Ridge Ranking
        self.logger.info("\nSTEP 2: Ridge Coefficient Ranking")
        ranked_features = self.ridge_coefficient_ranking(X_mmpc, y)
        
        # Step 3: Sequential Forward Feature Selection
        self.logger.info("\nSTEP 3: Sequential Forward Feature Selection")
        sffs_results = self.sequential_forward_feature_selection(X_mmpc, y, ranked_features)
        
        # Find best configuration
        self.logger.info("\nFinding best configuration...")
        best_config = self.find_best_configuration()
        
        if best_config:
            best_alg, best_n_features, best_features, best_score, best_metrics = best_config
            
            self.logger.info("="*60)
            self.logger.info("PIPELINE RESULTS")
            self.logger.info("="*60)
            self.logger.info(f"Best Algorithm: {best_alg}")
            self.logger.info(f"Number of Features: {best_n_features}")
            self.logger.info(f"Cross-validation Score: {best_score:.4f}")
            self.logger.info(f"AUC Score: {best_metrics['auc']:.4f}" if best_metrics['auc'] else "AUC: N/A")
            self.logger.info(f"Precision: {best_metrics['precision']:.4f}")
            self.logger.info(f"Recall: {best_metrics['recall']:.4f}")
            self.logger.info(f"F1-Score: {best_metrics['f1_score']:.4f}")
            
            self.logger.info(f"\nSelected Features:")
            for i, feature in enumerate(best_features, 1):
                self.logger.info(f"  {i:2d}. {feature}")
        
        # Store all results
        self.pipeline_results_ = {
            'mmpc_features': mmpc_features,
            'ridge_ranking': self.ridge_ranking_,
            'sffs_results': sffs_results,
            'best_config': {
                'algorithm': best_alg,
                'n_features': best_n_features,
                'features': best_features,
                'cv_score': best_score,
                'metrics': best_metrics
            }
        }
        
        return self
    
    def get_selected_features(self) -> List[str]:
        """Get the final selected features"""
        if self.best_config_ is None:
            raise ValueError("Must fit pipeline first")
        return self.best_config_[2]
    
    def get_best_algorithm(self) -> str:
        """Get the best performing algorithm"""
        if self.best_config_ is None:
            raise ValueError("Must fit pipeline first")
        return self.best_config_[0]
    
    def save_results(self, output_dir: str):
        """Save all pipeline results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save selected features
        with open(output_path / 'selected_features.txt', 'w') as f:
            for feature in self.get_selected_features():
                f.write(f"{feature}\n")
        
        # Save complete results
        with open(output_path / 'pipeline_results.pkl', 'wb') as f:
            pickle.dump(self.pipeline_results_, f)
        
        # Save Ridge ranking
        if self.ridge_ranking_ is not None:
            self.ridge_ranking_.to_csv(output_path / 'ridge_feature_ranking.csv', index=False)
        
        # Save SFFS results summary
        sffs_summary = []
        if self.sffs_results_:
            for alg_name, alg_results in self.sffs_results_.items():
                for result in alg_results:
                    sffs_summary.append({
                        'algorithm': alg_name,
                        'n_features': result['n_features'],
                        'cv_mean': result['cv_mean'],
                        'cv_std': result['cv_std'],
                        'auc': result.get('auc'),
                        'precision': result.get('precision'),
                        'recall': result.get('recall'),
                        'f1_score': result.get('f1_score')
                    })
            
            pd.DataFrame(sffs_summary).to_csv(output_path / 'sffs_results.csv', index=False)
        
        # Save best configuration summary
        if self.best_config_:
            best_summary = {
                'Algorithm': [self.best_config_[0]],
                'N_Features': [self.best_config_[1]],
                'CV_Score': [self.best_config_[3]],
                'AUC': [self.best_config_[4].get('auc', 'N/A')],
                'Precision': [self.best_config_[4].get('precision', 'N/A')],
                'Recall': [self.best_config_[4].get('recall', 'N/A')],
                'F1_Score': [self.best_config_[4].get('f1_score', 'N/A')]
            }
            pd.DataFrame(best_summary).to_csv(output_path / 'best_configuration.csv', index=False)
        
        self.logger.info(f"All results saved to {output_path}")

def main():
    """
    Main function to run the feature selection pipeline standalone
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Check if processed data exists
    data_dir = Path("data/processed")
    expression_file = data_dir / "combined_expression_data.csv"
    metadata_file = data_dir / "sample_metadata.csv"
    
    if not expression_file.exists() or not metadata_file.exists():
        logger.error("Processed data files not found!")
        logger.error("Please run the data download step first:")
        logger.error("python scripts/main.py --step download")
        return
    
    try:
        # Load data
        logger.info("Loading processed data...")
        expression_data = pd.read_csv(expression_file, index_col=0)
        metadata = pd.read_csv(metadata_file)
        
        # Prepare data (transpose so samples are rows, genes are columns)
        X = expression_data.T
        y = metadata['group'].map({'ALS': 1, 'Control': 0})
        
        # Remove samples with missing labels
        valid_samples = ~y.isna()
        X = X[valid_samples]
        y = y[valid_samples]
        
        logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} genes")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Run feature selection pipeline
        pipeline = FeatureSelectionPipeline()
        pipeline.fit(X, y)
        
        # Save results
        results_dir = "data/results/feature_selection"
        pipeline.save_results(results_dir)
        
        print("\n" + "="*60)
        print("FEATURE SELECTION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Results saved to: {results_dir}")
        print("\nNext step: Run SHAP analysis")
        print("python scripts/main.py --step shap_analysis")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()