import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import accuracy_score
import logging
from typing import List, Tuple, Dict
import pickle
from pathlib import Path

class MMPCSelector:
    """
    MMPC (Max-Min Parents and Children) feature selection
    Simplified implementation for demonstration
    """
    
    def __init__(self, max_k: int = 3, significance_threshold: float = 0.1):
        self.max_k = max_k
        self.significance_threshold = significance_threshold
        self.selected_features_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MMPCSelector':
        """
        Fit MMPC selector
        Note: This is a simplified implementation
        For production, use causal-learn library
        """
        from scipy.stats import chi2_contingency, pearsonr
        
        n_features = X.shape[1]
        feature_names = X.columns.tolist()
        selected_features = []
        
        # Calculate marginal associations
        marginal_associations = []
        for i, feature in enumerate(feature_names):
            # Calculate correlation with target
            corr, p_value = pearsonr(X.iloc[:, i], y)
            marginal_associations.append((i, feature, abs(corr), p_value))
        
        # Sort by correlation strength
        marginal_associations.sort(key=lambda x: x[2], reverse=True)
        
        # Select top features that meet significance threshold
        for i, feature, corr, p_value in marginal_associations:
            if p_value < self.significance_threshold and len(selected_features) < 50:
                selected_features.append(feature)
        
        self.selected_features_ = selected_features[:24]  # Limit to 24 as per paper
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to selected features"""
        if self.selected_features_ is None:
            raise ValueError("Must fit selector first")
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(X, y).transform(X)

class RidgeRanker:
    """
    Use Ridge Classifier coefficients to rank gene importance
    """
    
    def __init__(self, alpha_range: List[float] = None, cv_folds: int = 5):
        self.alpha_range = alpha_range or [0.01, 0.1, 1.0, 10.0]
        self.cv_folds = cv_folds
        self.best_model_ = None
        self.feature_ranking_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RidgeRanker':
        """
        Fit Ridge classifier and extract feature importance
        """
        # Grid search for best alpha
        ridge = RidgeClassifier()
        param_grid = {'alpha': self.alpha_range}
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        grid_search = GridSearchCV(ridge, param_grid, cv=cv, scoring='accuracy')
        
        grid_search.fit(X, y)
        self.best_model_ = grid_search.best_estimator_
        
        # Get feature importance from coefficients
        coefficients = np.abs(self.best_model_.coef_[0])
        
        # Create ranking
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': coefficients
        }).sort_values('importance', ascending=False)
        
        self.feature_ranking_ = feature_importance
        
        return self
    
    def get_ranked_features(self) -> List[str]:
        """Get features ranked by importance"""
        if self.feature_ranking_ is None:
            raise ValueError("Must fit ranker first")
        return self.feature_ranking_['feature'].tolist()

class SFFSOptimizer:
    """
    Sequential Forward Feature Selection with multiple algorithms
    """
    
    def __init__(self, max_features: int = 30, cv_folds: int = 4, scoring: str = 'accuracy'):
        self.max_features = max_features
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.results_ = {}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, ranked_features: List[str]) -> Dict:
        """
        Perform SFFS with multiple ML algorithms
        """
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from xgboost import XGBClassifier
        
        # Define algorithms to test
        algorithms = {
            'SVM': SVC(kernel='rbf', random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        results = {}
        
        for alg_name, algorithm in algorithms.items():
            print(f"\nTesting {alg_name}...")
            alg_results = []
            
            # Test different numbers of features
            for n_features in range(1, min(len(ranked_features), self.max_features) + 1):
                # Select top n features
                selected_features = ranked_features[:n_features]
                X_selected = X[selected_features]
                
                # Cross-validation
                cv_scores = []
                for train_idx, val_idx in cv.split(X_selected, y):
                    X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train and evaluate
                    algorithm.fit(X_train_scaled, y_train)
                    y_pred = algorithm.predict(X_val_scaled)
                    cv_scores.append(accuracy_score(y_val, y_pred))
                
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                
                alg_results.append({
                    'n_features': n_features,
                    'features': selected_features.copy(),
                    'cv_mean': mean_score,
                    'cv_std': std_score
                })
                
                print(f"  {n_features} features: {mean_score:.4f} (+/- {std_score:.4f})")
            
            results[alg_name] = alg_results
        
        self.results_ = results
        return results
    
    def get_best_configuration(self) -> Tuple[str, int, List[str], float]:
        """
        Get the best algorithm and feature combination
        """
        if not self.results_:
            raise ValueError("Must fit optimizer first")
        
        best_score = 0
        best_config = None
        
        for alg_name, alg_results in self.results_.items():
            for result in alg_results:
                if result['cv_mean'] > best_score:
                    best_score = result['cv_mean']
                    best_config = (alg_name, result['n_features'], result['features'], best_score)
        
        return best_config

class FeatureSelectionPipeline:
    """
    Complete feature selection pipeline combining MMPC, Ridge ranking, and SFFS
    Based on the methodology from the ALS research paper
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.mmpc_selector = MMPCSelector(
            max_k=self.config['mmpc']['max_k'],
            significance_threshold=self.config['mmpc']['significance_threshold']
        )
        
        self.ridge_ranker = RidgeRanker(
            alpha_range=self.config['ridge']['alpha_range'],
            cv_folds=self.config['ridge']['cv_folds']
        )
        
        self.sffs_optimizer = SFFSOptimizer(
            max_features=self.config['sffs']['max_features'],
            cv_folds=self.config['sffs']['cv_folds'],
            scoring=self.config['sffs']['scoring']
        )
        
        # Store results
        self.mmpc_features_ = None
        self.ranked_features_ = None
        self.best_config_ = None
        self.pipeline_results_ = {}
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'mmpc': {
                'max_k': 3,
                'significance_threshold': 0.1
            },
            'ridge': {
                'alpha_range': [0.01, 0.1, 1.0, 10.0],
                'cv_folds': 5
            },
            'sffs': {
                'max_features': 25,
                'cv_folds': 4,
                'scoring': 'accuracy'
            }
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelectionPipeline':
        """
        Run the complete feature selection pipeline
        
        Args:
            X: Gene expression data (samples x genes)
            y: Labels (ALS=1, Control=0)
        """
        self.logger.info("Starting feature selection pipeline...")
        self.logger.info(f"Input data shape: {X.shape}")
        
        # Step 1: MMPC feature selection
        self.logger.info("Step 1: MMPC feature selection...")
        X_mmpc = self.mmpc_selector.fit_transform(X, y)
        self.mmpc_features_ = self.mmpc_selector.selected_features_
        
        self.logger.info(f"MMPC selected {len(self.mmpc_features_)} features")
        self.pipeline_results_['mmpc_features'] = self.mmpc_features_
        
        # Step 2: Ridge ranking
        self.logger.info("Step 2: Ridge coefficient ranking...")
        self.ridge_ranker.fit(X_mmpc, y)
        self.ranked_features_ = self.ridge_ranker.get_ranked_features()
        
        self.logger.info("Ridge ranking completed")
        self.pipeline_results_['ranked_features'] = self.ranked_features_
        self.pipeline_results_['ridge_importance'] = self.ridge_ranker.feature_ranking_
        
        # Step 3: SFFS optimization
        self.logger.info("Step 3: SFFS optimization with multiple algorithms...")
        sffs_results = self.sffs_optimizer.fit(X_mmpc, y, self.ranked_features_)
        
        # Get best configuration
        self.best_config_ = self.sffs_optimizer.get_best_configuration()
        best_alg, best_n_features, best_features, best_score = self.best_config_
        
        self.logger.info(f"Best configuration: {best_alg} with {best_n_features} features")
        self.logger.info(f"Best cross-validation score: {best_score:.4f}")
        
        self.pipeline_results_['sffs_results'] = sffs_results
        self.pipeline_results_['best_config'] = {
            'algorithm': best_alg,
            'n_features': best_n_features,
            'features': best_features,
            'cv_score': best_score
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
        """Save pipeline results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save selected features
        with open(output_path / 'selected_features.txt', 'w') as f:
            for feature in self.get_selected_features():
                f.write(f"{feature}\n")
        
        # Save full results
        with open(output_path / 'pipeline_results.pkl', 'wb') as f:
            pickle.dump(self.pipeline_results_, f)
        
        # Save Ridge ranking
        if self.ridge_ranker.feature_ranking_ is not None:
            self.ridge_ranker.feature_ranking_.to_csv(
                output_path / 'ridge_feature_ranking.csv', index=False
            )
        
        # Save SFFS results summary
        sffs_summary = []
        for alg_name, alg_results in self.pipeline_results_['sffs_results'].items():
            for result in alg_results:
                sffs_summary.append({
                    'algorithm': alg_name,
                    'n_features': result['n_features'],
                    'cv_mean': result['cv_mean'],
                    'cv_std': result['cv_std']
                })
        
        pd.DataFrame(sffs_summary).to_csv(
            output_path / 'sffs_results.csv', index=False
        )
        
        self.logger.info(f"Results saved to {output_path}")

# Main execution script
def main():
    """
    Main function to run the feature selection pipeline
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load processed data
    data_dir = Path("data/processed")
    expression_file = data_dir / "combined_expression_data.csv"
    metadata_file = data_dir / "sample_metadata.csv"
    
    if not expression_file.exists() or not metadata_file.exists():
        print("Error: Processed data files not found!")
        print("Please run the data download script first.")
        return
    
    # Load data
    print("Loading processed data...")
    expression_data = pd.read_csv(expression_file, index_col=0)
    metadata = pd.read_csv(metadata_file)
    
    # Prepare data
    # Transpose so samples are rows and genes are columns
    X = expression_data.T
    
    # Create labels (ALS=1, Control=0)
    y = metadata['group'].map({'ALS': 1, 'Control': 0})
    
    # Remove any samples with missing labels
    valid_samples = ~y.isna()
    X = X[valid_samples]
    y = y[valid_samples]
    
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} genes")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Normalize data
    print("Normalizing data...")
    scaler = StandardScaler()
    X_normalized = pd.DataFrame(
        scaler.fit_transform(X),
        index=X.index,
        columns=X.columns
    )
    
    # Run feature selection pipeline
    print("\nStarting feature selection pipeline...")
    
    pipeline = FeatureSelectionPipeline()
    pipeline.fit(X_normalized, y)
    
    # Display results
    best_features = pipeline.get_selected_features()
    best_algorithm = pipeline.get_best_algorithm()
    best_score = pipeline.best_config_[3]
    
    print(f"\n=== RESULTS ===")
    print(f"Best algorithm: {best_algorithm}")
    print(f"Number of selected features: {len(best_features)}")
    print(f"Best cross-validation score: {best_score:.4f}")
    print(f"\nSelected genes:")
    for i, gene in enumerate(best_features, 1):
        print(f"  {i:2d}. {gene}")
    
    # Save results
    results_dir = "data/results/feature_selection"
    pipeline.save_results(results_dir)
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    main()