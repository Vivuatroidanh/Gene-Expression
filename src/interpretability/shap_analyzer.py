import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Any
import logging

class SHAPAnalyzer:
    """
    SHAP-based interpretability analysis for ALS diagnosis models
    """
    
    def __init__(self, model_type: str = 'SVM'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.explainer = None
        self.shap_values = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.logger = logging.getLogger(__name__)
        
    def _get_model(self) -> Any:
        """Get model instance based on type"""
        models = {
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
        }
        return models.get(self.model_type)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.1) -> Dict:
        """
        Train model and prepare SHAP explainer
        
        Args:
            X: Feature data (selected genes)
            y: Target labels
            test_size: Proportion of data for testing
            
        Returns:
            Training results dictionary
        """
        self.logger.info(f"Training {self.model_type} model...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = self._get_model()
        self.model.fit(X_train_scaled, y_train)
        
        # Store test data for SHAP analysis
        self.X_test = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        self.y_test = y_test
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
            'n_features': len(self.feature_names),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
        
        self.logger.info(f"Model performance: Accuracy={results['accuracy']:.4f}, AUC={results['auc']:.4f}")
        
        return results
    
    def create_shap_explainer(self, background_samples: int = 100):
        """
        Create SHAP explainer for the trained model
        
        Args:
            background_samples: Number of background samples for SHAP
        """
        if self.model is None:
            raise ValueError("Must train model first")
        
        self.logger.info(f"Creating SHAP explainer for {self.model_type}...")
        
        # Select background data
        background_data = shap.sample(self.X_test, background_samples)
        
        # Create appropriate explainer based on model type
        if self.model_type == 'XGBoost':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'RandomForest':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_type in ['SVM', 'LogisticRegression']:
            # Use Kernel explainer for non-tree models
            def model_predict(x):
                return self.model.predict_proba(x)[:, 1]
            self.explainer = shap.KernelExplainer(model_predict, background_data)
        
        self.logger.info("SHAP explainer created successfully")
    
    def calculate_shap_values(self, max_samples: int = None):
        """
        Calculate SHAP values for test samples
        
        Args:
            max_samples: Maximum number of samples to analyze (None for all)
        """
        if self.explainer is None:
            raise ValueError("Must create SHAP explainer first")
        
        # Limit samples if specified
        if max_samples is not None and len(self.X_test) > max_samples:
            sample_indices = np.random.choice(len(self.X_test), max_samples, replace=False)
            X_sample = self.X_test.iloc[sample_indices]
        else:
            X_sample = self.X_test
        
        self.logger.info(f"Calculating SHAP values for {len(X_sample)} samples...")
        
        # Calculate SHAP values
        if self.model_type in ['XGBoost', 'RandomForest']:
            self.shap_values = self.explainer.shap_values(X_sample)
            # For binary classification, take values for positive class
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]
        else:
            self.shap_values = self.explainer.shap_values(X_sample)
        
        self.logger.info("SHAP values calculated successfully")
    
    def plot_summary(self, output_dir: str = None, show_plot: bool = True):
        """
        Create SHAP summary plot
        """
        if self.shap_values is None:
            raise ValueError("Must calculate SHAP values first")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            self.X_test.iloc[:len(self.shap_values)], 
            feature_names=self.feature_names,
            show=False
        )
        plt.title(f'SHAP Summary Plot - {self.model_type} Model')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(f"{output_dir}/shap_summary_{self.model_type.lower()}.png", 
                       dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_feature_importance(self, output_dir: str = None, show_plot: bool = True):
        """
        Create feature importance plot based on mean SHAP values
        """
        if self.shap_values is None:
            raise ValueError("Must calculate SHAP values first")
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'gene': self.feature_names,
            'importance': mean_shap
        }).sort_values('importance', ascending=True)
        
        # Plot
        plt.figure(figsize=(10, max(6, len(self.feature_names) * 0.3)))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['gene'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title(f'Gene Importance - {self.model_type} Model')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(f"{output_dir}/gene_importance_{self.model_type.lower()}.png", 
                       dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return importance_df
    
    def plot_waterfall(self, sample_idx: int = 0, output_dir: str = None, show_plot: bool = True):
        """
        Create waterfall plot for individual sample
        """
        if self.shap_values is None:
            raise ValueError("Must calculate SHAP values first")
        
        if sample_idx >= len(self.shap_values):
            raise ValueError(f"Sample index {sample_idx} out of range")
        
        # Create explanation object
        if self.model_type in ['XGBoost', 'RandomForest']:
            explanation = shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.explainer.expected_value,
                data=self.X_test.iloc[sample_idx].values,
                feature_names=self.feature_names
            )
        else:
            explanation = shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=self.explainer.expected_value,
                data=self.X_test.iloc[sample_idx].values,
                feature_names=self.feature_names
            )
        
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)
        plt.title(f'SHAP Waterfall Plot - Sample {sample_idx} ({self.model_type})')
        
        if output_dir:
            plt.savefig(f"{output_dir}/waterfall_sample_{sample_idx}_{self.model_type.lower()}.png", 
                       dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def analyze_gene_interactions(self, output_dir: str = None) -> pd.DataFrame:
        """
        Analyze interactions between genes using SHAP values
        """
        if self.shap_values is None:
            raise ValueError("Must calculate SHAP values first")
        
        # Calculate correlation between SHAP values of different genes
        shap_df = pd.DataFrame(self.shap_values, columns=self.feature_names)
        correlation_matrix = shap_df.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.2f')
        plt.title('Gene Interaction Analysis (SHAP Value Correlations)')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(f"{output_dir}/gene_interactions_{self.model_type.lower()}.png", 
                       dpi=300, bbox_inches='tight')
            correlation_matrix.to_csv(f"{output_dir}/gene_correlations_{self.model_type.lower()}.csv")
        
        plt.show()
        
        return correlation_matrix
    
    def generate_interpretation_report(self, output_dir: str) -> Dict:
        """
        Generate comprehensive interpretation report
        """
        if self.shap_values is None:
            raise ValueError("Must calculate SHAP values first")
        
        # Calculate feature importance
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            'gene': self.feature_names,
            'mean_abs_shap': mean_shap,
            'mean_shap': self.shap_values.mean(axis=0),
            'std_shap': self.shap_values.std(axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        # Identify top contributing genes
        top_genes = importance_df.head(10)
        
        # Calculate positive and negative contributions
        pos_contributions = (self.shap_values > 0).sum(axis=0)
        neg_contributions = (self.shap_values < 0).sum(axis=0)
        
        contribution_df = pd.DataFrame({
            'gene': self.feature_names,
            'positive_contributions': pos_contributions,
            'negative_contributions': neg_contributions,
            'contribution_ratio': pos_contributions / (pos_contributions + neg_contributions)
        })
        
        # Generate report
        report = {
            'model_type': self.model_type,
            'n_features': len(self.feature_names),
            'n_samples_analyzed': len(self.shap_values),
            'top_genes': top_genes.to_dict('records'),
            'mean_feature_importance': float(mean_shap.mean()),
            'feature_importance_std': float(mean_shap.std()),
            'most_important_gene': {
                'name': importance_df.iloc[0]['gene'],
                'importance': float(importance_df.iloc[0]['mean_abs_shap'])
            }
        }
        
        # Save detailed results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(f"{output_dir}/detailed_gene_importance_{self.model_type.lower()}.csv", index=False)
        contribution_df.to_csv(f"{output_dir}/gene_contributions_{self.model_type.lower()}.csv", index=False)
        
        with open(f"{output_dir}/interpretation_report_{self.model_type.lower()}.pkl", 'wb') as f:
            pickle.dump(report, f)
        
        return report

def run_shap_analysis():
    """
    Main function to run SHAP analysis on the best model from feature selection
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Load processed data and feature selection results
    try:
        # Load expression data and metadata
        expression_data = pd.read_csv("data/processed/combined_expression_data.csv", index_col=0)
        metadata = pd.read_csv("data/processed/sample_metadata.csv")
        
        # Load feature selection results
        with open("data/results/feature_selection/pipeline_results.pkl", 'rb') as f:
            pipeline_results = pickle.load(f)
        
        best_config = pipeline_results['best_config']
        selected_features = best_config['features']
        best_algorithm = best_config['algorithm']
        
        logger.info(f"Loaded data and feature selection results")
        logger.info(f"Best algorithm: {best_algorithm}")
        logger.info(f"Selected features: {len(selected_features)}")
        
    except FileNotFoundError as e:
        logger.error(f"Required files not found: {e}")
        logger.error("Please run data processing and feature selection first")
        return
    
    # Prepare data
    X = expression_data.T[selected_features]  # Transpose and select features
    y = metadata['group'].map({'ALS': 1, 'Control': 0})
    
    # Remove samples with missing labels
    valid_samples = ~y.isna()
    X = X[valid_samples]
    y = y[valid_samples]
    
    logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize SHAP analyzer
    analyzer = SHAPAnalyzer(model_type=best_algorithm)
    
    # Train model and calculate SHAP values
    training_results = analyzer.fit(X, y)
    analyzer.create_shap_explainer(background_samples=50)
    analyzer.calculate_shap_values(max_samples=100)  # Limit for faster computation
    
    # Create output directory
    output_dir = "data/results/shap_analysis"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    logger.info("Generating SHAP visualizations...")
    analyzer.plot_summary(output_dir, show_plot=False)
    importance_df = analyzer.plot_feature_importance(output_dir, show_plot=False)
    analyzer.plot_waterfall(sample_idx=0, output_dir=output_dir, show_plot=False)
    analyzer.plot_waterfall(sample_idx=1, output_dir=output_dir, show_plot=False)
    
    # Analyze gene interactions
    correlation_matrix = analyzer.analyze_gene_interactions(output_dir)
    
    # Generate interpretation report
    report = analyzer.generate_interpretation_report(output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("SHAP ANALYSIS RESULTS")
    print("="*50)
    print(f"Model: {best_algorithm}")
    print(f"Training Accuracy: {training_results['accuracy']:.4f}")
    print(f"Training AUC: {training_results['auc']:.4f}")
    print(f"Number of features: {len(selected_features)}")
    print(f"\nTop 5 most important genes:")
    for i, (_, row) in enumerate(importance_df.head().iterrows(), 1):
        print(f"  {i}. {row['gene']}: {row['importance']:.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    print("Files generated:")
    print("  - SHAP summary plots")
    print("  - Gene importance rankings")
    print("  - Individual sample explanations")
    print("  - Gene interaction analysis")
    print("  - Comprehensive interpretation report")

if __name__ == "__main__":
    run_shap_analysis()