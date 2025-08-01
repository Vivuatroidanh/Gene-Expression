#!/usr/bin/env python3
"""
SHAP Interpretability Module for ALS Diagnosis
This module provides explainable AI capabilities using SHAP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime

# Set matplotlib backend for non-interactive environments
import matplotlib
matplotlib.use('Agg')

class SHAPAnalyzer:
    """
    SHAP-based interpretability analysis for ALS diagnosis models
    Provides comprehensive model explanation and gene importance analysis
    """
    
    def __init__(self, model_type: str = 'SVM'):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.explainer = None
        self.shap_values = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.training_results = {}
        self.logger = logging.getLogger(__name__)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
    def _get_model(self) -> Any:
        """Get model instance based on type with optimized parameters"""
        models = {
            'SVM': SVC(
                kernel='rbf', 
                probability=True, 
                random_state=42,
                C=1.0,
                gamma='scale'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=1.0
            ),
            'XGBoost': XGBClassifier(
                random_state=42, 
                eval_metric='logloss',
                verbosity=0,
                n_estimators=100,
                max_depth=6
            ),
            'AdaBoost': AdaBoostClassifier(
                random_state=42,
                n_estimators=100
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5
            ),
            'DecisionTree': DecisionTreeClassifier(
                random_state=42,
                max_depth=10
            )
        }
        return models.get(self.model_type)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.1) -> Dict:
        """
        Train model and prepare for SHAP analysis
        
        Args:
            X: Feature data (selected genes)
            y: Target labels (ALS=1, Control=0)
            test_size: Proportion of data for testing
            
        Returns:
            Training results dictionary
        """
        self.logger.info(f"Training {self.model_type} model for SHAP analysis...")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data stratified by target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store scaled data as DataFrames
        self.X_train = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        self.X_test = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        self.y_train = y_train.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        
        # Train model
        self.model = self._get_model()
        if self.model is None:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Get prediction probabilities if available
        train_proba = None
        test_proba = None
        if hasattr(self.model, 'predict_proba'):
            train_proba = self.model.predict_proba(X_train_scaled)[:, 1]
            test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        self.training_results = {
            'model_type': self.model_type,
            'n_features': len(self.feature_names),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'train_auc': roc_auc_score(y_train, train_proba) if train_proba is not None else None,
            'test_auc': roc_auc_score(y_test, test_proba) if test_proba is not None else None,
            'feature_names': self.feature_names
        }
        
        # For reporting, use test metrics as primary
        self.training_results['accuracy'] = self.training_results['test_accuracy']
        self.training_results['auc'] = self.training_results['test_auc']
        
        self.logger.info(f"Model training completed:")
        self.logger.info(f"  Train Accuracy: {self.training_results['train_accuracy']:.4f}")
        self.logger.info(f"  Test Accuracy: {self.training_results['test_accuracy']:.4f}")
        if train_proba is not None:
            self.logger.info(f"  Train AUC: {self.training_results['train_auc']:.4f}")
            self.logger.info(f"  Test AUC: {self.training_results['test_auc']:.4f}")
        
        return self.training_results
    
    def create_shap_explainer(self, background_samples: int = 100):
        """
        Create SHAP explainer for the trained model
        
        Args:
            background_samples: Number of background samples for SHAP
        """
        if self.model is None:
            raise ValueError("Must train model first")
        
        self.logger.info(f"Creating SHAP explainer for {self.model_type}...")
        
        # Use training data for background
        if len(self.X_train) > background_samples:
            background_data = shap.sample(self.X_train, background_samples)
        else:
            background_data = self.X_train
        
        # Create appropriate explainer based on model type
        if self.model_type in ['XGBoost', 'RandomForest', 'DecisionTree', 'AdaBoost']:
            try:
                # Try TreeExplainer first
                self.explainer = shap.TreeExplainer(self.model)
                self.logger.info("Using TreeExplainer")
            except:
                # Fallback to KernelExplainer
                self.logger.info("TreeExplainer failed, using KernelExplainer")
                def model_predict(x):
                    if hasattr(self.model, 'predict_proba'):
                        return self.model.predict_proba(x)[:, 1]
                    else:
                        return self.model.predict(x)
                self.explainer = shap.KernelExplainer(model_predict, background_data)
        else:
            # Use KernelExplainer for other models
            def model_predict(x):
                if hasattr(self.model, 'predict_proba'):
                    return self.model.predict_proba(x)[:, 1]
                else:
                    return self.model.predict(x)
            self.explainer = shap.KernelExplainer(model_predict, background_data)
            self.logger.info("Using KernelExplainer")
        
        self.logger.info("SHAP explainer created successfully")
    
    def calculate_shap_values(self, max_samples: int = None):
        """
        Calculate SHAP values for test samples
        
        Args:
            max_samples: Maximum number of samples to analyze (None for all)
        """
        if self.explainer is None:
            raise ValueError("Must create SHAP explainer first")
        
        # Determine samples to analyze
        if max_samples is not None and len(self.X_test) > max_samples:
            sample_indices = np.random.choice(len(self.X_test), max_samples, replace=False)
            X_sample = self.X_test.iloc[sample_indices]
        else:
            X_sample = self.X_test
        
        self.logger.info(f"Calculating SHAP values for {len(X_sample)} samples...")
        
        try:
            # Calculate SHAP values
            if self.model_type in ['XGBoost', 'RandomForest', 'DecisionTree', 'AdaBoost']:
                try:
                    self.shap_values = self.explainer.shap_values(X_sample)
                    # For binary classification, take values for positive class
                    if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
                        self.shap_values = self.shap_values[1]
                except:
                    # Fallback to slower but more robust calculation
                    self.logger.warning("TreeExplainer failed, recalculating with KernelExplainer")
                    self._fallback_to_kernel_explainer(X_sample)
            else:
                self.shap_values = self.explainer.shap_values(X_sample)
            
            # Store the sample data used for SHAP values
            self.X_shap_sample = X_sample
            
            self.logger.info("SHAP values calculated successfully")
            self.logger.info(f"SHAP values shape: {self.shap_values.shape}")
            
        except Exception as e:
            self.logger.error(f"Failed to calculate SHAP values: {str(e)}")
            raise
    
    def _fallback_to_kernel_explainer(self, X_sample):
        """Fallback to KernelExplainer if TreeExplainer fails"""
        background_data = shap.sample(self.X_train, 50)  # Smaller sample for speed
        
        def model_predict(x):
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(x)[:, 1]
            else:
                return self.model.predict(x)
        
        self.explainer = shap.KernelExplainer(model_predict, background_data)
        self.shap_values = self.explainer.shap_values(X_sample.iloc[:min(50, len(X_sample))])
    
    def plot_summary(self, output_dir: str = None, show_plot: bool = False):
        """Create SHAP summary plot"""
        if self.shap_values is None:
            raise ValueError("Must calculate SHAP values first")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values, 
            self.X_shap_sample, 
            feature_names=self.feature_names,
            show=False,
            max_display=min(20, len(self.feature_names))
        )
        plt.title(f'SHAP Summary Plot - {self.model_type} Model', fontsize=14, pad=20)
        plt.tight_layout()
        
        if output_dir:
            filename = f"{output_dir}/shap_summary_{self.model_type.lower()}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"SHAP summary plot saved to {filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_feature_importance(self, output_dir: str = None, show_plot: bool = False):
        """Create feature importance plot based on mean SHAP values"""
        if self.shap_values is None:
            raise ValueError("Must calculate SHAP values first")
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'gene': self.feature_names,
            'mean_abs_shap': mean_shap,
            'mean_shap': self.shap_values.mean(axis=0),
            'std_shap': self.shap_values.std(axis=0)
        }).sort_values('mean_abs_shap', ascending=True)
        
        # Plot
        plt.figure(figsize=(12, max(8, len(self.feature_names) * 0.4)))
        bars = plt.barh(range(len(importance_df)), importance_df['mean_abs_shap'])
        plt.yticks(range(len(importance_df)), importance_df['gene'])
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.title(f'Gene Importance - {self.model_type} Model', fontsize=14)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, importance_df['mean_abs_shap'])):
            plt.text(value + max(importance_df['mean_abs_shap']) * 0.01, i, 
                    f'{value:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if output_dir:
            filename = f"{output_dir}/gene_importance_{self.model_type.lower()}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            self.logger.info(f"Gene importance plot saved to {filename}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return importance_df.sort_values('mean_abs_shap', ascending=False)
    
    def plot_waterfall(self, sample_idx: int = 0, output_dir: str = None, show_plot: bool = False):
        """Create waterfall plot for individual sample"""
        if self.shap_values is None:
            raise ValueError("Must calculate SHAP values first")
        
        if sample_idx >= len(self.shap_values):
            sample_idx = 0
            self.logger.warning(f"Sample index out of range, using index 0")
        
        # Get the actual prediction for this sample
        sample_data = self.X_shap_sample.iloc[sample_idx:sample_idx+1]
        prediction = self.model.predict(sample_data)[0]
        if hasattr(self.model, 'predict_proba'):
            prediction_proba = self.model.predict_proba(sample_data)[0, 1]
        
        try:
            # Create explanation object
            if hasattr(self.explainer, 'expected_value'):
                expected_value = self.explainer.expected_value
                if isinstance(expected_value, np.ndarray):
                    expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
            else:
                expected_value = 0.5  # Default baseline for binary classification
            
            explanation = shap.Explanation(
                values=self.shap_values[sample_idx],
                base_values=expected_value,
                data=self.X_shap_sample.iloc[sample_idx].values,
                feature_names=self.feature_names
            )
            
            plt.figure(figsize=(12, 8))
            shap.waterfall_plot(explanation, show=False, max_display=15)
            
            # Add prediction information to title
            title = f'SHAP Waterfall Plot - Sample {sample_idx} ({self.model_type})\n'
            title += f'Prediction: {"ALS" if prediction == 1 else "Control"}'
            if hasattr(self.model, 'predict_proba'):
                title += f' (Probability: {prediction_proba:.3f})'
            
            plt.title(title, fontsize=12)
            
            if output_dir:
                filename = f"{output_dir}/waterfall_sample_{sample_idx}_{self.model_type.lower()}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                self.logger.info(f"Waterfall plot saved to {filename}")
            
            if show_plot:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            self.logger.warning(f"Could not create waterfall plot: {str(e)}")
            self._create_simple_waterfall(sample_idx, output_dir, show_plot)
    
    def _create_simple_waterfall(self, sample_idx: int, output_dir: str = None, show_plot: bool = False):
        """Create a simple waterfall-style plot when SHAP waterfall fails"""
        shap_values_sample = self.shap_values[sample_idx]
        feature_names = self.feature_names
        
        # Sort by absolute SHAP value
        sorted_indices = np.argsort(np.abs(shap_values_sample))[::-1][:15]  # Top 15
        
        plt.figure(figsize=(12, 8))
        colors = ['red' if val < 0 else 'blue' for val in shap_values_sample[sorted_indices]]
        
        plt.barh(range(len(sorted_indices)), shap_values_sample[sorted_indices], color=colors)
        plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
        plt.xlabel('SHAP Value')
        plt.title(f'Feature Contributions - Sample {sample_idx} ({self.model_type})')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        if output_dir:
            filename = f"{output_dir}/simple_waterfall_sample_{sample_idx}_{self.model_type.lower()}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def analyze_gene_interactions(self, output_dir: str = None) -> pd.DataFrame:
        """Analyze interactions between genes using SHAP values"""
        if self.shap_values is None:
            raise ValueError("Must calculate SHAP values first")
        
        # Calculate correlation between SHAP values of different genes
        shap_df = pd.DataFrame(self.shap_values, columns=self.feature_names)
        correlation_matrix = shap_df.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        
        sns.heatmap(correlation_matrix, 
                   mask=mask, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0, 
                   square=True, 
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        plt.title('Gene Interaction Analysis\n(SHAP Value Correlations)', fontsize=14)
        plt.tight_layout()
        
        if output_dir:
            filename = f"{output_dir}/gene_interactions_{self.model_type.lower()}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            correlation_matrix.to_csv(f"{output_dir}/gene_correlations_{self.model_type.lower()}.csv")
            self.logger.info(f"Gene interaction analysis saved to {filename}")
        
        plt.show() if output_dir is None else plt.close()
        
        return correlation_matrix
    
    def generate_interpretation_report(self, output_dir: str) -> Dict:
        """Generate comprehensive interpretation report"""
        if self.shap_values is None:
            raise ValueError("Must calculate SHAP values first")
        
        # Calculate feature importance statistics
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        mean_shap = self.shap_values.mean(axis=0)
        std_shap = self.shap_values.std(axis=0)
        
        # Create detailed importance DataFrame
        importance_df = pd.DataFrame({
            'gene': self.feature_names,
            'mean_abs_shap': mean_abs_shap,
            'mean_shap': mean_shap,
            'std_shap': std_shap,
            'importance_rank': range(1, len(self.feature_names) + 1)
        }).sort_values('mean_abs_shap', ascending=False)
        
        importance_df['importance_rank'] = range(1, len(importance_df) + 1)
        
        # Calculate positive and negative contributions
        pos_contributions = (self.shap_values > 0).sum(axis=0)
        neg_contributions = (self.shap_values < 0).sum(axis=0)
        total_samples = len(self.shap_values)
        
        # Add contribution statistics
        importance_df['positive_contributions'] = pos_contributions
        importance_df['negative_contributions'] = neg_contributions
        importance_df['positive_ratio'] = pos_contributions / total_samples
        importance_df['negative_ratio'] = neg_contributions / total_samples
        
        # Identify most consistent genes (low variance relative to mean)
        importance_df['consistency'] = importance_df['mean_abs_shap'] / (importance_df['std_shap'] + 1e-8)
        
        # Generate summary statistics
        top_genes = importance_df.head(10)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'n_features': len(self.feature_names),
            'n_samples_analyzed': len(self.shap_values),
            'training_results': self.training_results,
            'feature_importance_stats': {
                'mean_importance': float(mean_abs_shap.mean()),
                'std_importance': float(mean_abs_shap.std()),
                'max_importance': float(mean_abs_shap.max()),
                'min_importance': float(mean_abs_shap.min())
            },
            'top_genes': top_genes.to_dict('records'),
            'most_important_gene': {
                'name': importance_df.iloc[0]['gene'],
                'importance': float(importance_df.iloc[0]['mean_abs_shap']),
                'consistency': float(importance_df.iloc[0]['consistency'])
            },
            'most_consistent_genes': importance_df.nlargest(5, 'consistency')[['gene', 'mean_abs_shap', 'consistency']].to_dict('records')
        }
        
        # Save detailed results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save detailed gene importance
        importance_df.to_csv(f"{output_dir}/detailed_gene_importance_{self.model_type.lower()}.csv", index=False)
        
        # Save SHAP values
        shap_values_df = pd.DataFrame(self.shap_values, columns=self.feature_names)
        shap_values_df.to_csv(f"{output_dir}/shap_values_{self.model_type.lower()}.csv", index=False)
        
        # Save interpretation report
        with open(f"{output_dir}/interpretation_report_{self.model_type.lower()}.pkl", 'wb') as f:
            pickle.dump(report, f)
        
        # Save human-readable summary
        summary_text = f"""
ALS Diagnosis Model Interpretation Report
========================================
Generated: {report['timestamp']}
Model: {self.model_type}

Model Performance:
- Training Accuracy: {self.training_results.get('train_accuracy', 'N/A'):.4f}
- Test Accuracy: {self.training_results.get('test_accuracy', 'N/A'):.4f}
- Test AUC: {self.training_results.get('test_auc', 'N/A'):.4f}

Feature Analysis:
- Total Features: {len(self.feature_names)}
- Samples Analyzed: {len(self.shap_values)}

Top 10 Most Important Genes:
"""
        
        for i, (_, row) in enumerate(top_genes.iterrows(), 1):
            summary_text += f"{i:2d}. {row['gene']:15s} - Importance: {row['mean_abs_shap']:.4f} (Consistency: {row['consistency']:.2f})\n"
        
        summary_text += f"""
Most Consistent Genes (stable predictions):
"""
        for i, gene_info in enumerate(report['most_consistent_genes'], 1):
            summary_text += f"{i}. {gene_info['gene']:15s} - Consistency: {gene_info['consistency']:.2f}\n"
        
        with open(f"{output_dir}/interpretation_summary_{self.model_type.lower()}.txt", 'w') as f:
            f.write(summary_text)
        
        self.logger.info(f"Interpretation report saved to {output_dir}")
        
        return report

def run_shap_analysis():
    """
    Main function to run SHAP analysis on the best model from feature selection
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Load processed data and feature selection results
        logger.info("Loading data and feature selection results...")
        
        # Load expression data and metadata
        expression_data = pd.read_csv("data/processed/combined_expression_data.csv", index_col=0)
        metadata = pd.read_csv("data/processed/sample_metadata.csv")
        
        # Load feature selection results
        with open("data/results/feature_selection/pipeline_results.pkl", 'rb') as f:
            pipeline_results = pickle.load(f)
        
        best_config = pipeline_results['best_config']
        selected_features = best_config['features']
        best_algorithm = best_config['algorithm']
        
        logger.info(f"Best algorithm: {best_algorithm}")
        logger.info(f"Selected features: {len(selected_features)}")
        
    except FileNotFoundError as e:
        logger.error(f"Required files not found: {e}")
        logger.error("Please run feature selection first:")
        logger.error("python scripts/main.py --step feature_selection")
        return None, None, None
    
    # Prepare data
    X = expression_data.T[selected_features]  # Transpose and select features
    y = metadata['group'].map({'ALS': 1, 'Control': 0})
    
    # Remove samples with missing labels
    valid_samples = ~y.isna()
    X = X[valid_samples]
    y = y[valid_samples]
    
    logger.info(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Initialize SHAP analyzer
    analyzer = SHAPAnalyzer(model_type=best_algorithm)
    
    # Train model
    logger.info("Training model...")
    training_results = analyzer.fit(X, y, test_size=0.1)
    
    # Create SHAP explainer
    logger.info("Creating SHAP explainer...")
    analyzer.create_shap_explainer(background_samples=50)
    
    # Calculate SHAP values
    logger.info("Calculating SHAP values...")
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
    logger.info("Analyzing gene interactions...")
    correlation_matrix = analyzer.analyze_gene_interactions(output_dir)
    
    # Generate interpretation report
    logger.info("Generating interpretation report...")
    report = analyzer.generate_interpretation_report(output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("SHAP ANALYSIS COMPLETED")
    print("="*60)
    print(f"Model: {best_algorithm}")
    print(f"Test Accuracy: {training_results.get('test_accuracy', 'N/A'):.4f}")
    print(f"Test AUC: {training_results.get('test_auc', 'N/A'):.4f}")
    print(f"Number of features: {len(selected_features)}")
    print(f"\nTop 5 most important genes:")
    for i, (_, row) in enumerate(importance_df.head().iterrows(), 1):
        print(f"  {i}. {row['gene']}: {row['mean_abs_shap']:.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    print("Generated files:")
    print("  - SHAP summary and importance plots")
    print("  - Individual sample explanations (waterfall plots)")
    print("  - Gene interaction analysis")
    print("  - Comprehensive interpretation report")
    print("  - Detailed CSV files with all results")
    
    return analyzer, training_results, report

if __name__ == "__main__":
    run_shap_analysis()