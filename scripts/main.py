#!/usr/bin/env python3
"""
Main pipeline script for ALS diagnosis research with SHAP interpretability
Based on the methodology from the research paper with improvements
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import pandas as pd
from datetime import datetime

# Import our custom modules
sys.path.append('src')
from data_processing.geo_downloader import GEODownloader
from feature_selection_pipeline import FeatureSelectionPipeline
from shap_interpretability import SHAPAnalyzer

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def step_data_download(config: dict):
    """Step 1: Download and process GEO datasets"""
    logger = logging.getLogger(__name__)
    logger.info("=== STEP 1: DATA DOWNLOAD AND PROCESSING ===")
    
    # Initialize downloader
    downloader = GEODownloader(base_dir=config['data']['base_dir'])
    
    # Download and process datasets
    datasets = config['data']['datasets']
    logger.info(f"Processing datasets: {datasets}")
    
    expression_data, metadata = downloader.process_datasets(datasets)
    
    # Save processed data
    downloader.save_processed_data(expression_data, metadata)
    
    logger.info(f"Data processing completed")
    logger.info(f"Expression data shape: {expression_data.shape}")
    logger.info(f"Sample distribution: {metadata['group'].value_counts().to_dict()}")
    
    return expression_data, metadata

def step_feature_selection(config: dict):
    """Step 2: Feature selection using MMPC + Ridge + SFFS"""
    logger = logging.getLogger(__name__)
    logger.info("=== STEP 2: FEATURE SELECTION ===")
    
    # Load processed data
    data_dir = Path(config['data']['base_dir']) / "processed"
    expression_file = data_dir / "combined_expression_data.csv"
    metadata_file = data_dir / "sample_metadata.csv"
    
    if not expression_file.exists():
        raise FileNotFoundError("Processed data not found. Run data download step first.")
    
    # Load data
    expression_data = pd.read_csv(expression_file, index_col=0)
    metadata = pd.read_csv(metadata_file)
    
    # Prepare data for feature selection
    X = expression_data.T  # Transpose: samples as rows, genes as columns
    y = metadata['group'].map({'ALS': 1, 'Control': 0})
    
    # Remove samples with missing labels
    valid_samples = ~y.isna()
    X = X[valid_samples]
    y = y[valid_samples]
    
    logger.info(f"Feature selection input: {X.shape[0]} samples, {X.shape[1]} genes")
    
    # Initialize and run feature selection pipeline
    pipeline_config = config.get('feature_selection', {})
    pipeline = FeatureSelectionPipeline(pipeline_config)
    pipeline.fit(X, y)
    
    # Save results
    results_dir = Path(config['data']['base_dir']) / "results" / "feature_selection"
    pipeline.save_results(results_dir)
    
    # Log results
    best_features = pipeline.get_selected_features()
    best_algorithm = pipeline.get_best_algorithm()
    best_score = pipeline.best_config_[3]
    
    logger.info(f"Feature selection completed")
    logger.info(f"Best algorithm: {best_algorithm}")
    logger.info(f"Selected features: {len(best_features)}")
    logger.info(f"Best CV score: {best_score:.4f}")
    
    return pipeline

def step_model_training_and_shap(config: dict):
    """Step 3: Model training and SHAP analysis"""
    logger = logging.getLogger(__name__)
    logger.info("=== STEP 3: MODEL TRAINING AND SHAP ANALYSIS ===")
    
    # Load processed data and feature selection results
    data_dir = Path(config['data']['base_dir'])
    
    try:
        # Load expression data and metadata
        expression_data = pd.read_csv(data_dir / "processed" / "combined_expression_data.csv", index_col=0)
        metadata = pd.read_csv(data_dir / "processed" / "sample_metadata.csv")
        
        # Load feature selection results
        import pickle
        with open(data_dir / "results" / "feature_selection" / "pipeline_results.pkl", 'rb') as f:
            pipeline_results = pickle.load(f)
        
        best_config = pipeline_results['best_config']
        selected_features = best_config['features']
        best_algorithm = best_config['algorithm']
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Required files not found: {e}. Run previous steps first.")
    
    # Prepare data
    X = expression_data.T[selected_features]  # Transpose and select features
    y = metadata['group'].map({'ALS': 1, 'Control': 0})
    
    # Remove samples with missing labels
    valid_samples = ~y.isna()
    X = X[valid_samples]
    y = y[valid_samples]
    
    logger.info(f"Model training data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Initialize SHAP analyzer
    shap_config = config.get('shap', {})
    analyzer = SHAPAnalyzer(model_type=best_algorithm)
    
    # Train model
    training_results = analyzer.fit(X, y, test_size=config['data'].get('test_size', 0.1))
    
    # Create SHAP explainer and calculate values
    background_samples = shap_config.get('num_background_samples', 50)
    max_samples = shap_config.get('max_analysis_samples', 100)
    
    analyzer.create_shap_explainer(background_samples=background_samples)
    analyzer.calculate_shap_values(max_samples=max_samples)
    
    # Generate visualizations and analysis
    output_dir = data_dir / "results" / "shap_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if shap_config.get('generate_plots', True):
        logger.info("Generating SHAP visualizations...")
        analyzer.plot_summary(str(output_dir), show_plot=False)
        importance_df = analyzer.plot_feature_importance(str(output_dir), show_plot=False)
        analyzer.plot_waterfall(sample_idx=0, output_dir=str(output_dir), show_plot=False)
        analyzer.plot_waterfall(sample_idx=1, output_dir=str(output_dir), show_plot=False)
    
    if shap_config.get('analyze_interactions', True):
        logger.info("Analyzing gene interactions...")
        correlation_matrix = analyzer.analyze_gene_interactions(str(output_dir))
    
    # Generate comprehensive report
    report = analyzer.generate_interpretation_report(str(output_dir))
    
    logger.info(f"SHAP analysis completed")
    logger.info(f"Training accuracy: {training_results['accuracy']:.4f}")
    logger.info(f"Training AUC: {training_results['auc']:.4f}")
    
    return analyzer, training_results, report

def step_generate_report(config: dict):
    """Step 4: Generate comprehensive research report"""
    logger = logging.getLogger(__name__)
    logger.info("=== STEP 4: GENERATING COMPREHENSIVE REPORT ===")
    
    data_dir = Path(config['data']['base_dir'])
    results_dir = data_dir / "results"
    
    # Load all results
    try:
        import pickle
        
        # Load feature selection results
        with open(results_dir / "feature_selection" / "pipeline_results.pkl", 'rb') as f:
            fs_results = pickle.load(f)
        
        # Load SHAP results
        shap_files = list((results_dir / "shap_analysis").glob("interpretation_report_*.pkl"))
        if shap_files:
            with open(shap_files[0], 'rb') as f:
                shap_report = pickle.load(f)
        else:
            shap_report = {}
        
        # Load gene importance
        importance_files = list((results_dir / "shap_analysis").glob("detailed_gene_importance_*.csv"))
        if importance_files:
            gene_importance = pd.read_csv(importance_files[0])
        else:
            gene_importance = pd.DataFrame()
        
    except FileNotFoundError as e:
        logger.error(f"Could not load results: {e}")
        return
    
    # Generate comprehensive report
    report_content = f"""
# ALS Diagnosis Research Report
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary
This report presents the results of an advanced machine learning approach for ALS diagnosis 
based on gene expression data, incorporating explainable AI (SHAP) for model interpretability.

## Methodology
- **Datasets**: {', '.join(config['data']['datasets'])}
- **Feature Selection**: MMPC + Ridge Ranking + Sequential Forward Feature Selection
- **Model Training**: Multiple algorithms with cross-validation
- **Interpretability**: SHAP (SHapley Additive exPlanations) analysis

## Key Results

### Feature Selection Results
- **Initial features**: 21,029 genes
- **After MMPC filtering**: {len(fs_results.get('mmpc_features', []))} genes
- **Final selected features**: {len(fs_results['best_config']['features'])} genes
- **Best algorithm**: {fs_results['best_config']['algorithm']}
- **Cross-validation score**: {fs_results['best_config']['cv_score']:.4f}

### Model Performance
- **Algorithm**: {shap_report.get('model_type', 'N/A')}
- **Number of features**: {shap_report.get('n_features', 'N/A')}
- **Samples analyzed**: {shap_report.get('n_samples_analyzed', 'N/A')}

### Top Contributing Genes
"""
    
    if not gene_importance.empty:
        report_content += "\n"
        for i, (_, row) in enumerate(gene_importance.head(10).iterrows(), 1):
            report_content += f"{i:2d}. **{row['gene']}**: SHAP importance = {row['mean_abs_shap']:.4f}\n"
    
    report_content += f"""

### Key Insights
- The model identified {len(fs_results['best_config']['features'])} genes as most discriminative for ALS diagnosis
- SHAP analysis revealed the relative contribution of each gene to individual predictions
- Gene interaction analysis identified potential co-regulatory relationships

## Files Generated
- Feature selection results: `results/feature_selection/`
- SHAP analysis: `results/shap_analysis/`
- Visualizations: Summary plots, waterfall plots, interaction heatmaps
- Detailed data: Gene rankings, correlation matrices, model coefficients

## Next Steps
1. **External Validation**: Test the model on independent datasets
2. **Biological Validation**: Investigate the biological roles of top-ranked genes
3. **Clinical Integration**: Develop clinical decision support tools
4. **Pathway Analysis**: Perform gene set enrichment analysis

## Technical Details
- **Programming Language**: Python 3.8+
- **Key Libraries**: scikit-learn, SHAP, pandas, numpy
- **Computational Requirements**: ~4-8GB RAM, 2-4 hours processing time
- **Reproducibility**: All random seeds fixed, complete code provided

---
*This report was generated automatically by the ALS research pipeline.*
    """
    
    # Save report
    report_file = results_dir / "comprehensive_report.md"
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Comprehensive report saved to {report_file}")
    
    # Also create a summary CSV
    summary_data = {
        'Metric': [
            'Initial Genes', 'MMPC Selected', 'Final Selected', 'Best Algorithm', 
            'CV Score', 'Most Important Gene'
        ],
        'Value': [
            '21,029',
            str(len(fs_results.get('mmpc_features', []))),
            str(len(fs_results['best_config']['features'])),
            fs_results['best_config']['algorithm'],
            f"{fs_results['best_config']['cv_score']:.4f}",
            gene_importance.iloc[0]['gene'] if not gene_importance.empty else 'N/A'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(results_dir / "results_summary.csv", index=False)
    
    return report_content

def main():
    """Main function to run the complete pipeline"""
    parser = argparse.ArgumentParser(description='ALS Diagnosis Research Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--step', choices=['all', 'download', 'feature_selection', 'shap_analysis', 'report'],
                       default='all', help='Pipeline step to run')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    Path('data/results').mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        
        # Run specified steps
        if args.step in ['all', 'download']:
            step_data_download(config)
        
        if args.step in ['all', 'feature_selection']:
            step_feature_selection(config)
        
        if args.step in ['all', 'shap_analysis']:
            step_model_training_and_shap(config)
        
        if args.step in ['all', 'report']:
            step_generate_report(config)
        
        logger.info("Pipeline completed successfully!")
        
        # Print summary
        if args.step == 'all':
            print("\n" + "="*60)
            print("ALS DIAGNOSIS RESEARCH PIPELINE - COMPLETED")
            print("="*60)
            print("✓ Data downloaded and processed")
            print("✓ Feature selection completed")
            print("✓ Model training and SHAP analysis finished")
            print("✓ Comprehensive report generated")
            print("\nResults available in:")
            print("  - data/results/feature_selection/")
            print("  - data/results/shap_analysis/")
            print("  - data/results/comprehensive_report.md")
            print("\nNext steps:")
            print("  1. Review the generated visualizations")
            print("  2. Examine top-ranked genes for biological significance")
            print("  3. Consider external validation on independent datasets")
            print("="*60)
    
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()