import os
import pandas as pd
import numpy as np
import gzip
import logging
from typing import List, Dict, Tuple
from pathlib import Path

class GEODownloader:
    """
    Class to process local GEO datasets for ALS research
    Modified version that works with manually downloaded series matrix files
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def check_local_series_matrix(self, gse_id: str) -> str:
        """
        Check if series matrix file exists locally
        
        Args:
            gse_id: GEO Series ID (e.g., 'GSE112676')
            
        Returns:
            Path to local file if exists, None if not found
        """
        # Check for both compressed and uncompressed versions
        compressed_path = self.raw_dir / f"{gse_id}_series_matrix.txt.gz"
        uncompressed_path = self.raw_dir / f"{gse_id}_series_matrix.txt"
        
        if compressed_path.exists():
            self.logger.info(f"✅ Found local file: {compressed_path}")
            return str(compressed_path)
        elif uncompressed_path.exists():
            self.logger.info(f"✅ Found local file: {uncompressed_path}")
            return str(uncompressed_path)
        else:
            self.logger.error(f"❌ Series matrix file not found for {gse_id}")
            self.logger.error(f"Expected locations:")
            self.logger.error(f"  - {compressed_path}")
            self.logger.error(f"  - {uncompressed_path}")
            self.logger.error(f"Please download the file manually and place it in {self.raw_dir}")
            raise FileNotFoundError(f"Series matrix file not found for {gse_id}")
    
    def parse_series_matrix(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse series matrix file to extract expression data and sample metadata
        
        Args:
            file_path: Path to series matrix file
            
        Returns:
            Tuple of (expression_data, sample_metadata)
        """
        self.logger.info(f"📂 Parsing series matrix file: {file_path}")
        
        # Determine if file is compressed
        is_compressed = file_path.endswith('.gz')
        
        # Open file (compressed or uncompressed)
        if is_compressed:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                lines = f.readlines()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        
        # Find data section and extract metadata
        data_start_idx = None
        sample_info = {}
        
        self.logger.info("🔍 Extracting sample metadata...")
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Extract sample information
            if line.startswith('!Sample_geo_accession'):
                sample_ids = line.split('\t')[1:]
                self.logger.info(f"Found {len(sample_ids)} sample IDs")
                
            elif line.startswith('!Sample_title'):
                sample_titles = line.split('\t')[1:]
                self.logger.info(f"Found {len(sample_titles)} sample titles")
                
            elif line.startswith('!Sample_source_name'):
                sample_sources = line.split('\t')[1:]
                self.logger.info(f"Found {len(sample_sources)} sample sources")
                
            elif line.startswith('!Sample_characteristics'):
                # Parse sample characteristics (contains ALS/Control info)
                characteristics = line.split('\t')[1:]
                if 'group' in line.lower() or 'disease' in line.lower() or 'condition' in line.lower():
                    sample_info['group'] = characteristics
                    self.logger.info(f"Found group information in characteristics")
            
            # Find where actual expression data starts
            elif line.startswith('!series_matrix_table_begin'):
                data_start_idx = i + 1
                self.logger.info(f"Found data table starting at line {data_start_idx}")
                break
        
        if data_start_idx is None:
            raise ValueError("Could not find data section in series matrix file")
        
        # Read expression data
        self.logger.info("📊 Extracting expression data...")
        expression_lines = []
        
        for line in lines[data_start_idx:]:
            if line.startswith('!series_matrix_table_end'):
                break
            expression_lines.append(line.strip())
        
        # Parse expression data
        data_rows = []
        for line in expression_lines:
            if line and not line.startswith('!'):
                data_rows.append(line.split('\t'))
        
        if not data_rows:
            raise ValueError("No expression data found in file")
        
        # Create expression DataFrame
        self.logger.info("🔧 Creating expression DataFrame...")
        
        # First row contains sample IDs (headers)
        headers = data_rows[0]
        self.logger.info(f"Found {len(headers)} columns in expression data")
        
        # Remaining rows contain probe data
        probe_data = data_rows[1:]
        
        # Create DataFrame with probe IDs as index
        expression_df = pd.DataFrame(
            [row[1:] for row in probe_data],  # Skip first column (probe ID)
            columns=headers[1:],  # Skip first header ("ID_REF")
            index=[row[0] for row in probe_data]  # Use first column as index
        )
        
        # Convert to numeric
        self.logger.info("🔢 Converting expression values to numeric...")
        for col in expression_df.columns:
            expression_df[col] = pd.to_numeric(expression_df[col], errors='coerce')
        
        # Remove rows with all NaN values
        initial_probes = len(expression_df)
        expression_df = expression_df.dropna(how='all')
        final_probes = len(expression_df)
        
        if initial_probes != final_probes:
            self.logger.info(f"Removed {initial_probes - final_probes} probes with all missing values")
        
        # Create sample metadata DataFrame
        self.logger.info("📋 Creating sample metadata...")
        
        # Ensure we have all required variables
        if 'sample_ids' not in locals():
            sample_ids = headers[1:]  # Use column headers if sample_ids not found
        if 'sample_titles' not in locals():
            sample_titles = ['Sample_' + str(i) for i in range(len(sample_ids))]
        if 'sample_sources' not in locals():
            sample_sources = ['Unknown'] * len(sample_ids)
        
        metadata_df = pd.DataFrame({
            'sample_id': sample_ids,
            'title': sample_titles[:len(sample_ids)],  # Ensure same length
            'source': sample_sources[:len(sample_ids)]  # Ensure same length
        })
        
        # Extract group information from titles or sources
        metadata_df['group'] = metadata_df['title'].apply(self._extract_group_from_title)
        
        # If we found group info in characteristics, use that instead
        if 'group' in sample_info:
            group_chars = sample_info['group'][:len(sample_ids)]
            metadata_df['group'] = [self._extract_group_from_characteristics(char) for char in group_chars]
        
        self.logger.info(f"✅ Parsing completed!")
        self.logger.info(f"Expression data: {expression_df.shape[0]} probes x {expression_df.shape[1]} samples")
        self.logger.info(f"Sample groups: {metadata_df['group'].value_counts().to_dict()}")
        
        return expression_df, metadata_df
    
    def _extract_group_from_title(self, title: str) -> str:
        """
        Extract group (ALS/Control/Mimic) from sample title
        """
        if pd.isna(title):
            return 'Unknown'
        
        title_lower = title.lower()
        if 'als' in title_lower and 'mimic' not in title_lower:
            return 'ALS'
        elif 'control' in title_lower or 'normal' in title_lower or 'healthy' in title_lower:
            return 'Control'
        elif 'mimic' in title_lower:
            return 'Mimic'
        else:
            return 'Unknown'
    
    def _extract_group_from_characteristics(self, characteristics: str) -> str:
        """
        Extract group from sample characteristics field
        """
        if pd.isna(characteristics):
            return 'Unknown'
        
        char_lower = characteristics.lower()
        if 'als' in char_lower and 'mimic' not in char_lower:
            return 'ALS'
        elif 'control' in char_lower or 'normal' in char_lower or 'healthy' in char_lower:
            return 'Control'
        elif 'mimic' in char_lower:
            return 'Mimic'
        else:
            return 'Unknown'
    
    def download_platform_annotation(self, platform_id: str) -> str:
        """
        Check for local platform annotation file
        Note: This function is kept for compatibility but expects local files
        
        Args:
            platform_id: Platform ID (e.g., 'GPL6947')
            
        Returns:
            Path to annotation file if exists locally
        """
        local_path = self.raw_dir / "annotations" / f"{platform_id}.annot.gz"
        
        if local_path.exists():
            self.logger.info(f"✅ Found local annotation file: {local_path}")
            return str(local_path)
        else:
            self.logger.warning(f"⚠️ Annotation file not found: {local_path}")
            self.logger.warning("Continuing without platform annotations...")
            return None
    
    def process_datasets(self, gse_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process multiple GSE datasets from local files
        Following the methodology from the research paper
        
        Args:
            gse_ids: List of GSE IDs to process
            
        Returns:
            Tuple of (combined_expression_data, combined_metadata)
        """
        self.logger.info(f"🚀 Processing {len(gse_ids)} datasets: {gse_ids}")
        
        all_expression_data = []
        all_metadata = []
        
        # Process each dataset
        for gse_id in gse_ids:
            self.logger.info(f"\n📂 Processing {gse_id}...")
            
            # Check for local series matrix file
            matrix_file = self.check_local_series_matrix(gse_id)
            
            # Parse data
            expression_df, metadata_df = self.parse_series_matrix(matrix_file)
            
            # Add dataset source
            metadata_df['dataset'] = gse_id
            expression_df.name = gse_id
            
            all_expression_data.append(expression_df)
            all_metadata.append(metadata_df)
            
            self.logger.info(f"✅ {gse_id} processed: {expression_df.shape[0]} probes, {expression_df.shape[1]} samples")
        
        # Combine datasets (only common probes as per paper methodology)
        self.logger.info(f"\n🔗 Combining datasets...")
        
        # Find common probes across all datasets
        common_probes = set(all_expression_data[0].index)
        for df in all_expression_data[1:]:
            common_probes = common_probes.intersection(set(df.index))
        
        common_probes = list(common_probes)
        self.logger.info(f"Found {len(common_probes)} common probes across datasets")
        
        if len(common_probes) == 0:
            raise ValueError("No common probes found across datasets!")
        
        # Filter to common probes and combine
        self.logger.info("🔧 Filtering and combining expression data...")
        combined_expression = pd.concat([
            df.loc[common_probes] for df in all_expression_data
        ], axis=1)
        
        combined_metadata = pd.concat(all_metadata, ignore_index=True)
        
        # Filter out ALS-mimic samples (as per paper methodology)
        self.logger.info("🔍 Filtering samples...")
        
        initial_samples = len(combined_metadata)
        non_mimic_samples = combined_metadata[combined_metadata['group'] != 'Mimic']['sample_id']
        
        # Filter expression data and metadata
        combined_expression = combined_expression[non_mimic_samples]
        combined_metadata = combined_metadata[combined_metadata['group'] != 'Mimic'].reset_index(drop=True)
        
        final_samples = len(combined_metadata)
        
        self.logger.info(f"Sample filtering: {initial_samples} → {final_samples} samples")
        self.logger.info(f"Removed {initial_samples - final_samples} mimic samples")
        
        # Final summary
        sample_distribution = combined_metadata['group'].value_counts()
        self.logger.info(f"\n✅ Dataset combination completed!")
        self.logger.info(f"Final dataset: {len(combined_expression)} probes × {len(combined_metadata)} samples")
        self.logger.info(f"Sample distribution: {sample_distribution.to_dict()}")
        
        # Data quality checks
        self.logger.info(f"\n🔍 Data quality checks:")
        missing_values = combined_expression.isnull().sum().sum()
        self.logger.info(f"Missing expression values: {missing_values:,}")
        
        if missing_values > 0:
            self.logger.warning(f"⚠️ Found {missing_values} missing values in expression data")
        
        return combined_expression, combined_metadata
    
    def save_processed_data(self, expression_data: pd.DataFrame, metadata: pd.DataFrame):
        """
        Save processed data to files
        """
        expression_path = self.processed_dir / "combined_expression_data.csv"
        metadata_path = self.processed_dir / "sample_metadata.csv"
        
        self.logger.info(f"💾 Saving processed data...")
        
        # Save with progress indication
        self.logger.info(f"Saving expression data to {expression_path}...")
        expression_data.to_csv(expression_path)
        
        self.logger.info(f"Saving metadata to {metadata_path}...")
        metadata.to_csv(metadata_path, index=False)
        
        # Save summary statistics
        summary_path = self.processed_dir / "data_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"ALS Gene Expression Data Summary\n")
            f.write(f"================================\n\n")
            f.write(f"Expression Data:\n")
            f.write(f"  - Probes/Genes: {expression_data.shape[0]:,}\n")
            f.write(f"  - Samples: {expression_data.shape[1]:,}\n")
            f.write(f"  - Missing values: {expression_data.isnull().sum().sum():,}\n\n")
            f.write(f"Sample Metadata:\n")
            f.write(f"  - Total samples: {len(metadata)}\n")
            f.write(f"  - Sample distribution:\n")
            for group, count in metadata['group'].value_counts().items():
                f.write(f"    - {group}: {count}\n")
            f.write(f"  - Datasets included:\n")
            for dataset, count in metadata['dataset'].value_counts().items():
                f.write(f"    - {dataset}: {count} samples\n")
        
        self.logger.info(f"✅ All data saved successfully!")
        self.logger.info(f"Files created:")
        self.logger.info(f"  - {expression_path}")
        self.logger.info(f"  - {metadata_path}")
        self.logger.info(f"  - {summary_path}")

# Example usage and test
if __name__ == "__main__":
    # Initialize downloader
    downloader = GEODownloader()
    
    # Process the two datasets from the paper (using local files)
    gse_ids = ["GSE112676", "GSE112680"]
    
    print("🚀 ALS Gene Expression Data Processor")
    print("====================================")
    print(f"Expected files in {downloader.raw_dir}:")
    for gse_id in gse_ids:
        print(f"  - {gse_id}_series_matrix.txt.gz (or .txt)")
    print()
    
    try:
        expression_data, metadata = downloader.process_datasets(gse_ids)
        downloader.save_processed_data(expression_data, metadata)
        
        print("\n" + "="*50)
        print("✅ DATA PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"📊 Final Results:")
        print(f"  Expression data shape: {expression_data.shape}")
        print(f"  Metadata shape: {metadata.shape}")
        print(f"  Sample groups: {metadata['group'].value_counts().to_dict()}")
        print(f"  Datasets processed: {metadata['dataset'].unique().tolist()}")
        print("\n🎯 Ready for feature selection and machine learning analysis!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n📥 Please download the required files:")
        print("1. Go to NCBI GEO database")
        print("2. Search for GSE112676 and GSE112680")
        print("3. Download the series matrix files")
        print(f"4. Place them in: {downloader.raw_dir}")
        print("5. Run this script again")
        
    except Exception as e:
        print(f"\n❌ Error processing data: {str(e)}")
        raise