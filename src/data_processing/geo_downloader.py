import os
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
import gzip
import logging
from typing import List, Dict, Tuple
from pathlib import Path

class GEODownloader:
    """
    Class to download and process GEO datasets for ALS research
    Based on the methodology from the research paper
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
    
    def download_series_matrix(self, gse_id: str) -> str:
        """
        Download series matrix file for a given GSE ID
        
        Args:
            gse_id: GEO Series ID (e.g., 'GSE112676')
            
        Returns:
            Path to downloaded file
        """
        # Construct URL for series matrix file
        base_url = "https://ftp.ncbi.nlm.nih.gov/geo/series"
        series_num = gse_id[:5] + "nnn"  # GSE112676 -> GSE112nnn
        url = f"{base_url}/{series_num}/{gse_id}/matrix/{gse_id}_series_matrix.txt.gz"
        
        # Define local file path
        local_path = self.raw_dir / f"{gse_id}_series_matrix.txt.gz"
        
        if local_path.exists():
            self.logger.info(f"File {local_path} already exists, skipping download")
            return str(local_path)
        
        try:
            self.logger.info(f"Downloading {gse_id} series matrix...")
            urlretrieve(url, local_path)
            self.logger.info(f"Successfully downloaded {local_path}")
            return str(local_path)
        except Exception as e:
            self.logger.error(f"Failed to download {gse_id}: {str(e)}")
            raise
    
    def parse_series_matrix(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse series matrix file to extract expression data and sample metadata
        
        Args:
            file_path: Path to series matrix file
            
        Returns:
            Tuple of (expression_data, sample_metadata)
        """
        self.logger.info(f"Parsing series matrix file: {file_path}")
        
        # Open gzipped file
        with gzip.open(file_path, 'rt') as f:
            lines = f.readlines()
        
        # Find data section
        data_start_idx = None
        sample_info = {}
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Extract sample information
            if line.startswith('!Sample_geo_accession'):
                sample_ids = line.split('\t')[1:]
            elif line.startswith('!Sample_title'):
                sample_titles = line.split('\t')[1:]
            elif line.startswith('!Sample_source_name'):
                sample_sources = line.split('\t')[1:]
            elif line.startswith('!Sample_characteristics'):
                # Parse sample characteristics (contains ALS/Control info)
                characteristics = line.split('\t')[1:]
                if 'group' in line.lower() or 'disease' in line.lower():
                    sample_info['group'] = characteristics
            
            # Find where actual data starts
            if line.startswith('!series_matrix_table_begin'):
                data_start_idx = i + 1
                break
        
        if data_start_idx is None:
            raise ValueError("Could not find data section in series matrix file")
        
        # Read expression data
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
        
        # Create DataFrame
        if data_rows:
            # First row should be headers (sample IDs)
            headers = data_rows[0]
            data_values = data_rows[1:]
            
            # Create expression DataFrame
            expression_df = pd.DataFrame(data_values[1:], 
                                       columns=headers,
                                       index=[row[0] for row in data_values[1:]])
            
            # Convert to numeric
            for col in expression_df.columns[1:]:  # Skip first column (probe IDs)
                expression_df[col] = pd.to_numeric(expression_df[col], errors='coerce')
        
        # Create sample metadata DataFrame
        metadata_df = pd.DataFrame({
            'sample_id': sample_ids,
            'title': sample_titles,
            'source': sample_sources
        })
        
        # Extract group information from titles or sources
        metadata_df['group'] = metadata_df['title'].apply(self._extract_group_from_title)
        
        self.logger.info(f"Parsed {len(expression_df)} genes and {len(metadata_df)} samples")
        
        return expression_df, metadata_df
    
    def _extract_group_from_title(self, title: str) -> str:
        """
        Extract group (ALS/Control/Mimic) from sample title
        """
        title_lower = title.lower()
        if 'als' in title_lower and 'mimic' not in title_lower:
            return 'ALS'
        elif 'control' in title_lower or 'normal' in title_lower:
            return 'Control'
        elif 'mimic' in title_lower:
            return 'Mimic'
        else:
            return 'Unknown'
    
    def download_platform_annotation(self, platform_id: str) -> str:
        """
        Download platform annotation file for probe-to-gene mapping
        
        Args:
            platform_id: Platform ID (e.g., 'GPL6947')
            
        Returns:
            Path to annotation file
        """
        # Construct URL
        base_url = "https://ftp.ncbi.nlm.nih.gov/geo/platforms"
        platform_num = platform_id[:3] + "nnn"  # GPL6947 -> GPLnnn
        url = f"{base_url}/{platform_num}/{platform_id}/annot/{platform_id}.annot.gz"
        
        local_path = self.raw_dir / "annotations" / f"{platform_id}.annot.gz"
        local_path.parent.mkdir(exist_ok=True)
        
        if local_path.exists():
            self.logger.info(f"Annotation file {local_path} already exists")
            return str(local_path)
        
        try:
            self.logger.info(f"Downloading {platform_id} annotations...")
            urlretrieve(url, local_path)
            self.logger.info(f"Successfully downloaded {local_path}")
            return str(local_path)
        except Exception as e:
            self.logger.error(f"Failed to download {platform_id} annotations: {str(e)}")
            raise
    
    def process_datasets(self, gse_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Download and process multiple GSE datasets
        Following the methodology from the research paper
        
        Args:
            gse_ids: List of GSE IDs to process
            
        Returns:
            Tuple of (combined_expression_data, combined_metadata)
        """
        all_expression_data = []
        all_metadata = []
        
        # Process each dataset
        for gse_id in gse_ids:
            # Download series matrix
            matrix_file = self.download_series_matrix(gse_id)
            
            # Parse data
            expression_df, metadata_df = self.parse_series_matrix(matrix_file)
            
            # Add dataset source
            metadata_df['dataset'] = gse_id
            expression_df.name = gse_id
            
            all_expression_data.append(expression_df)
            all_metadata.append(metadata_df)
        
        # Combine datasets (only common genes as per paper methodology)
        self.logger.info("Combining datasets...")
        
        # Find common genes across all datasets
        common_genes = set(all_expression_data[0].index)
        for df in all_expression_data[1:]:
            common_genes = common_genes.intersection(set(df.index))
        
        common_genes = list(common_genes)
        self.logger.info(f"Found {len(common_genes)} common genes across datasets")
        
        # Filter to common genes and combine
        combined_expression = pd.concat([
            df.loc[common_genes] for df in all_expression_data
        ], axis=1)
        
        combined_metadata = pd.concat(all_metadata, ignore_index=True)
        
        # Filter out ALS-mimic samples (as per paper methodology)
        non_mimic_samples = combined_metadata[combined_metadata['group'] != 'Mimic']['sample_id']
        combined_expression = combined_expression[non_mimic_samples]
        combined_metadata = combined_metadata[combined_metadata['group'] != 'Mimic'].reset_index(drop=True)
        
        self.logger.info(f"Final dataset: {len(combined_expression)} genes, {len(combined_metadata)} samples")
        self.logger.info(f"Sample distribution: {combined_metadata['group'].value_counts().to_dict()}")
        
        return combined_expression, combined_metadata
    
    def save_processed_data(self, expression_data: pd.DataFrame, metadata: pd.DataFrame):
        """
        Save processed data to files
        """
        expression_path = self.processed_dir / "combined_expression_data.csv"
        metadata_path = self.processed_dir / "sample_metadata.csv"
        
        expression_data.to_csv(expression_path)
        metadata.to_csv(metadata_path, index=False)
        
        self.logger.info(f"Saved processed data to {expression_path} and {metadata_path}")

# Example usage and test
if __name__ == "__main__":
    # Initialize downloader
    downloader = GEODownloader()
    
    # Download and process the two datasets from the paper
    gse_ids = ["GSE112676", "GSE112680"]
    
    try:
        expression_data, metadata = downloader.process_datasets(gse_ids)
        downloader.save_processed_data(expression_data, metadata)
        
        print("\nData processing completed successfully!")
        print(f"Expression data shape: {expression_data.shape}")
        print(f"Metadata shape: {metadata.shape}")
        print(f"Sample groups: {metadata['group'].value_counts()}")
        
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        raise