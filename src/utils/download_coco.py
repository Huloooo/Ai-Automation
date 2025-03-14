import os
import requests
from tqdm import tqdm
import logging
from pathlib import Path
import zipfile
import hashlib
import argparse

class COCODownloader:
    """Utility class to download COCO dataset."""
    
    # COCO 2014 URLs and MD5 checksums
    URLS = {
        'train2014': {
            'url': 'http://images.cocodataset.org/zips/train2014.zip',
            'md5': '0da8c0bd3d6becc4dcb32757491aca88',
            'size_gb': 13.5
        },
        'val2014': {
            'url': 'http://images.cocodataset.org/zips/val2014.zip',
            'md5': 'a3d79f5ed8d289b7a7554ce06a5782b3',
            'size_gb': 6.6
        }
    }
    
    def __init__(self, download_dir: str):
        """
        Initialize downloader.
        
        Args:
            download_dir (str): Directory to download files to
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def calculate_md5(self, file_path: str, chunk_size: int = 8192) -> str:
        """Calculate MD5 checksum of a file."""
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    
    def download_file(self, url: str, filename: str, expected_md5: str = None) -> bool:
        """
        Download a file with progress bar.
        
        Args:
            url (str): URL to download from
            filename (str): Name to save the file as
            expected_md5 (str, optional): Expected MD5 checksum
            
        Returns:
            bool: True if download successful, False otherwise
        """
        file_path = self.download_dir / filename
        
        # Skip if file exists and MD5 matches
        if file_path.exists() and expected_md5:
            actual_md5 = self.calculate_md5(str(file_path))
            if actual_md5 == expected_md5:
                self.logger.info(f"File {filename} already exists and MD5 matches. Skipping download.")
                return True
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(file_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=8192):
                    size = f.write(data)
                    pbar.update(size)
            
            # Verify MD5 if provided
            if expected_md5:
                actual_md5 = self.calculate_md5(str(file_path))
                if actual_md5 != expected_md5:
                    self.logger.error(f"MD5 mismatch for {filename}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading {filename}: {str(e)}")
            if file_path.exists():
                file_path.unlink()
            return False
    
    def extract_zip(self, zip_path: str, extract_dir: str) -> bool:
        """
        Extract a zip file with progress bar.
        
        Args:
            zip_path (str): Path to zip file
            extract_dir (str): Directory to extract to
            
        Returns:
            bool: True if extraction successful, False otherwise
        """
        try:
            with zipfile.ZipFile(zip_path) as zf:
                # Get total size
                total_size = sum(info.file_size for info in zf.filelist)
                extracted_size = 0
                
                # Extract with progress bar
                with tqdm(desc=f"Extracting {Path(zip_path).name}",
                         total=total_size,
                         unit='iB',
                         unit_scale=True,
                         unit_divisor=1024) as pbar:
                    for info in zf.filelist:
                        zf.extract(info, extract_dir)
                        extracted_size += info.file_size
                        pbar.update(info.file_size)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error extracting {zip_path}: {str(e)}")
            return False
    
    def download_dataset(self, dataset_type: str) -> bool:
        """
        Download and extract a COCO dataset split.
        
        Args:
            dataset_type (str): Dataset split ('train2014' or 'val2014')
            
        Returns:
            bool: True if successful, False otherwise
        """
        if dataset_type not in self.URLS:
            self.logger.error(f"Invalid dataset type: {dataset_type}")
            return False
        
        dataset_info = self.URLS[dataset_type]
        zip_filename = f"{dataset_type}.zip"
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.download_dir)
            required_space = dataset_info['size_gb'] * 1024 * 1024 * 1024  # Convert GB to bytes
            if free < required_space:
                self.logger.error(
                    f"Not enough disk space. Required: {dataset_info['size_gb']:.1f}GB, "
                    f"Available: {free / (1024**3):.1f}GB"
                )
                return False
        except Exception as e:
            self.logger.warning(f"Could not check disk space: {str(e)}")
        
        # Download
        self.logger.info(f"Downloading {dataset_type}...")
        if not self.download_file(
            dataset_info['url'],
            zip_filename,
            dataset_info['md5']
        ):
            return False
        
        # Extract
        self.logger.info(f"Extracting {dataset_type}...")
        zip_path = self.download_dir / zip_filename
        extract_path = self.download_dir / dataset_type
        success = self.extract_zip(str(zip_path), str(extract_path))
        
        # Clean up zip file
        if success:
            zip_path.unlink()
            self.logger.info(f"Successfully downloaded and extracted {dataset_type}")
        
        return success

def main():
    parser = argparse.ArgumentParser(description='Download COCO dataset')
    parser.add_argument('--download-dir', type=str, default='data/raw/images',
                      help='Directory to download dataset to')
    parser.add_argument('--full', action='store_true',
                      help='Download full dataset (both train and val)')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Download dataset
    downloader = COCODownloader(args.download_dir)
    
    # First download validation set
    logging.info("Starting with validation set download...")
    if not downloader.download_dataset('val2014'):
        logging.error("Failed to download validation set")
        return
    
    # If --full flag is provided or user wants to download training set
    if args.full or input("\nDo you want to download the training set (13.5GB)? [y/N]: ").lower() == 'y':
        logging.info("\nDownloading training set...")
        if not downloader.download_dataset('train2014'):
            logging.error("Failed to download training set")
    else:
        logging.info("\nSkipping training set download")
    
    logging.info("Download process completed!")

if __name__ == "__main__":
    main() 