import pandas as pd
import logging
import sys
import os

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, filepath: str, url: str = None):
        self.filepath = filepath
        self.url = url

    def load_data(self) -> pd.DataFrame:
        """Loads the dataset from the specified filepath or downloads it if missing."""
        if not os.path.exists(self.filepath) and self.url:
            logger.info(f"File not found at {self.filepath}. Attempting to download from {self.url}...")
            try:
                import kaggle
                # Extract dataset name from URL (e.g., joniarroba/noshowappointments)
                if "kaggle.com/datasets/" in self.url:
                    dataset_name = self.url.split("kaggle.com/datasets/")[1]
                else:
                    dataset_name = self.url

                logger.info(f"Downloading dataset {dataset_name} using Kaggle API...")
                kaggle.api.authenticate()
                kaggle.api.dataset_download_files(dataset_name, path='.', unzip=True)
                
                # Check if the file exists now
                if not os.path.exists(self.filepath):
                    # Sometimes the downloaded file name might be different
                    # Let's look for csv files in the current directory
                    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
                    if csv_files:
                        # Prefer the one that matches the expected name if possible, otherwise take the first one
                        if "KaggleV2-May-2016.csv" in csv_files:
                             self.filepath = "KaggleV2-May-2016.csv"
                        else:
                             self.filepath = csv_files[0]
                        logger.info(f"Found downloaded CSV file: {self.filepath}")
                    else:
                        raise FileNotFoundError("Downloaded dataset but could not find any CSV file.")

                logger.info(f"Dataset downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download dataset: {e}")
                logger.error("Please ensure you have a valid kaggle.json file in ~/.kaggle/ or export KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
                sys.exit(1)

        logger.info(f"Loading data from {self.filepath}...")
        try:
            df = pd.read_csv(self.filepath)
            logger.info("Data loaded successfully.")
            return df
        except FileNotFoundError:
            logger.error(f"File not found at {self.filepath}. Please check the path or provide a valid URL.")
            sys.exit(1)
