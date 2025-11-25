import argparse
import logging
import sys
from typing import Optional
import pandas as pd

from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.feature_engineer import FeatureEngineer
from src.visualizer import Visualizer
from src.preprocessor import Preprocessor
from src.model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class NoShowPredictionPipeline:
    def __init__(self, filepath: str, url: str = None):
        self.filepath = filepath
        self.url = url
        self.df: Optional[pd.DataFrame] = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Initialize components
        self.loader = DataLoader(filepath, url)
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.visualizer = Visualizer()
        self.preprocessor = Preprocessor()
        self.trainer = ModelTrainer()

    def run(self, visualize=False):
        """Executes the full pipeline."""
        # 1. Load Data
        self.df = self.loader.load_data()
        
        # 2. Clean Data
        self.df = self.cleaner.clean_data(self.df)
        
        # 3. Feature Engineering
        self.df = self.engineer.engineer_features(self.df)
        
        # 4. Visualization (Optional)
        if visualize:
            self.visualizer.visualize_data(self.df)
        
        # 5. Preprocessing
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.preprocess(self.df)
        
        # 6. Train Models
        self.trainer.train_models(self.X_train, self.y_train)
        
        # 7. Evaluate Models
        self.trainer.evaluate_models(self.X_test, self.y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="No-Show Prediction Pipeline")
    parser.add_argument("--file", type=str, default="KaggleV2-May-2016.csv", help="Path to the dataset CSV file")
    parser.add_argument("--url", type=str, default="https://www.kaggle.com/datasets/joniarroba/noshowappointments", help="Kaggle dataset URL")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    args = parser.parse_args()
    
    pipeline = NoShowPredictionPipeline(args.file, args.url)
    pipeline.run(visualize=args.visualize)
