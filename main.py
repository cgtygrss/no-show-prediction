import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class NoShowPredictionPipeline:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df: Optional[pd.DataFrame] = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models: Dict[str, Any] = {}

    def load_data(self):
        """Loads the dataset from the specified filepath."""
        logger.info(f"Loading data from {self.filepath}...")
        try:
            self.df = pd.read_csv(self.filepath)
            logger.info("Data loaded successfully.")
        except FileNotFoundError:
            logger.error(f"File not found at {self.filepath}. Please check the path.")
            sys.exit(1)

    def clean_data(self):
        """Performs initial data cleaning and renaming."""
        logger.info("Cleaning data...")
        if self.df is None:
            raise ValueError("Dataframe is empty. Call load_data() first.")

        # Rename columns
        self.df.rename(columns={
            "Hipertension": "Hypertension",
            "Handcap": "Handicap",
            "SMS_received": "SMSReceived",
            "No-show": "NoShow"
        }, inplace=True)

        # Convert types
        self.df['PatientId'] = self.df['PatientId'].astype('int64')
        self.df['ScheduledDay'] = pd.to_datetime(self.df['ScheduledDay'])
        self.df['AppointmentDay'] = pd.to_datetime(self.df['AppointmentDay'])

        # Filter invalid ages
        initial_count = len(self.df)
        self.df = self.df[(self.df['Age'] < 115) & (self.df['Age'] > 0)]
        logger.info(f"Dropped {initial_count - len(self.df)} rows with invalid age.")

        # Drop unnecessary IDs
        self.df.drop(['PatientId', 'AppointmentID'], axis=1, inplace=True)

    def feature_engineering(self):
        """Creates new features from existing columns."""
        logger.info("Engineering features...")
        if self.df is None:
            raise ValueError("Dataframe is empty.")

        # Date features
        self.df['ScheduledMonth'] = self.df['ScheduledDay'].dt.month
        self.df['ScheduledDayofWeek'] = self.df['ScheduledDay'].dt.day_name()
        self.df['ScheduledHour'] = self.df['ScheduledDay'].dt.hour
        self.df['AppointmentMonth'] = self.df['AppointmentDay'].dt.month
        self.df['AppointmentDayofWeek'] = self.df['AppointmentDay'].dt.day_name()
        self.df['AppointmentHour'] = self.df['AppointmentDay'].dt.hour
        
        # Calculate waiting time (days between scheduled and appointment)
        self.df['WaitingDays'] = (self.df['AppointmentDay'] - self.df['ScheduledDay']).dt.days
        # Fix negative waiting days (impossible)
        self.df['WaitingDays'] = self.df['WaitingDays'].apply(lambda x: 0 if x < 0 else x)

    def visualize_data(self):
        """Generates exploratory data analysis plots."""
        logger.info("Generating visualizations...")
        if self.df is None:
            return

        # Set style
        sns.set_style('whitegrid')
        
        # List of columns to visualize against NoShow
        cols = ['Gender', 'Age', 'Scholarship', 'Hypertension', 'Handicap', 
                'Diabetes', 'Alcoholism', 'SMSReceived']
        
        for col in cols:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=col, hue='NoShow', data=self.df)
            plt.title(f'{col} vs NoShow')
            plt.savefig(f'{col}_vs_NoShow.png')
            plt.close()
            
        logger.info("Visualizations saved as PNG files.")

    def preprocess(self):
        """Encodes categorical variables and scales features."""
        logger.info("Preprocessing data...")
        
        # Binary encoding
        self.df['Gender'] = self.df['Gender'].apply(lambda x: 1 if x == 'M' else 0)
        self.df['NoShow'] = self.df['NoShow'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # One-hot encoding for categorical variables
        categorical_cols = ['Neighbourhood', 'ScheduledDayofWeek', 'AppointmentDayofWeek']
        self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)
        
        # Drop original date columns as we extracted features
        self.df.drop(['AppointmentDay', 'ScheduledDay'], axis=1, inplace=True)
        
        # Split features and target
        X = self.df.drop('NoShow', axis=1)
        y = self.df['NoShow']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )

    def train_models(self):
        """Trains multiple machine learning models."""
        logger.info("Training models...")
        
        self.models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB()
        }
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            logger.info(f"{name} trained.")

    def evaluate_models(self):
        """Evaluates trained models and prints reports."""
        logger.info("Evaluating models...")
        
        for name, model in self.models.items():
            print(f"\n{'='*20} {name} {'='*20}")
            predictions = model.predict(self.X_test)
            print(classification_report(self.y_test, predictions))
            print("Confusion Matrix:")
            print(confusion_matrix(self.y_test, predictions))

    def run(self, visualize=False):
        """Executes the full pipeline."""
        self.load_data()
        self.clean_data()
        self.feature_engineering()
        if visualize:
            self.visualize_data()
        self.preprocess()
        self.train_models()
        self.evaluate_models()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="No-Show Prediction Pipeline")
    parser.add_argument("--file", type=str, default="https://www.kaggle.com/datasets/muhammetgamal5/noshowappointmentskagglev2may2016csv", help="Path to the dataset CSV file")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization plots")
    args = parser.parse_args()
    
    pipeline = NoShowPredictionPipeline(args.file)
    pipeline.run(visualize=args.visualize)
