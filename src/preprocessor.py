import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Any

logger = logging.getLogger(__name__)

class Preprocessor:
    def preprocess(self, df: pd.DataFrame) -> Tuple[Any, Any, Any, Any]:
        """Encodes categorical variables and scales features."""
        logger.info("Preprocessing data...")
        
        df = df.copy()

        # Binary encoding
        df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'M' else 0)
        df['NoShow'] = df['NoShow'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        # One-hot encoding for categorical variables
        categorical_cols = ['Neighbourhood', 'ScheduledDayofWeek', 'AppointmentDayofWeek']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Drop original date columns as we extracted features
        df.drop(['AppointmentDay', 'ScheduledDay'], axis=1, inplace=True)
        
        # Split features and target
        X = df.drop('NoShow', axis=1)
        y = df['NoShow']
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
