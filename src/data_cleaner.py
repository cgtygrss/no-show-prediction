import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataCleaner:
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs initial data cleaning and renaming."""
        logger.info("Cleaning data...")
        if df is None:
            raise ValueError("Dataframe is empty.")

        df = df.copy()

        # Rename columns
        df.rename(columns={
            "Hipertension": "Hypertension",
            "Handcap": "Handicap",
            "SMS_received": "SMSReceived",
            "No-show": "NoShow"
        }, inplace=True)

        # Convert types
        df['PatientId'] = df['PatientId'].astype('int64')
        df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
        df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

        # Filter invalid ages
        initial_count = len(df)
        df = df[(df['Age'] < 115) & (df['Age'] > 0)]
        logger.info(f"Dropped {initial_count - len(df)} rows with invalid age.")

        # Drop unnecessary IDs
        df.drop(['PatientId', 'AppointmentID'], axis=1, inplace=True)
        
        return df
