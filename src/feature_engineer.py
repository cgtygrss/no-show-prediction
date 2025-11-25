import pandas as pd
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates new features from existing columns."""
        logger.info("Engineering features...")
        if df is None:
            raise ValueError("Dataframe is empty.")

        df = df.copy()

        # Date features
        df['ScheduledMonth'] = df['ScheduledDay'].dt.month
        df['ScheduledDayofWeek'] = df['ScheduledDay'].dt.day_name()
        df['ScheduledHour'] = df['ScheduledDay'].dt.hour
        df['AppointmentMonth'] = df['AppointmentDay'].dt.month
        df['AppointmentDayofWeek'] = df['AppointmentDay'].dt.day_name()
        df['AppointmentHour'] = df['AppointmentDay'].dt.hour
        
        # Calculate waiting time (days between scheduled and appointment)
        df['WaitingDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
        # Fix negative waiting days (impossible)
        df['WaitingDays'] = df['WaitingDays'].apply(lambda x: 0 if x < 0 else x)
        
        return df
