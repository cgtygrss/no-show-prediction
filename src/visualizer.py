import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    def visualize_data(self, df: pd.DataFrame):
        """Generates exploratory data analysis plots."""
        logger.info("Generating visualizations...")
        if df is None:
            return

        # Set style
        sns.set_style('whitegrid')
        
        # List of columns to visualize against NoShow
        cols = ['Gender', 'Age', 'Scholarship', 'Hypertension', 'Handicap', 
                'Diabetes', 'Alcoholism', 'SMSReceived']
        
        for col in cols:
            plt.figure(figsize=(10, 6))
            sns.countplot(x=col, hue='NoShow', data=df)
            plt.title(f'{col} vs NoShow')
            plt.savefig(f'{col}_vs_NoShow.png')
            plt.close()
            
        logger.info("Visualizations saved as PNG files.")
