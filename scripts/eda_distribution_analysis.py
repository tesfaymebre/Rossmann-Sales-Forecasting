import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class DistributionAnalyzer:
    def __init__(self, data):
        self.data = data
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("DistributionAnalyzer")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def compare_promo_distribution(self, test_data):
        """Compare promotion distribution between train and test datasets."""
        self.logger.info("Comparing promotion distribution...")
        plt.figure(figsize=(8, 6))
        sns.histplot(self.data['Promo'], label='Train', color='blue', kde=True)
        sns.histplot(test_data['Promo'], label='Test', color='red', kde=True)
        plt.title("Promotion Distribution (Train vs Test)")
        plt.legend()
        plt.show()

    def analyze_holiday_sales(self):
        """Compare sales behavior before, during, and after holidays."""
        self.logger.info("Analyzing sales behavior during holidays...")
        self.data['StateHoliday'] = self.data['StateHoliday'].replace(0, 'None')
        holiday_sales = self.data.groupby('StateHoliday')['Sales'].mean()
        holiday_sales.plot(kind='bar', figsize=(8, 6))
        plt.title("Average Sales During Holidays")
        plt.ylabel("Average Sales")
        plt.show()
