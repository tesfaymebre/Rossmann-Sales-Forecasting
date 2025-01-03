import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

class SeasonalAnalyzer:
    def __init__(self, data):
        self.data = data
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("SeasonalAnalyzer")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def analyze_monthly_sales(self):
        """Analyze monthly sales patterns."""
        self.logger.info("Analyzing monthly sales patterns...")
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        monthly_sales = self.data.groupby(self.data['Date'].dt.month)['Sales'].mean()
        monthly_sales.plot(kind='bar', figsize=(10, 6))
        plt.title("Average Monthly Sales")
        plt.xlabel("Month")
        plt.ylabel("Average Sales")
        plt.show()

    def analyze_holiday_trends(self):
        """Visualize sales trends during major holidays."""
        self.logger.info("Analyzing sales trends during major holidays...")
        holiday_sales = self.data[self.data['StateHoliday'].isin(['a', 'b', 'c'])]
        sns.lineplot(data=holiday_sales, x='Date', y='Sales', hue='StateHoliday')
        plt.title("Sales Trends During Holidays")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.xticks(rotation=45)
        plt.show()
