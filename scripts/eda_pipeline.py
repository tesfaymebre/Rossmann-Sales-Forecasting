from scripts.data_loader import DataLoader
from scripts.eda_visualizer import EDAVisualizer
from scripts.eda_analyzer import EDAAnalyzer
from scripts.eda_distribution_analysis import DistributionAnalyzer
from scripts.eda_seasonal_analysis import SeasonalAnalyzer

class EDAPipeline:
    def __init__(self, train_path, test_path):
        self.loader = DataLoader(train_path)
        self.test_loader = DataLoader(test_path)
        self.data = self.loader.load_data()
        self.test_data = self.test_loader.load_data()
        self.analyzer = EDAAnalyzer(self.data)
        self.visualizer = EDAVisualizer(self.data)
        self.distribution_analyzer = DistributionAnalyzer(self.data)
        self.seasonal_analyzer = SeasonalAnalyzer(self.data)

    def run(self):
        print("Summary Statistics:")
        print(self.analyzer.summary_statistics())

        print("\nChecking Missing Values:")
        print(self.analyzer.check_missing_values())

        print("\nPromo Distribution Analysis:")
        self.distribution_analyzer.compare_promo_distribution(self.test_data)

        print("\nSales During Holidays:")
        self.distribution_analyzer.analyze_holiday_sales()

        print("\nSeasonal Sales Patterns:")
        self.seasonal_analyzer.analyze_monthly_sales()
        self.seasonal_analyzer.analyze_holiday_trends()

if __name__ == "__main__":
    train_path = "data/preprocessed/store_preprocessed.csv"
    test_path = "data/raw/test.csv"
    pipeline = EDAPipeline(train_path, test_path)
    pipeline.run()
