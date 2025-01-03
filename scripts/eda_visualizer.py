import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class EDAVisualizer:
    def __init__(self, data):
        self.data = data

    def plot_distribution(self, column):
        plt.figure(figsize=(8, 6))
        sns.histplot(self.data[column], bins=30, kde=True)
        plt.title(f"Distribution of {column}")
        plt.show()

    def plot_correlation_heatmap(self):
        # Convert non-numeric columns (like Date) to numeric
        numeric_data = self.data.copy()
        for col in numeric_data.columns:
            if not pd.api.types.is_numeric_dtype(numeric_data[col]):
                numeric_data[col] = pd.factorize(numeric_data[col])[0]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_data.corr(), cmap='coolwarm', annot=True)
        plt.title("Correlation Heatmap (Numeric Features Only)")
        plt.show()
