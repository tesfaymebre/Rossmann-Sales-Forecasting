import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class EDAAnalyzer:
    def __init__(self, data):
        self.data = data

    def summary_statistics(self):
        return self.data.describe()

    def check_missing_values(self):
        return self.data.isnull().sum()
    
    def promo_analysis(self):
        promo_sales = self.data.groupby("Promo")["Sales"].mean()
        plt.figure(figsize=(8, 6))
        sns.barplot(x=promo_sales.index, y=promo_sales.values)
        plt.title("Average Sales During Promotions")
        plt.xlabel("Promo Active")
        plt.ylabel("Average Sales")
        plt.show()

    def holiday_sales_analysis(self):
        holiday_sales = self.data.groupby("StateHoliday")["Sales"].mean()
        plt.figure(figsize=(8, 6))
        sns.barplot(x=holiday_sales.index, y=holiday_sales.values)
        plt.title("Average Sales During Holidays")
        plt.xlabel("State Holiday Type")
        plt.ylabel("Average Sales")
        plt.show()

    def competitor_distance_analysis(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="CompetitionDistance", y="Sales", data=self.data)
        plt.title("Sales vs. Competition Distance")
        plt.xlabel("Competition Distance")
        plt.ylabel("Sales")
        plt.show()