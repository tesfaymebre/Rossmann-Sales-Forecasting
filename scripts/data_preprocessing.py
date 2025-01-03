import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from scipy import stats


class DataPreprocessor:
    def __init__(self, store_path: str, train_path: str, output_path: str):
        self.store_path = store_path
        self.train_path = train_path
        self.output_path = output_path
        self.data = None
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("DataPreprocessor")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def load_and_merge_data(self):
        self.logger.info("Loading and merging datasets.")
        store_df = pd.read_csv(self.store_path)
        train_df = pd.read_csv(self.train_path, low_memory=False)
        self.data = train_df.merge(store_df, on="Store", how="left")

    def process_datetime(self):
        self.logger.info("Processing datetime features.")
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data['Year'] = self.data['Date'].dt.year
        self.data['Month'] = self.data['Date'].dt.month
        self.data['Day'] = self.data['Date'].dt.day
        self.data['Weekday'] = self.data['Date'].dt.weekday
        self.data['IsWeekend'] = self.data['Weekday'].isin([5, 6]).astype(int)

    def handle_missing_data(self, num_strategy='mean', cat_strategy='mode', threshold=0.49):
        self.logger.info("Handling missing data.")
        missing_fraction = self.data.isnull().mean()
        columns_to_drop = missing_fraction[missing_fraction > threshold].index.tolist()
        
        if columns_to_drop:
            self.logger.info(f"Dropping columns with missing values > {threshold}: {columns_to_drop}")
            self.data.drop(columns=columns_to_drop, inplace=True)

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=[object]).columns

        if num_strategy == 'mean':
            self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        elif num_strategy == 'median':
            self.data[numeric_cols] = self.data[numeric_cols].fillna(self.data[numeric_cols].median())

        for col in categorical_cols:
            if cat_strategy == 'mode':
                mode_value = self.data[col].mode()
                if not mode_value.empty:
                    self.data[col] = self.data[col].fillna(mode_value.iloc[0])
            elif cat_strategy == 'unknown':
                self.data[col].fillna('Unknown', inplace=True)

    def detect_and_handle_outliers(self, method='iqr', threshold=3):
        self.logger.info(f"Detecting and handling outliers using {method} method.")
        numeric_cols = self.data.select_dtypes(include=[np.number])

        if method == 'zscore':
            z_scores = np.abs(stats.zscore(numeric_cols))
            self.data = self.data[(z_scores < threshold).all(axis=1)]
        elif method == 'iqr':
            Q1 = numeric_cols.quantile(0.25)
            Q3 = numeric_cols.quantile(0.75)
            IQR = Q3 - Q1
            self.data = self.data[~((numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)]

    def save_preprocessed_data(self):
        self.logger.info(f"Saving preprocessed data to {self.output_path}")
        output_dir = Path(self.output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(self.output_path, index=False)

    def preprocess(self):
        self.load_and_merge_data()
        self.process_datetime()
        self.handle_missing_data()
        self.detect_and_handle_outliers()
        self.save_preprocessed_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Rossmann Sales Forecasting data.")
    parser.add_argument("--store", type=str, required=True, help="Path to the store CSV file.")
    parser.add_argument("--train", type=str, required=True, help="Path to the train CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the preprocessed CSV file.")
    args = parser.parse_args()

    preprocessor = DataPreprocessor(store_path=args.store, train_path=args.train, output_path=args.output)
    preprocessor.preprocess()
