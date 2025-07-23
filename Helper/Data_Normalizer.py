import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from os import path
from typing import List
import threading


class DataNormalizer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DataNormalizer, cls).__new__(cls)
        return cls._instance

    def __init__(self, features: List[str], label: str, feature_range=(-1, 1),
                 scaler_path='Model/scaler.pkl', output_dir='Data'):
        if hasattr(self, "_initialized") and self._initialized:
            return  # Prevent reinitialization on multiple calls

        if not features or not label:
            raise ValueError("Both 'features' and 'label' must be provided.")

        self.features = features
        self.label = label
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.scaler_path = scaler_path
        self.output_dir = output_dir
        self._initialized = True

    def fit_transform_training_data(self, df: pd.DataFrame, save_csv=True) -> pd.DataFrame:
        """
        Fit the scaler to training data and transform it.
        Saves the fitted scaler and optionally the result as a CSV.
        """
        scaled_features = self.scaler.fit_transform(df[self.features])
        joblib.dump(self.scaler, self.scaler_path)

        scaled_df = pd.DataFrame(scaled_features, columns=self.features, index=df.index)
        result = pd.concat([scaled_df, df[[self.label]]], axis=1)

        if save_csv:
            output_path = path.join(self.output_dir, 'train_data_normalized.csv')
            result.to_csv(output_path, index=False)

        return result

    def transform_testing_data(self, df: pd.DataFrame, save_csv=True) -> pd.DataFrame:
        """
        Load the saved scaler and transform testing data.
        Optionally saves the transformed data to CSV.
        """
        self._load_scaler()
        scaled_features = self.scaler.transform(df[self.features])
        scaled_df = pd.DataFrame(scaled_features, columns=self.features, index=df.index)
        result = pd.concat([scaled_df, df[[self.label]]], axis=1)

        if save_csv:
            output_path = path.join(self.output_dir, 'testing_data_normalized.csv')
            result.to_csv(output_path, index=False)

        return result

    def transform_evaluation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform evaluation data (only features) using the saved scaler.
        """
        self._load_scaler()
        scaled_features = self.scaler.transform(df)
        return pd.DataFrame(scaled_features, columns=df.columns, index=df.index)

    def _load_scaler(self):
        """
        Load the scaler from disk if it exists.
        """
        if not path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found at '{self.scaler_path}'.")
        self.scaler = joblib.load(self.scaler_path)
