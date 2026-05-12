import pandas as pd
import numpy as np
import json
from pathlib import Path

# Defining standard column names for the NASA CMAPSS dataset
COLS = ['unit_id', 'cycle', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
       [f's{i}' for i in range(1, 22)]


def load_raw(path: str) -> pd.DataFrame:
    """Loading raw text data and assigning standardized column names."""
    return pd.read_csv(path, sep=' ', header=None, names=COLS)


def load_processed(path: str) -> pd.DataFrame:
    """Loading pre-processed tabular data."""
    return pd.read_csv(path)


def load_feature_cols(path: str = 'data/processed/feature_cols.json') -> list:
    """Loading the list of engineered feature columns from JSON."""
    with open(path) as f:
        return json.load(f)


def load_useful_sensors(path: str = 'data/processed/useful_sensors.json') -> list:
    """Loading the list of active and informative sensors from JSON."""
    with open(path) as f:
        return json.load(f)


def train_test_split_by_engine(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """Splitting the dataset into training and testing sets based on unique engine IDs to prevent data leakage."""
    engines = df['unit_id'].unique()
    np.random.seed(seed)
    test_engines = np.random.choice(engines, size=int(test_size * len(engines)), replace=False)
    
    train_mask = ~df['unit_id'].isin(test_engines)
    test_mask  =  df['unit_id'].isin(test_engines)
    
    return df[train_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)