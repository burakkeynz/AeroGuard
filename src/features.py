import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

STD_THRESHOLD = 0.5


def drop_flat_sensors(df: pd.DataFrame, threshold: float = STD_THRESHOLD) -> list:
    """Identifying and dropping sensors with near-zero variance (flat signals)."""
    sensor_cols = [f's{i}' for i in range(1, 22)]
    stds = df[sensor_cols].std()
    return stds[stds > threshold].index.tolist()


def add_op_clusters(df: pd.DataFrame, n_clusters: int = 3, seed: int = 42) -> pd.DataFrame:
    """Clustering operational settings to capture distinct flight conditions."""
    op_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[op_cols])
    
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    df = df.copy()
    df['op_cluster'] = km.fit_predict(scaled)
    return df


def add_rolling_features(df: pd.DataFrame, sensors: list, windows: list = [5, 10, 20]) -> pd.DataFrame:
    """Extracting moving averages and rolling standard deviations to capture temporal degradation trends."""
    df = df.copy()
    for w in windows:
        for s in sensors:
            grp = df.groupby('unit_id')[s]
            df[f'{s}_mean_{w}'] = grp.transform(lambda x: x.rolling(w, min_periods=1).mean())
            df[f'{s}_std_{w}']  = grp.transform(lambda x: x.rolling(w, min_periods=1).std().fillna(0))
    return df


def add_cycle_normalized(df: pd.DataFrame, sensors: list) -> pd.DataFrame:
    """Normalizing sensor values against the engine's operational lifecycle."""
    df = df.copy()
    max_cycles = df.groupby('unit_id')['cycle'].transform('max')
    for s in sensors:
        df[f'{s}_norm'] = df[s] / (max_cycles + 1e-8)
    return df


def add_cross_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generating cross-sensor interaction features to expose complex degradation patterns."""
    df = df.copy()
    pairs = [('s2', 's3'), ('s3', 's4'), ('s9', 's14'), ('s11', 's20')]
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            df[f'{a}_x_{b}'] = df[a] * df[b]
            
    if 's2' in df.columns and 's4' in df.columns:
        df['s2_div_s4'] = df['s2'] / (df['s4'] + 1e-8)
    return df


def add_piecewise_rul(df: pd.DataFrame, clip: int = 125) -> pd.DataFrame:
    """Capping the RUL to a maximum threshold (Piecewise Linear) to prevent over-optimistic early predictions."""
    df = df.copy()
    max_cycle = df.groupby('unit_id')['cycle'].transform('max')
    rul = max_cycle - df['cycle']
    df['RUL'] = rul.clip(upper=clip)
    return df


def build_features(df: pd.DataFrame, useful_sensors: list) -> pd.DataFrame:
    """Executing the complete feature engineering pipeline."""
    df = add_op_clusters(df)
    df = add_rolling_features(df, useful_sensors)
    df = add_cycle_normalized(df, useful_sensors)
    df = add_cross_sensor_features(df)
    return df