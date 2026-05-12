import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ── Baseline / Classical Models ───────────────────────────────────────────────

def get_linear():
    """Instantiating the baseline Linear Regression model."""
    return LinearRegression()


def get_ridge(alpha: float = 1.0):
    """Instantiating the Ridge Regression model with L2 regularization."""
    return Ridge(alpha=alpha)


def get_random_forest(n_estimators: int = 200, seed: int = 42):
    """Instantiating the Random Forest Regressor for robust non-linear baseline."""
    return RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)


def get_xgboost(seed: int = 42):
    """Instantiating the optimized XGBoost Regressor for high-performance tree boosting."""
    return xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1
    )

# ── Deep Learning (LSTM) ──────────────────────────────────────────────────────

def scale_features(df_train, df_test, feature_cols: list):
    """Applying MinMax scaling to bound features between [0, 1] for neural network stability."""
    scaler = MinMaxScaler()
    df_train = df_train.copy()
    df_test  = df_test.copy()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols]  = scaler.transform(df_test[feature_cols])
    return df_train, df_test, scaler


def build_sequences(df, feature_cols: list, window: int = 30):
    """Transforming time-series tabular data into 3D sequences for LSTM consumption."""
    X_list, y_list = [], []
    for _, group in df.groupby('unit_id'):
        group = group.sort_values('cycle')
        feats = group[feature_cols].values
        rul   = group['RUL'].values
        for i in range(window, len(feats)):
            X_list.append(feats[i - window:i])
            y_list.append(rul[i])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def build_lstm(input_shape: tuple):
    """Constructing the Deep Learning LSTM architecture with Batch Normalization and Dropout."""
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(1e-3), loss='huber', metrics=['mae'])
    return model


def train_lstm(model, X_train, y_train, epochs: int = 100, batch_size: int = 256):
    """Executing LSTM training with early stopping and dynamic learning rate reduction."""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return history