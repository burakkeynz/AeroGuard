import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, classification_report

# RUL Metrics ───────────────────────────────────────────────────────────────

def rmse(y_true, y_pred) -> float:
    """Calculating Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    """Calculating Mean Absolute Error (MAE)."""
    return float(mean_absolute_error(y_true, y_pred))


def nasa_score(y_true, y_pred) -> float:
    """
    Computing NASA's asymmetric scoring function.
    Early predictions (negative error) are penalized less heavily than late predictions.
    """
    diff = np.asarray(y_pred) - np.asarray(y_true)
    return float(np.sum(np.where(diff < 0,
                                 np.exp(-diff / 13) - 1,
                                 np.exp(diff / 10) - 1)))


def evaluate_rul(y_true, y_pred, model_name: str = '') -> dict:
    """Aggregating all regression metrics into a structured dictionary."""
    return {
        'Model':      model_name,
        'RMSE':       round(rmse(y_true, y_pred), 2),
        'MAE':        round(mae(y_true, y_pred), 2),
        'NASA_Score': round(nasa_score(y_true, y_pred), 1)
    }

# Alert / Classification System ─────────────────────────────────────────────

def rul_to_alert(rul: float) -> int:
    """Categorizing RUL values into actionable Early Warning System alerts (0=NORMAL, 1=WARNING, 2=CRITICAL)."""
    if rul < 30:   return 2
    elif rul < 60: return 1
    else:          return 0


LABEL_NAMES = ['NORMAL', 'WARNING', 'CRITICAL']


def alert_confusion(y_true_rul, y_pred_rul) -> np.ndarray:
    """Generating a confusion matrix for the Early Warning System."""
    y_true = np.array([rul_to_alert(r) for r in y_true_rul])
    y_pred = np.array([rul_to_alert(r) for r in y_pred_rul])
    return confusion_matrix(y_true, y_pred, labels=[0, 1, 2])


def alert_report(y_true_rul, y_pred_rul) -> str:
    """Generating a comprehensive classification report for alert performance."""
    y_true = np.array([rul_to_alert(r) for r in y_true_rul])
    y_pred = np.array([rul_to_alert(r) for r in y_pred_rul])
    return classification_report(y_true, y_pred, target_names=LABEL_NAMES)

# Benchmark Aggregation ─────────────────────────────────────────────────────

def build_benchmark(results: list) -> pd.DataFrame:
    """Compiling individual model results into a sorted benchmark DataFrame."""
    return pd.DataFrame(results).sort_values('RMSE').reset_index(drop=True)