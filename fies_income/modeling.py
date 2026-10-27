from collections.abc import Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_cols: Sequence[str], categorical_cols: Sequence[str]
) -> ColumnTransformer:
    """Build the shared imputation, scaling and encoding transformer.

    Args:
        numeric_cols: Columns to median-impute and standardize.
        categorical_cols: Columns to mode-impute and one-hot encode.

    Returns:
        An unfitted ``ColumnTransformer`` that drops all other columns.
    """
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop="first"
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_cols)),
            ("cat", categorical_transformer, list(categorical_cols)),
        ],
        remainder="drop",
    )


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute regression error metrics on the original (non-log) scale.

    Args:
        actual: Observed target values.
        predicted: Model predictions aligned with ``actual``.

    Returns:
        A mapping with keys ``MAE``, ``MAPE (%)``, ``R2`` and ``RMSE``.
    """
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MAPE (%)": mean_absolute_percentage_error(actual, predicted) * 100,
        "R2": r2_score(actual, predicted),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def reorder_confusion_matrix(
    cm: np.ndarray, labels: Sequence[str], order: Sequence[str]
) -> np.ndarray:
    """Reorder a confusion matrix and normalize each row to recall.

    ``LabelEncoder`` sorts classes alphabetically, which scrambles ordinal
    income brackets, so rows and columns are permuted into ``order`` first.

    Args:
        cm: Square confusion matrix whose axes follow ``labels``.
        labels: Class labels in the matrix's current axis order.
        order: Desired class order; must be a permutation of ``labels``.

    Returns:
        The reordered matrix with each row divided by its total.
    """
    idx = [list(labels).index(label) for label in order]
    reordered = cm[np.ix_(idx, idx)].astype(float)
    return reordered / reordered.sum(axis=1, keepdims=True)
