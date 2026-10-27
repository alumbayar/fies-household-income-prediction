import numpy as np
import pandas as pd
import pytest

from fies_income.modeling import (
    build_preprocessor,
    regression_metrics,
    reorder_confusion_matrix,
)


def test_build_preprocessor_imputes_scales_and_encodes():
    X = pd.DataFrame(
        {
            "num": [1.0, np.nan, 3.0],
            "cat": ["a", "b", None],
            "ignored": [9, 9, 9],
        }
    )

    out = build_preprocessor(["num"], ["cat"]).fit_transform(X)

    # Median fill gives [1, 2, 3]; mode fill picks "a"; drop="first" removes "a".
    np.testing.assert_allclose(out[:, 0], [-1.224745, 0.0, 1.224745], atol=1e-6)
    np.testing.assert_array_equal(out[:, 1], [0.0, 1.0, 0.0])
    assert out.shape == (3, 2)


@pytest.mark.filterwarnings("ignore:Found unknown categories")
def test_build_preprocessor_ignores_unknown_categories():
    train = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"]})
    test = pd.DataFrame({"num": [1.5], "cat": ["z"]})

    pre = build_preprocessor(["num"], ["cat"]).fit(train)

    np.testing.assert_array_equal(pre.transform(test)[:, 1], [0.0])


def test_regression_metrics():
    actual = np.array([100.0, 200.0, 400.0])
    predicted = np.array([110.0, 180.0, 400.0])

    metrics = regression_metrics(actual, predicted)

    assert metrics["MAE"] == pytest.approx(10.0)
    assert metrics["MAPE (%)"] == pytest.approx(20 / 3)
    assert metrics["RMSE"] == pytest.approx(np.sqrt(500 / 3))
    assert metrics["R2"] == pytest.approx(1 - 500 / (140_000 / 3))
    assert isinstance(metrics["RMSE"], float)


def test_reorder_confusion_matrix():
    cm = np.array(
        [
            [8, 2, 0],
            [1, 3, 0],
            [0, 5, 5],
        ]
    )

    out = reorder_confusion_matrix(cm, ["b", "a", "c"], ["a", "b", "c"])

    np.testing.assert_allclose(
        out,
        [
            [0.75, 0.25, 0.0],
            [0.2, 0.8, 0.0],
            [0.5, 0.0, 0.5],
        ],
    )
