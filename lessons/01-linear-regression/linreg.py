"""Linear regression from scratch.

Work through lesson.ipynb; each function here gets filled in at the section
that introduces it. test_linreg.py checks them.

Shapes used throughout:
    X  (n, d)  n examples, d features
    y  (n,)    the true outputs
    w  (d,)    one weight per feature
    b  float   the bias (intercept)
"""

import numpy as np


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Section 3. The model's prediction for every row of X, shape (n,)."""
    raise NotImplementedError


def mse(y_pred: np.ndarray, y: np.ndarray) -> float:
    """Section 3. Mean squared error between predictions and true values."""
    raise NotImplementedError


def gradients(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> tuple[np.ndarray, float]:
    """Section 5. Partial derivatives of the MSE loss: (dL/dw with shape (d,), dL/db)."""
    raise NotImplementedError


def fit(X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 1000) -> tuple[np.ndarray, float, list[float]]:
    """Section 6. Gradient descent from w = 0, b = 0.

    Returns (w, b, history), where history[i] is the loss at the start of epoch i.
    """
    raise NotImplementedError


def fit_closed_form(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Section 8. The exact least-squares (w, b), via the normal equation."""
    raise NotImplementedError
