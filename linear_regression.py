import numpy as np


class LinearRegression:
    """Linear Regression using Ordinary Least Squares (closed-form) or Gradient Descent."""

    def __init__(self, method="ols", learning_rate=0.01, n_iterations=1000, regularization=None, alpha=0.01):
        self.method = method              # "ols" or "gradient_descent"
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization  # None, "l1" (Lasso), or "l2" (Ridge)
        self.alpha = alpha                # regularization strength
        self.weights = None
        self.bias = None
        self.loss_history = []

    # ----------------------------
    # Fit
    # ----------------------------
    def fit(self, X, y):
        n_samples, n_features = X.shape

        if self.method == "ols" and self.regularization is None:
            self._fit_ols(X, y)
        elif self.method == "ols" and self.regularization == "l2":
            self._fit_ridge_closed(X, y)
        else:
            self._fit_gradient_descent(X, y)

    def _fit_ols(self, X, y):
        """Closed-form solution: w = (X^T X)^{-1} X^T y"""
        X_b = np.c_[np.ones(X.shape[0]), X]
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        self.bias = theta[0]
        self.weights = theta[1:]

    def _fit_ridge_closed(self, X, y):
        """Closed-form Ridge: w = (X^T X + alpha * I)^{-1} X^T y"""
        X_b = np.c_[np.ones(X.shape[0]), X]
        n = X_b.shape[1]
        I = np.eye(n)
        I[0, 0] = 0  # don't regularize bias
        theta = np.linalg.inv(X_b.T @ X_b + self.alpha * I) @ X_b.T @ y
        self.bias = theta[0]
        self.weights = theta[1:]

    def _fit_gradient_descent(self, X, y):
        """Gradient descent with optional L1/L2 regularization."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            # Gradients
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * np.sum(error)

            # Add regularization gradients
            if self.regularization == "l2":
                dw += (self.alpha / n_samples) * self.weights
            elif self.regularization == "l1":
                dw += (self.alpha / n_samples) * np.sign(self.weights)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Track loss
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

    def _compute_loss(self, y, y_pred):
        """MSE loss with optional regularization penalty."""
        n = len(y)
        mse = (1 / (2 * n)) * np.sum((y - y_pred) ** 2)

        if self.regularization == "l2":
            mse += (self.alpha / (2 * n)) * np.sum(self.weights ** 2)
        elif self.regularization == "l1":
            mse += (self.alpha / n) * np.sum(np.abs(self.weights))

        return mse

    # ----------------------------
    # Predict
    # ----------------------------
    def predict(self, X):
        return X @ self.weights + self.bias

    # ----------------------------
    # Metrics
    # ----------------------------
    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
