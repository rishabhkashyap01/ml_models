import numpy as np
from sklearn.datasets import load_diabetes, make_regression
from sklearn.model_selection import train_test_split
from linear_regression import LinearRegression


def test_ols():
    """Test OLS closed-form on diabetes dataset."""
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LinearRegression(method="ols")
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)

    r2 = lr.r2_score(y_test, preds)
    mse = lr.mse(y_test, preds)
    print(f"  R²: {r2:.4f}  MSE: {mse:.2f}")
    assert r2 > 0.4, f"R² too low: {r2}"
    print("  ✓ Passed")


def test_gradient_descent():
    """Test gradient descent on synthetic data."""
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    # Normalize for stable gradient descent
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LinearRegression(method="gradient_descent", learning_rate=0.1, n_iterations=500)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)

    r2 = lr.r2_score(y_test, preds)
    print(f"  R²: {r2:.4f}  Loss (final): {lr.loss_history[-1]:.4f}")
    assert r2 > 0.8, f"R² too low: {r2}"
    assert lr.loss_history[-1] < lr.loss_history[0], "Loss should decrease"
    print("  ✓ Passed")


def test_ridge_closed():
    """Test Ridge regression (closed-form)."""
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LinearRegression(method="ols", regularization="l2", alpha=1.0)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)

    r2 = lr.r2_score(y_test, preds)
    print(f"  R²: {r2:.4f}")
    assert r2 > 0.4, f"R² too low: {r2}"
    print("  ✓ Passed")


def test_ridge_gd():
    """Test Ridge regression (gradient descent)."""
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LinearRegression(method="gradient_descent", regularization="l2",
                          alpha=0.1, learning_rate=0.1, n_iterations=500)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)

    r2 = lr.r2_score(y_test, preds)
    print(f"  R²: {r2:.4f}")
    assert r2 > 0.8, f"R² too low: {r2}"
    print("  ✓ Passed")


def test_lasso_gd():
    """Test Lasso regression (gradient descent with L1)."""
    X, y = make_regression(n_samples=200, n_features=10, n_informative=3, noise=10, random_state=42)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    lr = LinearRegression(method="gradient_descent", regularization="l1",
                          alpha=0.5, learning_rate=0.1, n_iterations=500)
    lr.fit(X_train, y_train)
    preds = lr.predict(X_test)

    r2 = lr.r2_score(y_test, preds)
    n_near_zero = np.sum(np.abs(lr.weights) < 0.1)
    print(f"  R²: {r2:.4f}  Near-zero weights: {n_near_zero}/{len(lr.weights)}")
    assert r2 > 0.5, f"R² too low: {r2}"
    print("  ✓ Passed")


def test_perfect_fit():
    """Test on perfectly linear data (no noise)."""
    np.random.seed(42)
    X = np.random.randn(100, 3)
    true_weights = np.array([2.0, -3.0, 1.5])
    true_bias = 5.0
    y = X @ true_weights + true_bias

    lr = LinearRegression(method="ols")
    lr.fit(X, y)
    preds = lr.predict(X)

    r2 = lr.r2_score(y, preds)
    print(f"  R²: {r2:.6f}")
    assert np.allclose(lr.weights, true_weights, atol=1e-8), "Weights should match true weights"
    assert np.isclose(lr.bias, true_bias, atol=1e-8), "Bias should match true bias"
    print(f"  Recovered weights: {lr.weights}  bias: {lr.bias:.4f}")
    print("  ✓ Passed")


if __name__ == "__main__":
    print("=" * 50)
    print("Linear Regression Tests")
    print("=" * 50)

    print("\n[1] OLS (Diabetes dataset):")
    test_ols()

    print("\n[2] Gradient Descent (synthetic):")
    test_gradient_descent()

    print("\n[3] Ridge - closed form (Diabetes):")
    test_ridge_closed()

    print("\n[4] Ridge - gradient descent (synthetic):")
    test_ridge_gd()

    print("\n[5] Lasso - gradient descent (sparse features):")
    test_lasso_gd()

    print("\n[6] Perfect fit (no noise):")
    test_perfect_fit()

    print("\n✅ All linear regression tests passed!")
