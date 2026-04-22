import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.gradient_boosting import GradientBoostingClassifier
from model.xgboosting import XGBoostClassifier


# ----------------------------
# Gradient Boosting Tests
# ----------------------------
def test_gb_iris():
    """Test Gradient Boosting on Iris (multi-class)."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)
    gb.fit(X_train, y_train)
    preds = gb.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"  Accuracy: {acc:.2f}")
    assert acc > 0.85, f"Accuracy too low: {acc}"
    print("  ✓ Passed")


def test_gb_binary():
    """Test Gradient Boosting on Breast Cancer (binary)."""
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    gb = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3)
    gb.fit(X_train, y_train)
    preds = gb.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"  Accuracy: {acc:.2f}")
    assert acc > 0.90, f"Accuracy too low: {acc}"
    print("  ✓ Passed")


def test_gb_predict_proba():
    """Test Gradient Boosting predict_proba output."""
    X, y = load_iris(return_X_y=True)
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=2)
    gb.fit(X, y)
    probs = gb.predict_proba(X)

    assert probs.shape == (150, 3), f"Wrong shape: {probs.shape}"
    assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities must sum to 1"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities must be in [0, 1]"
    print("  ✓ Passed")


# ----------------------------
# XGBoost Tests
# ----------------------------
def test_xgb_iris():
    """Test XGBoost on Iris (multi-class)."""
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    xgb = XGBoostClassifier(
        n_estimators=50, learning_rate=0.1, max_depth=3,
        reg_lambda=1.0, gamma=0.0
    )
    xgb.fit(X_train, y_train)
    preds = xgb.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"  Accuracy: {acc:.2f}")
    assert acc > 0.85, f"Accuracy too low: {acc}"
    print("  ✓ Passed")


def test_xgb_binary():
    """Test XGBoost on Breast Cancer (binary)."""
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    xgb = XGBoostClassifier(
        n_estimators=50, learning_rate=0.1, max_depth=3,
        reg_lambda=1.0, gamma=0.0
    )
    xgb.fit(X_train, y_train)
    preds = xgb.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"  Accuracy: {acc:.2f}")
    assert acc > 0.90, f"Accuracy too low: {acc}"
    print("  ✓ Passed")


def test_xgb_regularization():
    """Test that increasing regularization reduces overfitting."""
    X, y = load_iris(return_X_y=True)

    xgb_low_reg = XGBoostClassifier(n_estimators=30, reg_lambda=0.01, gamma=0.0)
    xgb_low_reg.fit(X, y)
    preds_low = xgb_low_reg.predict(X)
    acc_low = accuracy_score(y, preds_low)

    xgb_high_reg = XGBoostClassifier(n_estimators=30, reg_lambda=10.0, gamma=5.0)
    xgb_high_reg.fit(X, y)
    preds_high = xgb_high_reg.predict(X)
    acc_high = accuracy_score(y, preds_high)

    print(f"  Low reg accuracy:  {acc_low:.2f}")
    print(f"  High reg accuracy: {acc_high:.2f}")
    assert acc_low >= acc_high, "Higher regularization should not increase training accuracy"
    print("  ✓ Passed")


def test_xgb_predict_proba():
    """Test XGBoost predict_proba output."""
    X, y = load_iris(return_X_y=True)
    xgb = XGBoostClassifier(n_estimators=20, learning_rate=0.1, max_depth=2)
    xgb.fit(X, y)
    probs = xgb.predict_proba(X)

    assert probs.shape == (150, 3), f"Wrong shape: {probs.shape}"
    assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities must sum to 1"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities must be in [0, 1]"
    print("  ✓ Passed")


if __name__ == "__main__":
    print("=" * 50)
    print("Gradient Boosting Tests")
    print("=" * 50)

    print("\n[1] Iris (multi-class):")
    test_gb_iris()

    print("\n[2] Breast Cancer (binary):")
    test_gb_binary()

    print("\n[3] predict_proba validation:")
    test_gb_predict_proba()

    print("\n" + "=" * 50)
    print("XGBoost Tests")
    print("=" * 50)

    print("\n[4] Iris (multi-class):")
    test_xgb_iris()

    print("\n[5] Breast Cancer (binary):")
    test_xgb_binary()

    print("\n[6] Regularization effect:")
    test_xgb_regularization()

    print("\n[7] predict_proba validation:")
    test_xgb_predict_proba()

    print("\n✅ All boosting tests passed!")
