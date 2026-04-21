import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from random_forest import RandomForest


def test_basic_functionality():
    """Test basic fit and predict functionality."""
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    rf = RandomForest(n_estimators=10, max_depth=5, max_features="sqrt")
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Basic test accuracy: {accuracy:.2f}")
    assert accuracy > 0.8, f"Accuracy too low: {accuracy}"
    print("✓ Basic functionality test passed")


def test_max_features_options():
    """Test different max_features options."""
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, 100)

    options = ["sqrt", "log2", 5, 0.5]
    for option in options:
        rf = RandomForest(n_estimators=5, max_features=option)
        rf.fit(X, y)
        preds = rf.predict(X)
        assert len(preds) == 100
        print(f"✓ max_features={option} test passed")


def test_bootstrap():
    """Test with and without bootstrap."""
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)

    rf_bootstrap = RandomForest(n_estimators=5, bootstrap=True)
    rf_bootstrap.fit(X, y)
    preds_bootstrap = rf_bootstrap.predict(X)

    rf_no_bootstrap = RandomForest(n_estimators=5, bootstrap=False)
    rf_no_bootstrap.fit(X, y)
    preds_no_bootstrap = rf_no_bootstrap.predict(X)

    assert len(preds_bootstrap) == 50
    assert len(preds_no_bootstrap) == 50
    print("✓ Bootstrap test passed")


def test_predict_proba():
    """Test predict_proba method."""
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 3, 50)

    rf = RandomForest(n_estimators=10)
    rf.fit(X, y)
    probs = rf.predict_proba(X)

    assert probs.shape[0] == 50
    assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities should sum to 1"
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities should be between 0 and 1"
    print("✓ predict_proba test passed")


def test_single_class():
    """Test with single class data."""
    X = np.random.randn(20, 3)
    y = np.zeros(20, dtype=int)

    rf = RandomForest(n_estimators=3)
    rf.fit(X, y)
    preds = rf.predict(X)

    assert np.all(preds == 0), "All predictions should be class 0"
    print("✓ Single class test passed")


def test_tree_consistency():
    """Test that predictions with same random_state are consistent."""
    np.random.seed(42)
    X = np.random.randn(30, 4)
    y = np.random.randint(0, 2, 30)

    rf1 = RandomForest(n_estimators=5, random_state=42)
    rf1.fit(X, y)
    preds1 = rf1.predict(X)

    rf2 = RandomForest(n_estimators=5, random_state=42)
    rf2.fit(X, y)
    preds2 = rf2.predict(X)

    assert np.array_equal(preds1, preds2), "Predictions with same random_state should be consistent"
    print("✓ Tree consistency test passed")


if __name__ == "__main__":
    print("Running Random Forest tests...\n")
    test_basic_functionality()
    test_max_features_options()
    test_bootstrap()
    test_predict_proba()
    test_single_class()
    test_tree_consistency()
    print("\n✅ All tests passed!")
