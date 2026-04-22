# ML Models

A collection of machine learning algorithms implemented from scratch for educational purposes. Each model includes comprehensive tests using scikit-learn datasets.

## Implemented Models

| Model | File | Description |
|-------|------|-------------|
| **Decision Tree** | `model/decision_tree.py` | CART classifier with entropy/gini criteria |
| **Random Forest** | `model/random_forest.py` | Ensemble of decision trees with bagging + feature subsampling |
| **Gradient Boosting** | `model/gradient_boosting.py` | Gradient boosting classifier with regression trees as weak learners |
| **XGBoost** | `model/xgboosting.py` | XGBoost classifier with L2 regularization (lambda) and min split gain (gamma) |
| **Linear Regression** | `model/linear_regression.py` | OLS closed-form, gradient descent, L1 (Lasso), and L2 (Ridge) regularization |

## Project Structure

```
ml_models/
├── model/
│   ├── __init__.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── gradient_boosting.py
│   ├── xgboosting.py
│   └── linear_regression.py
├── test/
│   ├── test_decision_tree.py
│   ├── test_random_forest.py
│   ├── test_boosting.py
│   └── test_linear_regression.py
└── README.md
```

## Usage

### Decision Tree
```python
from model.decision_tree import DecisionTree

clf = DecisionTree(max_depth=5, criterion="entropy")
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
```

### Random Forest
```python
from model.random_forest import RandomForest

rf = RandomForest(n_estimators=100, max_depth=10, max_features="sqrt")
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)
```

### Linear Regression
```python
from model.linear_regression import LinearRegression

# OLS closed-form
lr = LinearRegression(method="ols")
lr.fit(X_train, y_train)

# Gradient Descent with L2 regularization (Ridge)
lr = LinearRegression(method="gradient_descent", regularization="l2", alpha=0.1)
lr.fit(X_train, y_train)
```

## Testing

Run all tests:
```bash
python test/test_decision_tree.py
python test/test_random_forest.py
python test/test_boosting.py
python test/test_linear_regression.py
```

Or run individual test suites.

## Requirements

- numpy
- scikit-learn (for datasets and evaluation metrics only)


## This project is for educational purposes.
