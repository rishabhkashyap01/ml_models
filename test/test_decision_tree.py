import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.decision_tree import DecisionTree

# Sample dataset
X = np.array([
    [2.3, 1.5],
    [1.3, 3.5],
    [3.3, 2.5],
    [2.0, 2.0]
])

y = np.array([0, 1, 0, 1])

# Create model
clf = DecisionTree(max_depth=3, criterion="entropy")

# Train
clf.fit(X, y)

# Predict
predictions = clf.predict(X)

print("Predictions:", predictions)