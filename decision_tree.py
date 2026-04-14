import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, criterion="entropy"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None

    # ----------------------------
    # Impurity Measures
    # ----------------------------
    def entropy(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def gini(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1 - np.sum([p**2 for p in probs])

    def impurity(self, y):
        if self.criterion == "entropy":
            return self.entropy(y)
        else:
            return self.gini(y)

    # ----------------------------
    # Information Gain
    # ----------------------------
    def information_gain(self, X_column, y, threshold):
        parent_impurity = self.impurity(y)

        left_idxs = np.where(X_column <= threshold)[0]
        right_idxs = np.where(X_column > threshold)[0]

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)

        child_impurity = (n_l/n) * self.impurity(y[left_idxs]) + \
                         (n_r/n) * self.impurity(y[right_idxs])

        return parent_impurity - child_impurity

    # ----------------------------
    # Best Split
    # ----------------------------
    def best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None

        n_features = X.shape[1]

        for feature in range(n_features):
            X_column = X[:, feature]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self.information_gain(X_column, y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = threshold

        return split_idx, split_thresh

    # ----------------------------
    # Build Tree (Recursion)
    # ----------------------------
    def build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # stopping conditions
        if (depth >= self.max_depth or
            num_labels == 1 or
            num_samples < self.min_samples_split):
            leaf_value = self.most_common_label(y)
            return {"leaf": True, "value": leaf_value}

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return {"leaf": True, "value": self.most_common_label(y)}

        left_idxs = np.where(X[:, feature] <= threshold)[0]
        right_idxs = np.where(X[:, feature] > threshold)[0]

        left_subtree = self.build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_subtree = self.build_tree(X[right_idxs], y[right_idxs], depth + 1)

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree
        }

    def most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    # ----------------------------
    # Fit
    # ----------------------------
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    # ----------------------------
    # Predict
    # ----------------------------
    def predict(self, X):
        return np.array([self.traverse_tree(x, self.tree) for x in X])

    def traverse_tree(self, x, node):
        if node["leaf"]:
            return node["value"]

        if x[node["feature"]] <= node["threshold"]:
            return self.traverse_tree(x, node["left"])
        return self.traverse_tree(x, node["right"])