import numpy as np


class RegressionTree:
    """Simple regression tree used as a base learner in gradient boosting."""

    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.var(y) * len(y)

    def _best_split(self, X, y):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        current_mse = self._mse(y)

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                gain = current_mse - self._mse(y[left_mask]) - self._mse(y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            np.all(y == y[0])):
            return {"leaf": True, "value": np.mean(y)}

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return {"leaf": True, "value": np.mean(y)}

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


class GradientBoostingClassifier:
    """Gradient Boosting Classifier using softmax multi-class cross-entropy loss."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.n_classes = None
        self.classes_ = None

    def _softmax(self, raw_predictions):
        exp_preds = np.exp(raw_predictions - np.max(raw_predictions, axis=1, keepdims=True))
        return exp_preds / exp_preds.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        n_samples = X.shape[0]

        # One-hot encode targets
        y_onehot = np.zeros((n_samples, self.n_classes))
        for i, cls in enumerate(self.classes_):
            y_onehot[y == cls, i] = 1

        # Initialize raw predictions to zero
        raw_predictions = np.zeros((n_samples, self.n_classes))
        self.trees = []

        for m in range(self.n_estimators):
            probs = self._softmax(raw_predictions)
            trees_m = []

            for k in range(self.n_classes):
                # Pseudo-residuals: negative gradient of cross-entropy loss
                residuals = y_onehot[:, k] - probs[:, k]

                tree = RegressionTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split
                )
                tree.fit(X, residuals)
                trees_m.append(tree)

                raw_predictions[:, k] += self.learning_rate * tree.predict(X)

            self.trees.append(trees_m)

    def predict_proba(self, X):
        raw_predictions = np.zeros((X.shape[0], self.n_classes))

        for trees_m in self.trees:
            for k, tree in enumerate(trees_m):
                raw_predictions[:, k] += self.learning_rate * tree.predict(X)

        return self._softmax(raw_predictions)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
