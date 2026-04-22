import numpy as np


class XGBoostTree:
    """Custom tree for XGBoost that splits using gradient and hessian statistics."""

    def __init__(self, max_depth=3, min_samples_split=2, reg_lambda=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.tree = None

    def _calc_leaf_weight(self, gradients, hessians):
        return -np.sum(gradients) / (np.sum(hessians) + self.reg_lambda)

    def _calc_gain(self, gradients, hessians):
        return 0.5 * (np.sum(gradients) ** 2) / (np.sum(hessians) + self.reg_lambda)

    def _best_split(self, X, gradients, hessians):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        current_gain = self._calc_gain(gradients, hessians)

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() < 1 or right_mask.sum() < 1:
                    continue

                gain = (self._calc_gain(gradients[left_mask], hessians[left_mask]) +
                        self._calc_gain(gradients[right_mask], hessians[right_mask]) -
                        current_gain - self.gamma)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, gradients, hessians, depth=0):
        if (depth >= self.max_depth or
            len(gradients) < self.min_samples_split):
            return {"leaf": True, "weight": self._calc_leaf_weight(gradients, hessians)}

        feature, threshold, gain = self._best_split(X, gradients, hessians)

        if feature is None or gain <= 0:
            return {"leaf": True, "weight": self._calc_leaf_weight(gradients, hessians)}

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left_mask], gradients[left_mask], hessians[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], gradients[right_mask], hessians[right_mask], depth + 1),
        }

    def fit(self, X, gradients, hessians):
        self.tree = self._build_tree(X, gradients, hessians)

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["weight"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


class XGBoostClassifier:
    """XGBoost Classifier with L2 regularization and min split gain (gamma)."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, reg_lambda=1.0, gamma=0.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.gamma = gamma
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
                # First-order gradient and second-order hessian of cross-entropy
                gradients = probs[:, k] - y_onehot[:, k]
                hessians = probs[:, k] * (1 - probs[:, k])
                hessians = np.maximum(hessians, 1e-8)

                tree = XGBoostTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    reg_lambda=self.reg_lambda,
                    gamma=self.gamma,
                )
                tree.fit(X, gradients, hessians)
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
