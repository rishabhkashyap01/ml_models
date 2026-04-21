import numpy as np
from collections import Counter
from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2,
                 criterion="entropy", max_features="sqrt", bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []

    def _get_max_features(self, n_features):
        if self.max_features == "sqrt":
            return int(np.sqrt(n_features))
        elif self.max_features == "log2":
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return self.max_features
        elif isinstance(self.max_features, float):
            return int(self.max_features * n_features)
        else:
            return n_features

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape
        max_features = self._get_max_features(n_features)

        self.trees = []
        self.feature_indices = []

        for _ in range(self.n_estimators):
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y

            selected_features = np.random.choice(
                n_features, size=max_features, replace=False
            )
            self.feature_indices.append(selected_features)

            X_subset = X_sample[:, selected_features]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion
            )
            tree.fit(X_subset, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.n_estimators))

        for idx, tree in enumerate(self.trees):
            X_subset = X[:, self.feature_indices[idx]]
            predictions[:, idx] = tree.predict(X_subset)

        return np.array([Counter(row).most_common(1)[0][0] for row in predictions])

    def predict_proba(self, X):
        predictions = self.predict(X)
        unique_classes = np.unique(predictions)
        probs = np.zeros((X.shape[0], len(unique_classes)))

        for idx, tree in enumerate(self.trees):
            X_subset = X[:, self.feature_indices[idx]]
            tree_preds = tree.predict(X_subset)

        for i, pred in enumerate(predictions):
            class_counts = Counter(predictions[i])
            for j, cls in enumerate(unique_classes):
                probs[i, j] = class_counts.get(cls, 0) / self.n_estimators

        return probs
