import numpy as np
from .decision_tree import _DecisionTree, Node


class _ExtraDecisionTree(_DecisionTree):
    def _best_split(self, X, y, rng):
        n_samples, n_features = X.shape
        n_feats = self._get_n_features(n_features)
        feature_indices = rng.choice(n_features, size=n_feats, replace=False)
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        parent_impurity = self._impurity(y)

        for feat_idx in feature_indices:
            feat_min = X[:, feat_idx].min()
            feat_max = X[:, feat_idx].max()
            if feat_min == feat_max:
                continue
            threshold = rng.uniform(feat_min, feat_max)
            left_mask, right_mask = self._split(X, feat_idx, threshold)
            n_left = left_mask.sum()
            n_right = right_mask.sum()
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            left_imp = self._impurity(y[left_mask])
            right_imp = self._impurity(y[right_mask])
            gain = parent_impurity - (
                n_left / n_samples * left_imp + n_right / n_samples * right_imp
            )
            if gain > best_gain:
                best_gain = gain
                best_feature = feat_idx
                best_threshold = threshold

        return best_feature, best_threshold, best_gain


class _ExtraDecisionTreeClassifier(_ExtraDecisionTree):
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, random_state=None):
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf,
                         max_features, random_state)
        self.classes_ = None

    def _impurity(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        if self.criterion == 'gini':
            return 1.0 - np.sum(probs ** 2)
        elif self.criterion == 'entropy':
            probs = probs[probs > 0]
            return -np.sum(probs * np.log2(probs))
        return 0.0

    def _leaf_value(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return super().fit(X, y)

    def predict_proba(self, X):
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}

        def _proba_sample(x, node):
            if node.is_leaf():
                proba = np.zeros(n_classes)
                proba[class_to_idx[node.value]] = 1.0
                return proba
            if x[node.feature_idx] <= node.threshold:
                return _proba_sample(x, node.left)
            return _proba_sample(x, node.right)

        return np.array([_proba_sample(x, self.root) for x in X])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class _ExtraDecisionTreeRegressor(_ExtraDecisionTree):
    def __init__(self, criterion='mse', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, random_state=None):
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf,
                         max_features, random_state)

    def _impurity(self, y):
        if len(y) == 0:
            return 0.0
        if self.criterion == 'mse':
            return np.var(y)
        elif self.criterion == 'mae':
            return np.mean(np.abs(y - np.mean(y)))
        return 0.0

    def _leaf_value(self, y):
        return np.mean(y)

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)


class ExtraTreesClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=False,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimators_ = []
        self.classes_ = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.estimators_ = []

        for _ in range(self.n_estimators):
            seed = rng.randint(0, 2 ** 31 - 1)
            tree = _ExtraDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed,
            )
            if self.bootstrap:
                indices = rng.choice(n_samples, size=n_samples, replace=True)
            else:
                indices = np.arange(n_samples)
            tree.fit(X[indices], y[indices])
            self.estimators_.append(tree)

        importances = np.zeros(n_features)
        for tree in self.estimators_:
            importances += tree.feature_importances_
        self.feature_importances_ = importances / self.n_estimators
        return self

    def predict_proba(self, X):
        all_proba = np.zeros((X.shape[0], len(self.classes_)))
        for tree in self.estimators_:
            all_proba += tree.predict_proba(X)
        return all_proba / self.n_estimators

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return (f"ExtraTreesClassifier(n_estimators={self.n_estimators}, "
                f"max_depth={self.max_depth}, "
                f"max_features={self.max_features!r})")


class ExtraTreesRegressor:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=False,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.estimators_ = []

        for _ in range(self.n_estimators):
            seed = rng.randint(0, 2 ** 31 - 1)
            tree = _ExtraDecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed,
            )
            if self.bootstrap:
                indices = rng.choice(n_samples, size=n_samples, replace=True)
            else:
                indices = np.arange(n_samples)
            tree.fit(X[indices], y[indices])
            self.estimators_.append(tree)

        importances = np.zeros(n_features)
        for tree in self.estimators_:
            importances += tree.feature_importances_
        self.feature_importances_ = importances / self.n_estimators
        return self

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.estimators_])
        return np.mean(preds, axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)

    def __repr__(self):
        return (f"ExtraTreesRegressor(n_estimators={self.n_estimators}, "
                f"max_depth={self.max_depth}, "
                f"max_features={self.max_features!r})")
