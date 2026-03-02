import numpy as np


class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None,
                 value=None, n_samples=0, impurity=0.0):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.n_samples = n_samples
        self.impurity = impurity

    def is_leaf(self):
        return self.value is not None


class _DecisionTree:
    def __init__(self, criterion, max_depth, min_samples_split, min_samples_leaf,
                 max_features, random_state):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.n_features_ = None

    def _get_n_features(self, n_features):
        if self.max_features is None:
            return n_features
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        return n_features

    def _split(self, X, feature_idx, threshold):
        left_mask = X[:, feature_idx] <= threshold
        return left_mask, ~left_mask

    def _impurity(self, y):
        raise NotImplementedError

    def _leaf_value(self, y):
        raise NotImplementedError

    def _best_split(self, X, y, rng):
        n_samples, n_features = X.shape
        n_feats = self._get_n_features(n_features)
        feature_indices = rng.choice(n_features, size=n_feats, replace=False)
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        parent_impurity = self._impurity(y)

        for feat_idx in feature_indices:
            thresholds = np.unique(X[:, feat_idx])
            if len(thresholds) == 1:
                continue
            midpoints = (thresholds[:-1] + thresholds[1:]) / 2.0

            for threshold in midpoints:
                left_mask, right_mask = self._split(X, feat_idx, threshold)
                n_left = left_mask.sum()
                n_right = right_mask.sum()
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                left_imp = self._impurity(y[left_mask])
                right_imp = self._impurity(y[right_mask])
                gain = parent_impurity - (n_left / n_samples * left_imp + n_right / n_samples * right_imp)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth, rng):
        n_samples = len(y)
        node = Node(n_samples=n_samples, impurity=self._impurity(y))

        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            node.value = self._leaf_value(y)
            return node

        feature_idx, threshold, gain = self._best_split(X, y, rng)

        if feature_idx is None or gain <= 0:
            node.value = self._leaf_value(y)
            return node

        left_mask, right_mask = self._split(X, feature_idx, threshold)
        node.feature_idx = feature_idx
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1, rng)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1, rng)
        return node

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.n_features_ = X.shape[1]
        self.root = self._build_tree(X, y, 0, rng)
        self._compute_feature_importances()
        return self

    def _predict_sample(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _compute_feature_importances(self):
        importances = np.zeros(self.n_features_)

        def _traverse(node):
            if node.is_leaf():
                return
            imp_gain = node.n_samples * node.impurity - \
                       node.left.n_samples * node.left.impurity - \
                       node.right.n_samples * node.right.impurity
            importances[node.feature_idx] += imp_gain
            _traverse(node.left)
            _traverse(node.right)

        _traverse(self.root)
        total = importances.sum()
        if total > 0:
            importances /= total
        self.feature_importances_ = importances


class DecisionTreeClassifier(_DecisionTree):
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, random_state=None):
        super().__init__(criterion, max_depth, min_samples_split, min_samples_leaf,
                         max_features, random_state)
        self.classes_ = None

    def _impurity(self, y):
        if len(y) == 0:
            return 0.0
        classes, counts = np.unique(y, return_counts=True)
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

    def __repr__(self):
        return (f"DecisionTreeClassifier(criterion={self.criterion!r}, "
                f"max_depth={self.max_depth}, "
                f"min_samples_split={self.min_samples_split})")


class DecisionTreeRegressor(_DecisionTree):
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

    def __repr__(self):
        return (f"DecisionTreeRegressor(criterion={self.criterion!r}, "
                f"max_depth={self.max_depth}, "
                f"min_samples_split={self.min_samples_split})")
