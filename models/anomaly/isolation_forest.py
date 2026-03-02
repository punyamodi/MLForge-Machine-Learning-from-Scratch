import numpy as np


class _IsolationTree:
    class Node:
        def __init__(self, feature_idx=None, threshold=None, left=None, right=None, n_samples=0, is_leaf=False):
            self.feature_idx = feature_idx
            self.threshold = threshold
            self.left = left
            self.right = right
            self.n_samples = n_samples
            self.is_leaf = is_leaf

    def __init__(self, max_depth, random_state=None):
        self.max_depth = max_depth
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.root_ = None

    def c(self, n):
        if n > 2:
            return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
        elif n == 2:
            return 1.0
        else:
            return 0.0

    def fit(self, X, current_depth=0):
        if current_depth >= self.max_depth or X.shape[0] <= 1:
            node = self.Node(n_samples=X.shape[0], is_leaf=True)
            if current_depth == 0:
                self.root_ = node
            return node

        n_features = X.shape[1]
        feat = self.rng.randint(0, n_features)
        col = X[:, feat]
        col_min, col_max = col.min(), col.max()

        if col_min == col_max:
            node = self.Node(n_samples=X.shape[0], is_leaf=True)
            if current_depth == 0:
                self.root_ = node
            return node

        threshold = self.rng.uniform(col_min, col_max)
        left_mask = col <= threshold
        right_mask = ~left_mask

        if not left_mask.any() or not right_mask.any():
            node = self.Node(n_samples=X.shape[0], is_leaf=True)
            if current_depth == 0:
                self.root_ = node
            return node

        left_node = self.fit(X[left_mask], current_depth + 1)
        right_node = self.fit(X[right_mask], current_depth + 1)
        node = self.Node(
            feature_idx=feat,
            threshold=threshold,
            left=left_node,
            right=right_node,
            n_samples=X.shape[0]
        )
        if current_depth == 0:
            self.root_ = node
        return node

    def path_length(self, x, node=None, current_depth=0):
        if node is None:
            node = self.root_
        if node.is_leaf:
            return current_depth + self.c(node.n_samples)
        if x[node.feature_idx] <= node.threshold:
            return self.path_length(x, node.left, current_depth + 1)
        else:
            return self.path_length(x, node.right, current_depth + 1)


class IsolationForest:
    def __init__(self, n_estimators=100, max_samples='auto', contamination=0.1, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

    def _get_max_samples(self, n):
        if self.max_samples == 'auto':
            return min(256, n)
        elif isinstance(self.max_samples, float):
            return int(self.max_samples * n)
        else:
            return int(self.max_samples)

    def fit(self, X):
        n = X.shape[0]
        rng = np.random.RandomState(self.random_state if self.random_state is not None else 42)
        n_subsamples = self._get_max_samples(n)
        self.n_subsamples_ = n_subsamples
        max_depth = int(np.ceil(np.log2(max(n_subsamples, 2))))
        self.trees_ = []
        for i in range(self.n_estimators):
            indices = rng.choice(n, size=n_subsamples, replace=False)
            X_sub = X[indices]
            seed = rng.randint(0, 2**31 - 1)
            tree = _IsolationTree(max_depth=max_depth, random_state=seed)
            tree.fit(X_sub)
            self.trees_.append(tree)
        scores = self.score_samples(X)
        self.offset_ = np.percentile(scores, 100 * self.contamination)
        return self

    def score_samples(self, X):
        n = X.shape[0]
        path_lengths = np.zeros(n)
        for tree in self.trees_:
            for idx in range(n):
                path_lengths[idx] += tree.path_length(X[idx])
        mean_paths = path_lengths / len(self.trees_)
        c_n = self.trees_[0].c(self.n_subsamples_)
        if c_n == 0:
            return np.ones(n) * 0.5
        return np.power(2.0, -mean_paths / c_n)

    def decision_function(self, X):
        return self.score_samples(X) - self.offset_

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def __repr__(self):
        return f"IsolationForest(n_estimators={self.n_estimators}, contamination={self.contamination})"
