import numpy as np
from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


class _RandomForest:
    def __init__(self, n_estimators, max_depth, min_samples_split, min_samples_leaf,
                 max_features, bootstrap, oob_score, random_state):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.estimators_ = []

    def _make_estimator(self, random_state):
        raise NotImplementedError

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.estimators_ = []
        oob_predictions = np.zeros(n_samples) if self.oob_score else None
        oob_counts = np.zeros(n_samples) if self.oob_score else None

        for _ in range(self.n_estimators):
            seed = rng.randint(0, 2 ** 31 - 1)
            tree = self._make_estimator(seed)
            if self.bootstrap:
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                oob_indices = np.setdiff1d(np.arange(n_samples), indices)
            else:
                indices = np.arange(n_samples)
                oob_indices = np.array([], dtype=int)

            tree.fit(X[indices], y[indices])
            self.estimators_.append(tree)

            if self.oob_score and len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                oob_predictions[oob_indices] += oob_pred
                oob_counts[oob_indices] += 1

        if self.oob_score:
            mask = oob_counts > 0
            oob_predictions[mask] /= oob_counts[mask]
            self._compute_oob_score(y, oob_predictions, mask)

        importances = np.zeros(n_features)
        for tree in self.estimators_:
            importances += tree.feature_importances_
        self.feature_importances_ = importances / self.n_estimators

        return self

    def _compute_oob_score(self, y, oob_predictions, mask):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class RandomForestClassifier(_RandomForest):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                 oob_score=False, random_state=None):
        super().__init__(n_estimators, max_depth, min_samples_split, min_samples_leaf,
                         max_features, bootstrap, oob_score, random_state)
        self.classes_ = None

    def _make_estimator(self, random_state):
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=random_state,
        )

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return super().fit(X, y)

    def _compute_oob_score(self, y, oob_predictions, mask):
        self.oob_score_ = np.mean(oob_predictions[mask] == y[mask])

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
        return (f"RandomForestClassifier(n_estimators={self.n_estimators}, "
                f"max_depth={self.max_depth}, "
                f"max_features={self.max_features!r})")


class RandomForestRegressor(_RandomForest):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                 oob_score=False, random_state=None):
        super().__init__(n_estimators, max_depth, min_samples_split, min_samples_leaf,
                         max_features, bootstrap, oob_score, random_state)

    def _make_estimator(self, random_state):
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=random_state,
        )

    def _compute_oob_score(self, y, oob_predictions, mask):
        y_true = y[mask]
        y_pred = oob_predictions[mask]
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        self.oob_score_ = 1.0 - ss_res / (ss_tot + 1e-10)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.estimators_])
        return np.mean(preds, axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)

    def __repr__(self):
        return (f"RandomForestRegressor(n_estimators={self.n_estimators}, "
                f"max_depth={self.max_depth}, "
                f"max_features={self.max_features!r})")
