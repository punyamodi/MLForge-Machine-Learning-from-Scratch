import numpy as np
from .decision_tree import DecisionTreeRegressor


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        self.estimators_ = []
        self.init_prediction_ = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.init_prediction_ = np.mean(y)
        F = np.full(n_samples, self.init_prediction_)
        self.estimators_ = []

        for _ in range(self.n_estimators):
            residuals = y - F
            if self.subsample < 1.0:
                n_sub = max(1, int(self.subsample * n_samples))
                indices = rng.choice(n_samples, size=n_sub, replace=False)
            else:
                indices = np.arange(n_samples)

            seed = rng.randint(0, 2 ** 31 - 1)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=seed,
            )
            tree.fit(X[indices], residuals[indices])
            self.estimators_.append(tree)
            F += self.learning_rate * tree.predict(X)

        importances = np.zeros(n_features)
        for tree in self.estimators_:
            importances += tree.feature_importances_
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict(self, X):
        F = np.full(X.shape[0], self.init_prediction_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        return F

    def staged_predict(self, X):
        F = np.full(X.shape[0], self.init_prediction_)
        for tree in self.estimators_:
            F = F + self.learning_rate * tree.predict(X)
            yield F.copy()

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)

    def __repr__(self):
        return (f"GradientBoostingRegressor(n_estimators={self.n_estimators}, "
                f"learning_rate={self.learning_rate}, "
                f"max_depth={self.max_depth})")


class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        self.estimators_ = []
        self.init_prediction_ = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        p = np.clip(np.mean(y), 1e-10, 1.0 - 1e-10)
        self.init_prediction_ = np.log(p / (1.0 - p))
        F = np.full(n_samples, self.init_prediction_)
        self.estimators_ = []

        for _ in range(self.n_estimators):
            residuals = y - _sigmoid(F)
            if self.subsample < 1.0:
                n_sub = max(1, int(self.subsample * n_samples))
                indices = rng.choice(n_samples, size=n_sub, replace=False)
            else:
                indices = np.arange(n_samples)

            seed = rng.randint(0, 2 ** 31 - 1)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=seed,
            )
            tree.fit(X[indices], residuals[indices])
            self.estimators_.append(tree)
            F += self.learning_rate * tree.predict(X)

        importances = np.zeros(n_features)
        for tree in self.estimators_:
            importances += tree.feature_importances_
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0 else importances
        return self

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.init_prediction_)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        proba_pos = _sigmoid(F)
        return np.column_stack([1.0 - proba_pos, proba_pos])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def staged_predict(self, X):
        F = np.full(X.shape[0], self.init_prediction_)
        for tree in self.estimators_:
            F = F + self.learning_rate * tree.predict(X)
            yield (_sigmoid(F) >= 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return (f"GradientBoostingClassifier(n_estimators={self.n_estimators}, "
                f"learning_rate={self.learning_rate}, "
                f"max_depth={self.max_depth})")
