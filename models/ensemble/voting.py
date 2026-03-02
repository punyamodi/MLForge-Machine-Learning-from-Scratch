import numpy as np


class VotingClassifier:

    def __init__(self, estimators, voting="hard", weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.named_estimators_ = {}
        for name, est in self.estimators:
            est.fit(X, y)
            self.named_estimators_[name] = est
        return self

    def predict(self, X):
        if self.voting == "soft":
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]

        n = X.shape[0]
        n_est = len(self.estimators)
        weights = (
            np.array(self.weights, dtype=float)
            if self.weights is not None
            else np.ones(n_est, dtype=float)
        )

        all_preds = np.array(
            [est.predict(X) for _, est in self.estimators]
        )

        result = np.empty(n, dtype=all_preds.dtype)
        for j in range(n):
            col = all_preds[:, j]
            vote_totals = {}
            for k, cls in enumerate(col):
                key = cls
                vote_totals[key] = vote_totals.get(key, 0.0) + weights[k]
            result[j] = max(vote_totals, key=lambda k: vote_totals[k])
        return result

    def predict_proba(self, X):
        n_est = len(self.estimators)
        weights = (
            np.array(self.weights, dtype=float)
            if self.weights is not None
            else np.ones(n_est, dtype=float)
        )
        weights = weights / weights.sum()

        proba_sum = None
        for i, (_, est) in enumerate(self.estimators):
            p = est.predict_proba(X)
            if proba_sum is None:
                proba_sum = weights[i] * p
            else:
                proba_sum += weights[i] * p
        return proba_sum

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return (
            f"VotingClassifier(voting={self.voting!r}, "
            f"n_estimators={len(self.estimators)})"
        )


class VotingRegressor:

    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        self.named_estimators_ = {}
        for name, est in self.estimators:
            est.fit(X, y)
            self.named_estimators_[name] = est
        return self

    def predict(self, X):
        n_est = len(self.estimators)
        weights = (
            np.array(self.weights, dtype=float)
            if self.weights is not None
            else np.ones(n_est, dtype=float)
        )
        weights = weights / weights.sum()

        result = None
        for i, (_, est) in enumerate(self.estimators):
            preds = est.predict(X)
            if result is None:
                result = weights[i] * preds
            else:
                result += weights[i] * preds
        return result

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 0.0

    def __repr__(self):
        return f"VotingRegressor(n_estimators={len(self.estimators)})"
