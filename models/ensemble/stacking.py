import copy
import numpy as np


class _Stacking:

    def __init__(self, estimators, final_estimator, cv=5, passthrough=False):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.passthrough = passthrough

    def _kfold_indices(self, n, cv):
        indices = np.arange(n)
        fold_sizes = np.full(cv, n // cv, dtype=int)
        fold_sizes[: n % cv] += 1
        folds = []
        current = 0
        for size in fold_sizes:
            folds.append(indices[current : current + size])
            current += size
        result = []
        for k in range(cv):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(cv) if j != k])
            result.append((train_idx, val_idx))
        return result

    def fit(self, X, y):
        raise NotImplementedError

    def _get_meta_features(self, X):
        raise NotImplementedError


class StackingClassifier(_Stacking):

    def fit(self, X, y):
        n = X.shape[0]
        n_estimators = len(self.estimators)
        folds = self._kfold_indices(n, self.cv)

        col_indices = []
        meta_cols = []
        for i, (name, est) in enumerate(self.estimators):
            has_proba = hasattr(est, "predict_proba")
            if has_proba:
                n_classes = len(np.unique(y))
                col = np.zeros((n, n_classes))
            else:
                col = np.zeros((n, 1))

            for train_idx, val_idx in folds:
                fold_est = copy.deepcopy(est)
                fold_est.fit(X[train_idx], y[train_idx])
                if has_proba:
                    col[val_idx] = fold_est.predict_proba(X[val_idx])
                else:
                    col[val_idx, 0] = fold_est.predict(X[val_idx])

            meta_cols.append(col)

        meta_features = np.hstack(meta_cols)

        self.estimators_ = []
        for name, est in self.estimators:
            fitted = copy.deepcopy(est)
            fitted.fit(X, y)
            self.estimators_.append((name, fitted))

        if self.passthrough:
            meta_X = np.hstack([meta_features, X])
        else:
            meta_X = meta_features

        self.final_estimator_ = copy.deepcopy(self.final_estimator)
        self.final_estimator_.fit(meta_X, y)
        return self

    def _get_meta_features(self, X):
        cols = []
        for name, est in self.estimators_:
            if hasattr(est, "predict_proba"):
                cols.append(est.predict_proba(X))
            else:
                cols.append(est.predict(X).reshape(-1, 1))
        return np.hstack(cols)

    def predict(self, X):
        meta = self._get_meta_features(X)
        if self.passthrough:
            meta = np.hstack([meta, X])
        return self.final_estimator_.predict(meta)

    def predict_proba(self, X):
        meta = self._get_meta_features(X)
        if self.passthrough:
            meta = np.hstack([meta, X])
        return self.final_estimator_.predict_proba(meta)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return (
            f"StackingClassifier(n_estimators={len(self.estimators)}, "
            f"cv={self.cv}, "
            f"passthrough={self.passthrough})"
        )


class StackingRegressor(_Stacking):

    def fit(self, X, y):
        n = X.shape[0]
        n_estimators = len(self.estimators)
        folds = self._kfold_indices(n, self.cv)

        meta_features = np.zeros((n, n_estimators))

        for i, (name, est) in enumerate(self.estimators):
            for train_idx, val_idx in folds:
                fold_est = copy.deepcopy(est)
                fold_est.fit(X[train_idx], y[train_idx])
                meta_features[val_idx, i] = fold_est.predict(X[val_idx])

        self.estimators_ = []
        for name, est in self.estimators:
            fitted = copy.deepcopy(est)
            fitted.fit(X, y)
            self.estimators_.append((name, fitted))

        if self.passthrough:
            meta_X = np.hstack([meta_features, X])
        else:
            meta_X = meta_features

        self.final_estimator_ = copy.deepcopy(self.final_estimator)
        self.final_estimator_.fit(meta_X, y)
        return self

    def _get_meta_features(self, X):
        cols = []
        for name, est in self.estimators_:
            cols.append(est.predict(X).reshape(-1, 1))
        return np.hstack(cols)

    def predict(self, X):
        meta = self._get_meta_features(X)
        if self.passthrough:
            meta = np.hstack([meta, X])
        return self.final_estimator_.predict(meta)

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 0.0

    def __repr__(self):
        return (
            f"StackingRegressor(n_estimators={len(self.estimators)}, "
            f"cv={self.cv}, "
            f"passthrough={self.passthrough})"
        )
