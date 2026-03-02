import copy
import numpy as np


class _Bagging:

    def __init__(
        self,
        base_estimator,
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        random_state=None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.random_state = random_state

    def _get_n_samples(self, n):
        if isinstance(self.max_samples, float):
            return int(self.max_samples * n)
        return int(self.max_samples)

    def _get_n_features(self, p):
        if isinstance(self.max_features, float):
            return int(self.max_features * p)
        return int(self.max_features)

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n, p = X.shape
        n_samples = self._get_n_samples(n)
        n_features = self._get_n_features(p)

        self.estimators_ = []
        self.estimators_features_ = []
        self.estimators_samples_ = []

        for _ in range(self.n_estimators):
            if self.bootstrap:
                row_idx = rng.randint(0, n, size=n_samples)
            else:
                row_idx = rng.choice(n, size=n_samples, replace=False)

            if self.bootstrap_features:
                feat_idx = rng.randint(0, p, size=n_features)
            else:
                feat_idx = rng.choice(p, size=n_features, replace=False)

            est = copy.deepcopy(self.base_estimator)
            est.fit(X[row_idx][:, feat_idx], y[row_idx])

            self.estimators_.append(est)
            self.estimators_features_.append(feat_idx)
            self.estimators_samples_.append(row_idx)

        if self.oob_score:
            self._set_oob_score(X, y, n)

        return self

    def _set_oob_score(self, X, y, n):
        raise NotImplementedError


class BaggingClassifier(_Bagging):

    def _set_oob_score(self, X, y, n):
        all_indices = np.arange(n)
        oob_preds = {}

        for i, est in enumerate(self.estimators_):
            in_bag = set(self.estimators_samples_[i].tolist())
            oob_idx = np.array([j for j in all_indices if j not in in_bag])
            if len(oob_idx) == 0:
                continue
            feat_idx = self.estimators_features_[i]
            preds = est.predict(X[oob_idx][:, feat_idx])
            for idx, pred in zip(oob_idx, preds):
                if idx not in oob_preds:
                    oob_preds[idx] = []
                oob_preds[idx].append(pred)

        if len(oob_preds) == 0:
            self.oob_score_ = 0.0
            return

        oob_indices = sorted(oob_preds.keys())
        final_preds = []
        true_labels = []
        for idx in oob_indices:
            votes = oob_preds[idx]
            values, counts = np.unique(votes, return_counts=True)
            final_preds.append(values[np.argmax(counts)])
            true_labels.append(y[idx])

        final_preds = np.array(final_preds)
        true_labels = np.array(true_labels)
        self.oob_score_ = np.mean(final_preds == true_labels)

    def predict(self, X):
        all_preds = []
        for est, feat_idx in zip(self.estimators_, self.estimators_features_):
            preds = est.predict(X[:, feat_idx])
            all_preds.append(preds)
        all_preds = np.array(all_preds)
        n = X.shape[0]
        result = np.empty(n, dtype=all_preds.dtype)
        for j in range(n):
            col = all_preds[:, j]
            values, counts = np.unique(col, return_counts=True)
            result[j] = values[np.argmax(counts)]
        return result

    def predict_proba(self, X):
        proba_sum = None
        count = 0
        for est, feat_idx in zip(self.estimators_, self.estimators_features_):
            if not hasattr(est, "predict_proba"):
                continue
            p = est.predict_proba(X[:, feat_idx])
            if proba_sum is None:
                proba_sum = p.copy()
            else:
                proba_sum += p
            count += 1
        if count == 0:
            raise AttributeError("No estimators support predict_proba.")
        return proba_sum / count

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return (
            f"BaggingClassifier(n_estimators={self.n_estimators}, "
            f"max_samples={self.max_samples}, "
            f"max_features={self.max_features})"
        )


class BaggingRegressor(_Bagging):

    def _set_oob_score(self, X, y, n):
        all_indices = np.arange(n)
        oob_preds = {}

        for i, est in enumerate(self.estimators_):
            in_bag = set(self.estimators_samples_[i].tolist())
            oob_idx = np.array([j for j in all_indices if j not in in_bag])
            if len(oob_idx) == 0:
                continue
            feat_idx = self.estimators_features_[i]
            preds = est.predict(X[oob_idx][:, feat_idx])
            for idx, pred in zip(oob_idx, preds):
                if idx not in oob_preds:
                    oob_preds[idx] = []
                oob_preds[idx].append(pred)

        if len(oob_preds) == 0:
            self.oob_score_ = 0.0
            return

        oob_indices = sorted(oob_preds.keys())
        final_preds = np.array([np.mean(oob_preds[idx]) for idx in oob_indices])
        true_vals = np.array([y[idx] for idx in oob_indices])

        ss_res = np.sum((true_vals - final_preds) ** 2)
        ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
        self.oob_score_ = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 0.0

    def predict(self, X):
        all_preds = []
        for est, feat_idx in zip(self.estimators_, self.estimators_features_):
            all_preds.append(est.predict(X[:, feat_idx]))
        return np.mean(np.array(all_preds), axis=0)

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0.0 else 0.0

    def __repr__(self):
        return (
            f"BaggingRegressor(n_estimators={self.n_estimators}, "
            f"max_samples={self.max_samples}, "
            f"max_features={self.max_features})"
        )
