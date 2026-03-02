import numpy as np
from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


class AdaBoostClassifier:
    def __init__(self, n_estimators=50, learning_rate=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = np.array([])
        self.estimator_errors_ = np.array([])
        self.classes_ = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        weights = np.full(n_samples, 1.0 / n_samples)
        self.estimators_ = []
        alphas = []
        errors = []

        for _ in range(self.n_estimators):
            seed = rng.randint(0, 2 ** 31 - 1)
            tree = DecisionTreeClassifier(max_depth=1, random_state=seed)
            indices = rng.choice(n_samples, size=n_samples, replace=True, p=weights)
            tree.fit(X[indices], y[indices])

            y_pred = tree.predict(X)
            incorrect = (y_pred != y).astype(float)
            err = np.dot(weights, incorrect) / weights.sum()
            err = np.clip(err, 1e-10, 1.0 - 1e-10)

            if K == 2:
                alpha = self.learning_rate * np.log((1.0 - err) / err)
            else:
                alpha = self.learning_rate * (np.log((1.0 - err) / err) + np.log(K - 1))

            weights = weights * np.exp(alpha * incorrect)
            weights /= weights.sum()

            self.estimators_.append(tree)
            alphas.append(alpha)
            errors.append(err)

        self.estimator_weights_ = np.array(alphas)
        self.estimator_errors_ = np.array(errors)
        return self

    def predict(self, X):
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        votes = np.zeros((X.shape[0], n_classes))
        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = tree.predict(X)
            for j, p in enumerate(preds):
                votes[j, class_to_idx[p]] += alpha
        return self.classes_[np.argmax(votes, axis=1)]

    def predict_proba(self, X):
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        votes = np.zeros((X.shape[0], n_classes))
        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = tree.predict(X)
            for j, p in enumerate(preds):
                votes[j, class_to_idx[p]] += alpha
        row_sums = votes.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return votes / row_sums

    def staged_predict(self, X):
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        votes = np.zeros((X.shape[0], n_classes))
        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = tree.predict(X)
            for j, p in enumerate(preds):
                votes[j, class_to_idx[p]] += alpha
            yield self.classes_[np.argmax(votes, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return (f"AdaBoostClassifier(n_estimators={self.n_estimators}, "
                f"learning_rate={self.learning_rate})")


class AdaBoostRegressor:
    def __init__(self, n_estimators=50, learning_rate=1.0, loss='linear',
                 random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = np.array([])
        self.estimator_errors_ = np.array([])

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        weights = np.full(n_samples, 1.0 / n_samples)
        self.estimators_ = []
        alphas = []
        errors = []

        for _ in range(self.n_estimators):
            seed = rng.randint(0, 2 ** 31 - 1)
            tree = DecisionTreeRegressor(max_depth=3, random_state=seed)
            indices = rng.choice(n_samples, size=n_samples, replace=True, p=weights)
            tree.fit(X[indices], y[indices])

            y_pred = tree.predict(X)
            raw_errors = np.abs(y - y_pred)
            max_error = raw_errors.max()

            if max_error == 0.0:
                self.estimators_.append(tree)
                alphas.append(1.0)
                errors.append(0.0)
                break

            loss = raw_errors / max_error
            err = np.dot(weights, loss)
            err = np.clip(err, 1e-10, 1.0 - 1e-10)
            beta = err / (1.0 - err)

            weights = weights * np.power(beta, self.learning_rate * (1.0 - loss))
            weights /= weights.sum()

            alpha = np.log(1.0 / beta)
            self.estimators_.append(tree)
            alphas.append(alpha)
            errors.append(err)

        self.estimator_weights_ = np.array(alphas)
        self.estimator_errors_ = np.array(errors)
        return self

    def _weighted_median(self, predictions, weights):
        n_samples = predictions.shape[1]
        result = np.empty(n_samples)
        for i in range(n_samples):
            col = predictions[:, i]
            sorted_idx = np.argsort(col)
            sorted_vals = col[sorted_idx]
            sorted_w = weights[sorted_idx]
            cumulative = np.cumsum(sorted_w)
            median_pos = cumulative[-1] / 2.0
            result[i] = sorted_vals[np.searchsorted(cumulative, median_pos)]
        return result

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return self._weighted_median(predictions, self.estimator_weights_)

    def staged_predict(self, X):
        all_preds = []
        all_weights = []
        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            all_preds.append(tree.predict(X))
            all_weights.append(alpha)
            preds_arr = np.array(all_preds)
            weights_arr = np.array(all_weights)
            yield self._weighted_median(preds_arr, weights_arr)

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)

    def __repr__(self):
        return (f"AdaBoostRegressor(n_estimators={self.n_estimators}, "
                f"learning_rate={self.learning_rate}, "
                f"loss={self.loss!r})")
