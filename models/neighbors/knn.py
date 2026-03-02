import numpy as np


class _BaseKNN:
    def _compute_distances(self, X1, X2, metric):
        if metric == 'euclidean':
            X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
            X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
            cross = X1 @ X2.T
            dist_sq = X1_sq + X2_sq.T - 2 * cross
            dist_sq = np.maximum(dist_sq, 0.0)
            return np.sqrt(dist_sq)
        elif metric == 'manhattan':
            n1 = X1.shape[0]
            n2 = X2.shape[0]
            dists = np.empty((n1, n2), dtype=np.float64)
            for i in range(n1):
                dists[i] = np.sum(np.abs(X1[i] - X2), axis=1)
            return dists
        elif metric == 'chebyshev':
            n1 = X1.shape[0]
            n2 = X2.shape[0]
            dists = np.empty((n1, n2), dtype=np.float64)
            for i in range(n1):
                dists[i] = np.max(np.abs(X1[i] - X2), axis=1)
            return dists
        elif metric == 'minkowski':
            p = getattr(self, 'p', 2)
            n1 = X1.shape[0]
            n2 = X2.shape[0]
            dists = np.empty((n1, n2), dtype=np.float64)
            for i in range(n1):
                dists[i] = np.sum(np.abs(X1[i] - X2) ** p, axis=1) ** (1.0 / p)
            return dists
        else:
            raise ValueError("Unknown metric: {}".format(metric))


class KNeighborsClassifier(_BaseKNN):
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        self.X_train_ = np.array(X, dtype=np.float64)
        self.y_train_ = np.array(y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def kneighbors(self, X, n_neighbors=None):
        X = np.array(X, dtype=np.float64)
        k = n_neighbors if n_neighbors is not None else self.n_neighbors
        dists = self._compute_distances(X, self.X_train_, self.metric)
        indices = np.argsort(dists, axis=1)[:, :k]
        sorted_dists = np.take_along_axis(dists, indices, axis=1)
        return sorted_dists, indices

    def predict(self, X):
        dists, indices = self.kneighbors(X)
        y_neighbors = self.y_train_[indices]
        predictions = np.empty(X.shape[0] if hasattr(X, 'shape') else len(X), dtype=self.classes_.dtype)
        X = np.array(X, dtype=np.float64)
        for i in range(X.shape[0]):
            neighbor_labels = y_neighbors[i]
            neighbor_dists = dists[i]
            if self.weights == 'uniform':
                counts = {c: 0 for c in self.classes_}
                for label in neighbor_labels:
                    counts[label] += 1
                predictions[i] = max(counts, key=lambda c: counts[c])
            else:
                weights = np.where(neighbor_dists == 0, np.inf, 1.0 / neighbor_dists)
                has_zero = np.any(neighbor_dists == 0)
                class_weights = {c: 0.0 for c in self.classes_}
                for j, label in enumerate(neighbor_labels):
                    if has_zero:
                        class_weights[label] += (np.inf if neighbor_dists[j] == 0 else 0.0)
                    else:
                        class_weights[label] += weights[j]
                predictions[i] = max(class_weights, key=lambda c: class_weights[c])
        return predictions

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        dists, indices = self.kneighbors(X)
        y_neighbors = self.y_train_[indices]
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        class_index = {c: idx for idx, c in enumerate(self.classes_)}
        proba = np.zeros((n_samples, n_classes), dtype=np.float64)
        for i in range(n_samples):
            neighbor_labels = y_neighbors[i]
            neighbor_dists = dists[i]
            if self.weights == 'uniform':
                for label in neighbor_labels:
                    proba[i, class_index[label]] += 1.0
                proba[i] /= len(neighbor_labels)
            else:
                has_zero = np.any(neighbor_dists == 0)
                for j, label in enumerate(neighbor_labels):
                    if has_zero:
                        w = np.inf if neighbor_dists[j] == 0 else 0.0
                    else:
                        w = 1.0 / neighbor_dists[j]
                    proba[i, class_index[label]] += w
                total = np.sum(proba[i])
                if total > 0:
                    proba[i] /= total
        return proba

    def score(self, X, y):
        predictions = self.predict(X)
        y = np.array(y)
        return np.mean(predictions == y)

    def __repr__(self):
        return "KNeighborsClassifier(n_neighbors={}, weights={}, metric={})".format(
            self.n_neighbors, self.weights, self.metric
        )


class KNeighborsRegressor(_BaseKNN):
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        self.X_train_ = np.array(X, dtype=np.float64)
        self.y_train_ = np.array(y, dtype=np.float64)
        return self

    def kneighbors(self, X, n_neighbors=None):
        X = np.array(X, dtype=np.float64)
        k = n_neighbors if n_neighbors is not None else self.n_neighbors
        dists = self._compute_distances(X, self.X_train_, self.metric)
        indices = np.argsort(dists, axis=1)[:, :k]
        sorted_dists = np.take_along_axis(dists, indices, axis=1)
        return sorted_dists, indices

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        dists, indices = self.kneighbors(X)
        y_neighbors = self.y_train_[indices]
        n_samples = X.shape[0]
        predictions = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            neighbor_vals = y_neighbors[i]
            neighbor_dists = dists[i]
            if self.weights == 'uniform':
                predictions[i] = np.mean(neighbor_vals)
            else:
                has_zero = np.any(neighbor_dists == 0)
                if has_zero:
                    zero_mask = neighbor_dists == 0
                    predictions[i] = np.mean(neighbor_vals[zero_mask])
                else:
                    w = 1.0 / neighbor_dists
                    predictions[i] = np.sum(w * neighbor_vals) / np.sum(w)
        return predictions

    def score(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot

    def __repr__(self):
        return "KNeighborsRegressor(n_neighbors={}, weights={}, metric={})".format(
            self.n_neighbors, self.weights, self.metric
        )


class RadiusNeighborsClassifier(_BaseKNN):
    def __init__(self, radius=1.0, weights='uniform', metric='euclidean', outlier_label=None):
        self.radius = radius
        self.weights = weights
        self.metric = metric
        self.outlier_label = outlier_label

    def fit(self, X, y):
        self.X_train_ = np.array(X, dtype=np.float64)
        self.y_train_ = np.array(y)
        self.classes_ = np.unique(self.y_train_)
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        dists = self._compute_distances(X, self.X_train_, self.metric)
        n_samples = X.shape[0]
        predictions = np.empty(n_samples, dtype=self.classes_.dtype)
        class_index = {c: idx for idx, c in enumerate(self.classes_)}
        for i in range(n_samples):
            mask = dists[i] <= self.radius
            if not np.any(mask):
                predictions[i] = self.outlier_label
                continue
            neighbor_labels = self.y_train_[mask]
            neighbor_dists = dists[i][mask]
            if self.weights == 'uniform':
                counts = {c: 0 for c in self.classes_}
                for label in neighbor_labels:
                    counts[label] += 1
                predictions[i] = max(counts, key=lambda c: counts[c])
            else:
                has_zero = np.any(neighbor_dists == 0)
                class_weights = {c: 0.0 for c in self.classes_}
                for j, label in enumerate(neighbor_labels):
                    if has_zero:
                        class_weights[label] += (np.inf if neighbor_dists[j] == 0 else 0.0)
                    else:
                        class_weights[label] += 1.0 / neighbor_dists[j]
                predictions[i] = max(class_weights, key=lambda c: class_weights[c])
        return predictions

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        dists = self._compute_distances(X, self.X_train_, self.metric)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        class_index = {c: idx for idx, c in enumerate(self.classes_)}
        proba = np.zeros((n_samples, n_classes), dtype=np.float64)
        for i in range(n_samples):
            mask = dists[i] <= self.radius
            if not np.any(mask):
                proba[i] = np.full(n_classes, 1.0 / n_classes)
                continue
            neighbor_labels = self.y_train_[mask]
            neighbor_dists = dists[i][mask]
            if self.weights == 'uniform':
                for label in neighbor_labels:
                    proba[i, class_index[label]] += 1.0
                proba[i] /= len(neighbor_labels)
            else:
                has_zero = np.any(neighbor_dists == 0)
                for j, label in enumerate(neighbor_labels):
                    if has_zero:
                        w = np.inf if neighbor_dists[j] == 0 else 0.0
                    else:
                        w = 1.0 / neighbor_dists[j]
                    proba[i, class_index[label]] += w
                total = np.sum(proba[i])
                if total > 0:
                    proba[i] /= total
        return proba

    def score(self, X, y):
        predictions = self.predict(X)
        y = np.array(y)
        return np.mean(predictions == y)

    def __repr__(self):
        return "RadiusNeighborsClassifier(radius={}, weights={}, metric={})".format(
            self.radius, self.weights, self.metric
        )


class RadiusNeighborsRegressor(_BaseKNN):
    def __init__(self, radius=1.0, weights='uniform', metric='euclidean'):
        self.radius = radius
        self.weights = weights
        self.metric = metric

    def fit(self, X, y):
        self.X_train_ = np.array(X, dtype=np.float64)
        self.y_train_ = np.array(y, dtype=np.float64)
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        dists = self._compute_distances(X, self.X_train_, self.metric)
        n_samples = X.shape[0]
        predictions = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            mask = dists[i] <= self.radius
            if not np.any(mask):
                predictions[i] = np.mean(self.y_train_)
                continue
            neighbor_vals = self.y_train_[mask]
            neighbor_dists = dists[i][mask]
            if self.weights == 'uniform':
                predictions[i] = np.mean(neighbor_vals)
            else:
                has_zero = np.any(neighbor_dists == 0)
                if has_zero:
                    zero_mask = neighbor_dists == 0
                    predictions[i] = np.mean(neighbor_vals[zero_mask])
                else:
                    w = 1.0 / neighbor_dists
                    predictions[i] = np.sum(w * neighbor_vals) / np.sum(w)
        return predictions

    def score(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1.0 - ss_res / ss_tot

    def __repr__(self):
        return "RadiusNeighborsRegressor(radius={}, weights={}, metric={})".format(
            self.radius, self.weights, self.metric
        )
