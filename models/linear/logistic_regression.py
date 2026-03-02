import numpy as np


class LogisticRegression:
    def __init__(self, C=1.0, penalty='l2', max_iter=1000, tol=1e-4,
                 learning_rate=0.01, multi_class='ovr', random_state=None):
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.multi_class = multi_class
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _fit_binary(self, X, y_binary):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        w = rng.randn(n_features) * 0.01
        b = 0.0
        for i in range(self.max_iter):
            z = X @ w + b
            p = self._sigmoid(z)
            error = p - y_binary
            grad_w = X.T @ error / n_samples
            grad_b = np.mean(error)
            if self.penalty == 'l2':
                grad_w += (1.0 / self.C) * w
            elif self.penalty == 'l1':
                grad_w += (1.0 / self.C) * np.sign(w)
            w_new = w - self.learning_rate * grad_w
            b_new = b - self.learning_rate * grad_b
            if np.linalg.norm(w_new - w) < self.tol:
                w, b = w_new, b_new
                break
            w, b = w_new, b_new
        return w, b

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes == 2:
            y_binary = (y == self.classes_[1]).astype(float)
            w, b = self._fit_binary(X, y_binary)
            self.coef_ = w.reshape(1, -1)
            self.intercept_ = np.array([b])
        else:
            self.coef_ = np.zeros((n_classes, X.shape[1]))
            self.intercept_ = np.zeros(n_classes)
            for idx, cls in enumerate(self.classes_):
                y_binary = (y == cls).astype(float)
                w, b = self._fit_binary(X, y_binary)
                self.coef_[idx] = w
                self.intercept_[idx] = b
        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        n_classes = len(self.classes_)
        if n_classes == 2:
            p_pos = self._sigmoid(X @ self.coef_[0] + self.intercept_[0])
            return np.column_stack([1.0 - p_pos, p_pos])
        else:
            scores = np.zeros((X.shape[0], n_classes))
            for idx in range(n_classes):
                scores[:, idx] = self._sigmoid(X @ self.coef_[idx] + self.intercept_[idx])
            row_sums = scores.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            return scores / row_sums

    def predict(self, X):
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return (
            f"LogisticRegression(C={self.C}, penalty={self.penalty!r}, "
            f"max_iter={self.max_iter}, learning_rate={self.learning_rate}, "
            f"multi_class={self.multi_class!r})"
        )
