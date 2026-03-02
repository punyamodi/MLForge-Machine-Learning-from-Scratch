import numpy as np


class _SGDBase:
    def __init__(self, loss, penalty='l2', alpha=0.0001, learning_rate='invscaling',
                 eta0=0.01, power_t=0.5, max_iter=1000, tol=1e-3, batch_size=32,
                 random_state=None):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.max_iter = max_iter
        self.tol = tol
        self.batch_size = batch_size
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = np.zeros(1)
        self.loss_curve_ = []

    def _get_learning_rate(self, t):
        if self.learning_rate == 'constant':
            return self.eta0
        elif self.learning_rate == 'optimal':
            return self.eta0 / (1.0 + self.alpha * t)
        elif self.learning_rate == 'invscaling':
            return self.eta0 / (t ** self.power_t)
        elif self.learning_rate == 'adaptive':
            return self.eta0
        else:
            return self.eta0

    def _apply_penalty(self, coef, lr):
        if self.penalty == 'l2':
            coef *= (1.0 - lr * self.alpha)
        elif self.penalty == 'l1':
            threshold = lr * self.alpha
            coef = np.sign(coef) * np.maximum(np.abs(coef) - threshold, 0.0)
        elif self.penalty == 'elasticnet':
            threshold = lr * self.alpha * 0.5
            coef *= (1.0 - lr * self.alpha * 0.5)
            coef = np.sign(coef) * np.maximum(np.abs(coef) - threshold, 0.0)
        return coef


class SGDClassifier(_SGDBase):
    def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, learning_rate='invscaling',
                 eta0=0.01, power_t=0.5, max_iter=1000, tol=1e-3, batch_size=32,
                 random_state=None):
        super().__init__(loss=loss, penalty=penalty, alpha=alpha, learning_rate=learning_rate,
                         eta0=eta0, power_t=power_t, max_iter=max_iter, tol=tol,
                         batch_size=batch_size, random_state=random_state)
        self.classes_ = None
        self._label_map = None
        self._inv_label_map = None

    def _compute_loss_and_grad(self, X_batch, y_batch, coef, intercept):
        f = X_batch @ coef + intercept
        n = X_batch.shape[0]
        if self.loss == 'hinge':
            margin = y_batch * f
            mask = margin < 1.0
            loss_val = np.mean(np.maximum(0.0, 1.0 - margin))
            grad_coef = -X_batch.T @ (y_batch * mask) / n
            grad_intercept = -np.mean(y_batch * mask)
        elif self.loss == 'log':
            z = np.clip(y_batch * f, -500, 500)
            loss_val = np.mean(np.log1p(np.exp(-z)))
            p = 1.0 / (1.0 + np.exp(-z))
            grad_coef = -X_batch.T @ (y_batch * (1.0 - p)) / n
            grad_intercept = -np.mean(y_batch * (1.0 - p))
        elif self.loss == 'modified_huber':
            yf = y_batch * f
            loss_parts = np.where(yf >= 1.0, 0.0,
                          np.where(yf >= -1.0, (1.0 - yf) ** 2, -4.0 * yf))
            loss_val = np.mean(loss_parts)
            grad_parts = np.where(yf >= 1.0, 0.0,
                          np.where(yf >= -1.0, -2.0 * (1.0 - yf) * y_batch, -4.0 * y_batch))
            grad_coef = X_batch.T @ grad_parts / n
            grad_intercept = np.mean(grad_parts)
        else:
            f_diff = f - y_batch
            loss_val = np.mean(f_diff ** 2) * 0.5
            grad_coef = X_batch.T @ f_diff / n
            grad_intercept = np.mean(f_diff)
        return loss_val, grad_coef, grad_intercept

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        rng = np.random.RandomState(self.random_state)

        self.classes_ = np.unique(y)
        self._label_map = {cls: (1 if i == 1 else -1) for i, cls in enumerate(self.classes_)}
        self._inv_label_map = {v: k for k, v in self._label_map.items()}
        if len(self.classes_) == 2:
            self._inv_label_map[1] = self.classes_[1]
            self._inv_label_map[-1] = self.classes_[0]

        y_enc = np.array([self._label_map[lbl] for lbl in y], dtype=float)
        n_samples, n_features = X.shape
        self.coef_ = rng.randn(n_features) * 0.01
        self.intercept_ = np.zeros(1)
        self.loss_curve_ = []

        t = 1
        for epoch in range(self.max_iter):
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_enc[indices]
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_b = X_shuffled[start:end]
                y_b = y_shuffled[start:end]
                lr = self._get_learning_rate(t)
                loss_val, grad_coef, grad_intercept = self._compute_loss_and_grad(
                    X_b, y_b, self.coef_, self.intercept_[0]
                )
                self.coef_ -= lr * grad_coef
                self.coef_ = self._apply_penalty(self.coef_, lr)
                self.intercept_[0] -= lr * grad_intercept
                epoch_loss += loss_val
                n_batches += 1
                t += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_curve_.append(avg_loss)
            if len(self.loss_curve_) > 1 and abs(self.loss_curve_[-2] - avg_loss) < self.tol:
                break

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        scores = X @ self.coef_ + self.intercept_[0]
        raw = np.sign(scores)
        raw = np.where(raw == 0, 1, raw)
        return np.array([self._inv_label_map.get(int(s), self.classes_[0]) for s in raw])

    def predict_proba(self, X):
        if self.loss != 'log':
            raise ValueError("predict_proba is only available when loss='log'")
        X = np.array(X, dtype=float)
        z = np.clip(X @ self.coef_ + self.intercept_[0], -500, 500)
        p_pos = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1.0 - p_pos, p_pos])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))

    def __repr__(self):
        return (
            f"SGDClassifier(loss={self.loss!r}, penalty={self.penalty!r}, "
            f"alpha={self.alpha}, learning_rate={self.learning_rate!r}, "
            f"eta0={self.eta0}, max_iter={self.max_iter})"
        )


class SGDRegressor(_SGDBase):
    def __init__(self, loss='squared', penalty='l2', alpha=0.0001, learning_rate='invscaling',
                 eta0=0.01, power_t=0.5, max_iter=1000, tol=1e-3, batch_size=32,
                 random_state=None, epsilon=0.1):
        super().__init__(loss=loss, penalty=penalty, alpha=alpha, learning_rate=learning_rate,
                         eta0=eta0, power_t=power_t, max_iter=max_iter, tol=tol,
                         batch_size=batch_size, random_state=random_state)
        self.epsilon = epsilon

    def _compute_loss_and_grad(self, X_batch, y_batch, coef, intercept):
        f = X_batch @ coef + intercept
        residuals = f - y_batch
        n = X_batch.shape[0]
        if self.loss == 'squared':
            loss_val = 0.5 * np.mean(residuals ** 2)
            grad_coef = X_batch.T @ residuals / n
            grad_intercept = np.mean(residuals)
        elif self.loss == 'huber':
            abs_res = np.abs(residuals)
            huber_mask = abs_res <= self.epsilon
            loss_val = np.mean(
                np.where(huber_mask, 0.5 * residuals ** 2,
                         self.epsilon * abs_res - 0.5 * self.epsilon ** 2)
            )
            grad_r = np.where(huber_mask, residuals, self.epsilon * np.sign(residuals))
            grad_coef = X_batch.T @ grad_r / n
            grad_intercept = np.mean(grad_r)
        elif self.loss == 'epsilon_insensitive':
            excess = np.abs(residuals) - self.epsilon
            loss_val = np.mean(np.maximum(0.0, excess))
            grad_r = np.where(excess > 0, np.sign(residuals), 0.0)
            grad_coef = X_batch.T @ grad_r / n
            grad_intercept = np.mean(grad_r)
        else:
            loss_val = 0.5 * np.mean(residuals ** 2)
            grad_coef = X_batch.T @ residuals / n
            grad_intercept = np.mean(residuals)
        return loss_val, grad_coef, grad_intercept

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        rng = np.random.RandomState(self.random_state)

        n_samples, n_features = X.shape
        self.coef_ = rng.randn(n_features) * 0.01
        self.intercept_ = np.zeros(1)
        self.loss_curve_ = []

        t = 1
        for epoch in range(self.max_iter):
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_b = X_shuffled[start:end]
                y_b = y_shuffled[start:end]
                lr = self._get_learning_rate(t)
                loss_val, grad_coef, grad_intercept = self._compute_loss_and_grad(
                    X_b, y_b, self.coef_, self.intercept_[0]
                )
                self.coef_ -= lr * grad_coef
                self.coef_ = self._apply_penalty(self.coef_, lr)
                self.intercept_[0] -= lr * grad_intercept
                epoch_loss += loss_val
                n_batches += 1
                t += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_curve_.append(avg_loss)
            if len(self.loss_curve_) > 1 and abs(self.loss_curve_[-2] - avg_loss) < self.tol:
                break

        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        return X @ self.coef_ + self.intercept_[0]

    def score(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)

    def __repr__(self):
        return (
            f"SGDRegressor(loss={self.loss!r}, penalty={self.penalty!r}, "
            f"alpha={self.alpha}, learning_rate={self.learning_rate!r}, "
            f"eta0={self.eta0}, max_iter={self.max_iter}, epsilon={self.epsilon})"
        )
