import numpy as np


class LinearSVC:
    def __init__(self, C=1.0, penalty='l2', loss='hinge', max_iter=1000,
                 tol=1e-4, learning_rate=0.01, fit_intercept=True, random_state=None):
        self.C = C
        self.penalty = penalty
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.classes_ = np.unique(y)
        y_enc = np.where(y == self.classes_[0], -1.0, 1.0)
        n_samples, n_features = X.shape
        self.coef_ = rng.randn(n_features) * 0.01
        self.intercept_ = 0.0
        lr = self.learning_rate
        prev_loss = np.inf

        for iteration in range(self.max_iter):
            scores = X @ self.coef_ + (self.intercept_ if self.fit_intercept else 0.0)
            margins = y_enc * scores

            if self.loss == 'hinge':
                mask = margins < 1.0
            else:
                mask = margins < 1.0

            grad_coef = np.zeros(n_features)
            grad_intercept = 0.0

            if mask.any():
                grad_coef -= self.C * np.dot(y_enc[mask], X[mask])
                if self.fit_intercept:
                    grad_intercept -= self.C * np.sum(y_enc[mask])

            if self.penalty == 'l2':
                grad_coef += self.coef_
            elif self.penalty == 'l1':
                grad_coef += np.sign(self.coef_)

            self.coef_ -= lr * grad_coef
            if self.fit_intercept:
                self.intercept_ -= lr * grad_intercept

            if self.penalty == 'l2':
                reg_loss = 0.5 * np.dot(self.coef_, self.coef_)
            else:
                reg_loss = np.sum(np.abs(self.coef_))

            hinge_loss = np.sum(np.maximum(0.0, 1.0 - margins))
            current_loss = reg_loss + self.C * hinge_loss

            if abs(prev_loss - current_loss) < self.tol:
                break
            prev_loss = current_loss

        return self

    def decision_function(self, X):
        return X @ self.coef_ + (self.intercept_ if self.fit_intercept else 0.0)

    def predict(self, X):
        df = self.decision_function(X)
        return np.where(df >= 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def __repr__(self):
        return (f"LinearSVC(C={self.C}, penalty={self.penalty!r}, "
                f"loss={self.loss!r}, max_iter={self.max_iter})")


class LinearSVR:
    def __init__(self, C=1.0, epsilon=0.0, loss='epsilon_insensitive', max_iter=1000,
                 tol=1e-4, learning_rate=0.01, fit_intercept=True, random_state=None):
        self.C = C
        self.epsilon = epsilon
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        self.coef_ = rng.randn(n_features) * 0.01
        self.intercept_ = 0.0
        lr = self.learning_rate
        prev_loss = np.inf

        for iteration in range(self.max_iter):
            preds = X @ self.coef_ + (self.intercept_ if self.fit_intercept else 0.0)
            residuals = preds - y
            abs_residuals = np.abs(residuals)

            if self.loss == 'epsilon_insensitive':
                mask = abs_residuals > self.epsilon
                signs = np.sign(residuals)
            else:
                mask = abs_residuals > self.epsilon
                signs = np.sign(residuals)

            grad_coef = self.coef_.copy()
            grad_intercept = 0.0

            if mask.any():
                grad_coef += self.C * np.dot(signs[mask], X[mask])
                if self.fit_intercept:
                    grad_intercept += self.C * np.sum(signs[mask])

            self.coef_ -= lr * grad_coef
            if self.fit_intercept:
                self.intercept_ -= lr * grad_intercept

            reg_loss = 0.5 * np.dot(self.coef_, self.coef_)
            eps_loss = np.sum(np.maximum(0.0, abs_residuals - self.epsilon))
            current_loss = reg_loss + self.C * eps_loss

            if abs(prev_loss - current_loss) < self.tol:
                break
            prev_loss = current_loss

        return self

    def predict(self, X):
        return X @ self.coef_ + (self.intercept_ if self.fit_intercept else 0.0)

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot == 0:
            return 1.0
        return float(1.0 - ss_res / ss_tot)

    def __repr__(self):
        return (f"LinearSVR(C={self.C}, epsilon={self.epsilon}, "
                f"loss={self.loss!r}, max_iter={self.max_iter})")
