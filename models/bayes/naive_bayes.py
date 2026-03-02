import numpy as np


class GaussianNaiveBayes:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.theta_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.class_log_prior_ = np.zeros(n_classes)
        n_samples = X.shape[0]
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[i] = X_c.mean(axis=0)
            self.var_[i] = X_c.var(axis=0) + self.var_smoothing
            self.class_log_prior_[i] = np.log(X_c.shape[0] / n_samples)
        return self

    def _log_likelihood(self, X):
        log_likelihoods = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(len(self.classes_)):
            var = self.var_[i]
            mean = self.theta_[i]
            log_scale = -0.5 * np.log(2 * np.pi * var)
            log_exp = -0.5 * ((X - mean) ** 2) / var
            log_likelihoods[:, i] = np.sum(log_scale + log_exp, axis=1)
        return log_likelihoods

    def predict(self, X):
        log_posteriors = self.class_log_prior_ + self._log_likelihood(X)
        indices = np.argmax(log_posteriors, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        log_posteriors = self.class_log_prior_ + self._log_likelihood(X)
        log_posteriors -= np.max(log_posteriors, axis=1, keepdims=True)
        exp_posteriors = np.exp(log_posteriors)
        return exp_posteriors / exp_posteriors.sum(axis=1, keepdims=True)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return f"GaussianNaiveBayes(var_smoothing={self.var_smoothing})"


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        self.class_log_prior_ = np.zeros(n_classes)
        n_samples = X.shape[0]
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            count = X_c.sum(axis=0) + self.alpha
            total = count.sum()
            self.feature_log_prob_[i] = np.log(count / total)
            self.class_log_prior_[i] = np.log(X_c.shape[0] / n_samples)
        return self

    def predict(self, X):
        log_posteriors = self.class_log_prior_ + X @ self.feature_log_prob_.T
        indices = np.argmax(log_posteriors, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        log_posteriors = self.class_log_prior_ + X @ self.feature_log_prob_.T
        log_posteriors -= np.max(log_posteriors, axis=1, keepdims=True)
        exp_posteriors = np.exp(log_posteriors)
        return exp_posteriors / exp_posteriors.sum(axis=1, keepdims=True)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return f"MultinomialNaiveBayes(alpha={self.alpha})"


class BernoulliNaiveBayes:
    def __init__(self, alpha=1.0, binarize=0.0):
        self.alpha = alpha
        self.binarize = binarize

    def fit(self, X, y):
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        self.log_neg_prob_ = np.zeros((n_classes, n_features))
        self.class_log_prior_ = np.zeros(n_classes)
        n_samples = X.shape[0]
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            n_c = X_c.shape[0]
            p1 = (X_c.sum(axis=0) + self.alpha) / (n_c + 2 * self.alpha)
            self.feature_log_prob_[i] = np.log(p1)
            self.log_neg_prob_[i] = np.log(1 - p1)
            self.class_log_prior_[i] = np.log(n_c / n_samples)
        return self

    def predict(self, X):
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        log_posteriors = (
            self.class_log_prior_
            + X @ self.feature_log_prob_.T
            + (1 - X) @ self.log_neg_prob_.T
        )
        indices = np.argmax(log_posteriors, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        if self.binarize is not None:
            X = (X > self.binarize).astype(float)
        log_posteriors = (
            self.class_log_prior_
            + X @ self.feature_log_prob_.T
            + (1 - X) @ self.log_neg_prob_.T
        )
        log_posteriors -= np.max(log_posteriors, axis=1, keepdims=True)
        exp_posteriors = np.exp(log_posteriors)
        return exp_posteriors / exp_posteriors.sum(axis=1, keepdims=True)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return f"BernoulliNaiveBayes(alpha={self.alpha}, binarize={self.binarize})"
