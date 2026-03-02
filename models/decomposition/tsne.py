import numpy as np


class TSNE:
    def __init__(self, n_components=2, perplexity=30.0, early_exaggeration=12.0,
                 learning_rate=200.0, n_iter=1000, momentum=0.8, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.momentum = momentum
        self.random_state = random_state

    def _pairwise_sq_distances(self, X):
        sum_sq = np.sum(X ** 2, axis=1)
        D = sum_sq[:, np.newaxis] + sum_sq[np.newaxis, :] - 2 * X @ X.T
        return np.maximum(D, 0)

    def _binary_search_perplexity(self, dist_i, target_perplexity):
        beta_min = -np.inf
        beta_max = np.inf
        beta = 1.0
        tol = 1e-5
        max_tries = 50

        for _ in range(max_tries):
            exp_d = np.exp(-dist_i * beta)
            sum_exp = exp_d.sum() + 1e-10
            H = np.log(sum_exp) + beta * np.sum(dist_i * exp_d) / sum_exp
            p = exp_d / sum_exp
            perp = np.exp(H)

            if abs(perp - target_perplexity) < tol:
                break

            if perp > target_perplexity:
                beta_min = beta
                if beta_max == np.inf:
                    beta *= 2
                else:
                    beta = (beta + beta_max) / 2
            else:
                beta_max = beta
                if beta_min == -np.inf:
                    beta /= 2
                else:
                    beta = (beta + beta_min) / 2

        return p

    def _compute_P(self, X):
        n = X.shape[0]
        D = self._pairwise_sq_distances(X)
        P = np.zeros((n, n))
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            dist_i = D[i, mask]
            p_i = self._binary_search_perplexity(dist_i, self.perplexity)
            P[i, mask] = p_i

        P = (P + P.T) / (2 * n)
        P = np.maximum(P, 1e-12)
        return P

    def _compute_Q(self, Y):
        D = self._pairwise_sq_distances(Y)
        num = 1.0 / (1.0 + D)
        np.fill_diagonal(num, 0)
        Q = num / (num.sum() + 1e-10)
        Q = np.maximum(Q, 1e-12)
        return Q, num

    def _compute_gradient(self, P, Q, Y, num):
        n = Y.shape[0]
        PQ_diff = P - Q
        grad = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            grad[i] = 4 * np.sum((PQ_diff[:, i] * num[:, i])[:, np.newaxis] * diff, axis=0)
        return grad

    def fit_transform(self, X):
        rng = np.random.RandomState(self.random_state)
        n = X.shape[0]

        P = self._compute_P(X)
        P_exag = P * self.early_exaggeration

        Y = rng.randn(n, self.n_components) * 0.0001
        velocity = np.zeros_like(Y)

        for t in range(self.n_iter):
            if t == 250:
                P_exag = P

            Q, num = self._compute_Q(Y)
            grad = self._compute_gradient(P_exag, Q, Y, num)

            velocity = self.momentum * velocity - self.learning_rate * grad
            Y = Y + velocity

            Y = Y - Y.mean(axis=0)

        Q, _ = self._compute_Q(Y)
        self.kl_divergence_ = np.sum(P * np.log(P / Q + 1e-12))
        return Y

    def __repr__(self):
        return (f"TSNE(n_components={self.n_components}, perplexity={self.perplexity}, "
                f"n_iter={self.n_iter})")
