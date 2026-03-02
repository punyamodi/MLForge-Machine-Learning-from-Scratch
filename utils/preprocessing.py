import numpy as np
from itertools import combinations_with_replacement


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float)
        return X * self.scale_ + self.mean_


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        data_range = self.data_max_ - self.data_min_
        data_range[data_range == 0] = 1.0
        fmin, fmax = self.feature_range
        self.scale_ = (fmax - fmin) / data_range
        self.min_ = fmin - self.data_min_ * self.scale_
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X * self.scale_ + self.min_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.min_) / self.scale_


class RobustScaler:
    def __init__(self, quantile_range=(25.0, 75.0)):
        self.quantile_range = quantile_range
        self.center_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.center_ = np.median(X, axis=0)
        q_low, q_high = self.quantile_range
        iqr = np.percentile(X, q_high, axis=0) - np.percentile(X, q_low, axis=0)
        iqr[iqr == 0] = 1.0
        self.scale_ = iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X - self.center_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float)
        return X * self.scale_ + self.center_


class MaxAbsScaler:
    def __init__(self):
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.scale_ = np.max(np.abs(X), axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class Normalizer:
    def __init__(self, norm='l2'):
        self.norm = norm

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.norm == 'l2':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
        elif self.norm == 'l1':
            norms = np.sum(np.abs(X), axis=1, keepdims=True)
        elif self.norm == 'max':
            norms = np.max(np.abs(X), axis=1, keepdims=True)
        else:
            raise ValueError(f"Unknown norm: {self.norm}")
        norms[norms == 0] = 1.0
        return X / norms

    def fit_transform(self, X):
        return self.transform(X)


class LabelEncoder:
    def __init__(self):
        self.classes_ = None
        self._label_to_idx = None
        self._idx_to_label = None

    def fit(self, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self._label_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        self._idx_to_label = {idx: label for idx, label in enumerate(self.classes_)}
        return self

    def transform(self, y):
        y = np.asarray(y)
        return np.array([self._label_to_idx[label] for label in y])

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        y = np.asarray(y)
        return np.array([self._idx_to_label[idx] for idx in y])


class OneHotEncoder:
    def __init__(self, sparse=False, handle_unknown='ignore'):
        self.sparse = sparse
        self.handle_unknown = handle_unknown
        self.categories_ = None

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples = X.shape[0]
        parts = []
        for j, cats in enumerate(self.categories_):
            col = X[:, j]
            encoded = np.zeros((n_samples, len(cats)), dtype=float)
            for k, cat in enumerate(cats):
                encoded[:, k] = (col == cat).astype(float)
            if self.handle_unknown == 'ignore':
                unknown_mask = np.array([v not in cats for v in col])
                encoded[unknown_mask] = 0.0
            parts.append(encoded)
        return np.hstack(parts)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        result = np.empty((n_samples, len(self.categories_)), dtype=object)
        col_offset = 0
        for j, cats in enumerate(self.categories_):
            block = X[:, col_offset:col_offset + len(cats)]
            indices = np.argmax(block, axis=1)
            result[:, j] = np.array([cats[i] if block[row, i] != 0 else None for row, i in enumerate(indices)])
            col_offset += len(cats)
        return result


class OrdinalEncoder:
    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        result = np.zeros_like(X, dtype=float)
        for j, cats in enumerate(self.categories_):
            cat_to_idx = {c: i for i, c in enumerate(cats)}
            result[:, j] = np.array([cat_to_idx.get(v, -1) for v in X[:, j]], dtype=float)
        return result

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float)
        result = np.empty_like(X, dtype=object)
        for j, cats in enumerate(self.categories_):
            result[:, j] = np.array([cats[int(v)] if 0 <= int(v) < len(cats) else None for v in X[:, j]])
        return result


class LabelBinarizer:
    def __init__(self):
        self.classes_ = None
        self._sparse_input = False
        self._binary = False

    def fit(self, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self._binary = len(self.classes_) == 2
        return self

    def transform(self, y):
        y = np.asarray(y)
        if self._binary:
            return (y == self.classes_[1]).astype(int).reshape(-1, 1)
        n = len(y)
        result = np.zeros((n, len(self.classes_)), dtype=int)
        for i, c in enumerate(self.classes_):
            result[:, i] = (y == c).astype(int)
        return result

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, Y):
        Y = np.asarray(Y)
        if Y.ndim == 1 or Y.shape[1] == 1:
            Y_flat = Y.ravel()
            return np.array([self.classes_[1] if v else self.classes_[0] for v in Y_flat])
        return self.classes_[np.argmax(Y, axis=1)]


class PolynomialFeatures:
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias
        self._n_input_features = None
        self._powers = None

    def _build_powers(self, n_features):
        powers = []
        if self.include_bias:
            powers.append([0] * n_features)
        for d in range(1, self.degree + 1):
            for combo in combinations_with_replacement(range(n_features), d):
                power = [0] * n_features
                for idx in combo:
                    power[idx] += 1
                powers.append(power)
        self._powers = np.array(powers)

    def fit_transform(self, X):
        X = np.asarray(X, dtype=float)
        self._n_input_features = X.shape[1]
        self._build_powers(self._n_input_features)
        return np.column_stack([np.prod(X ** p, axis=1) for p in self._powers])

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self._powers is None:
            raise RuntimeError("Call fit_transform first.")
        return np.column_stack([np.prod(X ** p, axis=1) for p in self._powers])

    def get_feature_names_out(self, input_features=None):
        if self._powers is None:
            raise RuntimeError("Call fit_transform first.")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self._n_input_features)]
        names = []
        for power in self._powers:
            parts = []
            for feat, exp in zip(input_features, power):
                if exp == 0:
                    continue
                elif exp == 1:
                    parts.append(feat)
                else:
                    parts.append(f"{feat}^{exp}")
            names.append(" ".join(parts) if parts else "1")
        return names


class SimpleImputer:
    def __init__(self, strategy='mean', fill_value=None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == 'most_frequent':
            self.statistics_ = np.array([
                np.bincount(X[:, j][~np.isnan(X[:, j])].astype(int)).argmax()
                for j in range(X.shape[1])
            ], dtype=float)
        elif self.strategy == 'constant':
            val = self.fill_value if self.fill_value is not None else 0.0
            self.statistics_ = np.full(X.shape[1], val, dtype=float)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = self.statistics_[j]
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class Binarizer:
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return (X > self.threshold).astype(float)


class PowerTransformer:
    def __init__(self, method='yeo-johnson'):
        self.method = method
        self.lambdas_ = None
        self._scalers = None

    def _yeo_johnson(self, x, lam):
        out = np.zeros_like(x)
        pos = x >= 0
        neg = ~pos
        if lam != 0:
            out[pos] = (np.power(x[pos] + 1, lam) - 1) / lam
        else:
            out[pos] = np.log(x[pos] + 1)
        if lam != 2:
            out[neg] = -(np.power(-x[neg] + 1, 2 - lam) - 1) / (2 - lam)
        else:
            out[neg] = -np.log(-x[neg] + 1)
        return out

    def _box_cox(self, x, lam):
        if lam == 0:
            return np.log(x)
        return (np.power(x, lam) - 1) / lam

    def _neg_log_likelihood_yj(self, lam, x):
        x_t = self._yeo_johnson(x, lam)
        n = len(x)
        var = np.var(x_t)
        if var <= 0:
            return np.inf
        log_like = -0.5 * n * np.log(var)
        log_like += (lam - 1) * np.sum(np.sign(x) * np.log1p(np.abs(x)))
        return -log_like

    def _neg_log_likelihood_bc(self, lam, x):
        x_t = self._box_cox(x, lam)
        n = len(x)
        var = np.var(x_t)
        if var <= 0:
            return np.inf
        log_like = -0.5 * n * np.log(var)
        log_like += (lam - 1) * np.sum(np.log(x))
        return -log_like

    def _optimize_lambda(self, x):
        from functools import partial
        if self.method == 'yeo-johnson':
            obj = partial(self._neg_log_likelihood_yj, x=x)
            lam_range = np.linspace(-2, 2, 200)
        else:
            obj = partial(self._neg_log_likelihood_bc, x=x)
            lam_range = np.linspace(-2, 2, 200)
        losses = np.array([obj(l) for l in lam_range])
        return lam_range[np.argmin(losses)]

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.lambdas_ = np.array([self._optimize_lambda(X[:, j]) for j in range(X.shape[1])])
        X_t = self._apply(X)
        self._scalers = StandardScaler()
        self._scalers.fit(X_t)
        return self

    def _apply(self, X):
        X = np.asarray(X, dtype=float)
        result = np.zeros_like(X)
        for j in range(X.shape[1]):
            if self.method == 'yeo-johnson':
                result[:, j] = self._yeo_johnson(X[:, j], self.lambdas_[j])
            else:
                result[:, j] = self._box_cox(X[:, j], self.lambdas_[j])
        return result

    def transform(self, X):
        X_t = self._apply(X)
        return self._scalers.transform(X_t)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=float)
        X = self._scalers.inverse_transform(X)
        result = np.zeros_like(X)
        for j in range(X.shape[1]):
            lam = self.lambdas_[j]
            col = X[:, j]
            if self.method == 'yeo-johnson':
                out = np.zeros_like(col)
                pos = col >= 0
                neg = ~pos
                if lam != 0:
                    out[pos] = np.power(col[pos] * lam + 1, 1 / lam) - 1
                else:
                    out[pos] = np.exp(col[pos]) - 1
                if lam != 2:
                    out[neg] = 1 - np.power(-(2 - lam) * col[neg] + 1, 1 / (2 - lam))
                else:
                    out[neg] = 1 - np.exp(-col[neg])
                result[:, j] = out
            else:
                if lam == 0:
                    result[:, j] = np.exp(col)
                else:
                    result[:, j] = np.power(col * lam + 1, 1 / lam)
        return result


class QuantileTransformer:
    def __init__(self, n_quantiles=1000, output_distribution='uniform'):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.quantiles_ = None
        self._references = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self._references = np.linspace(0, 1, self.n_quantiles)
        self.quantiles_ = np.percentile(X, self._references * 100, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        result = np.zeros_like(X)
        for j in range(X.shape[1]):
            result[:, j] = np.interp(X[:, j], self.quantiles_[:, j], self._references)
        if self.output_distribution == 'normal':
            result = np.clip(result, 1e-7, 1 - 1e-7)
            result = np.sqrt(2) * _erfinv(2 * result - 1)
        return result

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def _erfinv(x):
    a = 0.147
    ln_term = np.log(1 - x ** 2)
    term1 = 2 / (np.pi * a) + ln_term / 2
    result = np.sign(x) * np.sqrt(np.sqrt(term1 ** 2 - ln_term / a) - term1)
    return result


class KBinsDiscretizer:
    def __init__(self, n_bins=5, strategy='quantile', encode='ordinal'):
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode
        self.bin_edges_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.bin_edges_ = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if self.strategy == 'uniform':
                edges = np.linspace(col.min(), col.max(), self.n_bins + 1)
            elif self.strategy == 'quantile':
                quantiles = np.linspace(0, 100, self.n_bins + 1)
                edges = np.percentile(col, quantiles)
            elif self.strategy == 'kmeans':
                centers = np.linspace(col.min(), col.max(), self.n_bins)
                edges = np.concatenate([[col.min()],
                                        [(centers[i] + centers[i + 1]) / 2 for i in range(self.n_bins - 1)],
                                        [col.max()]])
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            edges = np.unique(edges)
            self.bin_edges_.append(edges)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        result = np.zeros((X.shape[0], X.shape[1]), dtype=float)
        for j in range(X.shape[1]):
            edges = self.bin_edges_[j]
            result[:, j] = np.clip(np.searchsorted(edges[1:-1], X[:, j]), 0, len(edges) - 2).astype(float)
        if self.encode == 'onehot' or self.encode == 'onehot-dense':
            parts = []
            for j in range(X.shape[1]):
                n_bins_j = len(self.bin_edges_[j]) - 1
                one_hot = np.zeros((X.shape[0], n_bins_j), dtype=float)
                for i in range(X.shape[0]):
                    one_hot[i, int(result[i, j])] = 1.0
                parts.append(one_hot)
            return np.hstack(parts)
        return result

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class ColumnTransformer:
    def __init__(self, transformers, remainder='drop'):
        self.transformers = transformers
        self.remainder = remainder
        self._fitted_transformers = []
        self._all_columns = None

    def fit(self, X):
        X = np.asarray(X)
        self._all_columns = list(range(X.shape[1]))
        self._fitted_transformers = []
        covered = set()
        for name, transformer, cols in self.transformers:
            sub = X[:, cols]
            transformer.fit(sub)
            self._fitted_transformers.append((name, transformer, cols))
            covered.update(cols if hasattr(cols, '__iter__') else [cols])
        self._remainder_cols = [c for c in self._all_columns if c not in covered]
        return self

    def transform(self, X):
        X = np.asarray(X)
        parts = []
        for name, transformer, cols in self._fitted_transformers:
            sub = X[:, cols]
            parts.append(transformer.transform(sub))
        if self.remainder == 'passthrough' and self._remainder_cols:
            parts.append(X[:, self._remainder_cols].astype(float))
        return np.hstack(parts) if parts else np.empty((X.shape[0], 0))

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        X_current = X
        for name, step in self.steps[:-1]:
            if hasattr(step, 'fit_transform'):
                X_current = step.fit_transform(X_current)
            else:
                step.fit(X_current)
                X_current = step.transform(X_current)
        final_name, final_step = self.steps[-1]
        if y is not None and hasattr(final_step, 'fit'):
            final_step.fit(X_current, y)
        elif hasattr(final_step, 'fit_transform'):
            final_step.fit_transform(X_current)
        return self

    def transform(self, X):
        X_current = X
        for name, step in self.steps:
            X_current = step.transform(X_current)
        return X_current

    def fit_transform(self, X, y=None):
        X_current = X
        for name, step in self.steps[:-1]:
            if hasattr(step, 'fit_transform'):
                X_current = step.fit_transform(X_current)
            else:
                step.fit(X_current)
                X_current = step.transform(X_current)
        final_name, final_step = self.steps[-1]
        if hasattr(final_step, 'fit_transform'):
            X_current = final_step.fit_transform(X_current)
        else:
            X_current = final_step.transform(X_current)
        return X_current

    def predict(self, X):
        X_current = X
        for name, step in self.steps[:-1]:
            X_current = step.transform(X_current)
        final_name, final_step = self.steps[-1]
        return final_step.predict(X_current)

    def score(self, X, y):
        X_current = X
        for name, step in self.steps[:-1]:
            X_current = step.transform(X_current)
        final_name, final_step = self.steps[-1]
        return final_step.score(X_current, y)
