import numpy as np

def load_iris():
    data = np.array([
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [4.6, 3.1, 1.5, 0.2],
        [5.0, 3.6, 1.4, 0.2],
        [5.4, 3.9, 1.7, 0.4],
        [4.6, 3.4, 1.4, 0.3],
        [5.0, 3.4, 1.5, 0.2],
        [4.4, 2.9, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.1],
        [5.4, 3.7, 1.5, 0.2],
        [4.8, 3.4, 1.6, 0.2],
        [4.8, 3.0, 1.4, 0.1],
        [4.3, 3.0, 1.1, 0.1],
        [5.8, 4.0, 1.2, 0.2],
        [5.7, 4.4, 1.5, 0.4],
        [5.4, 3.9, 1.3, 0.4],
        [5.1, 3.5, 1.4, 0.3],
        [5.7, 3.8, 1.7, 0.3],
        [5.1, 3.8, 1.5, 0.3],
        [5.4, 3.4, 1.7, 0.2],
        [5.1, 3.7, 1.5, 0.4],
        [4.6, 3.6, 1.0, 0.2],
        [5.1, 3.3, 1.7, 0.5],
        [4.8, 3.4, 1.9, 0.2],
        [5.0, 3.0, 1.6, 0.2],
        [5.0, 3.4, 1.6, 0.4],
        [5.2, 3.5, 1.5, 0.2],
        [5.2, 3.4, 1.4, 0.2],
        [4.7, 3.2, 1.6, 0.2],
        [4.8, 3.1, 1.6, 0.2],
        [5.4, 3.4, 1.5, 0.4],
        [5.2, 4.1, 1.5, 0.1],
        [5.5, 4.2, 1.4, 0.2],
        [4.9, 3.1, 1.5, 0.2],
        [5.0, 3.2, 1.2, 0.2],
        [5.5, 3.5, 1.3, 0.2],
        [4.9, 3.6, 1.4, 0.1],
        [4.4, 3.0, 1.3, 0.2],
        [5.1, 3.4, 1.5, 0.2],
        [5.0, 3.5, 1.3, 0.3],
        [4.5, 2.3, 1.3, 0.3],
        [4.4, 3.2, 1.3, 0.2],
        [5.0, 3.5, 1.6, 0.6],
        [5.1, 3.8, 1.9, 0.4],
        [4.8, 3.0, 1.4, 0.3],
        [5.1, 3.8, 1.6, 0.2],
        [4.6, 3.2, 1.4, 0.2],
        [5.3, 3.7, 1.5, 0.2],
        [5.0, 3.3, 1.4, 0.2],
        [7.0, 3.2, 4.7, 1.4],
        [6.4, 3.2, 4.5, 1.5],
        [6.9, 3.1, 4.9, 1.5],
        [5.5, 2.3, 4.0, 1.3],
        [6.5, 2.8, 4.6, 1.5],
        [5.7, 2.8, 4.5, 1.3],
        [6.3, 3.3, 4.7, 1.6],
        [4.9, 2.4, 3.3, 1.0],
        [6.6, 2.9, 4.6, 1.3],
        [5.2, 2.7, 3.9, 1.4],
        [5.0, 2.0, 3.5, 1.0],
        [5.9, 3.0, 4.2, 1.5],
        [6.0, 2.2, 4.0, 1.0],
        [6.1, 2.9, 4.7, 1.4],
        [5.6, 2.9, 3.6, 1.3],
        [6.7, 3.1, 4.4, 1.4],
        [5.6, 3.0, 4.5, 1.5],
        [5.8, 2.7, 4.1, 1.0],
        [6.2, 2.2, 4.5, 1.5],
        [5.6, 2.5, 3.9, 1.1],
        [5.9, 3.2, 4.8, 1.8],
        [6.1, 2.8, 4.0, 1.3],
        [6.3, 2.5, 4.9, 1.5],
        [6.1, 2.8, 4.7, 1.2],
        [6.4, 2.9, 4.3, 1.3],
        [6.6, 3.0, 4.4, 1.4],
        [6.8, 2.8, 4.8, 1.4],
        [6.7, 3.0, 5.0, 1.7],
        [6.0, 2.9, 4.5, 1.5],
        [5.7, 2.6, 3.5, 1.0],
        [5.5, 2.4, 3.8, 1.1],
        [5.5, 2.4, 3.7, 1.0],
        [5.8, 2.7, 3.9, 1.2],
        [6.0, 2.7, 5.1, 1.6],
        [5.4, 3.0, 4.5, 1.5],
        [6.0, 3.4, 4.5, 1.6],
        [6.7, 3.1, 4.7, 1.5],
        [6.3, 2.3, 4.4, 1.3],
        [5.6, 3.0, 4.1, 1.3],
        [5.5, 2.5, 4.0, 1.3],
        [5.5, 2.6, 4.4, 1.2],
        [6.1, 3.0, 4.6, 1.4],
        [5.8, 2.6, 4.0, 1.2],
        [5.0, 2.3, 3.3, 1.0],
        [5.6, 2.7, 4.2, 1.3],
        [5.7, 3.0, 4.2, 1.2],
        [5.7, 2.9, 4.2, 1.3],
        [6.2, 2.9, 4.3, 1.3],
        [5.1, 2.5, 3.0, 1.1],
        [5.7, 2.8, 4.1, 1.3],
        [6.3, 3.3, 6.0, 2.5],
        [5.8, 2.7, 5.1, 1.9],
        [7.1, 3.0, 5.9, 2.1],
        [6.3, 2.9, 5.6, 1.8],
        [6.5, 3.0, 5.8, 2.2],
        [7.6, 3.0, 6.6, 2.1],
        [4.9, 2.5, 4.5, 1.7],
        [7.3, 2.9, 6.3, 1.8],
        [6.7, 2.5, 5.8, 1.8],
        [7.2, 3.6, 6.1, 2.5],
        [6.5, 3.2, 5.1, 2.0],
        [6.4, 2.7, 5.3, 1.9],
        [6.8, 3.0, 5.5, 2.1],
        [5.7, 2.5, 5.0, 2.0],
        [5.8, 2.8, 5.1, 2.4],
        [6.4, 3.2, 5.3, 2.3],
        [6.5, 3.0, 5.5, 1.8],
        [7.7, 3.8, 6.7, 2.2],
        [7.7, 2.6, 6.9, 2.3],
        [6.0, 2.2, 5.0, 1.5],
        [6.9, 3.2, 5.7, 2.3],
        [5.6, 2.8, 4.9, 2.0],
        [7.7, 2.8, 6.7, 2.0],
        [6.3, 2.7, 4.9, 1.8],
        [6.7, 3.3, 5.7, 2.1],
        [7.2, 3.2, 6.0, 1.8],
        [6.2, 2.8, 4.8, 1.8],
        [6.1, 3.0, 4.9, 1.8],
        [6.4, 2.8, 5.6, 2.1],
        [7.2, 3.0, 5.8, 1.6],
        [7.4, 2.8, 6.1, 1.9],
        [7.9, 3.8, 6.4, 2.0],
        [6.4, 2.8, 5.6, 2.2],
        [6.3, 2.8, 5.1, 1.5],
        [6.1, 2.6, 5.6, 1.4],
        [7.7, 3.0, 6.1, 2.3],
        [6.3, 3.4, 5.6, 2.4],
        [6.4, 3.1, 5.5, 1.8],
        [6.0, 3.0, 4.8, 1.8],
        [6.9, 3.1, 5.4, 2.1],
        [6.7, 3.1, 5.6, 2.4],
        [6.9, 3.1, 5.1, 2.3],
        [5.8, 2.7, 5.1, 1.9],
        [6.8, 3.2, 5.9, 2.3],
        [6.7, 3.3, 5.7, 2.5],
        [6.7, 3.0, 5.2, 2.3],
        [6.3, 2.5, 5.0, 1.9],
        [6.5, 3.0, 5.2, 2.0],
        [6.2, 3.4, 5.4, 2.3],
        [5.9, 3.0, 5.1, 1.8],
    ], dtype=float)
    target = np.array([0]*50 + [1]*50 + [2]*50)
    return {
        'data': data,
        'target': target,
        'feature_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
        'target_names': ['setosa', 'versicolor', 'virginica'],
    }


def load_wine():
    rng = np.random.RandomState(13)
    n_per_class = 59
    centers = np.array([[13.7, 2.0, 2.5, 17.0, 106.0, 2.8, 3.0, 0.29, 1.9, 5.5, 1.1, 3.1, 1100.0],
                        [12.5, 1.7, 2.2, 20.0, 95.0, 2.1, 2.0, 0.36, 1.6, 3.0, 1.0, 2.8, 500.0],
                        [13.1, 3.3, 2.4, 21.5, 99.0, 1.7, 0.8, 0.45, 1.4, 7.3, 0.7, 1.7, 630.0]])
    stds = np.array([0.5, 0.5, 0.3, 2.0, 15.0, 0.4, 0.5, 0.07, 0.4, 1.0, 0.2, 0.4, 300.0])
    data_list, target_list = [], []
    for c in range(3):
        data_list.append(centers[c] + stds * rng.randn(n_per_class, 13))
        target_list.append(np.full(n_per_class, c))
    data = np.vstack(data_list)
    target = np.concatenate(target_list)
    idx = rng.permutation(len(target))
    return {
        'data': data[idx],
        'target': target[idx].astype(int),
        'feature_names': ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
                          'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
                          'color_intensity', 'hue', 'od280/od315', 'proline'],
        'target_names': ['class_0', 'class_1', 'class_2'],
    }


def load_breast_cancer():
    rng = np.random.RandomState(17)
    n_malignant = 212
    n_benign = 357
    feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
                     'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
                     'mean fractal dimension']
    center_m = np.array([17.5, 21.6, 115.0, 980.0, 0.10, 0.15, 0.16, 0.09, 0.19, 0.063])
    center_b = np.array([12.1, 17.9, 78.0, 463.0, 0.09, 0.08, 0.05, 0.03, 0.17, 0.063])
    std_m = np.array([2.0, 4.0, 15.0, 250.0, 0.01, 0.04, 0.06, 0.03, 0.02, 0.007])
    std_b = np.array([1.5, 3.0, 10.0, 120.0, 0.01, 0.03, 0.03, 0.01, 0.02, 0.007])
    X_m = center_m + std_m * rng.randn(n_malignant, 10)
    X_b = center_b + std_b * rng.randn(n_benign, 10)
    data = np.vstack([X_m, X_b])
    target = np.concatenate([np.zeros(n_malignant), np.ones(n_benign)]).astype(int)
    idx = rng.permutation(len(target))
    return {
        'data': data[idx],
        'target': target[idx],
        'feature_names': feature_names,
        'target_names': ['malignant', 'benign'],
    }


def load_diabetes():
    rng = np.random.RandomState(19)
    n = 442
    feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
    X = rng.randn(n, 10) * 0.05
    true_coef = np.array([0.0, -236.0, 519.5, 321.8, -570.0, 250.0, -6.0, 100.0, 480.0, 60.0])
    y = X @ true_coef + 152.0 + rng.randn(n) * 50.0
    return {
        'data': X,
        'target': y,
        'feature_names': feature_names,
    }


def load_boston():
    rng = np.random.RandomState(23)
    n = 506
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    X = np.zeros((n, 13))
    X[:, 0] = np.abs(rng.exponential(3.6, n))
    X[:, 1] = rng.choice([0, 12.5, 25.0, 80.0], n, p=[0.7, 0.15, 0.1, 0.05])
    X[:, 2] = np.abs(rng.normal(11.14, 6.86, n))
    X[:, 3] = rng.choice([0, 1], n, p=[0.93, 0.07]).astype(float)
    X[:, 4] = np.clip(rng.normal(0.555, 0.116, n), 0.38, 0.87)
    X[:, 5] = np.clip(rng.normal(6.28, 0.7, n), 3.5, 8.8)
    X[:, 6] = np.clip(rng.normal(68.6, 28.1, n), 2.9, 100.0)
    X[:, 7] = np.clip(rng.exponential(3.8, n), 1.1, 12.0)
    X[:, 8] = rng.choice(np.arange(1, 25), n).astype(float)
    X[:, 9] = rng.normal(408.2, 168.5, n)
    X[:, 10] = np.clip(rng.normal(18.46, 2.16, n), 12.6, 22.0)
    X[:, 11] = np.clip(rng.normal(356.67, 91.29, n), 0.32, 396.9)
    X[:, 12] = np.clip(rng.exponential(12.65, n), 1.73, 37.97)
    y = (X[:, 5] * 3.8 - X[:, 12] * 0.95 + X[:, 0] * (-0.1) + X[:, 4] * (-17.0) +
         X[:, 7] * 0.5 + rng.normal(0, 2.5, n) + 10.0)
    y = np.clip(y, 5.0, 50.0)
    return {
        'data': X,
        'target': y,
        'feature_names': feature_names,
    }


def make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=2,
                        n_classes=2, random_state=None):
    rng = np.random.RandomState(random_state)
    n_informative = min(n_informative, n_features)
    n_redundant = min(n_redundant, n_features - n_informative)
    X_informative = rng.randn(n_samples, n_informative)
    if n_redundant > 0:
        B = rng.randn(n_informative, n_redundant)
        X_redundant = X_informative @ B
    else:
        X_redundant = np.empty((n_samples, 0))
    n_noise = n_features - n_informative - n_redundant
    X_noise = rng.randn(n_samples, max(n_noise, 0))
    X = np.hstack([X_informative, X_redundant, X_noise])
    weights = rng.randn(n_informative)
    scores = X_informative @ weights
    if n_classes == 2:
        y = (scores > np.median(scores)).astype(int)
    else:
        percentiles = np.linspace(0, 100, n_classes + 1)[1:-1]
        thresholds = np.percentile(scores, percentiles)
        y = np.digitize(scores, thresholds)
    return X, y


def make_regression(n_samples=100, n_features=20, n_informative=10, noise=0.0,
                    random_state=None):
    rng = np.random.RandomState(random_state)
    n_informative = min(n_informative, n_features)
    X = rng.randn(n_samples, n_features)
    coef = np.zeros(n_features)
    coef[:n_informative] = rng.randn(n_informative) * 10.0
    y = X @ coef + noise * rng.randn(n_samples)
    return X, y


def make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0,
               random_state=None):
    rng = np.random.RandomState(random_state)
    if isinstance(centers, int):
        center_points = rng.uniform(-10, 10, size=(centers, n_features))
    else:
        center_points = np.array(centers)
        centers = len(center_points)
    n_per = n_samples // centers
    remainder = n_samples % centers
    X_list, y_list = [], []
    for i in range(centers):
        ni = n_per + (1 if i < remainder else 0)
        X_list.append(center_points[i] + cluster_std * rng.randn(ni, n_features))
        y_list.append(np.full(ni, i))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    idx = rng.permutation(len(y))
    return X[idx], y[idx].astype(int)


def make_moons(n_samples=100, noise=0.1, random_state=None):
    rng = np.random.RandomState(random_state)
    n_half = n_samples // 2
    n_other = n_samples - n_half
    theta_outer = np.linspace(0, np.pi, n_half)
    theta_inner = np.linspace(0, np.pi, n_other)
    x_outer = np.cos(theta_outer)
    y_outer = np.sin(theta_outer)
    x_inner = 1 - np.cos(theta_inner)
    y_inner = 1 - np.sin(theta_inner) - 0.5
    X = np.vstack([np.column_stack([x_outer, y_outer]),
                   np.column_stack([x_inner, y_inner])])
    y = np.concatenate([np.zeros(n_half), np.ones(n_other)]).astype(int)
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    return X, y


def make_circles(n_samples=100, noise=0.1, factor=0.8, random_state=None):
    rng = np.random.RandomState(random_state)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer
    theta_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    theta_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    X_outer = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
    X_inner = factor * np.column_stack([np.cos(theta_inner), np.sin(theta_inner)])
    X = np.vstack([X_outer, X_inner])
    y = np.concatenate([np.zeros(n_outer), np.ones(n_inner)]).astype(int)
    if noise > 0:
        X += rng.normal(0, noise, X.shape)
    return X, y


def make_s_curve(n_samples=100, noise=0.1, random_state=None):
    rng = np.random.RandomState(random_state)
    t = 3 * np.pi * (rng.rand(n_samples) - 0.5)
    X = np.zeros((n_samples, 3))
    X[:, 0] = np.sin(t)
    X[:, 1] = 2.0 * rng.rand(n_samples)
    X[:, 2] = np.sign(t) * (np.cos(t) - 1)
    if noise > 0:
        X += noise * rng.randn(n_samples, 3)
    return X, t


def make_swiss_roll(n_samples=100, noise=0.0, random_state=None):
    rng = np.random.RandomState(random_state)
    t = 1.5 * np.pi * (1 + 2 * rng.rand(n_samples))
    X = np.zeros((n_samples, 3))
    X[:, 0] = t * np.cos(t)
    X[:, 1] = 21 * rng.rand(n_samples)
    X[:, 2] = t * np.sin(t)
    if noise > 0:
        X += noise * rng.randn(n_samples, 3)
    return X, t
