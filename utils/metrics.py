import numpy as np


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def _per_class_prf(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    precisions, recalls, f1s, supports = [], [], [], []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        support = np.sum(y_true == c)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        supports.append(support)
    return classes, np.array(precisions), np.array(recalls), np.array(f1s), np.array(supports)


def precision_score(y_true, y_pred, average='binary'):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes, precisions, recalls, f1s, supports = _per_class_prf(y_true, y_pred)
    if average == 'binary':
        pos = classes[-1]
        tp = np.sum((y_pred == pos) & (y_true == pos))
        fp = np.sum((y_pred == pos) & (y_true != pos))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    elif average == 'macro':
        return float(np.mean(precisions))
    elif average == 'weighted':
        return float(np.average(precisions, weights=supports))
    elif average == 'micro':
        tp_sum = sum(np.sum((y_pred == c) & (y_true == c)) for c in classes)
        fp_sum = sum(np.sum((y_pred == c) & (y_true != c)) for c in classes)
        return tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    raise ValueError(f"Unknown average: {average}")


def recall_score(y_true, y_pred, average='binary'):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes, precisions, recalls, f1s, supports = _per_class_prf(y_true, y_pred)
    if average == 'binary':
        pos = classes[-1]
        tp = np.sum((y_pred == pos) & (y_true == pos))
        fn = np.sum((y_pred != pos) & (y_true == pos))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    elif average == 'macro':
        return float(np.mean(recalls))
    elif average == 'weighted':
        return float(np.average(recalls, weights=supports))
    elif average == 'micro':
        tp_sum = sum(np.sum((y_pred == c) & (y_true == c)) for c in classes)
        fn_sum = sum(np.sum((y_pred != c) & (y_true == c)) for c in classes)
        return tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
    raise ValueError(f"Unknown average: {average}")


def f1_score(y_true, y_pred, average='binary'):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes, precisions, recalls, f1s, supports = _per_class_prf(y_true, y_pred)
    if average == 'binary':
        pos = classes[-1]
        tp = np.sum((y_pred == pos) & (y_true == pos))
        fp = np.sum((y_pred == pos) & (y_true != pos))
        fn = np.sum((y_pred != pos) & (y_true == pos))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    elif average == 'macro':
        return float(np.mean(f1s))
    elif average == 'weighted':
        return float(np.average(f1s, weights=supports))
    elif average == 'micro':
        p_micro = precision_score(y_true, y_pred, average='micro')
        r_micro = recall_score(y_true, y_pred, average='micro')
        return 2 * p_micro * r_micro / (p_micro + r_micro) if (p_micro + r_micro) > 0 else 0.0
    raise ValueError(f"Unknown average: {average}")


def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1
    return cm


def roc_curve(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thresholds = np.sort(np.unique(y_score))[::-1]
    fprs, tprs = [], []
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tprs.append(tp / n_pos if n_pos > 0 else 0.0)
        fprs.append(fp / n_neg if n_neg > 0 else 0.0)
    fprs = np.array([0.0] + fprs + [1.0])
    tprs = np.array([0.0] + tprs + [1.0])
    thresholds = np.concatenate([[thresholds[0] + 1], thresholds, [thresholds[-1] - 1]])
    return fprs, tprs, thresholds


def roc_auc_score(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.trapz(tpr, fpr))


def precision_recall_curve(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thresholds = np.sort(np.unique(y_score))[::-1]
    precisions, recalls = [], []
    n_pos = np.sum(y_true == 1)
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
    precisions = np.array(precisions + [1.0])
    recalls = np.array(recalls + [0.0])
    return precisions, recalls, thresholds


def classification_report(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes, precisions, recalls, f1s, supports = _per_class_prf(y_true, y_pred)
    report = {}
    for i, c in enumerate(classes):
        report[str(c)] = {
            'precision': precisions[i],
            'recall': recalls[i],
            'f1-score': f1s[i],
            'support': int(supports[i]),
        }
    total = np.sum(supports)
    report['macro avg'] = {
        'precision': float(np.mean(precisions)),
        'recall': float(np.mean(recalls)),
        'f1-score': float(np.mean(f1s)),
        'support': int(total),
    }
    report['weighted avg'] = {
        'precision': float(np.average(precisions, weights=supports)),
        'recall': float(np.average(recalls, weights=supports)),
        'f1-score': float(np.average(f1s, weights=supports)),
        'support': int(total),
    }
    report['accuracy'] = accuracy_score(y_true, y_pred)
    return report


def log_loss(y_true, y_proba):
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.clip(np.asarray(y_proba, dtype=float), 1e-15, 1 - 1e-15)
    if y_proba.ndim == 1:
        return float(-np.mean(y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba)))
    n = len(y_true)
    classes = np.unique(y_true.astype(int))
    Y = np.zeros((n, len(classes)))
    for i, c in enumerate(classes):
        Y[:, i] = (y_true == c).astype(float)
    return float(-np.mean(np.sum(Y * np.log(y_proba), axis=1)))


def cohen_kappa_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (n ** 2)
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0


def matthews_corrcoef(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    t = cm.sum(axis=1)
    p = cm.sum(axis=0)
    c = np.trace(cm)
    s = cm.sum()
    num = c * s - np.dot(t, p)
    den = np.sqrt((s ** 2 - np.dot(p, p)) * (s ** 2 - np.dot(t, t)))
    return float(num / den) if den != 0 else 0.0


def mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


def adjusted_r2_score(y_true, y_pred, n_features):
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    return float(1 - (1 - r2) * (n - 1) / (n - n_features - 1)) if (n - n_features - 1) > 0 else 0.0


def explained_variance_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    var_res = np.var(y_true - y_pred)
    var_y = np.var(y_true)
    return float(1 - var_res / var_y) if var_y != 0 else 0.0


def max_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.max(np.abs(y_true - y_pred)))


def median_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.median(np.abs(y_true - y_pred)))


def mean_squared_log_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))


def huber_loss(y_true, y_pred, delta=1.35):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = np.abs(y_true - y_pred)
    loss = np.where(residual <= delta, 0.5 * residual ** 2, delta * (residual - 0.5 * delta))
    return float(np.mean(loss))


def silhouette_samples(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = len(X)
    unique_labels = np.unique(labels)
    scores = np.zeros(n)
    for i in range(n):
        own_label = labels[i]
        own_mask = (labels == own_label)
        own_mask[i] = False
        if own_mask.sum() == 0:
            scores[i] = 0.0
            continue
        a = np.mean(np.linalg.norm(X[own_mask] - X[i], axis=1))
        b_vals = []
        for label in unique_labels:
            if label == own_label:
                continue
            other_mask = labels == label
            b_vals.append(np.mean(np.linalg.norm(X[other_mask] - X[i], axis=1)))
        if not b_vals:
            scores[i] = 0.0
            continue
        b = min(b_vals)
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return scores


def silhouette_score(X, labels):
    return float(np.mean(silhouette_samples(X, labels)))


def davies_bouldin_score(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    centers = np.array([X[labels == c].mean(axis=0) for c in unique_labels])
    s = np.array([np.mean(np.linalg.norm(X[labels == c] - centers[i], axis=1)) for i, c in enumerate(unique_labels)])
    db = 0.0
    for i in range(k):
        ratios = []
        for j in range(k):
            if i == j:
                continue
            d = np.linalg.norm(centers[i] - centers[j])
            ratios.append((s[i] + s[j]) / d if d > 0 else 0.0)
        db += max(ratios) if ratios else 0.0
    return float(db / k)


def calinski_harabasz_score(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    n = len(X)
    overall_mean = X.mean(axis=0)
    between = sum(
        len(X[labels == c]) * np.linalg.norm(X[labels == c].mean(axis=0) - overall_mean) ** 2
        for c in unique_labels
    )
    within = sum(
        np.sum(np.linalg.norm(X[labels == c] - X[labels == c].mean(axis=0), axis=1) ** 2)
        for c in unique_labels
    )
    if within == 0:
        return 0.0
    return float((between / (k - 1)) / (within / (n - k)))


def inertia_score(X, labels, centers):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    centers = np.asarray(centers, dtype=float)
    total = 0.0
    for i, center in enumerate(centers):
        mask = labels == i
        if mask.sum() > 0:
            total += np.sum(np.linalg.norm(X[mask] - center, axis=1) ** 2)
    return float(total)


def k_fold_indices(n_samples, n_splits=5, shuffle=True, random_state=None):
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)
    fold_sizes = np.full(n_splits, n_samples // n_splits)
    fold_sizes[:n_samples % n_splits] += 1
    folds = []
    current = 0
    for size in fold_sizes:
        folds.append(indices[current:current + size])
        current += size
    result = []
    for i in range(n_splits):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        result.append((train_idx, test_idx))
    return result


def cross_val_score(model, X, y, cv=5, scoring='accuracy'):
    X = np.asarray(X)
    y = np.asarray(y)
    folds = k_fold_indices(len(X), n_splits=cv, shuffle=True, random_state=0)
    scores = []
    scoring_funcs = {
        'accuracy': accuracy_score,
        'mse': mean_squared_error,
        'rmse': root_mean_squared_error,
        'mae': mean_absolute_error,
        'r2': r2_score,
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score,
    }
    score_fn = scoring_funcs.get(scoring, accuracy_score)
    for train_idx, test_idx in folds:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(score_fn(y_test, y_pred))
    return np.array(scores)


def train_test_split(X, y, test_size=0.2, random_state=None, stratify=None):
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)
    rng = np.random.RandomState(random_state)
    n_test = int(np.ceil(n * test_size))
    if stratify is not None:
        stratify = np.asarray(stratify)
        classes, counts = np.unique(stratify, return_counts=True)
        train_idx_list, test_idx_list = [], []
        for c in classes:
            c_idx = np.where(stratify == c)[0]
            rng.shuffle(c_idx)
            n_test_c = max(1, int(np.round(len(c_idx) * test_size)))
            test_idx_list.append(c_idx[:n_test_c])
            train_idx_list.append(c_idx[n_test_c:])
        train_idx = np.concatenate(train_idx_list)
        test_idx = np.concatenate(test_idx_list)
    else:
        perm = rng.permutation(n)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
