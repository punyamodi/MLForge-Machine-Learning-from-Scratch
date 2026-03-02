import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap


_PALETTE = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0",
    "#00BCD4", "#FF5722", "#3F51B5", "#009688", "#CDDC39",
]

_FIG_BG = "#0d1117"
_AXES_BG = "#161b22"
_TICK_COLOR = "#8b949e"
_LABEL_COLOR = "#c9d1d9"
_GRID_COLOR = "#21262d"


def _style_axes(ax):
    ax.set_facecolor(_AXES_BG)
    ax.tick_params(colors=_TICK_COLOR, labelsize=9)
    ax.xaxis.label.set_color(_LABEL_COLOR)
    ax.yaxis.label.set_color(_LABEL_COLOR)
    ax.title.set_color(_LABEL_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID_COLOR)
    ax.grid(True, color=_GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.6)


def _new_fig(w=9, h=5):
    fig = plt.figure(figsize=(w, h), facecolor=_FIG_BG)
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    idx_map = {c: i for i, c in enumerate(classes)}
    for t, p in zip(y_true, y_pred):
        cm[idx_map[t], idx_map[p]] += 1
    labels = class_names if class_names else [str(c) for c in classes]
    fig = _new_fig(max(5, n * 0.9 + 2), max(4, n * 0.8 + 1.5))
    ax = fig.add_subplot(111, facecolor=_AXES_BG)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9, color=_LABEL_COLOR)
    ax.set_yticklabels(labels, fontsize=9, color=_LABEL_COLOR)
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else _LABEL_COLOR, fontsize=10)
    ax.set_xlabel("Predicted Label", color=_LABEL_COLOR)
    ax.set_ylabel("True Label", color=_LABEL_COLOR)
    ax.set_title("Confusion Matrix", color=_LABEL_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID_COLOR)
    fig.tight_layout()
    return fig


def plot_roc_curve(y_true, y_score):
    from utils.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_score))
    auc = roc_auc_score(np.asarray(y_true), np.asarray(y_score))
    fig = _new_fig(7, 5)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    ax.plot(fpr, tpr, color="#58a6ff", lw=2, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#6e7681", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(facecolor=_AXES_BG, edgecolor=_GRID_COLOR, labelcolor=_LABEL_COLOR, fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    return fig


def plot_precision_recall_curve(y_true, y_score):
    from utils.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(np.asarray(y_true), np.asarray(y_score))
    fig = _new_fig(7, 5)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    ax.plot(recall, precision, color="#58a6ff", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    return fig


def plot_learning_curve(model, X, y, cv=5, train_sizes=None):
    from utils.metrics import k_fold_indices, accuracy_score, r2_score
    X = np.asarray(X)
    y = np.asarray(y)
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 8)
    n = len(X)
    train_scores_all, val_scores_all = [], []
    folds = k_fold_indices(n, n_splits=cv, shuffle=True, random_state=0)
    actual_sizes = []
    for frac in train_sizes:
        train_s, val_s = [], []
        for train_idx, val_idx in folds:
            n_take = max(2, int(len(train_idx) * frac))
            t_idx = train_idx[:n_take]
            try:
                model.fit(X[t_idx], y[t_idx])
                pred_train = model.predict(X[t_idx])
                pred_val = model.predict(X[val_idx])
                is_clf = y.dtype.kind in ("i", "u") or (len(np.unique(y)) < 20)
                if is_clf:
                    train_s.append(accuracy_score(y[t_idx], pred_train))
                    val_s.append(accuracy_score(y[val_idx], pred_val))
                else:
                    train_s.append(r2_score(y[t_idx], pred_train))
                    val_s.append(r2_score(y[val_idx], pred_val))
            except Exception:
                pass
        if train_s:
            train_scores_all.append(np.mean(train_s))
            val_scores_all.append(np.mean(val_s))
            actual_sizes.append(int(n * frac))
    if not actual_sizes:
        fig = _new_fig(7, 4)
        ax = fig.add_subplot(111)
        _style_axes(ax)
        ax.text(0.5, 0.5, "Learning curve unavailable", transform=ax.transAxes,
                ha="center", va="center", color=_LABEL_COLOR)
        return fig
    fig = _new_fig(9, 5)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    ax.plot(actual_sizes, train_scores_all, "o-", color="#58a6ff", label="Train")
    ax.plot(actual_sizes, val_scores_all, "s--", color="#f78166", label="Validation")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Score")
    ax.set_title("Learning Curve")
    ax.legend(facecolor=_AXES_BG, edgecolor=_GRID_COLOR, labelcolor=_LABEL_COLOR, fontsize=9)
    fig.tight_layout()
    return fig


def plot_feature_importance(importances, feature_names):
    importances = np.asarray(importances, dtype=float)
    n = len(importances)
    feature_names = list(feature_names)[:n]
    order = np.argsort(importances)
    fig = _new_fig(9, max(4, n * 0.35 + 1))
    ax = fig.add_subplot(111)
    _style_axes(ax)
    y_pos = np.arange(n)
    bars = ax.barh(y_pos, importances[order], color="#58a6ff", edgecolor=_GRID_COLOR, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=8, color=_LABEL_COLOR)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    return fig


def plot_residuals(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    fig = _new_fig(12, 4)
    ax1 = fig.add_subplot(121)
    _style_axes(ax1)
    ax1.scatter(y_pred, residuals, alpha=0.5, color="#58a6ff", s=20, edgecolors="none")
    ax1.axhline(0, color="#f85149", linewidth=1.5, linestyle="--")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Predicted")
    ax2 = fig.add_subplot(122)
    _style_axes(ax2)
    ax2.hist(residuals, bins=30, color="#58a6ff", edgecolor=_GRID_COLOR, alpha=0.85)
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual Distribution")
    fig.tight_layout()
    return fig


def plot_cluster_scatter(X_2d, labels):
    X_2d = np.asarray(X_2d)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    fig = _new_fig(8, 6)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = "#6e7681" if lbl == -1 else _PALETTE[i % len(_PALETTE)]
        label_str = "Noise" if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=label_str,
                   alpha=0.7, s=25, edgecolors="none")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("Cluster Assignments")
    ax.legend(facecolor=_AXES_BG, edgecolor=_GRID_COLOR, labelcolor=_LABEL_COLOR,
              fontsize=8, markerscale=1.5)
    fig.tight_layout()
    return fig


def plot_elbow_curve(k_range, inertias):
    fig = _new_fig(7, 4)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    ax.plot(list(k_range), inertias, "o-", color="#58a6ff", linewidth=2, markersize=6)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Curve")
    ax.set_xticks(list(k_range))
    fig.tight_layout()
    return fig


def plot_dendrogram(X, method="ward", truncate_p=30):
    from scipy.cluster.hierarchy import dendrogram, linkage
    X = np.asarray(X, dtype=float)
    if X.shape[0] > 500:
        X = X[:500]
    Z = linkage(X, method=method)
    fig = _new_fig(10, 5)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=truncate_p,
               color_threshold=0.7 * max(Z[:, 2]),
               above_threshold_color=_TICK_COLOR,
               leaf_font_size=8)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Distance")
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.tick_params(axis="x", labelcolor=_TICK_COLOR)
    fig.tight_layout()
    return fig


def plot_pca_variance(explained_variance_ratio):
    evr = np.asarray(explained_variance_ratio)
    cumulative = np.cumsum(evr)
    n = len(evr)
    fig = _new_fig(9, 4)
    ax1 = fig.add_subplot(121)
    _style_axes(ax1)
    ax1.bar(range(1, n + 1), evr * 100, color="#58a6ff", edgecolor=_GRID_COLOR)
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title("Explained Variance per Component")
    ax1.set_xticks(range(1, n + 1))
    ax2 = fig.add_subplot(122)
    _style_axes(ax2)
    ax2.plot(range(1, n + 1), cumulative * 100, "o-", color="#58a6ff", linewidth=2)
    ax2.axhline(95, color="#f85149", linewidth=1, linestyle="--", label="95%")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.set_title("Cumulative Explained Variance")
    ax2.set_xticks(range(1, n + 1))
    ax2.legend(facecolor=_AXES_BG, edgecolor=_GRID_COLOR, labelcolor=_LABEL_COLOR, fontsize=9)
    fig.tight_layout()
    return fig


def plot_2d_embedding(X_2d, labels=None, title="2D Embedding"):
    X_2d = np.asarray(X_2d)
    fig = _new_fig(8, 6)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    if labels is None:
        ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.6, color=_PALETTE[0], s=20, edgecolors="none")
    else:
        labels = np.asarray(labels)
        unique = np.unique(labels)
        for i, lbl in enumerate(unique):
            mask = labels == lbl
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], color=_PALETTE[i % len(_PALETTE)],
                       label=str(lbl), alpha=0.7, s=20, edgecolors="none")
        ax.legend(facecolor=_AXES_BG, edgecolor=_GRID_COLOR, labelcolor=_LABEL_COLOR,
                  fontsize=8, markerscale=2, title="Class",
                  title_fontsize=8)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_silhouette(X, labels):
    from utils.metrics import silhouette_samples
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    n = len(X)
    fig = _new_fig(8, max(4, len(unique_labels) * 1.0 + 2))
    ax = fig.add_subplot(111)
    _style_axes(ax)
    y_lower = 10
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        vals = np.sort(sample_silhouette_values[mask])
        size = mask.sum()
        y_upper = y_lower + size
        color = _PALETTE[i % len(_PALETTE)]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.7, color=color)
        ax.text(-0.05, y_lower + size / 2, str(lbl), color=_LABEL_COLOR, fontsize=8)
        y_lower = y_upper + 10
    mean_score = sample_silhouette_values.mean()
    ax.axvline(mean_score, color="#f85149", linewidth=1.5, linestyle="--",
               label=f"Mean = {mean_score:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    ax.set_title("Silhouette Plot")
    ax.legend(facecolor=_AXES_BG, edgecolor=_GRID_COLOR, labelcolor=_LABEL_COLOR, fontsize=9)
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_decision_boundary(model, X, y, resolution=200):
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[1] != 2:
        raise ValueError("plot_decision_boundary requires exactly 2 features.")
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    unique = np.unique(y)
    cmap = ListedColormap(_PALETTE[: len(unique)])
    fig = _new_fig(8, 6)
    ax = fig.add_subplot(111)
    _style_axes(ax)
    ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap)
    for i, lbl in enumerate(unique):
        mask = y == lbl
        ax.scatter(X[mask, 0], X[mask, 1], color=_PALETTE[i % len(_PALETTE)],
                   label=str(lbl), edgecolors=_GRID_COLOR, s=30, linewidth=0.4)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Decision Boundary")
    ax.legend(facecolor=_AXES_BG, edgecolor=_GRID_COLOR, labelcolor=_LABEL_COLOR, fontsize=9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()
    return fig


def plot_correlation_matrix(X, feature_names=None):
    X = np.asarray(X, dtype=float)
    n = X.shape[1]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n)]
    corr = np.corrcoef(X.T)
    fig = _new_fig(max(6, n * 0.6 + 2), max(5, n * 0.55 + 1.5))
    ax = fig.add_subplot(111, facecolor=_AXES_BG)
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=_TICK_COLOR, labelsize=8)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_names = [s[:12] for s in feature_names]
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7, color=_LABEL_COLOR)
    ax.set_yticklabels(short_names, fontsize=7, color=_LABEL_COLOR)
    ax.set_title("Correlation Matrix", color=_LABEL_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID_COLOR)
    if n <= 15:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(corr[i, j]) > 0.5 else _LABEL_COLOR)
    fig.tight_layout()
    return fig
