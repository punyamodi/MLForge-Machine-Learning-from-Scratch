import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from utils.datasets import load_iris, load_breast_cancer, make_blobs
from utils.preprocessing import StandardScaler

_DATASET_OPTIONS = ["Blobs with Outliers", "Iris", "Breast Cancer"]

_MODEL_NAMES = [
    "Isolation Forest",
    "Local Outlier Factor",
    "Elliptic Envelope",
    "One-Class SVM",
]


def _inject_outliers(X, contamination, random_state=42):
    rng = np.random.RandomState(random_state)
    n_outliers = max(1, int(len(X) * contamination))
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    spread = (hi - lo) * 3.0
    center = (lo + hi) / 2.0
    outliers = rng.uniform(center - spread, center + spread, size=(n_outliers, X.shape[1]))
    X_out = np.vstack([X, outliers])
    true_labels = np.concatenate([np.ones(len(X), dtype=int), -np.ones(n_outliers, dtype=int)])
    return X_out, true_labels


def _build_model(name, params):
    if name == "Isolation Forest":
        from models.anomaly import IsolationForest
        return IsolationForest(
            n_estimators=params["n_estimators"],
            contamination=params["contamination"],
        )
    if name == "Local Outlier Factor":
        from models.anomaly import LocalOutlierFactor
        return LocalOutlierFactor(
            n_neighbors=params["n_neighbors"],
            contamination=params["contamination"],
        )
    if name == "Elliptic Envelope":
        from models.anomaly import EllipticEnvelope
        return EllipticEnvelope(contamination=params["contamination"])
    if name == "One-Class SVM":
        from models.anomaly import OneClassSVM
        return OneClassSVM(nu=params["nu"], kernel=params["kernel"])
    raise ValueError(f"Unknown model: {name}")


def render():
    st.title("Anomaly Detection")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Data")
    dataset_name = st.sidebar.selectbox("Dataset", _DATASET_OPTIONS, key="anom_dataset")

    n_samples = n_features = None
    if dataset_name == "Blobs with Outliers":
        n_samples = st.sidebar.slider("n_samples (clean)", 100, 1000, 300, 50, key="anom_ns")
        n_features = st.sidebar.slider("n_features", 2, 10, 2, key="anom_nf")

    st.sidebar.subheader("Algorithm")
    model_name = st.sidebar.selectbox("Algorithm", _MODEL_NAMES, key="anom_model")

    st.sidebar.subheader("Parameters")
    params = {}
    params["contamination"] = st.sidebar.slider("contamination", 0.01, 0.49, 0.1, 0.01, key="anom_cont")

    if model_name == "Isolation Forest":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 10, 200, 100, 10, key="anom_nest")
    elif model_name == "Local Outlier Factor":
        params["n_neighbors"] = st.sidebar.slider("n_neighbors", 5, 50, 20, 5, key="anom_k")
    elif model_name == "One-Class SVM":
        params["nu"] = st.sidebar.slider("nu", 0.01, 0.49, 0.1, 0.01, key="anom_nu")
        params["kernel"] = st.sidebar.selectbox("kernel", ["rbf", "linear", "poly"], key="anom_kernel")

    fit_clicked = st.sidebar.button("Fit Model", type="primary", key="anom_fit")

    if not fit_clicked:
        st.info("Configure the detector in the sidebar and click **Fit Model**.")
        return

    with st.spinner("Preparing data..."):
        if dataset_name == "Blobs with Outliers":
            data = make_blobs(n_samples=n_samples, n_features=n_features, centers=3)
            X_clean = data["X"]
        elif dataset_name == "Breast Cancer":
            data = load_breast_cancer()
            X_clean = data["X"]
        else:
            data = load_iris()
            X_clean = data["X"]

        X, true_labels = _inject_outliers(X_clean, contamination=params["contamination"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        model = _build_model(model_name, params)
    except Exception as exc:
        st.error(f"Model construction failed: {exc}")
        return

    with st.spinner(f"Fitting {model_name}..."):
        try:
            if hasattr(model, "fit_predict"):
                pred_labels = model.fit_predict(X_scaled)
            else:
                model.fit(X_scaled)
                pred_labels = model.predict(X_scaled)
        except Exception as exc:
            st.error(f"Fitting failed: {exc}")
            return

    pred_labels = np.asarray(pred_labels)
    n_outliers = int((pred_labels == -1).sum())
    n_inliers = int((pred_labels == 1).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", len(X_scaled))
    c2.metric("Outliers Detected", n_outliers)
    c3.metric("Inliers", n_inliers)

    st.markdown("---")

    tab_scatter, tab_hist = st.tabs(["Scatter Plot", "Score Distribution"])

    with tab_scatter:
        if X_scaled.shape[1] > 2:
            from models.decomposition import PCA
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_scaled)
        else:
            X_2d = X_scaled[:, :2]

        inlier_mask = pred_labels == 1
        outlier_mask = pred_labels == -1

        fig, ax = plt.subplots(figsize=(9, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        ax.scatter(
            X_2d[inlier_mask, 0], X_2d[inlier_mask, 1],
            c="#2da44e", label=f"Inliers ({inlier_mask.sum()})",
            alpha=0.5, s=20, edgecolors="none",
        )
        ax.scatter(
            X_2d[outlier_mask, 0], X_2d[outlier_mask, 1],
            c="#f85149", label=f"Outliers ({outlier_mask.sum()})",
            alpha=0.9, s=60, marker="x", linewidths=1.5,
        )
        ax.set_xlabel("Component 1", color="#c9d1d9")
        ax.set_ylabel("Component 2", color="#c9d1d9")
        ax.set_title(f"{model_name} — Anomaly Detection", color="#c9d1d9")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#21262d")
        ax.grid(True, color="#21262d", linewidth=0.5, linestyle="--", alpha=0.6)
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab_hist:
        scores = None
        score_label = "Score"

        if hasattr(model, "score_samples"):
            try:
                scores = model.score_samples(X_scaled)
                score_label = "Anomaly Score"
            except Exception:
                pass

        if scores is None and hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(X_scaled)
                score_label = "Decision Score"
            except Exception:
                pass

        if scores is not None:
            scores = np.asarray(scores)
            fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0d1117")
            ax.set_facecolor("#161b22")
            ax.hist(
                scores[inlier_mask], bins=40, alpha=0.7,
                color="#2da44e", label="Inliers", edgecolor="#21262d",
            )
            ax.hist(
                scores[outlier_mask], bins=40, alpha=0.7,
                color="#f85149", label="Outliers", edgecolor="#21262d",
            )
            ax.set_xlabel(score_label, color="#c9d1d9")
            ax.set_ylabel("Count", color="#c9d1d9")
            ax.set_title("Score Distribution", color="#c9d1d9")
            ax.tick_params(colors="#8b949e")
            for sp in ax.spines.values():
                sp.set_edgecolor("#21262d")
            ax.grid(True, color="#21262d", linewidth=0.5, linestyle="--", alpha=0.6)
            ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Score distribution is not available for this model variant.")

    with st.expander("Injected vs Detected Outliers"):
        n_injected = int((true_labels == -1).sum())
        true_positives = int(((pred_labels == -1) & (true_labels == -1)).sum())
        false_positives = int(((pred_labels == -1) & (true_labels == 1)).sum())
        false_negatives = int(((pred_labels == 1) & (true_labels == -1)).sum())
        precision = true_positives / (true_positives + false_positives + 1e-9)
        recall = true_positives / (true_positives + false_negatives + 1e-9)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Injected Outliers", n_injected)
        c2.metric("True Positives", true_positives)
        c3.metric("Precision", f"{precision:.3f}")
        c4.metric("Recall", f"{recall:.3f}")
