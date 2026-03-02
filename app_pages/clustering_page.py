import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from utils.datasets import load_iris, load_wine, make_blobs
from utils.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from utils.preprocessing import StandardScaler
from utils.visualization import (
    plot_cluster_scatter,
    plot_elbow_curve,
    plot_dendrogram,
    plot_silhouette,
)

_DATASET_OPTIONS = ["make_blobs", "Iris", "Wine"]

_MODEL_NAMES = [
    "KMeans",
    "DBSCAN",
    "Agglomerative Clustering",
    "Gaussian Mixture",
    "MeanShift",
    "Spectral Clustering",
]


def _build_model(name, params):
    if name == "KMeans":
        from models.clustering import KMeans
        return KMeans(
            n_clusters=params["n_clusters"],
            init=params.get("init", "k-means++"),
            max_iter=params.get("max_iter", 300),
        )
    if name == "DBSCAN":
        from models.clustering import DBSCAN
        return DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
    if name == "Agglomerative Clustering":
        from models.clustering import AgglomerativeClustering
        return AgglomerativeClustering(
            n_clusters=params["n_clusters"],
            linkage=params.get("linkage", "ward"),
        )
    if name == "Gaussian Mixture":
        from models.clustering import GaussianMixture
        return GaussianMixture(
            n_components=params["n_clusters"],
            covariance_type=params.get("covariance_type", "full"),
        )
    if name == "MeanShift":
        from models.clustering import MeanShift
        return MeanShift(bandwidth=params["bandwidth"])
    if name == "Spectral Clustering":
        from models.clustering import SpectralClustering
        return SpectralClustering(n_clusters=params["n_clusters"])
    raise ValueError(f"Unknown model: {name}")


def _fit_predict(model, X):
    if hasattr(model, "fit_predict"):
        return model.fit_predict(X)
    model.fit(X)
    if hasattr(model, "labels_"):
        return model.labels_
    if hasattr(model, "predict"):
        return model.predict(X)
    raise RuntimeError("Model has no fit_predict, labels_, or predict.")


def render():
    st.title("Clustering")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Source")
    dataset_name = st.sidebar.selectbox("Dataset", _DATASET_OPTIONS, key="clust_dataset")

    n_samples = n_features = centers = cluster_std = None
    if dataset_name == "make_blobs":
        n_samples = st.sidebar.slider("n_samples", 100, 2000, 500, 100, key="clust_ns")
        n_features = st.sidebar.slider("n_features", 2, 20, 2, key="clust_nf")
        centers = st.sidebar.slider("centers", 2, 10, 4, key="clust_centers")
        cluster_std = st.sidebar.slider("cluster_std", 0.1, 3.0, 1.0, 0.1, key="clust_std")

    st.sidebar.subheader("Algorithm")
    model_name = st.sidebar.selectbox("Algorithm", _MODEL_NAMES, key="clust_model")

    st.sidebar.subheader("Hyperparameters")
    params = {}
    needs_k = model_name in ("KMeans", "Agglomerative Clustering", "Gaussian Mixture", "Spectral Clustering")
    if needs_k:
        params["n_clusters"] = st.sidebar.slider("n_clusters", 2, 15, 4, key="clust_k")
    if model_name == "KMeans":
        params["init"] = st.sidebar.selectbox("init", ["k-means++", "random"], key="clust_init")
        params["max_iter"] = st.sidebar.slider("max_iter", 100, 500, 300, 50, key="clust_maxiter")
    if model_name == "DBSCAN":
        params["eps"] = st.sidebar.number_input("eps", 0.01, 5.0, 0.5, key="clust_eps")
        params["min_samples"] = st.sidebar.slider("min_samples", 1, 30, 5, key="clust_minpts")
    if model_name == "Agglomerative Clustering":
        params["linkage"] = st.sidebar.selectbox(
            "linkage", ["ward", "complete", "average", "single"], key="clust_linkage"
        )
    if model_name == "Gaussian Mixture":
        params["covariance_type"] = st.sidebar.selectbox(
            "covariance_type", ["full", "tied", "diag", "spherical"], key="clust_cov"
        )
    if model_name == "MeanShift":
        params["bandwidth"] = st.sidebar.number_input("bandwidth", 0.1, 10.0, 2.0, key="clust_bw")

    fit_clicked = st.sidebar.button("Fit Model", type="primary", key="clust_fit")

    if not fit_clicked:
        st.info("Configure the clustering algorithm in the sidebar and click **Fit Model**.")
        return

    with st.spinner("Preparing data..."):
        if dataset_name == "make_blobs":
            data = make_blobs(
                n_samples=n_samples,
                n_features=n_features,
                centers=centers,
                cluster_std=cluster_std,
            )
        elif dataset_name == "Iris":
            data = load_iris()
        else:
            data = load_wine()

    X = data["X"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        model = _build_model(model_name, params)
    except Exception as exc:
        st.error(f"Model construction failed: {exc}")
        return

    with st.spinner(f"Fitting {model_name}..."):
        try:
            labels = _fit_predict(model, X_scaled)
        except Exception as exc:
            st.error(f"Fitting failed: {exc}")
            return

    labels = np.asarray(labels)
    unique_labels = set(labels)
    n_found = len(unique_labels - {-1})

    valid_mask = labels != -1
    sil_score = db_score = ch_score = None
    if n_found >= 2 and valid_mask.sum() > n_found:
        try:
            sil_score = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
        except Exception:
            pass
        try:
            db_score = davies_bouldin_score(X_scaled[valid_mask], labels[valid_mask])
        except Exception:
            pass
        try:
            ch_score = calinski_harabasz_score(X_scaled[valid_mask], labels[valid_mask])
        except Exception:
            pass

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clusters Found", n_found)
    c2.metric("Silhouette Score", f"{sil_score:.4f}" if sil_score is not None else "N/A")
    c3.metric("Davies-Bouldin", f"{db_score:.4f}" if db_score is not None else "N/A")
    c4.metric("Calinski-Harabasz", f"{ch_score:.1f}" if ch_score is not None else "N/A")

    st.markdown("---")

    if X_scaled.shape[1] > 2:
        from models.decomposition import PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_scaled)
    else:
        X_2d = X_scaled[:, :2]

    tab_scatter, tab_elbow, tab_dendro, tab_sil = st.tabs(
        ["Scatter", "Elbow Curve", "Dendrogram", "Silhouette"]
    )

    with tab_scatter:
        try:
            fig = plot_cluster_scatter(X_2d, labels)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as exc:
            st.error(f"Scatter plot: {exc}")

    with tab_elbow:
        if model_name == "KMeans":
            with st.spinner("Computing elbow curve (k=2..10)..."):
                try:
                    from models.clustering import KMeans as _KM
                    k_vals = list(range(2, 11))
                    inertias = []
                    for k in k_vals:
                        km = _KM(n_clusters=k, init=params.get("init", "k-means++"), max_iter=200)
                        km.fit(X_scaled)
                        inertias.append(float(km.inertia_))
                    fig = plot_elbow_curve(k_vals, inertias)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception as exc:
                    st.error(f"Elbow curve: {exc}")
        else:
            st.info("Elbow curve is only available for KMeans.")

    with tab_dendro:
        if model_name == "Agglomerative Clustering":
            try:
                linkage_method = params.get("linkage", "ward")
                fig = plot_dendrogram(X_scaled, method=linkage_method)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                st.error(f"Dendrogram: {exc}")
        else:
            st.info("Dendrogram is only available for Agglomerative Clustering.")

    with tab_sil:
        if sil_score is not None:
            try:
                fig = plot_silhouette(X_scaled[valid_mask], labels[valid_mask])
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                st.error(f"Silhouette plot: {exc}")
        else:
            st.info("Silhouette plot requires at least 2 valid clusters with sufficient samples.")
