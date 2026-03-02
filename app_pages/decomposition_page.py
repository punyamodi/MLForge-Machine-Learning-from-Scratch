import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from utils.datasets import load_iris, load_breast_cancer, load_wine
from utils.preprocessing import StandardScaler
from utils.visualization import plot_2d_embedding, plot_pca_variance

_DATASET_LOADERS = {
    "Iris": load_iris,
    "Breast Cancer": load_breast_cancer,
    "Wine": load_wine,
}

_METHODS = ["PCA", "LDA", "t-SNE", "NMF", "TruncatedSVD", "KernelPCA", "ICA", "Factor Analysis"]


def _build_model(method, n_components, extra):
    if method == "PCA":
        from models.decomposition import PCA
        return PCA(n_components=n_components)
    if method == "LDA":
        from models.decomposition import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis(n_components=n_components)
    if method == "t-SNE":
        from models.decomposition import TSNE
        return TSNE(n_components=n_components, perplexity=extra.get("perplexity", 30))
    if method == "NMF":
        from models.decomposition import NMF
        return NMF(n_components=n_components)
    if method == "TruncatedSVD":
        from models.decomposition import TruncatedSVD
        return TruncatedSVD(n_components=n_components)
    if method == "KernelPCA":
        from models.decomposition import KernelPCA
        return KernelPCA(n_components=n_components, kernel=extra.get("kernel", "rbf"))
    if method == "ICA":
        from models.decomposition import IndependentComponentAnalysis
        return IndependentComponentAnalysis(n_components=n_components)
    if method == "Factor Analysis":
        from models.decomposition import FactorAnalysis
        return FactorAnalysis(n_components=n_components)
    raise ValueError(f"Unknown method: {method}")


def render():
    st.title("Dimensionality Reduction")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset")
    dataset_name = st.sidebar.selectbox("Dataset", list(_DATASET_LOADERS.keys()), key="decomp_ds")
    preprocessing = st.sidebar.selectbox(
        "Preprocessing", ["StandardScaler", "None"], key="decomp_preproc"
    )

    st.sidebar.subheader("Method")
    method = st.sidebar.selectbox("Algorithm", _METHODS, key="decomp_method")

    n_components = st.sidebar.slider("n_components", 2, 10, 2, key="decomp_nc")

    st.sidebar.subheader("Method Parameters")
    extra = {}
    if method == "t-SNE":
        extra["perplexity"] = st.sidebar.slider("perplexity", 5, 50, 30, key="decomp_perp")
    elif method == "KernelPCA":
        extra["kernel"] = st.sidebar.selectbox(
            "kernel", ["rbf", "poly", "cosine", "sigmoid"], key="decomp_kernel"
        )

    run_clicked = st.sidebar.button("Run", type="primary", key="decomp_run")

    if not run_clicked:
        st.info("Select a method and click **Run**.")
        return

    with st.spinner("Loading data..."):
        data = _DATASET_LOADERS[dataset_name]()

    X = data["X"]
    y = data["y"]
    feature_names = data.get("feature_names", [f"feature_{i}" for i in range(X.shape[1])])

    if preprocessing == "StandardScaler":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    n_components_safe = min(n_components, X.shape[0] - 1, X.shape[1])

    needs_nonneg = method == "NMF"
    if needs_nonneg:
        X = X - X.min(axis=0)

    try:
        model = _build_model(method, n_components_safe, extra)
    except Exception as exc:
        st.error(f"Model construction failed: {exc}")
        return

    with st.spinner(f"Running {method}..."):
        try:
            if method == "LDA":
                X_transformed = model.fit_transform(X, y)
            else:
                X_transformed = model.fit_transform(X)
        except Exception as exc:
            st.error(f"Transformation failed: {exc}")
            return

    X_transformed = np.asarray(X_transformed)

    st.success("Transformation complete.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Original Dimensions", X.shape[1])
    c2.metric("Reduced Dimensions", X_transformed.shape[1])
    c3.metric("Samples", X.shape[0])

    st.markdown("---")

    tab_embed, tab_var, tab_data = st.tabs(["2D Embedding", "Explained Variance", "Transformed Data"])

    with tab_embed:
        if X_transformed.shape[1] >= 2:
            try:
                fig = plot_2d_embedding(X_transformed[:, :2], y, title=f"{method} — 2D Projection")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                st.error(f"Embedding plot: {exc}")
        else:
            st.info("At least 2 components are required for a 2D scatter plot.")

    with tab_var:
        if method == "PCA":
            evr = getattr(model, "explained_variance_ratio_", None)
            if evr is not None:
                try:
                    fig = plot_pca_variance(evr)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                except Exception as exc:
                    st.error(f"Variance plot: {exc}")
            else:
                st.info("Explained variance ratio not available on this PCA instance.")
        else:
            st.info("Explained variance plot is available for PCA only.")

    with tab_data:
        comp_names = [f"Component_{i + 1}" for i in range(X_transformed.shape[1])]
        df_out = pd.DataFrame(X_transformed, columns=comp_names)
        df_out.insert(0, "label", y)
        st.dataframe(df_out.head(100), use_container_width=True)
        st.markdown("---")
        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Transformed Data",
            data=csv,
            file_name="transformed.csv",
            mime="text/csv",
        )

    if method == "PCA":
        recon = None
        try:
            if hasattr(model, "inverse_transform"):
                X_recon = model.inverse_transform(X_transformed)
                recon_err = float(np.mean((X - X_recon) ** 2))
                st.metric("Reconstruction MSE", f"{recon_err:.6f}")
        except Exception:
            pass
