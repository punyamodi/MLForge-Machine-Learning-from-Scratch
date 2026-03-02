import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.datasets import load_iris, load_breast_cancer, load_diabetes, load_wine, load_boston
from utils.visualization import plot_correlation_matrix

_LOADERS = {
    "Iris": load_iris,
    "Breast Cancer": load_breast_cancer,
    "Diabetes": load_diabetes,
    "Wine": load_wine,
    "Boston": load_boston,
}

_CLASSIFICATION = {"Iris", "Breast Cancer", "Wine"}


@st.cache_data
def _load_builtin(name):
    return _LOADERS[name]()


def render():
    st.title("Dataset Explorer")

    source = st.radio("Data Source", ["Built-in Dataset", "Upload CSV"], horizontal=True)

    X = y = feature_names = target_name = None
    is_classification = False

    if source == "Built-in Dataset":
        dataset_name = st.selectbox("Select Dataset", list(_LOADERS.keys()))
        with st.spinner("Loading..."):
            data = _load_builtin(dataset_name)
        X = data["X"]
        y = data["y"]
        feature_names = data.get("feature_names", [f"feature_{i}" for i in range(X.shape[1])])
        target_name = data.get("target_name", "target")
        is_classification = dataset_name in _CLASSIFICATION
    else:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV file to begin exploration.")
            return
        raw_df = pd.read_csv(uploaded)
        target_col = st.selectbox("Select target column", raw_df.columns.tolist())
        feature_cols = [c for c in raw_df.columns if c != target_col]
        if not feature_cols:
            st.error("The file must have at least one feature column.")
            return
        try:
            X = raw_df[feature_cols].values.astype(float)
        except ValueError as exc:
            st.error(f"Could not convert features to numeric: {exc}")
            return
        y = raw_df[target_col].values
        feature_names = feature_cols
        target_name = target_col
        is_classification = raw_df[target_col].nunique() <= 20

    if X is None:
        return

    feature_names = list(feature_names)
    df = pd.DataFrame(X, columns=feature_names)
    df[target_name] = y

    st.markdown("---")

    tab_overview, tab_stats, tab_corr, tab_dist, tab_target = st.tabs(
        ["Overview", "Statistics", "Correlation", "Distributions", "Target"]
    )

    with tab_overview:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Samples", X.shape[0])
        col2.metric("Features", X.shape[1])
        unique_y = np.unique(y)
        if is_classification:
            col3.metric("Classes", len(unique_y))
        else:
            col3.metric("Target Range", f"{float(y.min()):.2f} — {float(y.max()):.2f}")
        col4.metric("Missing Values", int(df.isnull().sum().sum()))

        st.subheader("Data Preview")
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("Column Information")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Dtype": [str(dt) for dt in df.dtypes.values],
            "Non-Null": df.count().values,
            "Null": df.isnull().sum().values,
        })
        st.dataframe(info_df, use_container_width=True)

    with tab_stats:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    with tab_corr:
        st.subheader("Correlation Matrix")
        if X.shape[1] < 2:
            st.info("Need at least 2 features for a correlation matrix.")
        else:
            try:
                fig = plot_correlation_matrix(X, feature_names)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                st.error(f"Correlation matrix error: {exc}")

    with tab_dist:
        st.subheader("Feature Distributions")
        n_features = X.shape[1]
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), facecolor="#0d1117")
        axes_flat = np.array(axes).flatten()
        for i, fname in enumerate(feature_names):
            ax = axes_flat[i]
            ax.set_facecolor("#161b22")
            ax.hist(X[:, i], bins=30, color="#58a6ff", edgecolor="#21262d", alpha=0.85)
            ax.set_title(fname, fontsize=8, color="#c9d1d9")
            ax.tick_params(colors="#8b949e", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#21262d")
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab_target:
        st.subheader("Target Distribution")
        if is_classification:
            unique_vals, counts = np.unique(y, return_counts=True)
            fig, ax = plt.subplots(figsize=(max(6, len(unique_vals) * 0.8 + 2), 4),
                                   facecolor="#0d1117")
            ax.set_facecolor("#161b22")
            ax.bar([str(v) for v in unique_vals], counts, color="#58a6ff", edgecolor="#21262d")
            ax.set_xlabel("Class", color="#c9d1d9")
            ax.set_ylabel("Count", color="#c9d1d9")
            ax.set_title("Class Balance", color="#c9d1d9")
            ax.tick_params(colors="#8b949e")
            for sp in ax.spines.values():
                sp.set_edgecolor("#21262d")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            balance_df = pd.DataFrame({
                "Class": unique_vals,
                "Count": counts,
                "Proportion": (counts / counts.sum()).round(4),
            })
            st.dataframe(balance_df, use_container_width=True)
        else:
            y_float = y.astype(float)
            fig, ax = plt.subplots(figsize=(8, 4), facecolor="#0d1117")
            ax.set_facecolor("#161b22")
            ax.hist(y_float, bins=40, color="#58a6ff", edgecolor="#21262d", alpha=0.85)
            ax.set_xlabel(target_name, color="#c9d1d9")
            ax.set_ylabel("Frequency", color="#c9d1d9")
            ax.set_title("Target Distribution", color="#c9d1d9")
            ax.tick_params(colors="#8b949e")
            for sp in ax.spines.values():
                sp.set_edgecolor("#21262d")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("---")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Dataset as CSV",
        data=csv_bytes,
        file_name="dataset.csv",
        mime="text/csv",
    )
