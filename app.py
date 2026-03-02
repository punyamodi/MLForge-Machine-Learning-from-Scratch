import streamlit as st

st.set_page_config(
    page_title="MLForge",
    layout="wide",
    initial_sidebar_state="expanded",
)

_CSS = """
<style>
[data-testid="stSidebar"] {
    background-color: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * {
    color: #c9d1d9;
}
.stRadio > div {
    gap: 0.3rem;
}
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1.1rem 1rem;
    text-align: center;
}
.metric-card .mc-value {
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
    line-height: 1;
    margin-bottom: 0.35rem;
}
.metric-card .mc-label {
    font-size: 0.8rem;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.feature-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.5rem;
}
.feature-card h4 {
    color: #f0f6fc;
    margin: 0 0 0.3rem 0;
    font-size: 0.95rem;
}
.feature-card p {
    color: #8b949e;
    font-size: 0.82rem;
    margin: 0;
}
.block-container {
    padding-top: 1.5rem;
}
h1, h2, h3, h4 {
    color: #f0f6fc;
}
.stButton > button {
    background-color: #238636;
    color: #ffffff;
    border: 1px solid #2ea043;
    border-radius: 6px;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #2ea043;
    border-color: #3fb950;
}
</style>
"""

_PAGES = [
    "Home",
    "Datasets",
    "Classification",
    "Regression",
    "Clustering",
    "Dimensionality Reduction",
    "Anomaly Detection",
    "Neural Networks",
    "Benchmarking",
]


def _render_home():
    st.title("MLForge")
    st.markdown("##### 50+ machine learning algorithms implemented purely in NumPy")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "50+", "Algorithms"),
        (c2, "10", "Model Families"),
        (c3, "8", "Datasets"),
        (c4, "0", "External ML Deps"),
    ]
    for col, value, label in cards:
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="mc-value">{value}</div>'
                f'<div class="mc-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("Feature Categories")

    categories = [
        ("Classification",
         "9 classifiers: Logistic Regression, SVM, Random Forest, Gradient Boosting, MLP, AdaBoost, KNN, Naive Bayes, and Decision Tree."),
        ("Regression",
         "10 regressors from simple linear methods to ensemble and neural network approaches."),
        ("Clustering",
         "6 algorithms: KMeans, DBSCAN, Agglomerative, Gaussian Mixture, MeanShift, and Spectral Clustering."),
        ("Dimensionality Reduction",
         "8 methods: PCA, LDA, t-SNE, NMF, TruncatedSVD, KernelPCA, ICA, and Factor Analysis."),
        ("Anomaly Detection",
         "4 detectors: Isolation Forest, Local Outlier Factor, Elliptic Envelope, and One-Class SVM."),
        ("Neural Networks",
         "Perceptron, MLP Classifier, MLP Regressor, Autoencoder, and RBF Network — all from scratch."),
        ("Benchmarking",
         "Compare multiple models side-by-side with cross-validation and timing statistics."),
        ("Dataset Explorer",
         "Built-in datasets with EDA: statistics, correlations, distributions, and class balance."),
    ]

    cols = st.columns(2)
    for i, (title, desc) in enumerate(categories):
        with cols[i % 2]:
            st.markdown(
                f'<div class="feature-card"><h4>{title}</h4><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )


def main():
    st.markdown(_CSS, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## MLForge")
        st.markdown("---")
        page = st.radio("Navigation", _PAGES, label_visibility="collapsed")
        st.markdown("---")
        st.markdown(
            '<span style="font-size:0.75rem;color:#6e7681;">NumPy-only ML implementations</span>',
            unsafe_allow_html=True,
        )

    if page == "Home":
        _render_home()
    elif page == "Datasets":
        from app_pages.datasets_page import render
        render()
    elif page == "Classification":
        from app_pages.classification_page import render
        render()
    elif page == "Regression":
        from app_pages.regression_page import render
        render()
    elif page == "Clustering":
        from app_pages.clustering_page import render
        render()
    elif page == "Dimensionality Reduction":
        from app_pages.decomposition_page import render
        render()
    elif page == "Anomaly Detection":
        from app_pages.anomaly_page import render
        render()
    elif page == "Neural Networks":
        from app_pages.neural_page import render
        render()
    elif page == "Benchmarking":
        from app_pages.benchmark_page import render
        render()


if __name__ == "__main__":
    main()
