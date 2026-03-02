import time

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from utils.datasets import (
    load_iris, load_breast_cancer, load_wine,
    load_diabetes, load_boston,
    make_classification, make_regression,
)
from utils.metrics import (
    accuracy_score,
    r2_score,
    cross_val_score,
    train_test_split,
)
from utils.preprocessing import StandardScaler

_CLF_DATASETS = {
    "Iris": load_iris,
    "Breast Cancer": load_breast_cancer,
    "Wine": load_wine,
    "Synthetic": None,
}

_REG_DATASETS = {
    "Diabetes": load_diabetes,
    "Boston": load_boston,
    "Synthetic": None,
}

_CLF_MODELS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "SVC",
    "KNN",
    "Gaussian Naive Bayes",
    "AdaBoost",
]

_REG_MODELS = [
    "Linear Regression",
    "Ridge",
    "Lasso",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "SVR",
    "KNN Regressor",
]


def _get_clf_model(name):
    if name == "Logistic Regression":
        from models.linear import LogisticRegression
        return LogisticRegression()
    if name == "Decision Tree":
        from models.trees import DecisionTreeClassifier
        return DecisionTreeClassifier()
    if name == "Random Forest":
        from models.trees import RandomForestClassifier
        return RandomForestClassifier(n_estimators=50)
    if name == "Gradient Boosting":
        from models.trees import GradientBoostingClassifier
        return GradientBoostingClassifier(n_estimators=50)
    if name == "SVC":
        from models.svm import SVC
        return SVC()
    if name == "KNN":
        from models.neighbors import KNeighborsClassifier
        return KNeighborsClassifier()
    if name == "Gaussian Naive Bayes":
        from models.bayes import GaussianNaiveBayes
        return GaussianNaiveBayes()
    if name == "AdaBoost":
        from models.trees import AdaBoostClassifier
        return AdaBoostClassifier(n_estimators=50)
    raise ValueError(f"Unknown classifier: {name}")


def _get_reg_model(name):
    if name == "Linear Regression":
        from models.linear import LinearRegression
        return LinearRegression()
    if name == "Ridge":
        from models.linear import RidgeRegression
        return RidgeRegression()
    if name == "Lasso":
        from models.linear import LassoRegression
        return LassoRegression()
    if name == "Decision Tree Regressor":
        from models.trees import DecisionTreeRegressor
        return DecisionTreeRegressor()
    if name == "Random Forest Regressor":
        from models.trees import RandomForestRegressor
        return RandomForestRegressor(n_estimators=50)
    if name == "Gradient Boosting Regressor":
        from models.trees import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=50)
    if name == "SVR":
        from models.svm import SVR
        return SVR()
    if name == "KNN Regressor":
        from models.neighbors import KNeighborsRegressor
        return KNeighborsRegressor()
    raise ValueError(f"Unknown regressor: {name}")


def render():
    st.title("Model Benchmarking")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Task")
    task = st.sidebar.radio("Task", ["Classification", "Regression"], key="bench_task")

    st.sidebar.subheader("Dataset")
    if task == "Classification":
        dataset_name = st.sidebar.selectbox(
            "Dataset", list(_CLF_DATASETS.keys()), key="bench_ds_clf"
        )
        model_options = _CLF_MODELS
        scoring = "accuracy"
    else:
        dataset_name = st.sidebar.selectbox(
            "Dataset", list(_REG_DATASETS.keys()), key="bench_ds_reg"
        )
        model_options = _REG_MODELS
        scoring = "r2"

    selected_models = st.sidebar.multiselect(
        "Models to Compare",
        model_options,
        default=model_options[:4],
        key="bench_models",
    )

    preprocessing = st.sidebar.selectbox(
        "Preprocessing", ["None", "StandardScaler"], key="bench_preproc"
    )
    cv_folds = st.sidebar.slider("CV Folds", 2, 10, 5, key="bench_cv")

    run_clicked = st.sidebar.button("Run Benchmark", type="primary", key="bench_run")

    if not run_clicked:
        st.info("Select models to compare and click **Run Benchmark**.")
        return

    if not selected_models:
        st.warning("Select at least one model.")
        return

    with st.spinner("Loading data..."):
        if task == "Classification":
            if dataset_name == "Synthetic":
                data = make_classification(n_samples=500, n_features=20, n_informative=10)
            else:
                data = _CLF_DATASETS[dataset_name]()
        else:
            if dataset_name == "Synthetic":
                data = make_regression(n_samples=500, n_features=20, n_informative=10, noise=0.1)
            else:
                data = _REG_DATASETS[dataset_name]()

    X = data["X"]
    y = data["y"]

    if preprocessing == "StandardScaler":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    results = []
    progress = st.progress(0)
    status = st.empty()

    for idx, model_name in enumerate(selected_models):
        status.text(f"Benchmarking {model_name} ({idx + 1}/{len(selected_models)})...")
        try:
            if task == "Classification":
                model = _get_clf_model(model_name)
            else:
                model = _get_reg_model(model_name)

            t_cv_start = time.perf_counter()
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring)
            cv_time = time.perf_counter() - t_cv_start

            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

            if task == "Classification":
                m2 = _get_clf_model(model_name)
            else:
                m2 = _get_reg_model(model_name)

            t0 = time.perf_counter()
            m2.fit(X_tr, y_tr)
            train_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            y_pred = m2.predict(X_te)
            predict_time = time.perf_counter() - t0

            results.append({
                "Model": model_name,
                "Mean Score": round(float(np.mean(cv_scores)), 4),
                "Std": round(float(np.std(cv_scores)), 4),
                "Train Time (s)": round(train_time, 4),
                "Predict Time (s)": round(predict_time, 6),
                "_cv_scores": cv_scores.tolist(),
            })
        except Exception as exc:
            st.warning(f"{model_name} failed: {exc}")

        progress.progress((idx + 1) / len(selected_models))

    status.empty()
    progress.empty()

    if not results:
        st.error("All selected models failed. Check your configuration.")
        return

    results_df = pd.DataFrame(results)
    display_df = (
        results_df.drop(columns=["_cv_scores"])
        .sort_values("Mean Score", ascending=False)
        .reset_index(drop=True)
    )

    st.subheader("Results")
    st.dataframe(
        display_df.style.format({"Mean Score": "{:.4f}", "Std": "{:.4f}"}),
        use_container_width=True,
    )

    st.markdown("---")

    tab_bar, tab_box = st.tabs(["Bar Chart", "Box Plot"])

    with tab_bar:
        sorted_df = results_df.sort_values("Mean Score", ascending=True)
        fig, ax = plt.subplots(figsize=(9, max(4, len(results) * 0.65 + 1)), facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        ax.barh(
            sorted_df["Model"],
            sorted_df["Mean Score"],
            xerr=sorted_df["Std"],
            color="#58a6ff",
            edgecolor="#21262d",
            capsize=4,
            ecolor="#8b949e",
            height=0.6,
        )
        ax.set_xlabel(f"Mean {scoring.upper()} Score ({cv_folds}-fold CV)", color="#c9d1d9")
        ax.set_title("Model Comparison", color="#c9d1d9")
        ax.tick_params(colors="#8b949e", labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor("#21262d")
        ax.grid(True, axis="x", color="#21262d", linewidth=0.5, linestyle="--", alpha=0.6)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab_box:
        cv_data = [r["_cv_scores"] for r in results]
        model_names_plot = [r["Model"] for r in results]
        fig, ax = plt.subplots(
            figsize=(max(8, len(results) * 1.5 + 1), 5), facecolor="#0d1117"
        )
        ax.set_facecolor("#161b22")
        bp = ax.boxplot(
            cv_data,
            patch_artist=True,
            labels=model_names_plot,
            medianprops={"color": "#f85149", "linewidth": 1.5},
            whiskerprops={"color": "#8b949e"},
            capprops={"color": "#8b949e"},
            flierprops={"markerfacecolor": "#8b949e", "markersize": 4},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor("#58a6ff")
            patch.set_alpha(0.6)
            patch.set_edgecolor("#21262d")
        ax.set_ylabel(f"{scoring.upper()} Score", color="#c9d1d9")
        ax.set_title("CV Score Distribution", color="#c9d1d9")
        ax.tick_params(colors="#8b949e", labelsize=8)
        plt.xticks(rotation=30, ha="right")
        for sp in ax.spines.values():
            sp.set_edgecolor("#21262d")
        ax.grid(True, axis="y", color="#21262d", linewidth=0.5, linestyle="--", alpha=0.6)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results CSV",
        data=csv,
        file_name="benchmark_results.csv",
        mime="text/csv",
    )
