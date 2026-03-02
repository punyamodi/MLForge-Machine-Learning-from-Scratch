import time

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from utils.datasets import load_diabetes, load_boston, make_regression
from utils.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    adjusted_r2_score,
    train_test_split,
)
from utils.preprocessing import StandardScaler, MinMaxScaler
from utils.visualization import plot_residuals, plot_learning_curve, plot_feature_importance

_DATASET_LOADERS = {
    "Diabetes": load_diabetes,
    "Boston": load_boston,
    "Synthetic": None,
}

_MODEL_NAMES = [
    "Linear Regression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "Gradient Boosting Regressor",
    "SVR",
    "KNN Regressor",
    "MLP Regressor",
]


def _build_model(name, params):
    if name == "Linear Regression":
        from models.linear import LinearRegression
        return LinearRegression()
    if name == "Ridge":
        from models.linear import RidgeRegression
        return RidgeRegression(alpha=params["alpha"])
    if name == "Lasso":
        from models.linear import LassoRegression
        return LassoRegression(alpha=params["alpha"])
    if name == "ElasticNet":
        from models.linear import ElasticNet
        return ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"])
    if name == "Decision Tree Regressor":
        from models.trees import DecisionTreeRegressor
        return DecisionTreeRegressor(max_depth=params["max_depth"])
    if name == "Random Forest Regressor":
        from models.trees import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
        )
    if name == "Gradient Boosting Regressor":
        from models.trees import GradientBoostingRegressor
        return GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
        )
    if name == "SVR":
        from models.svm import SVR
        return SVR(C=params["C"], kernel=params["kernel"])
    if name == "KNN Regressor":
        from models.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(n_neighbors=params["n_neighbors"])
    if name == "MLP Regressor":
        from models.neural import MLPRegressor
        return MLPRegressor(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            activation=params["activation"],
            learning_rate_init=params["learning_rate_init"],
            max_iter=params["max_iter"],
        )
    raise ValueError(f"Unknown model: {name}")


def _sidebar_hyperparams(model_name):
    params = {}
    if model_name in ("Ridge", "Lasso"):
        params["alpha"] = st.sidebar.number_input("alpha", 0.0001, 100.0, 1.0, key="reg_alpha")
    elif model_name == "ElasticNet":
        params["alpha"] = st.sidebar.number_input("alpha", 0.0001, 100.0, 1.0, key="reg_alpha")
        params["l1_ratio"] = st.sidebar.slider("l1_ratio", 0.0, 1.0, 0.5, 0.05, key="reg_l1")
    elif model_name == "Decision Tree Regressor":
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 20, 5, key="reg_maxdepth")
    elif model_name == "Random Forest Regressor":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 10, 300, 100, 10, key="reg_nest")
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 20, 5, key="reg_maxdepth")
    elif model_name == "Gradient Boosting Regressor":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 10, 300, 100, 10, key="reg_nest")
        params["learning_rate"] = st.sidebar.number_input("learning_rate", 0.001, 1.0, 0.1, key="reg_lr")
    elif model_name == "SVR":
        params["C"] = st.sidebar.number_input("C", 0.01, 100.0, 1.0, key="reg_C")
        params["kernel"] = st.sidebar.selectbox("kernel", ["rbf", "linear", "poly"], key="reg_kernel")
    elif model_name == "KNN Regressor":
        params["n_neighbors"] = st.sidebar.slider("n_neighbors", 1, 30, 5, key="reg_k")
    elif model_name == "MLP Regressor":
        n_layers = st.sidebar.slider("Hidden layers", 1, 5, 2, key="reg_nlayers")
        neurons = st.sidebar.slider("Neurons per layer", 16, 256, 64, 16, key="reg_neurons")
        params["hidden_layer_sizes"] = tuple([neurons] * n_layers)
        params["activation"] = st.sidebar.selectbox("Activation", ["relu", "tanh", "sigmoid"], key="reg_act")
        params["learning_rate_init"] = st.sidebar.number_input(
            "learning_rate", 0.0001, 0.1, 0.001, format="%.4f", key="reg_lr"
        )
        params["max_iter"] = st.sidebar.slider("max_iter", 50, 500, 200, 25, key="reg_maxiter")
    return params


def render():
    st.title("Regression")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset")
    dataset_name = st.sidebar.selectbox("Dataset", list(_DATASET_LOADERS.keys()), key="reg_dataset")
    test_size = st.sidebar.slider("Test Split", 0.1, 0.5, 0.2, 0.05, key="reg_test")
    preprocessing = st.sidebar.selectbox(
        "Preprocessing", ["None", "StandardScaler", "MinMaxScaler"], key="reg_preproc"
    )

    st.sidebar.subheader("Model")
    model_name = st.sidebar.selectbox("Model", _MODEL_NAMES, key="reg_model")

    st.sidebar.subheader("Hyperparameters")
    params = _sidebar_hyperparams(model_name)

    train_clicked = st.sidebar.button("Train Model", type="primary", key="reg_train")

    if not train_clicked:
        st.info("Configure your model in the sidebar and click **Train Model**.")
        return

    with st.spinner("Loading dataset..."):
        if dataset_name == "Synthetic":
            data = make_regression(n_samples=500, n_features=20, n_informative=10, noise=0.1)
        else:
            data = _DATASET_LOADERS[dataset_name]()

    X = data["X"]
    y = data["y"].astype(float)
    feature_names = data.get("feature_names", [f"feature_{i}" for i in range(X.shape[1])])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if preprocessing == "StandardScaler":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif preprocessing == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    try:
        model = _build_model(model_name, params)
    except Exception as exc:
        st.error(f"Model construction failed: {exc}")
        return

    with st.spinner(f"Training {model_name}..."):
        t0 = time.perf_counter()
        try:
            model.fit(X_train, y_train)
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            return
        train_time = time.perf_counter() - t0

    try:
        y_pred = model.predict(X_test).astype(float)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = adjusted_r2_score(y_test, y_pred, X_test.shape[1])

    st.success(f"Training complete — {train_time:.3f}s")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MSE", f"{mse:.4f}")
    c2.metric("RMSE", f"{rmse:.4f}")
    c3.metric("MAE", f"{mae:.4f}")
    c4.metric("R\u00b2", f"{r2:.4f}")
    c5.metric("Adjusted R\u00b2", f"{adj_r2:.4f}")

    st.markdown("---")

    tab_resid, tab_pred, tab_fi, tab_lc = st.tabs(
        ["Residuals", "Actual vs Predicted", "Feature Importance", "Learning Curve"]
    )

    with tab_resid:
        try:
            fig = plot_residuals(y_test, y_pred)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as exc:
            st.error(f"Residuals plot: {exc}")

    with tab_pred:
        fig, ax = plt.subplots(figsize=(7, 6), facecolor="#0d1117")
        ax.set_facecolor("#161b22")
        ax.scatter(y_test, y_pred, alpha=0.6, color="#58a6ff", s=25, edgecolors="none")
        mn = min(float(y_test.min()), float(y_pred.min()))
        mx = max(float(y_test.max()), float(y_pred.max()))
        ax.plot([mn, mx], [mn, mx], color="#f85149", linewidth=1.5, linestyle="--")
        ax.set_xlabel("Actual", color="#c9d1d9")
        ax.set_ylabel("Predicted", color="#c9d1d9")
        ax.set_title("Actual vs Predicted", color="#c9d1d9")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#21262d")
        ax.grid(True, color="#21262d", linewidth=0.5, linestyle="--", alpha=0.6)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab_fi:
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_).ravel()
            importances = np.abs(coef)[: len(feature_names)]

        if importances is not None:
            try:
                fig = plot_feature_importance(importances, feature_names)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                st.error(f"Feature importance: {exc}")
        else:
            st.info("Feature importance is not available for this model.")

    with tab_lc:
        with st.spinner("Computing learning curve..."):
            try:
                model_lc = _build_model(model_name, params)
                fig = plot_learning_curve(model_lc, X_train, y_train, cv=3)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                st.error(f"Learning curve: {exc}")

    st.markdown("---")
    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    csv = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
