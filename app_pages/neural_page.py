import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from utils.datasets import (
    load_iris, load_breast_cancer, load_wine,
    load_diabetes, make_classification, make_regression,
)
from utils.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    train_test_split,
)
from utils.preprocessing import StandardScaler
from utils.visualization import plot_confusion_matrix, plot_residuals

_CLF_DATASETS = {
    "Iris": load_iris,
    "Breast Cancer": load_breast_cancer,
    "Wine": load_wine,
    "Synthetic Classification": None,
}

_REG_DATASETS = {
    "Diabetes": load_diabetes,
    "Synthetic Regression": None,
}

_CLF_MODELS = ["MLPClassifier", "Perceptron", "RBFNetwork"]
_REG_MODELS = ["MLPRegressor", "RBFNetwork"]


def _load_data(name, task):
    if name == "Iris":
        return load_iris()
    if name == "Breast Cancer":
        return load_breast_cancer()
    if name == "Wine":
        return load_wine()
    if name == "Diabetes":
        return load_diabetes()
    if name == "Synthetic Classification":
        return make_classification(n_samples=500, n_features=20, n_informative=10)
    if name == "Synthetic Regression":
        return make_regression(n_samples=500, n_features=20, n_informative=10, noise=0.1)
    raise ValueError(f"Unknown dataset: {name}")


def _build_clf_model(name, arch_params, train_params):
    hidden = arch_params["hidden_layer_sizes"]
    act = arch_params["activation"]
    lr = train_params["learning_rate_init"]
    iters = train_params["max_iter"]

    if name == "MLPClassifier":
        from models.neural import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=act,
            learning_rate_init=lr,
            max_iter=iters,
        )
    if name == "Perceptron":
        from models.neural import Perceptron
        return Perceptron(max_iter=iters, eta0=lr)
    if name == "RBFNetwork":
        from models.neural import RBFNetwork
        return RBFNetwork(n_centers=hidden[0])
    raise ValueError(f"Unknown classifier: {name}")


def _build_reg_model(name, arch_params, train_params):
    hidden = arch_params["hidden_layer_sizes"]
    act = arch_params["activation"]
    lr = train_params["learning_rate_init"]
    iters = train_params["max_iter"]

    if name == "MLPRegressor":
        from models.neural import MLPRegressor
        return MLPRegressor(
            hidden_layer_sizes=hidden,
            activation=act,
            learning_rate_init=lr,
            max_iter=iters,
        )
    if name == "RBFNetwork":
        from models.neural import RBFNetwork
        return RBFNetwork(n_centers=hidden[0])
    raise ValueError(f"Unknown regressor: {name}")


def _build_autoencoder(arch_params, train_params):
    from models.neural import Autoencoder
    return Autoencoder(
        hidden_layers=arch_params["hidden_layer_sizes"],
        activation=arch_params["activation"],
        learning_rate_init=train_params["learning_rate_init"],
        max_iter=train_params["max_iter"],
    )


def render():
    st.title("Neural Networks")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Task")
    task = st.sidebar.radio("Task Type", ["Classification", "Regression", "Autoencoder"], key="nn_task")

    st.sidebar.subheader("Dataset")
    if task == "Classification":
        dataset_name = st.sidebar.selectbox("Dataset", list(_CLF_DATASETS.keys()), key="nn_ds_clf")
        model_choices = _CLF_MODELS
    elif task == "Regression":
        dataset_name = st.sidebar.selectbox("Dataset", list(_REG_DATASETS.keys()), key="nn_ds_reg")
        model_choices = _REG_MODELS
    else:
        dataset_name = st.sidebar.selectbox(
            "Dataset", list(_CLF_DATASETS.keys()), key="nn_ds_ae"
        )
        model_choices = ["Autoencoder"]

    if task != "Autoencoder":
        model_name = st.sidebar.selectbox("Model", model_choices, key="nn_model")
    else:
        model_name = "Autoencoder"

    test_size = st.sidebar.slider("Test Split", 0.1, 0.5, 0.2, 0.05, key="nn_test")

    st.sidebar.subheader("Architecture")
    n_hidden = st.sidebar.slider("Hidden Layers", 1, 5, 2, key="nn_nlayers")
    neurons = st.sidebar.slider("Neurons per Layer", 8, 256, 64, 8, key="nn_neurons")
    activation = st.sidebar.selectbox("Activation", ["relu", "tanh", "sigmoid"], key="nn_act")

    st.sidebar.subheader("Training")
    learning_rate = st.sidebar.number_input(
        "Learning Rate", 0.0001, 0.1, 0.001, format="%.4f", key="nn_lr"
    )
    max_iter = st.sidebar.slider("Epochs", 50, 500, 100, 25, key="nn_epochs")

    arch_params = {
        "hidden_layer_sizes": tuple([neurons] * n_hidden),
        "activation": activation,
    }
    train_params = {
        "learning_rate_init": learning_rate,
        "max_iter": max_iter,
    }

    train_clicked = st.sidebar.button("Train", type="primary", key="nn_train")

    if not train_clicked:
        st.info("Configure your model in the sidebar and click **Train**.")
        return

    with st.spinner("Loading data..."):
        try:
            data = _load_data(dataset_name, task)
        except Exception as exc:
            st.error(f"Data loading failed: {exc}")
            return

    X = data["X"]
    y = data["y"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    try:
        if task == "Autoencoder":
            model = _build_autoencoder(arch_params, train_params)
        elif task == "Classification":
            model = _build_clf_model(model_name, arch_params, train_params)
        else:
            model = _build_reg_model(model_name, arch_params, train_params)
    except Exception as exc:
        st.error(f"Model construction failed: {exc}")
        return

    with st.spinner(f"Training {model_name}..."):
        try:
            if task == "Autoencoder":
                model.fit(X_train)
            else:
                model.fit(X_train, y_train)
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            return

    st.success("Training complete.")

    if task == "Autoencoder":
        try:
            X_recon = model.reconstruct(X_test)
            recon_mse = float(np.mean((X_test - X_recon) ** 2))
        except Exception as exc:
            st.error(f"Reconstruction failed: {exc}")
            return

        st.metric("Reconstruction MSE", f"{recon_mse:.6f}")

        tab_recon, tab_loss = st.tabs(["Original vs Reconstructed", "Loss Curve"])

        with tab_recon:
            n_show = min(6, len(X_test))
            fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5), facecolor="#0d1117")
            for i in range(n_show):
                for row, (vals, title) in enumerate([(X_test[i], "Original"), (X_recon[i], "Reconstructed")]):
                    ax = axes[row, i]
                    ax.set_facecolor("#161b22")
                    ax.bar(range(len(vals)), vals, color="#58a6ff" if row == 0 else "#f78166",
                           edgecolor="#21262d", width=0.9)
                    ax.set_title(f"{title} {i + 1}", fontsize=7, color="#c9d1d9")
                    ax.tick_params(colors="#8b949e", labelsize=6)
                    ax.set_xticks([])
                    for sp in ax.spines.values():
                        sp.set_edgecolor("#21262d")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with tab_loss:
            loss_curve = getattr(model, "reconstruction_loss_", None)
            if loss_curve:
                _plot_loss_curve(loss_curve)
            else:
                st.info("Loss curve not available for this model.")

    elif task == "Classification":
        try:
            y_pred = model.predict(X_test)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        acc = accuracy_score(y_test, y_pred)
        st.metric("Test Accuracy", f"{acc:.4f}")

        tab_cm, tab_loss = st.tabs(["Confusion Matrix", "Loss Curve"])

        with tab_cm:
            try:
                fig = plot_confusion_matrix(y_test, y_pred)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                st.error(f"Confusion matrix: {exc}")

        with tab_loss:
            loss_curve = getattr(model, "loss_curve_", None)
            if loss_curve:
                _plot_loss_curve(loss_curve)
            else:
                st.info("Loss curve not available for this model.")

    else:
        try:
            y_pred = model.predict(X_test).astype(float)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        r2 = r2_score(y_test.astype(float), y_pred)
        mse = mean_squared_error(y_test.astype(float), y_pred)

        c1, c2 = st.columns(2)
        c1.metric("R\u00b2 Score", f"{r2:.4f}")
        c2.metric("MSE", f"{mse:.6f}")

        tab_resid, tab_loss = st.tabs(["Residuals", "Loss Curve"])

        with tab_resid:
            try:
                fig = plot_residuals(y_test.astype(float), y_pred)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                st.error(f"Residuals plot: {exc}")

        with tab_loss:
            loss_curve = getattr(model, "loss_curve_", None)
            if loss_curve:
                _plot_loss_curve(loss_curve)
            else:
                st.info("Loss curve not available for this model.")


def _plot_loss_curve(loss_curve):
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    ax.plot(range(1, len(loss_curve) + 1), loss_curve, color="#58a6ff", linewidth=1.5)
    ax.set_xlabel("Epoch", color="#c9d1d9")
    ax.set_ylabel("Loss", color="#c9d1d9")
    ax.set_title("Training Loss Curve", color="#c9d1d9")
    ax.tick_params(colors="#8b949e")
    for sp in ax.spines.values():
        sp.set_edgecolor("#21262d")
    ax.grid(True, color="#21262d", linewidth=0.5, linestyle="--", alpha=0.6)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
