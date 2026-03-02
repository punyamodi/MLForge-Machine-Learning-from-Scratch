import time

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from utils.datasets import load_iris, load_breast_cancer, load_wine, make_classification
from utils.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    train_test_split,
    classification_report,
)
from utils.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance,
    plot_learning_curve,
)

_DATASET_LOADERS = {
    "Iris": load_iris,
    "Breast Cancer": load_breast_cancer,
    "Wine": load_wine,
    "Synthetic": None,
}

_MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "SVC",
    "KNN",
    "Gaussian Naive Bayes",
    "MLP Classifier",
    "AdaBoost",
]


def _build_model(name, params):
    if name == "Logistic Regression":
        from models.linear import LogisticRegression
        return LogisticRegression(C=params["C"], max_iter=params["max_iter"])
    if name == "Decision Tree":
        from models.trees import DecisionTreeClassifier
        return DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
        )
    if name == "Random Forest":
        from models.trees import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
        )
    if name == "Gradient Boosting":
        from models.trees import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
        )
    if name == "SVC":
        from models.svm import SVC
        return SVC(C=params["C"], kernel=params["kernel"])
    if name == "KNN":
        from models.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            metric=params["metric"],
        )
    if name == "Gaussian Naive Bayes":
        from models.bayes import GaussianNaiveBayes
        return GaussianNaiveBayes()
    if name == "MLP Classifier":
        from models.neural import MLPClassifier
        return MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            activation=params["activation"],
            learning_rate_init=params["learning_rate_init"],
            max_iter=params["max_iter"],
        )
    if name == "AdaBoost":
        from models.trees import AdaBoostClassifier
        return AdaBoostClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
        )
    raise ValueError(f"Unknown model: {name}")


def _sidebar_hyperparams(model_name):
    params = {}
    if model_name == "Logistic Regression":
        params["C"] = st.sidebar.number_input("C (regularization)", 0.01, 100.0, 1.0, key="cls_C")
        params["max_iter"] = st.sidebar.slider("max_iter", 100, 1000, 300, 50, key="cls_maxiter")
    elif model_name == "Decision Tree":
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 20, 5, key="cls_maxdepth")
        params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 20, 2, key="cls_minsplit")
    elif model_name == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 10, 300, 100, 10, key="cls_nest")
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 20, 5, key="cls_maxdepth")
    elif model_name == "Gradient Boosting":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 10, 300, 100, 10, key="cls_nest")
        params["learning_rate"] = st.sidebar.number_input("learning_rate", 0.001, 1.0, 0.1, key="cls_lr")
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 10, 3, key="cls_maxdepth")
    elif model_name == "SVC":
        params["C"] = st.sidebar.number_input("C", 0.01, 100.0, 1.0, key="cls_C")
        params["kernel"] = st.sidebar.selectbox("kernel", ["rbf", "linear", "poly"], key="cls_kernel")
    elif model_name == "KNN":
        params["n_neighbors"] = st.sidebar.slider("n_neighbors", 1, 30, 5, key="cls_k")
        params["metric"] = st.sidebar.selectbox("metric", ["euclidean", "manhattan", "cosine"], key="cls_metric")
    elif model_name == "Gaussian Naive Bayes":
        pass
    elif model_name == "MLP Classifier":
        n_layers = st.sidebar.slider("Hidden layers", 1, 5, 2, key="cls_nlayers")
        neurons = st.sidebar.slider("Neurons per layer", 16, 256, 64, 16, key="cls_neurons")
        params["hidden_layer_sizes"] = tuple([neurons] * n_layers)
        params["activation"] = st.sidebar.selectbox("Activation", ["relu", "tanh", "sigmoid"], key="cls_act")
        params["learning_rate_init"] = st.sidebar.number_input(
            "learning_rate", 0.0001, 0.1, 0.001, format="%.4f", key="cls_lr"
        )
        params["max_iter"] = st.sidebar.slider("max_iter", 50, 500, 200, 25, key="cls_maxiter")
    elif model_name == "AdaBoost":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 10, 200, 50, 10, key="cls_nest")
        params["learning_rate"] = st.sidebar.number_input("learning_rate", 0.01, 2.0, 1.0, key="cls_lr")
    return params


def render():
    st.title("Classification")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset")
    dataset_name = st.sidebar.selectbox("Dataset", list(_DATASET_LOADERS.keys()), key="cls_dataset")
    test_size = st.sidebar.slider("Test Split", 0.1, 0.5, 0.2, 0.05, key="cls_test")
    preprocessing = st.sidebar.selectbox(
        "Preprocessing", ["None", "StandardScaler", "MinMaxScaler"], key="cls_preproc"
    )

    st.sidebar.subheader("Model")
    model_name = st.sidebar.selectbox("Model", _MODEL_NAMES, key="cls_model")

    st.sidebar.subheader("Hyperparameters")
    params = _sidebar_hyperparams(model_name)

    train_clicked = st.sidebar.button("Train Model", type="primary", key="cls_train")

    if not train_clicked:
        st.info("Configure your model in the sidebar and click **Train Model**.")
        return

    with st.spinner("Loading dataset..."):
        if dataset_name == "Synthetic":
            data = make_classification(n_samples=600, n_features=20, n_informative=10)
        else:
            data = _DATASET_LOADERS[dataset_name]()

    X = data["X"]
    y = data["y"]
    feature_names = data.get("feature_names", [f"feature_{i}" for i in range(X.shape[1])])

    le = LabelEncoder()
    y = le.fit_transform(y)

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
        y_pred = model.predict(X_test)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)
    except Exception:
        pass

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    n_classes = len(np.unique(y))
    roc_auc = None
    if y_prob is not None and n_classes == 2:
        try:
            roc_auc = roc_auc_score(y_test, y_prob[:, 1])
        except Exception:
            pass

    st.success(f"Training complete — {train_time:.3f}s")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("Precision", f"{prec:.4f}")
    c3.metric("Recall", f"{rec:.4f}")
    c4.metric("F1 Score", f"{f1:.4f}")
    c5.metric("ROC-AUC", f"{roc_auc:.4f}" if roc_auc is not None else "N/A")

    st.markdown("---")

    tab_cm, tab_roc, tab_fi, tab_report, tab_lc = st.tabs(
        ["Confusion Matrix", "ROC Curve", "Feature Importance", "Report", "Learning Curve"]
    )

    with tab_cm:
        try:
            fig = plot_confusion_matrix(y_test, y_pred)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as exc:
            st.error(f"Confusion matrix: {exc}")

    with tab_roc:
        if y_prob is not None and n_classes == 2:
            try:
                fig = plot_roc_curve(y_test, y_prob[:, 1])
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                st.error(f"ROC curve: {exc}")
        else:
            st.info("ROC curve requires binary classification with probability output.")

    with tab_fi:
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
            importances = importances[: len(feature_names)]

        if importances is not None:
            try:
                fig = plot_feature_importance(importances, feature_names)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                st.error(f"Feature importance: {exc}")
        else:
            st.info("Feature importance is not available for this model.")

    with tab_report:
        try:
            report_dict = classification_report(y_test, y_pred)
            report_dict.pop("accuracy", None)
            report_df = pd.DataFrame(report_dict).T
            st.dataframe(report_df.style.format("{:.4f}", na_rep="-"), use_container_width=True)
        except Exception as exc:
            st.error(f"Classification report: {exc}")

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
    if y_prob is not None:
        for i in range(y_prob.shape[1]):
            pred_df[f"prob_class_{i}"] = y_prob[:, i]
    csv = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Predictions", data=csv, file_name="predictions.csv", mime="text/csv")
