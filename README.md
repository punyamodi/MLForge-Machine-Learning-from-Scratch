# MLForge -- Machine Learning from Scratch

A comprehensive machine learning library with 50+ algorithms implemented purely in NumPy, paired with an interactive Streamlit web application for training, visualization, and benchmarking.

---

## Overview

MLForge is a ground-up implementation of over 50 machine learning algorithms using only NumPy as its
computational backbone. The project bridges the gap between theoretical understanding and practical
implementation, offering both a clean Python API and an interactive Streamlit-based web interface for
experimentation, visualization, and benchmarking.

Every algorithm is implemented without relying on scikit-learn or any other high-level ML framework.
You can read the source code of any model and trace the math directly, from gradient descent in linear
regression to information gain splits in decision trees to the kernel trick in support vector machines.

The Streamlit application wraps all algorithms in an interactive UI that lets you load datasets,
configure hyperparameters, train models, visualize decision boundaries and learning curves, and compare
model performance side by side, all without writing a single line of code.

---

## Features

- 50+ machine learning algorithms implemented from scratch using NumPy
- Interactive Streamlit web application with real-time training and visualization
- Zero scikit-learn dependency for model logic
- Classification, regression, clustering, dimensionality reduction, and anomaly detection
- 5 built-in benchmark datasets and 7 synthetic data generators
- Preprocessing pipeline with scalers, encoders, imputers, and feature transformers
- Side-by-side model benchmarking with cross-validation and timing statistics
- Decision boundary visualization for 2D classification
- Confusion matrix, ROC curve, precision-recall curve, and residual plot support
- Modular architecture allowing individual model imports

---

## Model Library

| Category                 | Models                                                                                                                                                           | Count |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------|
| Linear Models            | Linear Regression, Logistic Regression, Ridge, Lasso, ElasticNet, Polynomial Regression, Huber Regression, Quantile Regression, Bayesian Ridge, Poisson Regression, SGD Classifier, SGD Regressor | 12    |
| Tree-Based Models        | Decision Tree Classifier, Decision Tree Regressor, Random Forest Classifier, Random Forest Regressor, Gradient Boosting Classifier, Gradient Boosting Regressor, AdaBoost Classifier, AdaBoost Regressor, Extra Trees Classifier, Extra Trees Regressor | 10    |
| Support Vector Machines  | SVC (SMO with RBF/linear/poly/sigmoid kernels), SVR, Linear SVC, Linear SVR, Nu-SVC                                                                            | 5     |
| Bayesian Models          | Gaussian Naive Bayes, Multinomial Naive Bayes, Bernoulli Naive Bayes, Bayesian Linear Regression, Gaussian Process Regressor, Gaussian Process Classifier        | 6     |
| Nearest Neighbors        | KNN Classifier, KNN Regressor, Radius Neighbors Classifier, Radius Neighbors Regressor, Local Outlier Factor                                                    | 5     |
| Ensemble Methods         | Bagging Classifier, Bagging Regressor, Voting Classifier, Voting Regressor, Stacking Classifier, Stacking Regressor                                             | 6     |
| Neural Networks          | Perceptron, MLP Classifier, MLP Regressor, Autoencoder, RBF Network, Self-Organizing Map                                                                        | 6     |
| Clustering               | K-Means (with k-means++ init), DBSCAN, Agglomerative Clustering, Gaussian Mixture Model, Mean Shift, Spectral Clustering, BIRCH                                 | 7     |
| Dimensionality Reduction | PCA, Linear Discriminant Analysis, Kernel PCA, t-SNE, NMF, Truncated SVD, Independent Component Analysis, Factor Analysis                                       | 8     |
| Anomaly Detection        | Isolation Forest, Local Outlier Factor, Elliptic Envelope, One-Class SVM                                                                                        | 4     |

**Total: 69 model classes across 10 categories**

---

## Project Structure

```
MLFromScratch/
|-- app.py                              Main Streamlit application entry point
|-- requirements.txt                    Python dependencies
|-- setup.py                            Package installation configuration
|-- README.md
|
|-- models/
|   |-- __init__.py
|   |-- base.py                         Abstract base classes for all model types
|   |
|   |-- linear/
|   |   |-- __init__.py
|   |   |-- linear_regression.py        OLS with normal equation and gradient descent
|   |   |-- logistic_regression.py      Binary and multiclass with L1/L2 regularization
|   |   |-- ridge.py                    Closed-form L2 regularized regression
|   |   |-- lasso.py                    Coordinate descent L1 regularized regression
|   |   |-- elastic_net.py             Combined L1/L2 regularization
|   |   |-- polynomial_regression.py    Polynomial feature expansion + linear fit
|   |   |-- huber_regression.py         Robust regression with Huber loss (IRLS)
|   |   |-- quantile_regression.py      Pinball loss minimization
|   |   |-- bayesian_ridge.py           Evidence maximization with posterior inference
|   |   |-- poisson_regression.py       GLM with log link (IRLS)
|   |   +-- sgd.py                      SGDClassifier and SGDRegressor
|   |
|   |-- trees/
|   |   |-- __init__.py
|   |   |-- decision_tree.py            Classifier and Regressor with Gini/entropy/MSE
|   |   |-- random_forest.py            Bootstrap aggregation with feature subsampling
|   |   |-- gradient_boosting.py        Stagewise additive boosting with decision trees
|   |   |-- adaboost.py                 SAMME algorithm with decision stump weak learners
|   |   +-- extra_trees.py              Extremely randomized trees
|   |
|   |-- svm/
|   |   |-- __init__.py
|   |   |-- svm.py                      SVC and SVR with SMO algorithm
|   |   |-- linear_svm.py              LinearSVC and LinearSVR with hinge/squared loss
|   |   +-- nu_svm.py                   Nu-SVM formulation
|   |
|   |-- bayes/
|   |   |-- __init__.py
|   |   |-- naive_bayes.py              Gaussian, Multinomial, and Bernoulli NB
|   |   |-- bayesian_linear.py          Bayesian linear regression with uncertainty
|   |   +-- gaussian_process.py         GP Regressor and Classifier with multiple kernels
|   |
|   |-- neighbors/
|   |   |-- __init__.py
|   |   +-- knn.py                      KNN, Radius Neighbors (classifier and regressor)
|   |
|   |-- ensemble/
|   |   |-- __init__.py
|   |   |-- bagging.py                  Bagging Classifier and Regressor
|   |   |-- voting.py                   Hard/soft voting Classifier and Regressor
|   |   +-- stacking.py                 Cross-validated stacking with meta-learner
|   |
|   |-- neural/
|   |   |-- __init__.py
|   |   |-- perceptron.py               Single-layer perceptron with step activation
|   |   |-- mlp.py                      MLP with backprop, Adam/SGD/RMSprop, dropout
|   |   |-- autoencoder.py              Encoder-decoder with reconstruction loss
|   |   |-- rbf_network.py              Radial basis function network
|   |   +-- som.py                      Self-Organizing Map (Kohonen network)
|   |
|   |-- clustering/
|   |   |-- __init__.py
|   |   |-- kmeans.py                   K-Means with random and k-means++ initialization
|   |   |-- dbscan.py                   Density-based clustering
|   |   |-- hierarchical.py             Agglomerative with single/complete/average/ward
|   |   |-- gmm.py                      Gaussian Mixture Model (EM algorithm)
|   |   |-- mean_shift.py              Kernel density based clustering
|   |   |-- spectral.py                Graph Laplacian based clustering
|   |   +-- birch.py                    Balanced Iterative Reducing and Clustering
|   |
|   |-- decomposition/
|   |   |-- __init__.py
|   |   |-- pca.py                      SVD and eigendecomposition based PCA
|   |   |-- lda.py                      Linear Discriminant Analysis
|   |   |-- kernel_pca.py              Kernel PCA with RBF/poly/cosine/sigmoid
|   |   |-- tsne.py                     t-SNE with KL divergence minimization
|   |   |-- nmf.py                      Non-negative Matrix Factorization
|   |   |-- truncated_svd.py           Randomized SVD
|   |   |-- ica.py                      FastICA (negentropy maximization)
|   |   +-- factor_analysis.py          EM-based factor analysis
|   |
|   +-- anomaly/
|       |-- __init__.py
|       |-- isolation_forest.py         Random partitioning anomaly scorer
|       |-- lof.py                      Local Outlier Factor
|       |-- elliptic_envelope.py        Robust covariance (MCD) anomaly detection
|       +-- one_class_svm.py            One-class SVM for novelty detection
|
|-- utils/
|   |-- __init__.py
|   |-- metrics.py                      30+ classification, regression, and clustering metrics
|   |-- preprocessing.py                Scalers, encoders, imputers, pipeline, transformers
|   |-- datasets.py                     5 benchmark datasets and 7 synthetic generators
|   +-- visualization.py               15 plotting utilities (matplotlib)
|
+-- app_pages/
    |-- __init__.py
    |-- datasets_page.py                Dataset exploration and EDA
    |-- classification_page.py          Classification model training and evaluation
    |-- regression_page.py              Regression model training and evaluation
    |-- clustering_page.py              Clustering analysis and visualization
    |-- decomposition_page.py           Dimensionality reduction visualization
    |-- anomaly_page.py                 Anomaly detection analysis
    |-- neural_page.py                  Neural network training with architecture builder
    +-- benchmark_page.py               Multi-model comparison with cross-validation
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/punyamodi/Logistic-Regression.git
cd Logistic-Regression
```

Create and activate a virtual environment (recommended):

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

The application opens in your default browser at `http://localhost:8501`. Use the left sidebar to navigate
between pages.

---

## Usage

### Exploring Datasets

The Datasets page lets you load and inspect built-in datasets before training any model. Built-in tabular
datasets include Iris, Wine, Breast Cancer, Diabetes, and Boston Housing. You can also upload a custom CSV
file. For each dataset you can view feature distributions, correlation matrices, class balance charts, and
descriptive statistics.

### Training Classification Models

Navigate to the Classification page, select a dataset, then pick any classifier from the sidebar. Configure
hyperparameters using sliders and input fields. Click "Train Model" to fit and view the confusion matrix,
ROC curve, and classification report including precision, recall, F1 score, and support per class.

### Regression Analysis

The Regression page works the same way for regression tasks. Output includes predicted-vs-actual scatter,
residual plot, and metrics: MSE, RMSE, MAE, R-squared, and adjusted R-squared.

### Clustering

Choose a dataset and clustering algorithm. After fitting, the app displays cluster assignment scatter plots
(with centroids where applicable), silhouette scores, and DBSCAN noise point visualization.

### Dimensionality Reduction

Project datasets down to 2D for visualization. Select PCA, t-SNE, LDA, NMF, or other methods. For PCA, the
explained variance ratio is plotted alongside the projection.

### Anomaly Detection

Fit outlier detection models, visualize inlier/outlier separation, and inspect anomaly score distributions.

### Benchmarking

Compare multiple models on the same dataset using k-fold cross-validation. Results are displayed as bar
charts and sortable tables with accuracy/R2, standard deviation, training time, and prediction time.

---

## Algorithm Details

### Linear Models

Linear regression uses the ordinary least squares closed-form solution (normal equations) with an optional
gradient descent solver. Ridge and Lasso add L2 and L1 regularization respectively; Lasso uses coordinate
descent. ElasticNet combines both penalties. Logistic regression supports binary and multiclass (one-vs-rest)
with configurable regularization. Huber regression uses the Huber loss for robustness to outliers via IRLS.
Bayesian Ridge places a Gaussian prior over weights and uses evidence maximization. The SGD module supports
hinge, log, and squared loss with L1/L2/ElasticNet penalties and multiple learning rate schedules.

### Tree-Based Models

Decision trees use recursive binary splitting with Gini impurity or entropy (classifier) and MSE or MAE
(regressor). Random Forest applies bootstrap aggregation with random feature subsets. Gradient Boosting fits
trees sequentially on negative gradients. AdaBoost uses SAMME with decision stump weak learners. Extra Trees
randomize both feature selection and split thresholds for faster training.

### Support Vector Machines

SVC and SVR are implemented using the SMO (Sequential Minimal Optimization) algorithm with support for
linear, RBF, polynomial, and sigmoid kernels. LinearSVC/SVR use hinge or squared loss with SGD. Nu-SVM
uses the nu parameter to control the fraction of support vectors.

### Bayesian Models

Gaussian, Multinomial, and Bernoulli Naive Bayes implement class-conditional likelihood estimation with
Laplace smoothing. Bayesian Linear Regression provides posterior uncertainty over predictions. Gaussian
Process Regressor and Classifier support RBF, Matern, Linear, Periodic, and Rational Quadratic kernels.

### Nearest Neighbors

KNN supports Euclidean, Manhattan, Chebyshev, and Minkowski distances with uniform or distance-based
weighting. Radius Neighbors uses a fixed-radius neighborhood. Both classifiers and regressors are provided.

### Ensemble Methods

Bagging uses bootstrap samples with configurable base estimators. Voting supports hard and soft voting with
per-estimator weights. Stacking trains a meta-learner on cross-validated out-of-fold predictions.

### Neural Networks

MLP supports arbitrary layer configurations with ReLU, sigmoid, tanh, and softmax activations.
Backpropagation with Adam, SGD, or RMSprop optimizers. Dropout regularization is configurable per layer.
The Autoencoder uses an encoder-decoder architecture for dimensionality reduction. The RBF Network uses
Gaussian radial basis functions with K-Means initialized centers. The Self-Organizing Map implements
competitive learning on a rectangular grid.

### Clustering

K-Means uses Lloyd's algorithm with k-means++ initialization. DBSCAN identifies core, border, and noise
points. Agglomerative Clustering supports single, complete, average, and Ward linkage. The Gaussian Mixture
Model uses EM with full, tied, diagonal, and spherical covariance types. Mean Shift uses kernel density
estimation. Spectral Clustering uses the graph Laplacian eigenvectors. BIRCH uses a CF-tree structure.

### Dimensionality Reduction

PCA via eigendecomposition or SVD. Kernel PCA supports RBF, polynomial, cosine, linear, and sigmoid
kernels. LDA maximizes between-class vs within-class scatter. t-SNE minimizes KL divergence with early
exaggeration and momentum. NMF uses multiplicative update rules. Truncated SVD uses randomized SVD. FastICA
maximizes negentropy. Factor Analysis uses EM for latent factor estimation.

### Anomaly Detection

Isolation Forest uses random partitioning trees with isolation depth scoring. Local Outlier Factor computes
local reachability density ratios. Elliptic Envelope uses minimum covariance determinant for robust
Mahalanobis distance. One-Class SVM learns a boundary around normal data using an RBF kernel.

---

## Utilities

### metrics.py

30+ metrics covering classification (accuracy, precision, recall, F1, ROC-AUC, log-loss, confusion matrix,
Cohen's kappa, Matthews correlation), regression (MSE, RMSE, MAE, MAPE, R2, adjusted R2, explained
variance, max error, Huber loss), clustering (silhouette, Davies-Bouldin, Calinski-Harabasz), and model
selection (cross-validation, train-test split, k-fold indices).

### preprocessing.py

StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, LabelEncoder, OneHotEncoder,
OrdinalEncoder, LabelBinarizer, PolynomialFeatures, SimpleImputer, Binarizer, PowerTransformer,
QuantileTransformer, KBinsDiscretizer, ColumnTransformer, and Pipeline. All follow the fit/transform API.

### datasets.py

Built-in datasets: Iris (150 samples, exact values), Wine, Breast Cancer, Diabetes, Boston Housing.
Generators: make_classification, make_regression, make_blobs, make_moons, make_circles, make_s_curve,
make_swiss_roll.

### visualization.py

15 plotting functions: confusion matrix, ROC curve, precision-recall curve, learning curve, feature
importance, decision boundary, residuals, correlation matrix, pairplot, cluster scatter, dendrogram,
elbow curve, silhouette plot, PCA variance, and 2D embedding visualization.

---

## API Reference

All models follow a consistent interface:

```python
from models.linear import LogisticRegression
from models.trees import RandomForestClassifier
from utils.datasets import make_moons
from utils.metrics import accuracy_score, train_test_split
from utils.preprocessing import StandardScaler

X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

Classifiers expose `predict_proba()`. Regressors return R2 from `score()`. Clustering models support
`fit_predict()`. Decomposition models support `fit_transform()` and `inverse_transform()`.

---

## Contributing

Contributions are welcome. To add a new algorithm:

1. Create a new module under the appropriate subdirectory in `models/`.
2. Implement `fit`, `predict`, and `score` methods following the existing interface.
3. Add the model to the corresponding `__init__.py` and Streamlit page dropdown.
4. Open a pull request with a description of the algorithm and implementation notes.

To report a bug or request a feature, open an issue on GitHub.

---

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
