import numpy as np


class _MLP:
    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_deriv(self, z):
        return (z > 0).astype(float)

    def _sigmoid(self, z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def _sigmoid_deriv(self, a):
        return a * (1 - a)

    def _tanh(self, z):
        return np.tanh(z)

    def _tanh_deriv(self, a):
        return 1 - a ** 2

    def _softmax(self, z):
        z_shifted = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def _linear(self, z):
        return z

    def _linear_deriv(self, a):
        return np.ones_like(a)

    def _apply_activation(self, z, name):
        if name == 'relu':
            return self._relu(z)
        elif name == 'sigmoid':
            return self._sigmoid(z)
        elif name == 'tanh':
            return self._tanh(z)
        elif name == 'softmax':
            return self._softmax(z)
        elif name == 'linear':
            return self._linear(z)
        else:
            raise ValueError(f"Unknown activation: {name}")

    def _apply_activation_deriv(self, z, a, name):
        if name == 'relu':
            return self._relu_deriv(z)
        elif name == 'sigmoid':
            return self._sigmoid_deriv(a)
        elif name == 'tanh':
            return self._tanh_deriv(a)
        elif name == 'linear':
            return self._linear_deriv(a)
        else:
            raise ValueError(f"Unknown activation deriv: {name}")

    def _init_weights(self, layer_sizes):
        rng = np.random.RandomState(self.random_state)
        self.weights_ = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            if self.activation == 'relu':
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            W = rng.randn(fan_in, fan_out) * scale
            b = np.zeros(fan_out)
            self.weights_.append((W, b))

    def _forward(self, X, training=False):
        rng = np.random.RandomState(self.random_state)
        layer_outputs = []
        a = X
        n_layers = len(self.weights_)
        for i, (W, b) in enumerate(self.weights_):
            z = a @ W + b
            if i < n_layers - 1:
                a_new = self._apply_activation(z, self.activation)
                if training and self.dropout_rate > 0.0:
                    mask = (rng.rand(*a_new.shape) > self.dropout_rate).astype(float)
                    a_new = a_new * mask / (1.0 - self.dropout_rate + 1e-10)
                layer_outputs.append((z, a_new))
                a = a_new
            else:
                a_new = self._apply_activation(z, self._output_activation)
                layer_outputs.append((z, a_new))
                a = a_new
        return layer_outputs

    def _backward(self, layer_outputs, X, y_target):
        n_samples = X.shape[0]
        grads = []
        n_layers = len(self.weights_)

        z_out, a_out = layer_outputs[-1]

        if self._output_activation == 'softmax':
            delta = a_out - y_target
        elif self._output_activation == 'sigmoid':
            delta = a_out - y_target
        else:
            delta = (a_out - y_target) * 2.0 / n_samples

        grads_list = [None] * n_layers

        for i in reversed(range(n_layers)):
            z_i, a_i = layer_outputs[i]
            if i == 0:
                a_prev = X
            else:
                a_prev = layer_outputs[i - 1][1]

            dW = (a_prev.T @ delta) / n_samples + self.alpha * self.weights_[i][0]
            db = delta.mean(axis=0)
            grads_list[i] = (dW, db)

            if i > 0:
                W_i = self.weights_[i][0]
                z_prev, a_prev_val = layer_outputs[i - 1]
                d_act = self._apply_activation_deriv(z_prev, a_prev_val, self.activation)
                delta = (delta @ W_i.T) * d_act

        return grads_list

    def _adam_init(self):
        self._m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.weights_]
        self._v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.weights_]
        self._t = 0

    def _adam_update(self, grads):
        self._t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        lr = self.learning_rate_init
        new_weights = []
        new_m = []
        new_v = []
        for i, (dW, db) in enumerate(grads):
            mW, mb = self._m[i]
            vW, vb = self._v[i]

            mW_new = beta1 * mW + (1 - beta1) * dW
            mb_new = beta1 * mb + (1 - beta1) * db
            vW_new = beta2 * vW + (1 - beta2) * dW ** 2
            vb_new = beta2 * vb + (1 - beta2) * db ** 2

            mW_hat = mW_new / (1 - beta1 ** self._t)
            mb_hat = mb_new / (1 - beta1 ** self._t)
            vW_hat = vW_new / (1 - beta2 ** self._t)
            vb_hat = vb_new / (1 - beta2 ** self._t)

            W_new = self.weights_[i][0] - lr * mW_hat / (np.sqrt(vW_hat) + eps)
            b_new = self.weights_[i][1] - lr * mb_hat / (np.sqrt(vb_hat) + eps)

            new_weights.append((W_new, b_new))
            new_m.append((mW_new, mb_new))
            new_v.append((vW_new, vb_new))

        self.weights_ = new_weights
        self._m = new_m
        self._v = new_v

    def _sgd_update(self, grads):
        new_weights = []
        for i, (dW, db) in enumerate(grads):
            W_new = self.weights_[i][0] - self.learning_rate_init * dW
            b_new = self.weights_[i][1] - self.learning_rate_init * db
            new_weights.append((W_new, b_new))
        self.weights_ = new_weights

    def _fit_loop(self, X, y_target):
        n_samples = X.shape[0]
        self._adam_init()
        self.loss_curve_ = []
        rng = np.random.RandomState(self.random_state)

        prev_loss = np.inf
        for epoch in range(self.max_iter):
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_target[indices]
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                layer_outputs = self._forward(X_batch, training=True)
                a_out = layer_outputs[-1][1]

                if self._output_activation == 'softmax':
                    eps = 1e-12
                    loss = -np.mean(np.sum(y_batch * np.log(a_out + eps), axis=1))
                elif self._output_activation == 'sigmoid':
                    eps = 1e-12
                    loss = -np.mean(y_batch * np.log(a_out + eps) + (1 - y_batch) * np.log(1 - a_out + eps))
                else:
                    loss = np.mean((a_out - y_batch) ** 2)

                grads = self._backward(layer_outputs, X_batch, y_batch)

                if self.solver == 'adam':
                    self._adam_update(grads)
                else:
                    self._sgd_update(grads)

                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.loss_curve_.append(avg_loss)

            if abs(prev_loss - avg_loss) < self.tol:
                break
            prev_loss = avg_loss


class MLPClassifier(_MLP):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam',
                 alpha=0.0001, batch_size=32, learning_rate_init=0.001,
                 max_iter=200, tol=1e-4, dropout_rate=0.0, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.dropout_rate = dropout_rate
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        if n_classes == 2:
            self._output_activation = 'sigmoid'
            y_enc = (y == self.classes_[1]).astype(float).reshape(-1, 1)
            n_outputs = 1
        else:
            self._output_activation = 'softmax'
            y_enc = np.zeros((len(y), n_classes))
            for i, cls in enumerate(self.classes_):
                y_enc[y == cls, i] = 1.0
            n_outputs = n_classes

        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_outputs]
        self._init_weights(layer_sizes)
        self._fit_loop(X, y_enc)
        return self

    def predict_proba(self, X):
        layer_outputs = self._forward(X, training=False)
        probs = layer_outputs[-1][1]
        if self._output_activation == 'sigmoid':
            return np.hstack([1 - probs, probs])
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def __repr__(self):
        return (f"MLPClassifier(hidden_layer_sizes={self.hidden_layer_sizes}, "
                f"activation='{self.activation}', solver='{self.solver}', "
                f"max_iter={self.max_iter})")


class MLPRegressor(_MLP):
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam',
                 alpha=0.0001, batch_size=32, learning_rate_init=0.001,
                 max_iter=200, tol=1e-4, dropout_rate=0.0, random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.tol = tol
        self.dropout_rate = dropout_rate
        self.random_state = random_state

    def fit(self, X, y):
        self._output_activation = 'linear'
        y = np.array(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_features = X.shape[1]
        n_outputs = y.shape[1]
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_outputs]
        self._init_weights(layer_sizes)
        self._fit_loop(X, y)
        return self

    def predict(self, X):
        layer_outputs = self._forward(X, training=False)
        out = layer_outputs[-1][1]
        return out.ravel() if out.shape[1] == 1 else out

    def score(self, X, y):
        y_pred = self.predict(X)
        y = np.array(y, dtype=float)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / (ss_tot + 1e-10)

    def __repr__(self):
        return (f"MLPRegressor(hidden_layer_sizes={self.hidden_layer_sizes}, "
                f"activation='{self.activation}', solver='{self.solver}', "
                f"max_iter={self.max_iter})")
