import numpy as np


class Autoencoder:
    def __init__(self, hidden_layers=(64, 32), activation='relu', solver='adam',
                 alpha=0.0001, learning_rate_init=0.001, max_iter=200,
                 batch_size=32, random_state=None):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state

    def _activation(self, z, name):
        if name == 'relu':
            return np.maximum(0, z)
        elif name == 'sigmoid':
            return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))
        elif name == 'tanh':
            return np.tanh(z)
        elif name == 'linear':
            return z
        else:
            raise ValueError(f"Unknown activation: {name}")

    def _activation_deriv(self, z, a, name):
        if name == 'relu':
            return (z > 0).astype(float)
        elif name == 'sigmoid':
            return a * (1 - a)
        elif name == 'tanh':
            return 1 - a ** 2
        elif name == 'linear':
            return np.ones_like(a)
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

    def _forward(self, X):
        layer_outputs = []
        a = X
        n_layers = len(self.weights_)
        for i, (W, b) in enumerate(self.weights_):
            z = a @ W + b
            if i < n_layers - 1:
                a_new = self._activation(z, self.activation)
            else:
                a_new = self._activation(z, 'linear')
            layer_outputs.append((z, a_new))
            a = a_new
        return layer_outputs

    def _backward(self, layer_outputs, X):
        n_samples = X.shape[0]
        n_layers = len(self.weights_)
        grads = [None] * n_layers

        z_out, a_out = layer_outputs[-1]
        delta = 2.0 * (a_out - X) / n_samples

        for i in reversed(range(n_layers)):
            z_i, a_i = layer_outputs[i]
            a_prev = X if i == 0 else layer_outputs[i - 1][1]

            dW = a_prev.T @ delta + self.alpha * self.weights_[i][0]
            db = delta.mean(axis=0)
            grads[i] = (dW, db)

            if i > 0:
                W_i = self.weights_[i][0]
                z_prev, a_prev_val = layer_outputs[i - 1]
                act_name = self.activation if i - 1 < n_layers - 1 else 'linear'
                d_act = self._activation_deriv(z_prev, a_prev_val, act_name)
                delta = (delta @ W_i.T) * d_act

        return grads

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

    def fit(self, X):
        n_samples, n_features = X.shape
        hidden = list(self.hidden_layers)
        decoder_layers = list(reversed(hidden[:-1]))
        layer_sizes = [n_features] + hidden + decoder_layers + [n_features]
        self._latent_idx = len(hidden)
        self._init_weights(layer_sizes)
        self._adam_init()

        rng = np.random.RandomState(self.random_state)
        self.reconstruction_loss_ = []
        prev_loss = np.inf

        for epoch in range(self.max_iter):
            indices = rng.permutation(n_samples)
            X_shuffled = X[indices]
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]

                layer_outputs = self._forward(X_batch)
                a_out = layer_outputs[-1][1]
                loss = np.mean((a_out - X_batch) ** 2)
                grads = self._backward(layer_outputs, X_batch)
                self._adam_update(grads)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            self.reconstruction_loss_.append(avg_loss)

            if abs(prev_loss - avg_loss) < 1e-6:
                break
            prev_loss = avg_loss

        return self

    def encode(self, X):
        a = X
        for i in range(self._latent_idx):
            W, b = self.weights_[i]
            z = a @ W + b
            a = self._activation(z, self.activation)
        return a

    def decode(self, Z):
        a = Z
        n_layers = len(self.weights_)
        for i in range(self._latent_idx, n_layers):
            W, b = self.weights_[i]
            z = a @ W + b
            if i < n_layers - 1:
                a = self._activation(z, self.activation)
            else:
                a = self._activation(z, 'linear')
        return a

    def transform(self, X):
        return self.encode(X)

    def reconstruct(self, X):
        return self.decode(self.encode(X))

    def __repr__(self):
        return (f"Autoencoder(hidden_layers={self.hidden_layers}, "
                f"activation='{self.activation}', max_iter={self.max_iter})")
