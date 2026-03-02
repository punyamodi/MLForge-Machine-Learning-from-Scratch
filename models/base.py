from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y): pass

    @abstractmethod
    def predict(self, X): pass


class BaseClassifier(BaseModel):
    @abstractmethod
    def predict_proba(self, X): pass

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class BaseRegressor(BaseModel):
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)


class BaseClusterer(ABC):
    @abstractmethod
    def fit(self, X): pass

    @abstractmethod
    def predict(self, X): pass

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


class BaseDecomposer(ABC):
    @abstractmethod
    def fit(self, X): pass

    @abstractmethod
    def transform(self, X): pass

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
