import numpy as np


class CFEntry:
    def __init__(self, point):
        self.n = 1
        self.linear_sum = point.copy()
        self.sq_sum = np.sum(point ** 2)

    @property
    def centroid(self):
        return self.linear_sum / self.n

    def radius(self):
        mean = self.centroid
        return float(np.sqrt(max(self.sq_sum / self.n - np.sum(mean ** 2), 0)))

    def dist_to(self, point):
        return float(np.sqrt(np.sum((self.centroid - point) ** 2)))

    def merge(self, other):
        self.n += other.n
        self.linear_sum += other.linear_sum
        self.sq_sum += other.sq_sum


class BirchClustering:
    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters

    def fit(self, X):
        subclusters = []
        for point in X:
            if len(subclusters) == 0:
                subclusters.append(CFEntry(point))
                continue
            dists = np.array([sc.dist_to(point) for sc in subclusters])
            nearest_idx = int(np.argmin(dists))
            if dists[nearest_idx] <= self.threshold:
                sc = subclusters[nearest_idx]
                sc.n += 1
                sc.linear_sum += point
                sc.sq_sum += float(np.sum(point ** 2))
            else:
                subclusters.append(CFEntry(point))

        self.subcluster_centers_ = np.array([sc.centroid for sc in subclusters])

        if len(subclusters) <= self.n_clusters:
            subcluster_labels = np.arange(len(subclusters))
        else:
            from .hierarchical import AgglomerativeClustering
            agg = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward')
            agg.fit(self.subcluster_centers_)
            subcluster_labels = agg.labels_

        labels = np.zeros(X.shape[0], dtype=int)
        assignment_idx = 0
        subcluster_point_counts = [sc.n for sc in subclusters]

        i = 0
        for sc_idx, count in enumerate(subcluster_point_counts):
            for _ in range(count):
                if i < X.shape[0]:
                    labels[i] = int(subcluster_labels[sc_idx])
                    i += 1

        self.labels_ = labels
        self.subclusters_ = subclusters
        return self

    def predict(self, X):
        labels = np.zeros(X.shape[0], dtype=int)
        for i, point in enumerate(X):
            dists = np.sqrt(np.sum((self.subcluster_centers_ - point) ** 2, axis=1))
            nearest_sc = int(np.argmin(dists))
            labels[i] = int(self.labels_[nearest_sc]) if nearest_sc < len(self.labels_) else 0
        return labels

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def __repr__(self):
        return (f"BirchClustering(threshold={self.threshold}, "
                f"branching_factor={self.branching_factor}, "
                f"n_clusters={self.n_clusters})")
