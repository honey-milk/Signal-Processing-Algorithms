# -*- coding: utf-8 -*-
"""
@File    : kmeans.py
@Time    : 2023/4/12 23:42:14
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""
import numpy as np


class KMeans:
    """
    K-means clustering
    """
    def __init__(self, n_clusters, init_clusters=None, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.clusters = init_clusters.copy()
        self.max_iter = max_iter
        self.tol = tol
        self.label = None
        self.eps = 1e-10

    def fit(self, X):
        """
        fit

        :param X: shape: (n_samples, n_features)
        :return:
            label: shape (n_samples)
        """
        assert len(X.shape) == 2, 'X必须是2维'
        assert len(X) >= self.n_clusters, 'X数量小于类别数'
        # 初始化聚类中心
        if self.clusters is None:
            index = np.random.choice(range(len(X)), self.n_clusters, replace=False)
            clusters = X[index]
        else:
            clusters = self.clusters
            assert self.clusters.shape == (self.n_clusters, X.shape[1]), 'clusters shape有误'

        # 迭代
        last_total_distance = None
        for iter in range(self.max_iter):
            # M步骤
            # X = X[:, None]  # (n_samples, 1, n_features)
            # clusters = clusters[None]   # (1, n_clusters, n_features)
            # distances = X - clusters # (n_samples, n_clusters, n_features)
            distances = np.linalg.norm(X[:, None] - clusters[None], 2, axis=2)  # (n_samples, n_clusters)
            self.label = np.argmin(distances, axis=1)    # (n_samples)

            # 判断是否收敛
            total_distance = distances.min(axis=1).sum()
            if last_total_distance is not None:
                delta = np.abs(total_distance - last_total_distance) / (last_total_distance + self.eps)
                if delta <= self.tol:
                    break
            last_total_distance = total_distance

            # E步骤
            for i in range(self.n_clusters):
                mask = self.label == i
                if mask.sum() == 0:
                    continue
                clusters[i] = np.mean(X[mask])

        return self.label

    def predict(self, X):
        """
        predict

        :param X: shape: (n_samples, n_features)
        :return:
            label: shape (n_samples)
        """
        assert self.clusters is not None, 'clusters为空'
        assert len(X.shape) == 2 and X.shape[1] == self.clusters.shape[1], 'X必须是2维'

        distances = np.linalg.norm(X[:, None] - self.clusters[None], 2, axis=2)  # (n_samples, n_clusters)
        label = np.argmin(distances, axis=1)  # (n_samples)

        return label
