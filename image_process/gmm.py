# -*- coding: utf-8 -*-
"""
@File    : gmm.py
@Time    : 2023/4/15 21:35:42
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""

import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    """
    Gaussian Mixture Model clustering
    """

    def __init__(self, n_clusters, num_gaussian=1, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.num_gaussian = num_gaussian
        self.max_iter = max_iter
        self.tol = tol
        self.label = None
        self.eps = 1e-10
        self.miu = None
        self.sigma = None
        self.weight = None

    def fit(self, X):
        """
        fit

        :param X: shape: (n_samples, n_features)
        :return:
            label: shape (n_samples)
        """
        assert len(X.shape) == 2, 'X必须是2维'
        assert len(X) >= self.n_clusters, 'X数量小于类别数'
        n_samples, n_features = X.shape

        # 初始化参数
        self.miu = np.zeros((self.n_clusters, self.num_gaussian, n_features))
        self.sigma = np.zeros((self.n_clusters, self.num_gaussian, n_features, n_features))
        self.sigma[:, :] = np.eye(n_features)
        self.weight = np.full((self.n_clusters, self.num_gaussian,),  1.0 / self.num_gaussian)
        for i in range(self.num_gaussian):
            index = np.random.choice(range(len(X)), self.n_clusters, replace=False)
            self.miu[:, i] = X[index]

        # 迭代
        last_total_prod = None
        for iter in range(self.max_iter):
            # M步骤
            prob = np.zeros((n_samples, self.n_clusters, self.num_gaussian))
            for k in range(self.n_clusters):
                for i in range(self.num_gaussian):
                    prob[:, k, i] = multivariate_normal.pdf(X, mean=self.miu[k, i], cov=self.sigma[k, i])

            prob_cluster = (prob * self.weight).sum(axis=-1)  # shape: (n_samples, n_clusters)
            self.label = np.argmax(prob_cluster, axis=-1)  # shape: (n_samples)

            # 判断是否收敛
            total_prod = prob_cluster.max(axis=1).prod()
            if last_total_prod is not None:
                delta = np.abs(total_prod - last_total_prod) / (last_total_prod + self.eps)
                if delta <= self.tol:
                    break
            last_total_prod = total_prod

            # E步骤
            for i in range(self.n_clusters):
                mask = self.label == i
                if mask.sum() == 0:
                    continue
                prob_ = prob[mask][:, i]  # shape: (N, num_gaussian)
                X_ = X[mask]  # shape: (N, n_features)
                # miu[i]: shape (num_gaussian, n_features)
                self.miu[i] = (prob_[..., None] * X_[:, None]).sum(axis=0) / (prob_[..., None].sum(axis=0) + self.eps)
                # sigma[i]: shape (num_gaussian, n_features, n_features)
                dist = np.matmul((X_[:, None] - self.miu[i]).transpose((0, 2, 1)), X_[:, None] - self.miu[i])
                self.sigma[i] = (prob_[..., None, None] * dist[:, None]).sum(axis=0) \
                           / (prob_[..., None, None].sum(axis=0) + self.eps)
                # weight[i]: shape (num_gaussian)
                self.weight[i] = prob_.sum(axis=0) / mask.sum()

        return self.label

    def predict(self, X):
        """
        predict

        :param X: shape: (n_samples, n_features)
        :return:
            label: shape (n_samples)
        """
        assert self.miu is not None, 'miu为空'
        assert len(X.shape) == 2 and X.shape[1] == self.miu.shape[1], 'X必须是2维'
        n_samples, n_features = X.shape

        prob = np.zeros((n_samples, self.n_clusters, self.num_gaussian))
        for k in range(self.n_clusters):
            for i in range(self.num_gaussian):
                prob[:, k, i] = multivariate_normal.pdf(X, mean=self.miu[k, i], cov=self.sigma[k, i])

        prob_cluster = (prob * self.weight).sum(axis=-1)  # shape: (n_samples, n_clusters)
        label = np.argmax(prob_cluster, axis=-1)  # shape: (n_samples)

        return label
