# -*- coding: utf-8 -*-
"""
@File    : dtw.py
@Time    : 2023/4/30 21:11:50
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""

import numpy as np
import matplotlib.pyplot as plt


def dtw(x, y):
    """
    Dynamic Time Warping

    :param x: shape (M)
    :param y: shape (N)
    :return:
        indices: shape (max(M, N), 2)
        distance
    """
    M, N = len(x), len(y)

    # 计算距离矩阵 shape (M, N)
    dist_matrix = np.abs(x[:, None] - y[None])

    # 动态规划
    acc_matrix = np.zeros((M + 1, N + 1))
    acc_matrix[0] = np.inf
    acc_matrix[:, 0] = np.inf
    acc_matrix[0, 0] = 0
    parent_matrix = np.zeros((M + 1, N + 1, 2), dtype='int32')
    for i in range(M):
        for j in range(N):
            d1 = acc_matrix[i, j] + dist_matrix[i, j]
            d2 = acc_matrix[i + 1, j] + dist_matrix[i, j]
            d3 = acc_matrix[i, j + 1] + dist_matrix[i, j]
            if d1 <= d2 and d1 <= d3:
                acc_matrix[i + 1, j + 1] = d1
                parent_matrix[i + 1, j + 1] = [i, j]
            elif d2 < d1 and d2 < d3:
                acc_matrix[i + 1, j + 1] = d2
                parent_matrix[i + 1, j + 1] = [i + 1, j]
            else:
                acc_matrix[i + 1, j + 1] = d3
                parent_matrix[i + 1, j + 1] = [i, j + 1]

    distance = acc_matrix[-1, -1]

    # 回溯
    i, j = M, N
    indices = [(i - 1, j - 1)]
    while i > 1 and j > 1:
        i, j = parent_matrix[i, j]
        indices.append((i - 1, j - 1))
    indices = indices[::-1]

    return indices, distance


def dtw_demo():
    """
    dtw demo

    :return:
    """
    N = 100
    T1 = 2
    T2 = 1
    t1 = np.linspace(0, T1, N)
    t2 = np.linspace(0, T2, N)
    x = np.sin(2 * np.pi / T1 * t1)
    y = np.sin(2 * np.pi / T2 * t2)

    indices, distance = dtw(x, y)

    # 显示测试数据
    plt.figure()
    plt.scatter(t1, x)
    plt.scatter(t2, y)
    for i, j in indices:
        plt.plot((t1[i], t2[j]), (x[i], y[j]))
    plt.show()


if __name__ == '__main__':
    dtw_demo()
