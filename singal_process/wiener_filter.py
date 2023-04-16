# -*- coding: utf-8 -*-
"""
@File    : wiener_filter.py
@Time    : 2023/4/16 20:27:45
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


def wiener_filter(x, d, n):
    """
    1-D wiener filter

    :param x: shape (M)
    :param d: shape (M)
    :param n:
    :return:
        y: shape (M)
        e: shape (M)
        w: shape (n + 1)
    """
    assert len(x.shape) == 1 and len(d.shape) == 1, '输入必须为1维'
    assert x.shape == d.shape, 'x和d长度必须一致'

    # 输入信号延时
    d = d[None]
    nx = len(x)
    x_tmp = np.zeros((n + 1, nx))
    x_tmp[0] = x
    for i in range(1, n + 1):
        x_tmp[i, i:] = x[:-i]

    # 计算n阶自相关矩阵
    R_xx = np.matmul(x_tmp, x_tmp.T)
    R_xx = scipy.linalg.toeplitz(R_xx[0])   # shape (n + 1, n + 1)

    # 计算n阶互相关向量
    R_dx = np.matmul(d, x_tmp.T)  # shape (1, n + 1)

    # 计算维纳滤波器系数
    w = np.matmul(np.linalg.inv(R_xx), R_dx.T).T  # shape (1, n + 1)

    # 进行维纳滤波
    y = np.matmul(w, x_tmp)   # shape (1, nx)

    # 误差
    e = d - y   # shape (1, nx)

    return y[0], e[0], w[0]


def wiener_filter_demo():
    """
    wiener filter demo

    :return:
    """
    fs = 8000
    T = 1
    t = np.arange(0, T, 1 / fs)
    freq1 = 10
    freq2 = 50

    # 产生信号
    signal_1 = np.sin(2 * np.pi * freq1 * t)    # 正弦波
    signal_2 = (np.sin(2 * np.pi * freq2 * t) >= 0).astype('float32')   # 方波
    signal_noise = (np.sin(2 * np.pi * freq2 * (t + 0.25 / freq2)) >= 0).astype('float32')   # 方波
    signal_mix = signal_1 + 0.5 * signal_2

    # 维纳滤波
    y, e, w = wiener_filter(signal_noise, signal_mix, 50)

    # 绘制时域波形
    plt.figure()
    plt.subplot(6, 1, 1)
    plt.plot(t, signal_1)
    plt.subplot(6, 1, 2)
    plt.plot(t, signal_2)
    plt.subplot(6, 1, 3)
    plt.plot(t, signal_noise)
    plt.subplot(6, 1, 4)
    plt.plot(t, signal_mix)
    plt.subplot(6, 1, 5)
    plt.plot(t, y)
    plt.subplot(6, 1, 6)
    plt.plot(t, e)
    plt.show()


if __name__ == '__main__':
    wiener_filter_demo()
