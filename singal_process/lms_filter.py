# -*- coding: utf-8 -*-
"""
@File    : lms_filter.py
@Time    : 2023/4/18 23:01:58
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg


def lms_filter(x, d, n, miu=1e-2):
    """
    1-D least mean square filter

    :param x: shape (M)
    :param d: shape (M)
    :param n:
    :param miu:
    :return:
        y: shape (M)
        e: shape (M)
        w: shape (n + 1)
    """
    assert len(x.shape) == 1 and len(d.shape) == 1, '输入必须为1维'
    assert x.shape == d.shape, 'x和d长度必须一致'

    # 输入信号延时
    nx = len(x)
    x_tmp = np.zeros((n + 1, nx))
    x_tmp[0] = x
    for i in range(1, n + 1):
        x_tmp[i, i:] = x[:-i]

    # 初始化参数
    w = np.zeros((n + 1,))
    y = np.zeros((nx,))
    e = np.zeros((nx,))

    # 滤波
    for i in range(nx):
        y[i] = (w * x_tmp[:, i]).sum()
        e[i] = d[i] - y[i]
        w = w + 2 * miu * e[i] * x_tmp[:, i]

    return y, e, w


def lms_filter_demo():
    """
    lms filter demo

    :return:
    """
    fs = 8000
    T = 1
    t = np.arange(0, T, 1 / fs)
    freq1 = 10
    freq2 = 40

    # 产生信号
    signal_1 = np.sin(2 * np.pi * freq1 * t)    # 正弦波
    signal_2 = np.sin(2 * np.pi * freq2 * t)
    signal_noise = np.sin(2 * np.pi * freq2 * (t + 0.25 / freq2))
    signal_mix = signal_1 + 0.5 * signal_2

    # 自适应滤波
    y, e, w = lms_filter(signal_noise, signal_mix, 200, 0.0005)

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
    lms_filter_demo()
