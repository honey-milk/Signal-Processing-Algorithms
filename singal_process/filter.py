# -*- coding: utf-8 -*-
"""
@File    : filter.py
@Time    : 2023/4/1 10:53:51
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""

import numpy as np


def filter1d(x, kernel, padding='zero', dtype=None):
    """
    image filter

    :param x: shape (N)
    :param kernel: shape (M)
    :param padding: ['zero', 'replicate', 'duplicate'], default: 'zero'
    :param dtype: default: x.dtype
    :return:
        y: with the same shape as x
    """
    N = len(x)
    M = len(kernel)
    assert M % 2 == 1, 'M必须是奇数'
    assert padding in ['zero', 'replicate', 'duplicate'], 'padding not support'
    assert N >= M, 'x size less than kernel size'
    dtype = x.dtype if dtype is None else dtype

    # padding
    si = (M - 1) // 2
    padding_x = np.zeros((N + M - 1), dtype='float32')
    padding_x[si:-si] = x
    if padding == 'replicate':
        padding_x[:si] = padding_x[si:2 * si][::-1]
        padding_x[-si:] = padding_x[-2 * si - 1: -si - 1][::-1]
    elif padding == 'duplicate':
        padding_x[:si] = padding_x[si]
        padding_x[-si:] = padding_x[-si - 1]

    # filter
    y = np.zeros_like(x, dtype='float32')
    for i in range(0, N):
        y[i] = (kernel * padding_x[i:i + M]).sum()
    y = y.astype(dtype)

    return y


def filter2d(image, kernel, padding='zero', dtype=None):
    """
    image filter

    :param image: shape (H, W) or (H, W, C)
    :param kernel: shape (M, N)
    :param padding: ['zero', 'replicate', 'duplicate'], default: 'zero'
    :param dtype: default: image.dtype
    :return:
        result: with the same shape as image
    """
    assert len(image.shape) in [2, 3], '图像形状的维度必须为2或3'
    assert len(kernel.shape) == 2, '核形状的维度必须为2'
    if len(image.shape) == 2:
        H, W = image.shape
    else:
        H, W, C = image.shape
    M, N = kernel.shape
    assert M % 2 == 1 and N % 2 == 1, 'M和N必须是奇数'
    assert padding in ['zero', 'replicate', 'duplicate'], 'padding not support'
    assert H >= M and W >= N, 'image size less than kernel size'
    dtype = image.dtype if dtype is None else dtype

    # padding
    sy, sx = (M - 1) // 2, (N - 1) // 2
    if len(image.shape) == 2:
        padding_image = np.zeros((H + M - 1, W + N - 1), dtype='float32')
    else:
        padding_image = np.zeros((H + M - 1, W + N - 1, C), dtype='float32')
    padding_image[sy:-sy, sx:-sx] = image
    if padding == 'replicate':
        padding_image[:sy] = padding_image[sy:2 * sy][::-1]
        padding_image[-sy:] = padding_image[-2 * sy - 1: -sy - 1][::-1]
        padding_image[:, :sx] = padding_image[:, sx:2 * sx][:, ::-1]
        padding_image[:, -sx:] = padding_image[:, -2 * sx - 1:-sx - 1][:, ::-1]
    elif padding == 'duplicate':
        padding_image[:sy] = padding_image[sy]
        padding_image[-sy:] = padding_image[-sy - 1]
        padding_image[:, :sx] = padding_image[:, sx:sx + 1]
        padding_image[:, -sx:] = padding_image[:, -sx - 1:-sx]

    # filter
    result = np.zeros_like(image, dtype='float32')
    for y in range(0, H):
        for x in range(0, W):
            result[y, x] = (kernel * padding_image[y:y + M, x:x + N]).sum()
    result = result.astype(dtype)

    return result


def gaussian2d(kernel_size, sigma=1.0):
    """
    Generate gaussian kernel

    :param kernel_size:
    :param sigma:
    :return:
    """
    assert sigma > 0, 'sigma must greater than 0'
    sigma_3 = 3 * sigma
    xs = np.linspace(-sigma_3, sigma_3, kernel_size)
    ys = np.linspace(-sigma_3, sigma_3, kernel_size)
    ys, xs = np.meshgrid(ys, xs)
    kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(- (xs ** 2 + ys ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel
