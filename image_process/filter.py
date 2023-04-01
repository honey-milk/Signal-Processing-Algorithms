# -*- coding: utf-8 -*-
"""
@File    : filter.py
@Time    : 2023/4/1 10:53:51
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""

import numpy as np


def filter(image, kernel, padding='zero'):
    """
    image filter

    :param image: shape (H, W) or (H, W, C)
    :param kernel: shape (M, N)
    :param padding: ['zero', 'replicate', 'duplicate'], default: 'zero'
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
    return result
