# -*- coding: utf-8 -*-
"""
@File    : histeq.py
@Time    : 2023/4/1 20:34:32
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from image_process import filter2d
from image_process import sobel


def imhist(image, weight=None):
    """
     calculate the histogram for the image
    :param image: shape (H, W) or (H, W, 3)
    :param weight: shape (H, W)
    :return:
        hist: shape (256)
    """
    assert len(image.shape) in [2, 3], '图像形状的维度必须为2或3'
    if weight is not None:
        assert image.shape[:2] == weight.shape

    # 转灰度图
    if len(image.shape) == 2:
        gray = image.copy()
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 统计每个灰度级的点数
    hist = np.zeros((256,), dtype='float32')
    for level in range(256):
        mask = gray == level
        if weight is None:
            hist[level] = mask.sum()
        else:
            hist[level] = weight[mask].sum()

    return hist


def histeq(image):
    """
    Histogram equalization

    :param image: shape (H, W) or (H, W, 3)
    :return:
        result: shape (H, W)
    """
    assert len(image.shape) in [2, 3], '图像形状的维度必须为2或3'
    # 转灰度图
    if len(image.shape) == 2:
        gray = image.copy()
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 直方图均衡化
    hist = imhist(gray)
    # 计算归一化直方图
    hist = hist / hist.sum()
    # 计算灰度映射关系
    hist = (np.cumsum(hist) * 255).astype('int32')

    # 根据映射关系，进行灰度转换
    result = np.zeros_like(gray)
    for level in range(256):
        mask = gray == level
        result[mask] = hist[level]

    return result


def improved_histeq(image):
    """
    Improved histogram equalization

    :param image: shape (H, W) or (H, W, 3)
    :return:
        result: shape (H, W)
    """
    assert len(image.shape) in [2, 3], '图像形状的维度必须为2或3'
    # 转灰度图
    if len(image.shape) == 2:
        gray = image.copy()
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算梯度
    kernel_y1 = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype='float32')
    kernel_y2 = kernel_y1[::-1]
    kernel_x1 = kernel_y1.T
    kernel_x2 = kernel_y2.T
    grad_y1 = filter2d(gray, kernel_y1)
    grad_y2 = filter2d(gray, kernel_y2)
    grad_x1 = filter2d(gray, kernel_x1)
    grad_x2 = filter2d(gray, kernel_x2)

    # 正梯度有效
    grad_y1 = np.clip(grad_y1, 0, None)
    grad_y2 = np.clip(grad_y2, 0, None)
    grad_x1 = np.clip(grad_x1, 0, None)
    grad_x2 = np.clip(grad_x2, 0, None)

    # 梯度相加
    image_grad = grad_y1 + grad_y2 + grad_x1 + grad_x2

    # 直方图均衡化
    hist = imhist(gray, image_grad)
    # 计算归一化直方图
    hist = hist / hist.sum()
    # 计算灰度映射关系
    hist = (np.cumsum(hist) * 255).astype('int32')

    # 根据映射关系，进行灰度转换
    result = np.zeros_like(gray)
    for level in range(256):
        mask = gray == level
        result[mask] = hist[level]

    return result


def histeq_demo(src_file, dst_file):
    """
    Histogram equalization demo

    :param src_file:
    :param dst_file:
    :return:
    """
    image = cv2.imread(src_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq_result = histeq(gray)
    improve_eq_result = improved_histeq(gray)

    # show
    gray_hist = imhist(gray)
    eq_result_hist = imhist(eq_result)
    improve_eq_result_hist = imhist(improve_eq_result)
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.bar(range(256), gray_hist)
    plt.subplot(3, 1, 2)
    plt.bar(range(256), eq_result_hist)
    plt.subplot(3, 1, 3)
    plt.bar(range(256), improve_eq_result_hist)
    plt.show()

    image = np.hstack([gray, eq_result, improve_eq_result])
    cv2.namedWindow('image', 0)
    cv2.imshow('image', image)
    cv2.waitKey()

    # save image
    cv2.imwrite(dst_file, image)


if __name__ == '__main__':
    src_file = 'E:/Signal-Processing-Algorithms/src/image/histeq.jpg'
    dst_file = 'result.jpg'
    histeq_demo(src_file, dst_file)
