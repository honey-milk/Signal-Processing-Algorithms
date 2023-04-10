# -*- coding: utf-8 -*-
"""
@File    : binarize.py
@Time    : 2023/4/10 22:02:43
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""

import cv2
import numpy as np
from scipy.ndimage.filters import generic_filter
from image_process import imhist, filter1d, filter2d
import image_process as ip


def ostu_thresh(image):
    """
    OSTU threshold

    :param image: shape (H, W) or (H, W, 3)
    :return:
        thresh
    """
    assert len(image.shape) in [2, 3], '图像形状的维度必须为2或3'
    # 转灰度图
    if len(image.shape) == 2:
        gray = image.copy()
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算归一化直方图
    hist = imhist(gray)
    hist = hist / hist.sum()

    # 求累计概率分布
    hist_sum = np.cumsum(hist)

    # 求累计均值
    ave_sum = np.cumsum(np.arange(256) * hist)

    # 计算类间方差
    sigma2 = ((ave_sum[-1] * hist_sum) - ave_sum) ** 2 / (hist_sum * (1 - hist_sum) + 1e-10)

    # 求最佳阈值
    indices = np.nonzero(sigma2 == sigma2.max())
    thresh = indices[0].mean()

    return thresh


def global_thresh(image, delta=0.5):
    """
    global threshold

    :param image: shape (H, W) or (H, W, 3)
    :param delta:
    :return:
        thresh
    """
    assert len(image.shape) in [2, 3], '图像形状的维度必须为2或3'
    # 转灰度图
    if len(image.shape) == 2:
        gray = image.copy()
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 初始阈值
    thresh = gray.mean()

    # 迭代完成标记
    done = False

    while not done:
        mask = gray > thresh
        if mask.sum() == 0 or (~mask).sum() == 0:
            break
        thresh_next = 0.5 * (gray[mask].mean() + gray[~mask].mean())
        if np.abs(thresh_next - thresh) < delta:
            done = True
        thresh = thresh_next

    return thresh


def local_thresh(image, kernel_size=3, alpha=30, belta=1, meantype='local'):
    """
    local threshold

    :param image: shape (H, W) or (H, W, 3)
    :param kernel_size:
    :param alpha:
    :param belta:
    :param meantype: choose from ['local', 'global'], default: 'local'
    :return:
        result: shape (H, W)
    """
    assert len(image.shape) in [2, 3], '图像形状的维度必须为2或3'
    assert meantype in ['local', 'global'], 'meantype不支持'
    # 转灰度图
    if len(image.shape) == 2:
        gray = image.copy()
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算标准差
    sigma = generic_filter(gray, np.std, size=kernel_size)

    # 计算均值
    if meantype == 'global':
        mean = np.mean(gray)
    else:
        kernel = np.full((kernel_size, kernel_size), 1.0 / kernel_size)
        mean = filter2d(gray, kernel, padding='replicate')

    # 二值化
    result = (gray > alpha * sigma) & (gray > belta * mean)

    return result


def moving_thresh(image, kernel_size=3, K=0.5):
    """
    moving threshold

    :param image: shape (H, W) or (H, W, 3)
    :param kernel_size:
    :param K:
    :return:
        result: shape (H, W)
    """
    assert len(image.shape) in [2, 3], '图像形状的维度必须为2或3'
    # 转灰度图
    if len(image.shape) == 2:
        gray = image.copy()
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    H, W = gray.shape

    # 扫描
    gray[1::2] = gray[1::2, ::-1]
    gray = gray.reshape((-1))
    gray = gray.astype('float32')

    # 计算移动均值
    kernel = np.full((kernel_size,), 1 / kernel_size)
    filtered_gray = filter1d(gray, kernel)

    # 阈值处理
    result = gray > filtered_gray * K

    # 反扫描
    result = result.reshape((H, W))
    result[1::2] = result[1::2, ::-1]

    return result


def threshold(image, maxval=255, type=ip.THRESH_OTSU, **kwargs):
    """
    threshold

    :param image: shape (H, W) or (H, W, 3)
    :param maxval:
    :param type:
    :param kwargs:
    :return:
        result: shape (H, W)
    """
    assert type in [ip.THRESH_OTSU, ip.THRESH_GLOBAL, ip.THRESH_LOCAL,
                    ip.THRESH_MOVING], '类型不支持'
    assert len(image.shape) in [2, 3], '图像形状的维度必须为2或3'
    # 转灰度图
    if len(image.shape) == 2:
        gray = image.copy()
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if type == ip.THRESH_OTSU:
        thresh = ostu_thresh(image, **kwargs)
        result = (gray > thresh).astype('uint8') * maxval
    elif type == ip.THRESH_GLOBAL:
        thresh = global_thresh(gray, **kwargs)
        result = (gray > thresh).astype('uint8') * maxval
    elif type == ip.THRESH_LOCAL:
        result = local_thresh(image, **kwargs)
        result = result.astype('uint8') * maxval
    elif type == ip.THRESH_MOVING:
        result = moving_thresh(image, **kwargs)
        result = result.astype('uint8') * maxval
    else:
        result = None

    return result


def binarization_demo(src_file, dst_file):
    """
    binarization demo

    :param src_file:
    :param dst_file:
    :return:
    """
    image = cv2.imread(src_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    result1 = threshold(gray, type=ip.THRESH_OTSU)
    result2 = threshold(gray, type=ip.THRESH_GLOBAL, delta=0.5)
    result3 = threshold(gray, type=ip.THRESH_LOCAL, kernel_size=3, alpha=30, belta=1, meantype='local')
    result4 = threshold(gray, type=ip.THRESH_MOVING, kernel_size=3, K=0.5)

    # show
    image = np.hstack([gray, result, result1, result2, result3, result4])
    cv2.namedWindow('image', 0)
    cv2.imshow('image', image)
    cv2.waitKey()

    # save image
    cv2.imwrite(dst_file, image)


if __name__ == '__main__':
    src_file = 'E:/Signal-Processing-Algorithms/src/image/Fig1019(a).tif'
    dst_file = 'result.jpg'
    binarization_demo(src_file, dst_file)