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
from kmeans import KMeans
from gmm import GMM


THRESH_OTSU = 1
THRESH_GLOBAL = 2
THRESH_LOCAL = 3
THRESH_MOVING = 4
THRESH_KMEANS = 5
THRESH_GMM = 6


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


def kmeans_thresh(image, K=2, **kwargs):
    """
    kmeans threshold

    :param image: shape (H, W) or (H, W, 3)
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

    # 输入图像f处理
    gray = gray.reshape((-1, 1))
    gray = gray.astype('float32')

    # K-means聚类
    gray_rank = np.linspace(0, 255, K)
    kmeans = KMeans(n_clusters=K, init_clusters=gray_rank.reshape((-1, 1)))
    label = kmeans.fit(gray)

    # k值化
    for i in range(K):
        gray[label == i] = gray_rank[i]

    # 向量转为矩阵
    gray = gray.reshape((H, W)).astype('uint8')

    return gray


def gmm_thresh(image, K=2, num_gaussian=1, **kwargs):
    """
    gmm threshold

    :param image: shape (H, W) or (H, W, 3)
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

    # 输入图像f处理
    gray = gray.reshape((-1, 1))
    gray = gray.astype('float32')

    # K-means聚类
    gray_rank = np.linspace(0, 255, K)
    gmm = GMM(n_clusters=K, num_gaussian=num_gaussian)
    label = gmm.fit(gray)

    # k值化
    for i in range(K):
        gray[label == i] = gray_rank[i]

    # 向量转为矩阵
    gray = gray.reshape((H, W)).astype('uint8')

    return gray


def threshold(image, maxval=255, type=THRESH_OTSU, **kwargs):
    """
    threshold

    :param image: shape (H, W) or (H, W, 3)
    :param maxval:
    :param type:
    :param kwargs:
    :return:
        result: shape (H, W)
    """
    assert type in [THRESH_OTSU, THRESH_GLOBAL, THRESH_LOCAL,
                    THRESH_MOVING, THRESH_KMEANS, THRESH_GMM], '类型不支持'
    assert len(image.shape) in [2, 3], '图像形状的维度必须为2或3'
    # 转灰度图
    if len(image.shape) == 2:
        gray = image.copy()
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if type == THRESH_OTSU:
        thresh = ostu_thresh(image, **kwargs)
        result = (gray > thresh).astype('uint8') * maxval
    elif type == THRESH_GLOBAL:
        thresh = global_thresh(gray, **kwargs)
        result = (gray > thresh).astype('uint8') * maxval
    elif type == THRESH_LOCAL:
        result = local_thresh(image, **kwargs)
        result = result.astype('uint8') * maxval
    elif type == THRESH_MOVING:
        result = moving_thresh(image, **kwargs)
        result = result.astype('uint8') * maxval
    elif type == THRESH_KMEANS:
        result = kmeans_thresh(image, **kwargs)
    elif type == THRESH_GMM:
        result = gmm_thresh(image, **kwargs)
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
    # result = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]
    # result1 = threshold(gray, type=THRESH_OTSU)
    # result2 = threshold(gray, type=THRESH_GLOBAL, delta=0.5)
    # result3 = threshold(gray, type=THRESH_LOCAL, kernel_size=3, alpha=30, belta=1, meantype='local')
    # result4 = threshold(gray, type=THRESH_MOVING, kernel_size=3, K=0.5)
    # result5 = threshold(gray, type=THRESH_KMEANS, K=2)
    result6 = threshold(gray, type=THRESH_GMM, K=2, num_gaussian=1)

    # show
    image = np.hstack([gray, result6])
    # image = np.hstack([gray, result, result1, result2, result3, result4, result5])
    cv2.namedWindow('image', 0)
    cv2.imshow('image', image)
    cv2.waitKey()

    # save image
    cv2.imwrite(dst_file, image)


if __name__ == '__main__':
    src_file = 'E:/Signal-Processing-Algorithms/src/image/Fig1019(a).tif'
    dst_file = 'result.jpg'
    binarization_demo(src_file, dst_file)