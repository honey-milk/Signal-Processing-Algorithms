# -*- coding: utf-8 -*-
"""
@File    : ostuthresh.py
@Time    : 2023/4/9 12:03:20
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""
import cv2
import numpy as np
from image_process import imhist


def ostuthresh(image):
    """

    :param image: shape (H, W) or (H, W, 3)
    :return:
        thresh
    """
    assert len(image.shape) in [2, 3], '图像形状的维度必须为2或3'
    # 转灰度图
    if len(image.shape) == 2:
        gray = image
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


def ostuthresh_demo(src_file, dst_file):
    """
    OSTU threshold demo

    :param src_file:
    :param dst_file:
    :return:
    """
    image = cv2.imread(src_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = ostuthresh(gray)
    result = ((gray > thresh) * 255).astype('uint8')
    result2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

    # show
    image = np.hstack([gray, result, result2])
    cv2.namedWindow('image', 0)
    cv2.imshow('image', image)
    cv2.waitKey()

    # save image
    cv2.imwrite(dst_file, image)


if __name__ == '__main__':
    src_file = 'E:/Signal-Processing-Algorithms/src/image/Fig1019(a).tif'
    dst_file = 'result.jpg'
    ostuthresh_demo(src_file, dst_file)
