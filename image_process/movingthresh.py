# -*- coding: utf-8 -*-
"""
@File    : movingthresh.py
@Time    : 2023/4/9 21:40:03
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""

import cv2
import numpy as np
from filter import filter


def moving_thresh(image, kernel_size=3, K=0.5):
    """

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
    gray = gray.reshape((1, -1))
    gray = gray.astype('float32')

    # 计算移动均值
    kernel = np.full((1, kernel_size), 1 / kernel_size)
    filtered_gray = filter(gray, kernel)

    # 阈值处理
    result = (gray > filtered_gray * K).astype('uint8') * 255

    # 反扫描
    result = result.reshape((H, W))
    result[1::2] = result[1::2, ::-1]

    return result


def moving_thresh_demo(src_file, dst_file):
    """
    OSTU threshold demo

    :param src_file:
    :param dst_file:
    :return:
    """
    image = cv2.imread(src_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = moving_thresh(gray)
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
    moving_thresh_demo(src_file, dst_file)

