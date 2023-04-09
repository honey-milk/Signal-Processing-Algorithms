# -*- coding: utf-8 -*-
"""
@File    : globalthresh.py
@Time    : 2023/4/9 21:27:16
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""

import cv2
import numpy as np


def global_thresh(image, delta=0.5):
    """

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


def globalthresh_demo(src_file, dst_file):
    """
    OSTU threshold demo

    :param src_file:
    :param dst_file:
    :return:
    """
    image = cv2.imread(src_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = global_thresh(gray)
    result = ((gray >= thresh) * 255).astype('uint8')
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
    globalthresh_demo(src_file, dst_file)
