# -*- coding: utf-8 -*-
"""
@File    : sobel.py
@Time    : 2023/3/31 22:05:37
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""

import cv2
import numpy as np
from image_process import filter


def sobel(image):
    """
    Edge detection by Sobel operator

    :param image: shape (H, W) or (H, W, 3)
    :return:
        edge: shape (H, W)
        edge_x: shape (H, W)
        edge_y: shape (H, W)
    """
    assert len(image.shape) in [2, 3], '图像形状的维度必须为2或3'
    # 转灰度图
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 边缘检测
    H_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    H_x = H_y.T
    edge_x = filter(gray, H_x, padding='replicate', dtype='float32')
    edge_y = filter(gray, H_y, padding='replicate', dtype='float32')
    edge_x = np.abs(edge_x)
    edge_y = np.abs(edge_y)
    edge = np.sqrt(edge_x ** 2 + edge_y ** 2)

    # 归一化
    edge = (edge / (edge.max() + 1e-10) * 255).astype('uint8')
    edge_x = (edge_x / (edge_x.max() + 1e-10) * 255).astype('uint8')
    edge_y = (edge_y / (edge_y.max() + 1e-10) * 255).astype('uint8')

    return edge, edge_x, edge_y


def sobel_demo(src_file, dst_file):
    """
    sobel demo

    :param src_file:
    :param dst_file:
    :return:
    """
    image = cv2.imread(src_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge, edge_x, edge_y = sobel(image)

    # show
    image = np.vstack([np.hstack([gray, edge]), np.hstack([edge_x, edge_y])])
    cv2.namedWindow('image', 0)
    cv2.imshow('image', image)
    cv2.waitKey()

    # save image
    cv2.imwrite(dst_file, image)


if __name__ == '__main__':
    src_file = 'E:/Signal-Processing-Algorithms/src/image/lena.jpg'
    dst_file = 'lena.jpg'
    sobel_demo(src_file, dst_file)
    # image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 3]])
    # kernel = np.ones((3, 3))
    # imfilter(image, kernel, padding='zero')
