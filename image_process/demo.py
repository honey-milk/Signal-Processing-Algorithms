# -*- coding: utf-8 -*-
"""
@File    : demo.py
@Time    : 2023/3/30 22:15:29
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""

import cv2
import numpy as np


def show_image(src_file, dst_file):
    """
    show image
    
    :param src_file: shape (H, W, 3)
    :param dst_file: shape (H, W, 3)
    :return: 
    """
    image = cv2.imread(src_file)
    cv2.namedWindow('image', 0)
    cv2.imshow('image', image)
    cv2.waitKey()
    cv2.imwrite(dst_file, image)


def flip_image(src_file, dst_file):
    """
    flip image

    :param src_file: shape (H, W, 3)
    :param dst_file: shape (H, W, 3)
    :return:
    """
    image = cv2.imread(src_file)
    # image = image[::-1, ::-1]
    image += np.array([1, 2, 3], dtype='uint8')  # bgr
    cv2.imwrite(dst_file, image)


if __name__ == '__main__':
    src_file = 'E:/Signal-Processing-Algorithms/src/image/lena.jpg'
    dst_file = 'lena.jpg'
    flip_image(src_file, dst_file)
