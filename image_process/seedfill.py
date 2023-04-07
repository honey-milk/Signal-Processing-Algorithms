# -*- coding: utf-8 -*-
"""
@File    : seedfill.py
@Time    : 2023/4/7 21:35:28
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""
import cv2
import numpy as np


def seed_fill(bw, seeds, conn=4):
    """
    Seed filling in 2-D binary image

    :param bw: shape (H, W)
    :param seeds: shape (N, 2)
    :param conn: 4 or 8, default: 4
    :return:
        label: shape (H, W)
    """
    assert len(bw.shape) == 2, '输入图像BW必须为二值图'
    assert len(seeds.shape) == 2 and seeds.shape[-1] == 2, '输入种子的形状必须为N * 2'
    assert conn in [4, 8], '参数conn必须为4或8'

    if conn == 4:
        neighbors = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
    else:
        neighbors = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])

    # 遍历图片
    H, W = bw.shape
    label = np.zeros_like(bw, dtype='bool')
    for x, y in seeds:
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        if not bw[y, x] or label[y, x]:
            continue
        stack = [(y, x)]
        label[y, x] = True
        while stack:
            seed_y, seed_x = stack.pop()
            for dx, dy in neighbors:
                x2, y2 = seed_x + dx, seed_y + dy
                if x2 < 0 or x2 >= W or y2 < 0 or y2 >= H:
                    continue
                if not bw[y2, x2] or label[y2, x2]:
                    continue
                stack.append((y2, x2))
                label[y2, x2] = True

    return label


def flood_fill(image, seeds, newval, thresh, conn=4):
    """
    Flood filling

    :param image: shape (H, W, 3) or (H, W)
    :param seeds: shape (N, 2)
    :param newval: filled color
    :param thresh: type: int or list with shape (3)
    :param conn: 4 or 8, default: 4
    :return:
        result: with the same shape as input image
    """
    result = image.copy()
    image = image.astype('int32')
    bw = np.zeros(image.shape[:2], dtype='bool')
    for seed in seeds:
        x, y = seed
        value = image[y, x]
        bw |= (np.abs(image - value) <= thresh).all(axis=-1)
        mask = seed_fill(bw, seeds, conn)
        result[mask] = newval

    return result


def seed_fill_demo(src_file, dst_file):
    """
    Seed filling demo

    :param src_file:
    :param dst_file:
    :return:
    """
    image = cv2.imread(src_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

    # 种子填充
    seeds = np.array([[89, 148], [240, 73]])
    label = seed_fill(bw, seeds, conn=4)
    label = (label * 255).astype('uint8')

    # 可视化
    image = np.hstack([bw, label])
    cv2.namedWindow('image', 0)
    cv2.imshow('image', image)
    cv2.waitKey()

    # save image
    cv2.imwrite(dst_file, image)


def flood_fill_demo(src_file, dst_file):
    """
    Flood filling demo

    :param src_file:
    :param dst_file:
    :return:
    """
    image = cv2.imread(src_file)

    # 种子填充
    seeds = np.array([[371, 279]])
    result = flood_fill(image, seeds, newval=[0, 255, 0], thresh=10, conn=4)

    # 可视化
    image = np.hstack([image, result])
    cv2.namedWindow('image', 0)
    cv2.imshow('image', image)
    cv2.waitKey()

    # save image
    cv2.imwrite(dst_file, image)


if __name__ == '__main__':
    src_file = 'E:/Signal-Processing-Algorithms/src/image/floodfill.jpg'
    dst_file = 'result.jpg'
    flood_fill_demo(src_file, dst_file)
