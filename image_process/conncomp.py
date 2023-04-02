# -*- coding: utf-8 -*-
"""
@File    : conncomp.py
@Time    : 2023/4/2 15:28:51
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""

import cv2
import numpy as np


def bwlabel(bw, conn=4):
    """
    Label connected components in 2-D binary image

    :param bw: shape (H, W)
    :param conn: 4 or 8, default: 4
    :return:
        label: shape (H, W)
    """
    assert len(bw.shape) == 2, '输入图像BW必须为二值图'
    assert conn in [4, 8], '参数conn必须为4或8'

    if conn == 4:
        neighbors = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
    else:
        neighbors = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])

    # 遍历图片
    H, W = bw.shape
    label = np.zeros_like(bw, dtype='int32')
    number = 0
    for y in range(H):
        for x in range(W):
            if not bw[y, x] or label[y, x]:
                continue
            number += 1
            stack = [(y, x)]
            label[y, x] = number
            while stack:
                seed_y, seed_x = stack.pop()
                for dx, dy in neighbors:
                    x2, y2 = seed_x + dx, seed_y + dy
                    if x2 < 0 or x2 >= W or y2 < 0 or y2 >= H:
                        continue
                    if not bw[y2, x2] or label[y2, x2]:
                        continue
                    stack.append((y2, x2))
                    label[y2, x2] = number

    return label


def conncomp(bw, conn=4):
    """
    Find connected components in binary image

    :param bw: shape (H, W)
    :param conn: 4 or 8, default: 4
    :return:
        label: shape (H, W)
        infos:
    """
    # 连通域检测
    label = bwlabel(bw, conn)

    # 计算连通域参数
    infos = []
    number = label.max()
    for i in range(number):
        mask = label == (i + 1)
        area = mask.sum()
        ys, xs = np.nonzero(mask)
        center = np.array([xs.mean(), ys.mean()])
        x_min = xs.min()
        x_max = xs.max()
        y_min = ys.min()
        y_max = ys.max()
        weight = x_max - x_min + 1
        height = y_max - y_min + 1
        bbox = np.array([x_min, y_min, weight, height])
        info = {
            'area': area,
            'center': center,
            'bbox': bbox
        }
        infos.append(info)

    return label, infos


def conncomp_demo(src_file, dst_file):
    """
    Connected components extraction demo

    :param src_file:
    :param dst_file:
    :return:
    """
    image = cv2.imread(src_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[1]

    # 连通域检测
    label, infos = conncomp(bw)

    # 可视化
    result = np.zeros_like(image)
    number = len(infos)
    for i in range(number):
        mask = label == (i + 1)
        color = [int(val * 255) for val in np.random.rand(3)]
        result[mask] = color
        info = infos[i]
        area = info['area']
        center = info['center'].astype('int32')
        bbox = info['bbox']
        cv2.circle(image, center, 2, color=color, thickness=-1)
        pt1 = (bbox[0], bbox[1])
        pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        cv2.rectangle(image, pt1, pt2, color=color, thickness=1)
        cv2.putText(image, str(area), (center[0], center[1] - 5), color=color,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, thickness=1)

    image = np.hstack([image, result])
    cv2.namedWindow('image', 0)
    cv2.imshow('image', image)
    cv2.waitKey()

    # save image
    cv2.imwrite(dst_file, image)


if __name__ == '__main__':
    src_file = 'E:/Signal-Processing-Algorithms/src/image/cc.jpg'
    dst_file = 'result.jpg'
    conncomp_demo(src_file, dst_file)
