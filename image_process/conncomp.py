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


def _boundary_by_seed(bw, seed, dierction, conn=4):
    """
    Extract boundary by seed

    :param bw: shape (H, W)
    :param seed: (x, y)
    :param dierction: (dx, dy)
    :param conn: 4 or 8, default: 4
    :return:
        boundary
    """
    assert len(bw.shape) == 2, '输入图像BW必须为二值图'
    assert conn in [4, 8], '参数conn必须为4或8'

    if conn == 4:
        neighbors = [[-1, 0], [0, 1], [1, 0], [0, -1]]
    else:
        neighbors = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
    assert dierction in neighbors, '输入方向有误'
    H, W = bw.shape[:2]
    assert 0 <= seed[0] < W and 0 <= seed[1] < H and bw[seed[1], seed[0]], '种子点有误'

    # 初始化
    boundary = []
    area = bw.sum()
    if area == 0:
        return boundary
    elif area == 1:
        return np.array(seed)

    boundary.append(seed)
    last_point = seed
    neighbor_index = neighbors.index(dierction)

    # 遍历图像
    H, W = bw.shape
    finish = False
    while not finish:
        for i in range(conn):
            current_index = (neighbor_index + conn // 2 + 1 + i) % conn
            dx, dy = neighbors[current_index]
            x2, y2 = last_point[0] + dx, last_point[1] + dy
            if x2 < 0 or x2 >= W or y2 < 0 or y2 >= H:
                continue
            if bw[y2, x2]:
                current_point = (x2, y2)
                if current_point == seed:
                    finish = True
                else:
                    last_point = current_point
                    boundary.append(current_point)
                    neighbor_index = current_index
                break

    return np.array(boundary)


def bwboundary(bw, conn=4):
    """
    Extract boundary

    :param bw: shape (H, W)
    :param conn: 4 or 8, default: 4
    :return:
        boundaries
    """
    assert len(bw.shape) == 2, '输入图像BW必须为二值图'
    assert bw.sum(), '输入图像非零像素个数不能为0'

    seeds = []
    directions = []

    # 寻找外轮廓起始点
    ys, xs = np.nonzero(bw)
    index = np.argmax(xs)
    x, y = xs[index], ys[index]
    seeds.append((x, y))
    directions.append([0, -1])

    # 寻找内轮廓起始点
    H, W = bw.shape
    padding_bw = np.zeros((H + 2, W + 2), dtype=bw.dtype)
    padding_bw[1:-1, 1:-1] = bw
    padding_bw = padding_bw == 0
    label = bwlabel(padding_bw, conn=4)
    for i in range(1, label.max() + 1):
        mask = label == i
        ys, xs = np.nonzero(mask)
        index = np.argmax(xs)
        x, y = xs[index] + 1, ys[index]
        if x >= W + 2:
            continue
        seeds.append((x - 1, y - 1))
        directions.append([0, 1])

    # 寻找内外轮廓
    boundaries = []
    for seed, direction in zip(seeds, directions):
        boundary = _boundary_by_seed(bw, seed, direction, conn)
        boundaries.append(boundary)

    return boundaries


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
        boundaries = bwboundary(mask, conn=conn)
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
            'bbox': bbox,
            'boundaries': boundaries
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
    label, infos = conncomp(bw, conn=4)

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
        boundaries = info['boundaries']
        # cv2.circle(image, center, 2, color=color, thickness=-1)
        # pt1 = (bbox[0], bbox[1])
        # pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        # cv2.rectangle(image, pt1, pt2, color=color, thickness=1)
        # cv2.putText(image, str(area), (center[0], center[1] - 5), color=color,
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, thickness=1)
        for boundary in boundaries:
            for pt in boundary:
                pt = (pt[0], pt[1])
                cv2.rectangle(image, pt, pt, color=color, thickness=1)

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
