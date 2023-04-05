# -*- coding: utf-8 -*-
"""
@File    : similarity.py
@Time    : 2023/4/3 22:54:30
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""

import cv2
import numpy as np
import scipy.ndimage
from image_process import conncomp


def similarity(image1, image2):
    """
    Compute the similarity of two images

    :param image1: shape (H, W) or (H, W, 3)
    :param image2: shape (H, W) or (H, W, 3)
    :return:
        score
    """
    assert len(image1.shape) in [2, 3], '图像形状的维度必须为2或3'
    assert len(image2.shape) in [2, 3], '图像形状的维度必须为2或3'

    # 转灰度图
    if len(image1.shape) == 2:
        gray1 = image1
    else:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    if len(image2.shape) == 2:
        gray2 = image2
    else:
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 二值化
    bw1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_OTSU)[1]
    bw2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_OTSU)[1]

    # 连通域处理, 找出图像中最大的物体
    _, infos1 = conncomp(bw1)
    _, infos2 = conncomp(bw2)
    info1 = sorted(infos1, key=lambda x: x['area'])[-1]
    info2 = sorted(infos2, key=lambda x: x['area'])[-1]

    # 计算轮廓到中心点的距离
    boundary1, boundary2 = info1['boundary'], info2['boundary']
    center1, center2 = info1['center'], info2['center']
    distances1 = np.linalg.norm(boundary1 - center1, 2, axis=-1)
    distances2 = np.linalg.norm(boundary2 - center2, 2, axis=-1)

    # 插值，使点数相等, 解决伸缩问题
    max_len = max(len(distances1), len(distances2))
    xp1 = np.linspace(0, 1, len(distances1))
    xp2 = np.linspace(0, 1, len(distances2))
    x = np.linspace(0, 1, max_len)
    distances1 = np.interp(x, xp1, distances1)
    distances2 = np.interp(x, xp2, distances2)

    # 计算周期相关，解决旋转问题
    corr = scipy.ndimage.correlate1d(distances2, distances1, mode='wrap', origin=-(max_len // 2))
    index = corr.argmax()
    distances1 = np.roll(distances1, index)

    # 计算相关系数（余弦距离）
    score = np.corrcoef(distances1, distances2)[1, 0]

    return score


def similarity_demo(candidate_file_list, test_file, dst_file=None):
    """
    Similarity demo

    :param candidate_file_list:
    :param test_file:
    :param dst_file
    :return:
    """
    candidate_images = [cv2.imread(src_file) for src_file in candidate_file_list]
    test_image = cv2.imread(test_file)

    # 计算相似度
    scores = [similarity(255 - image, 255 - test_image) for image in candidate_images]

    # 可视化
    max_height, total_width = test_image.shape[:2]
    heights = [max_height]
    accum_width = [total_width]
    for image in candidate_images:
        height, width = image.shape[:2]
        max_height = max(max_height, height)
        total_width += width
        accum_width.append(total_width)
        heights.append(height)
    result = np.zeros((max_height, total_width, 3), dtype='uint8')
    result[:heights[0], :accum_width[0]] = test_image
    for i, image in enumerate(candidate_images):
        cv2.putText(image, '%.3f' % scores[i], (20, 20), color=[0, 0, 255],
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
        result[:heights[i + 1], accum_width[i]:accum_width[i + 1]] = image

    cv2.namedWindow('image', 0)
    cv2.imshow('image', result)
    cv2.waitKey()

    # 保存结果
    if dst_file is not None:
        cv2.imwrite(dst_file, result)


if __name__ == '__main__':
    candidate_file_list = [
        '../src/image/leaf_training_1.jpg',
        '../src/image/leaf_training_2.jpg',
        '../src/image/leaf_training_3.jpg',
        '../src/image/leaf_training_4.jpg',
        '../src/image/leaf_training_5.jpg'
    ]
    test_file = '../src/image/leaf_test.jpg'
    # candidate_file_list = [
    #     '../src/image/rectangle1.jpg',
    #     '../src/image/square1.jpg',
    #     '../src/image/triangle1.jpg',
    #     '../src/image/ellipse1.jpg',
    # ]
    # test_file = '../src/image/ellipse2.jpg'
    dst_file = 'result.jpg'
    similarity_demo(candidate_file_list, test_file, dst_file=dst_file)
