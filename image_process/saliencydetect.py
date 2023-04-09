# -*- coding: utf-8 -*-
"""
@File    : saliencydetect.py
@Time    : 2023/4/2 15:17:41
@Author  : xjhan
@Contact : xjhansgg@whu.edu.cn
"""
import cv2
import numpy as np
from image_process import filter, conncomp, gaussian


def saliency_detect(image, threshold=0):
    """
    Saliency detect

    :param image: shape (H, W) or (H, W, 3)
    :param threshold: area threshold
    :return:
        saliency_map: shape (H, W)
        infos
    """
    assert len(image.shape) in [2, 3], '图像形状的维度必须为2或3'

    # 转灰度图
    if len(image.shape) == 2:
        gray = image.copy()
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算频谱
    fft = np.fft.fft2(gray)
    log_amplitude = np.log(np.abs(fft))
    phase = np.angle(fft)

    # 对幅度谱做均值滤波
    kernel = np.full((3, 3), 1 / 9)
    smoothed_log_amplitude = filter(log_amplitude, kernel, 'replicate')

    # 计算幅度谱残差
    spectral_residual = log_amplitude - smoothed_log_amplitude

    # 恢复成图像
    new_fft = np.exp(spectral_residual + 1j * phase)
    saliency_map = np.abs(np.fft.ifft2(new_fft))
    saliency_map = saliency_map / (saliency_map.max() + 1e-10)
    saliency_map = (saliency_map * 255).astype('uint8')

    # 二值化
    kernel = gaussian(11, 2.5)
    saliency_map = filter(saliency_map, kernel)
    bw = (saliency_map >= 2 * saliency_map.mean()).astype('uint8') * 255

    # 连通域处理
    _, infos = conncomp(bw)
    infos = [info for info in infos if info['area'] >= threshold]

    return saliency_map, infos


def saliency_detect_demo(src_file, dst_file):
    """
    saliency detect

    :param src_file:
    :param dst_file:
    :return:
    """
    image = cv2.imread(src_file)
    saliency_map, infos = saliency_detect(image, 0)
    saliency_map = np.repeat(saliency_map[..., None], 3, axis=-1)

    # show
    number = len(infos)
    for i in range(number):
        color = [int(val * 255) for val in np.random.rand(3)]
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
    image = np.hstack([image, saliency_map])
    cv2.namedWindow('image', 0)
    cv2.imshow('image', image)
    cv2.waitKey()

    # save image
    cv2.imwrite(dst_file, image)


if __name__ == '__main__':
    src_file = 'E:/Signal-Processing-Algorithms/src/image/cc.jpg'
    dst_file = 'result.jpg'
    saliency_detect_demo(src_file, dst_file)