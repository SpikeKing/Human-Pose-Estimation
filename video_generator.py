#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/11/5
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from matchstick.MatchstickSkeleton import MatchstickSkeleton
from root_dir import *
from utils.project_utils import *


def process_bkg(img_path, is_clear=True):
    """
    将图像中的火柴人去除，支持保留阴影和不保留
    :param img_path: 图像路径
    :param is_clear: 不保留，完全去除
    :return: 去除后的背景，已去除的物体位置
    """
    img_opencv = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()
    img = img_opencv.copy()  # 高，宽
    h, w = img.shape[:2]  # h, w
    print("[Info] img shape: {}".format(img.shape))

    # TODO: 预设的定位
    thr_piny = cv2.inRange(img[150:300, 150:300, :], np.array([0, 0, 0]), np.array([80, 80, 80]))
    scan = np.ones((3, 3), np.uint8)
    if is_clear:
        thr_piny = cv2.dilate(thr_piny, scan, iterations=2)
    print('[Info] thr_piny shape: {}'.format(thr_piny.shape))

    thr_bkg = np.zeros(img.shape[:2])
    thr_bkg[150:300, 150:300] = thr_piny
    thr_bkg = thr_bkg.astype(np.uint8)

    res_idx = np.where(thr_bkg == 255)
    x_min, x_max = np.min(res_idx[0]), np.max(res_idx[0])
    y_min, y_max = np.min(res_idx[1]), np.max(res_idx[1])
    thr_min, thr_max = (x_min, y_min), (x_max, y_max)
    print('[Info] 左上角: {}, 右下角: {}'.format(thr_min, thr_max))

    blank_bkg = cv2.inpaint(img_opencv, thr_bkg, 4, cv2.INPAINT_TELEA)  # 去除的图像
    blank_bkg_rgb = cv2.cvtColor(blank_bkg, cv2.COLOR_BGR2RGB)
    plt.imshow(blank_bkg_rgb)
    plt.show()

    return blank_bkg, (thr_min, thr_max)


def gen_canvas(i, os_thr0, data_lines, nd, thr_min, scale, f_skt, frame_shape, blank_bkg_copy):
    """
    生成姿态的函数
    :param i: 帧的位置
    :param os_thr0: 水平位置的偏移量
    :param data_lines: 已有的运动数据
    :param nd: 数据总量
    :param thr_min: 最小值
    :param scale: 缩放
    :param f_skt: 第一个骨骼
    :param frame_shape: 帧的尺寸
    :param blank_bkg_copy: 背景
    :return: 图
    """
    data_line = data_lines[i % nd]  # 获取数据
    skt = json.loads(data_line)
    skt = [tuple(s) for s in skt]
    thr_min_m = (thr_min[0] + os_thr0, thr_min[1] - 8)  # 起始点的偏移
    skt = MatchstickSkeleton.resize_skt(skt, scale, thr_min_m, f_skt)
    canvas = MatchstickSkeleton.draw(frame_shape, skt, blank_bkg_copy)

    return canvas


def write_video(img_path, vid_path):
    """
    :param img_path: 背景图像
    :param vid_path: 源视频，提取帧率
    :return: None
    """
    blank_bkg, (thr_min, thr_max) = process_bkg(img_path)  # 处理背景
    blank_bkg_t, (_, _) = process_bkg(img_path, is_clear=False)  # 渐变的背景

    h_blank_bkg, w_blank_bkg = blank_bkg.shape[:2]  # h, w
    thr_x_len = thr_max[1] - thr_min[1]  # 长度

    video_name = vid_path.split('/')[-1]
    print('[Info] 视频名称: {}'.format(video_name))

    cap = cv2.VideoCapture(vid_path)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率24

    print('[Info] 总帧数: {}, 帧率: {}'.format(n_frame, fps))

    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # height = 210 * 5
    # width = 297 * 5
    height = h_blank_bkg
    width = w_blank_bkg
    frame_shape = (height, width, 3)  # (1024, 576, 3)
    print('[Info] 宽: {}, 高: {}, frame: {}'.format(width, height, frame_shape))

    o_vid_path = os.path.join(DATA_DIR, 'video.mp4')
    print('[Info] 写入视频的路径: {}'.format(o_vid_path))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the lower case，可以

    vw = cv2.VideoWriter(filename=o_vid_path, fourcc=fourcc, fps=fps, frameSize=(width, height), isColor=True)

    skt_file = os.path.join(DATA_DIR, 'skt_file.m.txt')  # TODO: 预设的骨骼
    data_lines = read_file(skt_file)
    nd = len(data_lines)  # TODO: 已经预设的总帧数
    print('[Info] 总帧数: {}'.format(nd))

    first_frame = data_lines[0]
    f_skt = json.loads(first_frame)
    f_skt = [tuple(s) for s in f_skt]
    skt_min, skt_max = MatchstickSkeleton.get_skt_min_and_max(f_skt)
    skt_x_len = skt_max[1] - skt_min[1]
    scale = thr_x_len / skt_x_len + 0.08  # TODO: 预设的比例
    print('[Info] scale: {}'.format(scale))

    img_path = os.path.join(DATA_DIR, 'test.jpg')
    img_opencv = cv2.imread(img_path)

    for i in range(5):  # 写入原图
        vw.write(img_opencv)

    for i in range(5):  # 写入渐变图
        vw.write(blank_bkg_t)

    for i in range(200):
        blank_bkg_copy = blank_bkg.copy()

        canvas = gen_canvas(i, 5, data_lines, nd, thr_min, scale, f_skt, frame_shape, blank_bkg_copy)

        if i > 40:
            canvas = gen_canvas(i - 40, 105, data_lines, nd, thr_min, scale, f_skt, frame_shape, blank_bkg_copy)

        if i > 80:
            canvas = gen_canvas(i - 80, 205, data_lines, nd, thr_min, scale, f_skt, frame_shape, blank_bkg_copy)

        if i > 120:
            canvas = gen_canvas(i - 120, 305, data_lines, nd, thr_min, scale, f_skt, frame_shape, blank_bkg_copy)

        if i > 160:
            canvas = gen_canvas(i - 160, 405, data_lines, nd, thr_min, scale, f_skt, frame_shape, blank_bkg_copy)

        vw.write(canvas)
        print('[Info] 写入 {} / {} 帧完成'.format(i, nd))

    vw.release()
    print('[Info] 视频写入完成! ')


def write_video_test():
    video_dir = os.path.join(DATA_DIR, 'videos')
    vid_path = os.path.join(video_dir, 'dance_video.mp4')

    img_path = os.path.join(DATA_DIR, 'test.jpg')

    write_video(img_path, vid_path)


def process_bkg_test():
    img_path = os.path.join(DATA_DIR, 'test.jpg')
    process_bkg(img_path)


def main():
    write_video_test()


if __name__ == '__main__':
    main()
