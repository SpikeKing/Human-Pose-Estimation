#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/11/11
"""

import os
import cv2
import numpy as np

from matchstick.matchstick_skeleton import MatchstickSkeleton
from matchstick.parts_segmentation import PartsSegmentation
from root_dir import DATA_DIR
from utils.project_utils import *


def generate_body_parts():
    """
    由PNG图像生成10个身体部分
    :return: 10个身体部分
    """
    img_draw_path = os.path.join(DATA_DIR, 'custom', 'x.png')
    img_png_path = os.path.join(DATA_DIR, 'custom', 'trans.png')
    img_config_path = os.path.join(DATA_DIR, 'configs', 'parts_config.json')

    ps = PartsSegmentation(img_png_path, img_config_path, img_draw_path)
    png_parts = ps.process()

    return png_parts


def draw_custom():
    """
    绘制用户自定义的图像
    """
    png_parts = generate_body_parts()

    skt_file = os.path.join(DATA_DIR, 'skt_file.m.txt')
    data_lines = read_file(skt_file)
    n_data = len(data_lines)
    print('[Info] 运动序列总数: {}'.format(n_data))

    first_frame = data_lines[1]

    frame_shape = (1024, 576, 4)
    canvas = np.ones(frame_shape) * 255
    canvas_alpha = np.zeros(frame_shape[0:2])
    canvas[:, :, 3] = canvas_alpha

    skt = json.loads(first_frame)
    skt = [tuple(s) for s in skt]
    # skt = MatchstickSkeleton.resize_skt(skt, 0., (0, 0))

    canvas = MatchstickSkeleton.draw_png(frame_shape, skt, canvas, png_parts)
    o_vid_path = os.path.join(DATA_DIR, 'custom', 'first_frame.jpg')
    cv2.imwrite(o_vid_path, canvas)


def main():
    draw_custom()


if __name__ == '__main__':
    main()
