#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/11/20
"""

import os
import cv2
import json

from matchstick.matchstick_skeleton import MatchstickSkeleton
from matchstick.skeleton_generator import SkeletonGenerator, read_file
from root_dir import DATA_DIR
from utils.project_utils import get_current_time_str


class SkeletonMain(object):

    @staticmethod
    def write_video_with_parts(o_vid_path, skt_file, fps, frame_shape, png_parts):
        """
        根据骨骼文件, 写入视频
        :param o_vid_path: 视频输出路径
        :param skt_file: 骨骼文件
        :param fps: FPS
        :param frame_shape: Frame Shape
        :return: None
        """
        # o_vid_path = os.path.join(DATA_DIR, 'video.mp4')
        (h, w, _) = frame_shape  # (1024, 576, 3)

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the lower case，可以
        vw = cv2.VideoWriter(filename=o_vid_path, fourcc=fourcc, fps=fps, frameSize=(w, h), isColor=True)

        data_lines = read_file(skt_file)
        n_data = len(data_lines)
        print('[Info] 总帧数: {}'.format(n_data))

        for i, data_line in enumerate(data_lines):
            skt = json.loads(data_line)
            skt = [tuple(s) for s in skt]
            # skt = MatchstickSkeleton.resize_skt(skt, 0.3, (150, 150))  # 测试偏移和缩放
            # white_bkg = np.ones(frame_shape) * 255  # 需要实时更新
            # canvas = MatchstickSkeleton.draw(frame_shape, skt, white_bkg)

            # PNG版本
            png_bkg = MatchstickSkeleton.generate_png_canvas(frame_shape)
            canvas = MatchstickSkeleton.draw_png(frame_shape, skt, png_bkg, png_parts)
            # canvas = MatchstickSkeleton.draw_png_demo(frame_shape, skt, png_bkg, png_parts)
            canvas = MatchstickSkeleton.convert_transparent_png(canvas)

            vw.write(canvas)
            print('[Info] 写入 {} / {} 帧完成'.format(i, n_data))

        vw.release()
        print('[Info] 视频写入完成! ')


def skeleton_main_test():
    """
    测试写入视频
    """
    frame_shape = (1024, 576, 4)  # 背景尺寸
    fps = 24  # FPS
    o_vid_path = os.path.join(DATA_DIR, 'video.{}.mp4'.format(get_current_time_str()))
    skt_file = os.path.join(DATA_DIR, 'skt_file.20191119145920.m.txt')

    img_png_path = os.path.join(DATA_DIR, 'custom', 'trans.png')
    # img_draw_path = os.path.join(DATA_DIR, 'custom', 'trans.png')
    img_draw_path = os.path.join(DATA_DIR, 'custom', 'x.png')
    img_config_path = os.path.join(DATA_DIR, 'configs', 'parts_config.json')

    # 生成PNG
    png_parts = SkeletonGenerator.generate_body_parts(img_png_path, img_draw_path, img_config_path)

    # 写入视频
    SkeletonMain.write_video_with_parts(o_vid_path, skt_file, fps, frame_shape, png_parts)


def main():
    skeleton_main_test()


if __name__ == '__main__':
    main()
