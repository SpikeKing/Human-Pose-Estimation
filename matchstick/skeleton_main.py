#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/11/20
"""

import os
import cv2
import json

import imageio
import numpy as np
from PIL import Image

from matchstick.matchstick_skeleton import MatchstickSkeleton
from matchstick.skeleton_generator import SkeletonGenerator, read_file
from root_dir import DATA_DIR
from utils.project_utils import get_current_time_str


class SkeletonMain(object):

    def __init__(self):
        self.frame_shape = (1024, 576, 4)  # 背景尺寸
        self.fps = 24
        self.skt_templates = self.get_skt_templates()  # 模板文件
        self.skeleton_bkg = os.path.join(DATA_DIR, 'configs', 'skeleton_bkg.png')  # 骨骼背景
        self.skeleton_config = os.path.join(DATA_DIR, 'configs', 'skeleton_config.json')  # 骨骼背景参数

    @staticmethod
    def get_skt_templates():
        skt_file = os.path.join(DATA_DIR, 'configs', 'skt_file.0.skt')
        return [skt_file]

    def generate_matchstick_api(self, custom_png, skt_idx):
        """
        生成火柴人运动的API
        :param custom_png: 用户的PNG
        :param skt_idx: 骨骼ID
        :return: 头部的PNG, 用户的PNG, Gif存储路径
        """
        img_draw = custom_png
        img_bkg = cv2.imread(self.skeleton_bkg, cv2.IMREAD_UNCHANGED)
        png_parts = SkeletonGenerator.generate_body_parts(img_draw, img_bkg, self.skeleton_config)

        head_png = png_parts[0]

        skt_file = self.skt_templates[skt_idx]

        # 写入视频
        o_gif_path = os.path.join(DATA_DIR, 'tmp', 'matchstick.{}.gif'.format(get_current_time_str()))
        SkeletonMain.write_gif_with_parts(o_gif_path, skt_file, self.fps, self.frame_shape, png_parts)

        return head_png, custom_png, o_gif_path

    @staticmethod
    def write_video_with_parts(o_vid_path, skt_file, fps, frame_shape, png_parts):
        """
        根据骨骼文件, 写入视频
        :param o_vid_path: 视频输出路径
        :param skt_file: 骨骼文件
        :param fps: FPS
        :param frame_shape: Frame Shape
        :param png_parts: PNG部分
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

    @staticmethod
    def write_gif_with_parts(o_gif_path, skt_file, fps, frame_shape, png_parts):
        """
        根据骨骼文件, 写入视频
        :param o_gif_path: 视频输出路径
        :param skt_file: 骨骼文件
        :param fps: FPS
        :param frame_shape: Frame Shape
        :return: None
        """
        (h, w, _) = frame_shape  # (1024, 576, 3)

        data_lines = read_file(skt_file)
        n_data = len(data_lines)
        print('[Info] 总帧数: {}'.format(n_data))

        images = []

        for i, data_line in enumerate(data_lines):
            skt = json.loads(data_line)
            skt = [tuple(s) for s in skt]

            # PNG版本
            png_bkg = MatchstickSkeleton.generate_png_canvas(frame_shape)
            canvas = MatchstickSkeleton.draw_png(frame_shape, skt, png_bkg, png_parts)
            canvas = MatchstickSkeleton.convert_transparent_png(canvas)

            # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            # canvas_pil = Image.fromarray(canvas)
            # images.append(canvas_pil)

            images.append(canvas)

        imageio.mimsave(o_gif_path, images)
        # images[0].save('moving_ball.gif', format='GIF', append_images=images[1:], save_all=True, duration=100, loop=0)
        print('[Info] 视频写入完成! ')


def skeleton_main_for_video_test():
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


def skeleton_main_for_gif_test():
    """
    测试写入gif
    """
    frame_shape = (1024, 576, 4)  # 背景尺寸
    fps = 24  # FPS
    o_gif_path = os.path.join(DATA_DIR, 'img_gif.{}.gif'.format(get_current_time_str()))
    skt_file = os.path.join(DATA_DIR, 'skt_file.20191119145920.m.txt')

    img_png_path = os.path.join(DATA_DIR, 'custom', 'trans.png')
    img_draw_path = os.path.join(DATA_DIR, 'custom', 'x.png')
    img_config_path = os.path.join(DATA_DIR, 'configs', 'parts_config.json')

    # 生成PNG
    png_parts = SkeletonGenerator.generate_body_parts(img_png_path, img_draw_path, img_config_path)

    # 写入视频
    SkeletonMain.write_gif_with_parts(o_gif_path, skt_file, fps, frame_shape, png_parts)


def matchstick_api_test():
    """
    火柴人API测试
    """
    custom_png_path = os.path.join(DATA_DIR, 'custom', 'x.png')
    custom_png = cv2.imread(custom_png_path, cv2.IMREAD_UNCHANGED)  # 注意OpenCV格式
    skt_idx = 0  # 火柴人模板的索引

    sm = SkeletonMain()
    head_png, custom_png, o_gif_path = sm.generate_matchstick_api(custom_png=custom_png, skt_idx=skt_idx)

    # 生成的结果
    head_png_path = os.path.join(DATA_DIR, 'tmp', 'head.{}.png'.format(get_current_time_str()))
    custom_png_path = os.path.join(DATA_DIR, 'tmp', 'custom.{}.png'.format(get_current_time_str()))

    cv2.imwrite(head_png_path, head_png)
    cv2.imwrite(custom_png_path, custom_png)
    print('[Info] Gif的路径: {}'.format(o_gif_path))


def main():
    # skeleton_main_for_video_test()
    # skeleton_main_for_gif_test()
    matchstick_api_test()


if __name__ == '__main__':
    main()
