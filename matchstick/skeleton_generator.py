#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/11/18
"""

import os
import sys
import cv2
import numpy as np

from matchstick.parts_segmentation import PartsSegmentation

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if p not in sys.path:
    sys.path.append(p)

from config_reader import config_reader
from matchstick.matchstick_skeleton import MatchstickSkeleton
from model.cmu_model import get_testing_model
from processing import extract_parts
from root_dir import DATA_DIR, ROOT_DIR
from utils.project_utils import *


class SkeletonGenerator(object):
    """
    骨骼生成的类
    """

    def __init__(self, vid_path, weights_path, config_path, skt_file, skt_m_file):
        self.vid_path = vid_path
        self.video_name = self.vid_path.split('/')[-1]
        print('[Info] 视频名称: {}'.format(self.video_name))

        self.weights_path = weights_path
        self.config_path = config_path
        self.model, self.params, self.model_params = self.load_model()

        self.skt_file = skt_file
        self.skt_m_file = skt_m_file

    def load_model(self):
        """
        加载模型
        """
        model = get_testing_model()
        model.load_weights(self.weights_path)
        params, model_params = config_reader(self.config_path)
        return model, params, model_params

    def generate(self, p_frame=-1):
        """
        生成骨架
        """
        cap = cv2.VideoCapture(self.vid_path)
        n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # 24

        print('[Info] 总帧数: {}, 帧率: {}'.format(n_frame, fps))

        skt_prev = None  # 前一个骨骼点，用于补充

        # 写入的frame值
        if p_frame == -1 or p_frame > n_frame:
            p_frame = n_frame

        for i in range(0, p_frame):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)

            ret, frame = cap.read()

            all_peaks, subset, candidate = extract_parts(frame, self.params, self.model, self.model_params)

            # 原始的骨架
            ms = MatchstickSkeleton(subset, candidate, frame)
            skt = ms.get_skeleton()

            # 修改之后的骨架
            skt_m = MatchstickSkeleton.generate_skeleton_v2(skt, skt_prev, None)  # 根据前一个骨骼点补充
            skt_prev = skt_m

            # 写入原始骨架
            skt_json = json.dumps(skt)
            write_line(self.skt_file, skt_json)

            # 写入之后的骨架
            skt_f_json = json.dumps(skt_m)
            write_line(self.skt_m_file, skt_f_json)

            print('[Info] 已处理帧数: {} / {}'.format(i, n_frame))

        print('[Info] 视频提取完成! ')

    @staticmethod
    def modify_skt(skt_file, skt_m_file):
        """
        修改骨骼文件，补充一些点，写入新的文件
        """
        # skt_file = os.path.join(DATA_DIR, 'skt_file.txt')  # 输入文件
        # skt_m_file = os.path.join(DATA_DIR, 'skt_file.m.txt')  # 输出文件

        create_file(skt_m_file)

        data_lines = read_file(skt_file)
        n_data = len(data_lines)
        print('[Info] 总帧数: {}'.format(n_data))

        skt_prev = None  # 前一个骨骼点，用于补充

        for i, data_line in enumerate(data_lines):
            skt = json.loads(data_line)
            skt_f = MatchstickSkeleton.generate_skeleton_v2(skt, skt_prev, None)  # 根据前一个骨骼点补充
            skt_prev = skt_f
            skt_json = json.dumps(skt_f)
            write_line(skt_m_file, skt_json)
            print('[Info] 已处理帧数: {} / {}'.format(i, n_data))

        print('[Info] 骨骼修改完成')

    @staticmethod
    def write_video(o_vid_path, skt_file, fps, frame_shape):
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
            white_bkg = np.ones(frame_shape) * 255  # 需要实时更新
            canvas = MatchstickSkeleton.draw(frame_shape, skt, white_bkg)
            vw.write(canvas)
            print('[Info] 写入 {} / {} 帧完成'.format(i, n_data))

        vw.release()
        print('[Info] 视频写入完成! ')

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

    @staticmethod
    def generate_body_parts(img_draw, img_png, img_config_path):
        """
        由PNG图像生成10个身体部分
        :return: 10个身体部分
        """

        ps = PartsSegmentation(img_draw, img_png, img_config_path)
        png_parts = ps.process()

        return png_parts


def write_video_test():
    """
    测试写入视频
    """
    frame_shape = (1024, 576, 4)
    fps = 24
    o_vid_path = os.path.join(DATA_DIR, 'video.{}.mp4'.format(get_current_time_str()))
    skt_file = os.path.join(DATA_DIR, 'skt_file.20191119145920.m.txt')
    # SkeletonGenerator.write_video(o_vid_path, skt_file, fps, frame_shape)

    img_png_path = os.path.join(DATA_DIR, 'custom', 'trans.png')
    # img_draw_path = os.path.join(DATA_DIR, 'custom', 'trans.png')
    img_draw_path = os.path.join(DATA_DIR, 'custom', 'x.png')
    img_config_path = os.path.join(DATA_DIR, 'configs', 'parts_config.json')

    png_parts = SkeletonGenerator.generate_body_parts(img_png_path, img_draw_path, img_config_path)
    SkeletonGenerator.write_video_with_parts(o_vid_path, skt_file, fps, frame_shape, png_parts)


def skeleton_generator_test():
    """
    测试骨骼生成
    """
    video_path = os.path.join(DATA_DIR, 'videos', 'dance_video.mp4')
    keras_weights_file = os.path.join(ROOT_DIR, "model/keras/model.h5")
    config_file = os.path.join(DATA_DIR, 'configs', 'model_config')
    skt_file = os.path.join(DATA_DIR, "skt_file.{}.txt".format(get_current_time_str()))
    skt_m_file = os.path.join(DATA_DIR, "skt_file.{}.m.txt".format(get_current_time_str()))

    create_file(skt_file)  # 创建新的文件
    create_file(skt_m_file)  # 创建新的文件

    sg = SkeletonGenerator(video_path, keras_weights_file, config_file, skt_file, skt_m_file)
    sg.generate(p_frame=100)


def main():
    # skeleton_generator_test()
    write_video_test()


if __name__ == '__main__':
    main()
