#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/11/18
"""

import os
import cv2

from config_reader import config_reader
from matchstick.matchstick_skeleton import MatchstickSkeleton
from model.cmu_model import get_testing_model
from processing import extract_parts
from root_dir import DATA_DIR, ROOT_DIR
from utils.project_utils import *


class SkeletonGenerator(object):
    def __init__(self, vid_path, weights_path, config_path, skt_file):
        self.vid_path = vid_path
        self.video_name = self.vid_path.split('/')[-1]
        print('[Info] 视频名称: {}'.format(self.video_name))

        self.weights_path = weights_path
        self.config_path = config_path
        self.model, self.params, self.model_params = self.load_model()

        self.skt_file = skt_file

    def load_model(self):
        model = get_testing_model()
        model.load_weights(self.weights_path)
        params, model_params = config_reader(self.config_path)
        return model, params, model_params

    def generate(self):
        cap = cv2.VideoCapture(self.vid_path)
        n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # 24

        print('[Info] 总帧数: {}, 帧率: {}'.format(n_frame, fps))

        for i in range(0, n_frame - 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            all_peaks, subset, candidate = extract_parts(frame, self.params, self.model, self.model_params)
            ms = MatchstickSkeleton(subset, candidate, frame)
            skt = ms.get_skeleton()
            skt_json = json.dumps(skt)
            write_line(self.skt_file, skt_json)
            print('[Info] 已处理帧数: {} / {}'.format(i, n_frame))

            # canvas = ms.draw(frame.shape, skt)
            # output_img = os.path.join(DATA_DIR, 'errors', video_name + ".{}.rx.jpg".format(i))
            # print('[Info] writer: {}'.format(output_img))
            # cv2.imwrite(output_img, canvas)
            # canvas = (canvas * 255).astype(np.uint8)
            # vw.write(canvas)
            # print('[Info] count: {}, total: {}'.format(i, n_frame))

        # vw.release()
        print('[Info] 视频提取完成! ')


def skeleton_generator_test():
    video_path = os.path.join(DATA_DIR, 'videos', 'dance_video.mp4')
    keras_weights_file = os.path.join(ROOT_DIR, "model/keras/model.h5")
    config_file = os.path.join(DATA_DIR, 'configs', 'model_config')
    skt_file = os.path.join(DATA_DIR, "skt_file.{}.txt".format(get_current_time_str()))
    create_file(skt_file)  # 创建新的文件

    sg = SkeletonGenerator(video_path, keras_weights_file, config_file, skt_file)
    sg.generate()


def main():
    skeleton_generator_test()


if __name__ == '__main__':
    main()
