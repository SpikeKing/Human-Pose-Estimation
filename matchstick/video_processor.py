#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/10/30
"""

import os
import cv2
import numpy as np

from config_reader import config_reader
from matchstick.MatchstickSkeleton import MatchstickSkeleton
from model.cmu_model import get_testing_model
from processing import extract_parts, draw
from root_dir import DATA_DIR, ROOT_DIR

from utils.project_utils import *


def process_video(vid_path):
    """
    从视频中，生成人体姿态的数据，写入文件
    :param vid_path:
    :return:
    """
    video_name = vid_path.split('/')[-1]
    print('[Info] 视频名称: {}'.format(video_name))

    cap = cv2.VideoCapture(vid_path)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 24

    print('[Info] 总帧数: {}, 帧率: {}'.format(n_frame, fps))

    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # o_vid_path = os.path.join(DATA_DIR, 'video.mp4')
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the lower case，可以

    # vw = cv2.VideoWriter(filename=o_vid_path, fourcc=fourcc, fps=fps, frameSize=(width, height), isColor=True)

    keras_weights_file = os.path.join(ROOT_DIR, "model/keras/model.h5")
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    params, model_params = config_reader()

    skt_file = os.path.join(DATA_DIR, 'skt_file.txt')  # TODO: 原始数据的写入文件
    create_file(skt_file)  # 创建新的文件

    for i in range(0, n_frame - 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        all_peaks, subset, candidate = extract_parts(frame, params, model, model_params)
        ms = MatchstickSkeleton(subset, candidate, frame)
        skt = ms.get_skeleton()
        skt_json = json.dumps(skt)
        write_line(skt_file, skt_json)
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


def modify_skt():
    """
    修改骨骼文件，补充一些点，写入新的文件
    :return: None
    """
    skt_file = os.path.join(DATA_DIR, 'skt_file.txt')  # 输入文件

    skt_m_file = os.path.join(DATA_DIR, 'skt_file.m.txt')  # 输出文件
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


def write_video_pose(vid_path):
    """
    将姿态写入视频
    :param vid_path: 视频路径
    :return: None
    """
    video_name = vid_path.split('/')[-1]
    print('[Info] 视频名称: {}'.format(video_name))

    cap = cv2.VideoCapture(vid_path)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 24

    print('[Info] 总帧数: {}, 帧率: {}'.format(n_frame, fps))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_shape = (height, width, 3)  # (1024, 576, 3)

    o_vid_path = os.path.join(DATA_DIR, 'video.pose.mp4')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the lower case，可以

    vw = cv2.VideoWriter(filename=o_vid_path, fourcc=fourcc, fps=fps, frameSize=(width, height), isColor=True)

    keras_weights_file = os.path.join(ROOT_DIR, "model/keras/model.h5")
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    params, model_params = config_reader()

    skt_file = os.path.join(DATA_DIR, 'skt_file.m.txt')
    data_lines = read_file(skt_file)
    n_data = len(data_lines)
    print('[Info] 总帧数: {}'.format(n_data))

    n_frame = 165
    for i in range(0, n_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()

        # 写入原有骨骼的逻辑
        all_peaks, subset, candidate = extract_parts(frame, params, model, model_params)
        canvas = draw(frame, all_peaks, subset, candidate)

        vw.write(canvas)
        print('[Info] count: {}, total: {}'.format(i, n_frame))

    vw.release()
    print('[Info] 视频提取完成! ')


def write_video(vid_path):
    """
    将火柴人写入视频
    :param vid_path: 视频路径
    :return: None
    """
    video_name = vid_path.split('/')[-1]
    print('[Info] 视频名称: {}'.format(video_name))

    cap = cv2.VideoCapture(vid_path)
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 24
    ret, frame = cap.read()
    print(frame.shape)

    print('[Info] 总帧数: {}, 帧率: {}'.format(n_frame, fps))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # height = 210 * 5
    # width = 297 * 5
    frame_shape = (height, width, 3)  # (1024, 576, 3)
    print('[Info] 宽: {}, 高: {}, frame: {}'.format(width, height, frame_shape))

    o_vid_path = os.path.join(DATA_DIR, 'video.mp4')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # note the lower case，可以

    vw = cv2.VideoWriter(filename=o_vid_path, fourcc=fourcc, fps=fps, frameSize=(width, height), isColor=True)

    skt_file = os.path.join(DATA_DIR, 'skt_file.m.txt')
    data_lines = read_file(skt_file)
    n_data = len(data_lines)
    print('[Info] 总帧数: {}'.format(n_data))

    # first_frame = data_lines[0]
    # skt = json.loads(first_frame)
    # skt = [tuple(s) for s in skt]
    # skt = MatchstickSkeleton.resize_skt(skt, 0.5, (150, 150))
    # canvas = MatchstickSkeleton.draw(frame_shape, skt)
    # o_vid_path = os.path.join(DATA_DIR, 'first_frame.jpg')
    # cv2.imwrite(o_vid_path, canvas)

    for i, data_line in enumerate(data_lines):
        skt = json.loads(data_line)
        skt = [tuple(s) for s in skt]
        # skt = MatchstickSkeleton.resize_skt(skt, 0.3, (150, 150))  # 测试偏移和缩放
        canvas = MatchstickSkeleton.draw(frame_shape, skt, np.ones(frame_shape) * 255)
        vw.write(canvas)
        print('[Info] 写入 {} / {} 帧完成'.format(i, n_data))

    vw.release()
    print('[Info] 视频写入完成! ')


def process_img(img_path, output_img_path):
    keras_weights_file = os.path.join(ROOT_DIR, "model/keras/model.h5")
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    params, model_params = config_reader()
    input_image = cv2.imread(img_path)  # B,G,R order

    all_peaks, subset, candidate = extract_parts(input_image, params, model, model_params)
    canvas = draw(input_image, all_peaks, subset, candidate)

    cv2.imwrite(output_img_path, canvas)
    print('[Info] 图像处理完成! ')


def process_video_test():
    video_dir = os.path.join(DATA_DIR, 'videos')
    vid_path = os.path.join(video_dir, 'dance_video.mp4')
    process_video(vid_path)


def write_video_test():
    video_dir = os.path.join(DATA_DIR, 'videos')
    vid_path = os.path.join(video_dir, 'dance_video.mp4')
    write_video(vid_path)  # 生成火柴人
    # write_video_pose(vid_path)  # 生成姿态点


def process_img_test():
    name = 'dance_video.mp4_0.jpg'
    img_path = os.path.join(DATA_DIR, 'frames', name)
    output_img_path = os.path.join(DATA_DIR, 'frames', name + ".r.jpg")
    process_img(img_path, output_img_path)


def main():
    # process_video_test()  # 第1步，生成关键点
    # modify_skt()  # 第二步修改关键点
    write_video_test()


if __name__ == '__main__':
    main()
