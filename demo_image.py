#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/10/18
"""

import os
import cv2
import time
import argparse

from root_dir import ROOT_DIR, DATA_DIR

from processing import extract_parts, draw, draw_skeleton

from config_reader import config_reader
from model.cmu_model import get_testing_model

from utils.project_utils import *


def std_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image')
    parser.add_argument('--output', type=str, default='result.png', help='output image')
    parser.add_argument('--model', type=str, default='model/keras/model.h5', help='path to the weights file')

    args = parser.parse_args()
    image_path = args.image
    output = args.output
    keras_weights_file = args.model

    tic = time.time()
    print('start processing...')

    # load model

    # authors of original model don't use
    # vgg normalization (subtracting mean) on input images
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    # load config
    params, model_params = config_reader()

    input_image = cv2.imread(image_path)  # B,G,R order

    all_peaks, subset, candidate = extract_parts(input_image, params, model, model_params)
    canvas = draw(input_image, all_peaks, subset, candidate)

    toc = time.time()
    print('processing time is %.5f' % (toc - tic))

    cv2.imwrite(output, canvas)

    cv2.destroyAllWindows()


def predict_img(image_path, model, output_img, output_pos):
    # load config
    params, model_params = config_reader()

    input_image = cv2.imread(image_path)  # B,G,R order

    all_peaks, subset, candidate = extract_parts(input_image, params, model, model_params)

    for peak in all_peaks:
        if peak:
            write_line(output_pos, "{}".format(peak[0]))
        else:
            write_line(output_pos, "")

    # canvas = draw(input_image, all_peaks, subset, candidate)
    canvas = draw_skeleton(input_image, subset, candidate)

    cv2.imwrite(output_img, canvas)
    cv2.destroyAllWindows()


def process_img_batch():
    keras_weights_file = os.path.join(ROOT_DIR, "model/keras/model.h5")
    output_img_dir = os.path.join(DATA_DIR, "results")
    output_pos_dir = os.path.join(DATA_DIR, "pos")
    img_dir = os.path.join(DATA_DIR, 'frames-p')

    model = get_testing_model()
    model.load_weights(keras_weights_file)

    paths_list, names_list = traverse_dir_files(img_dir)

    for path, name in zip(paths_list, names_list):
        print('[Info] name: {}'.format(name))
        output_img = os.path.join(output_img_dir, name + ".r.jpg")
        output_pos = os.path.join(output_pos_dir, name + ".p.txt")
        predict_img(path, model, output_img, output_pos)


def process_one_img():
    img_path = os.path.join(DATA_DIR, 'test_img.jpg')
    keras_weights_file = os.path.join(ROOT_DIR, "model/keras/model.h5")
    model = get_testing_model()
    model.load_weights(keras_weights_file)

    output_img = os.path.join(img_path + ".r.jpg")
    output_pos = os.path.join(img_path + ".p.txt")
    predict_img(img_path, model, output_img, output_pos)


def main():
    process_one_img()  # 处理一张图像
    # process_img_batch()  # 处理多张图像


if __name__ == '__main__':
    main()
