#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/11/8
"""

import math

import cv2
import numpy as np


class MatchstickSkeleton(object):
    def __init__(self, subset, candidate, image):
        self.s = self.get_main_person(subset)
        self.candidate = candidate
        self.image = image
        # [鼻子0, 颈部1, 左胯8，右胯11, 右肘3,
        # 左肘6, 右腕4, 左腕7, 右膝盖9, 左膝盖12,
        # 右脚10, 左脚13]
        self.n_parts = 12
        self.parts = [int(self.s[i]) for i in [0, 1, 8, 11, 3, 6, 4, 7, 9, 12, 10, 13]]
        self.skt = self.generate_skeleton()

    def get_skeleton(self):
        return self.skt

    @staticmethod
    def get_main_person(subset):
        """
        获取核心的人
        :param subset: 人的集合
        :return: 核心的人
        """
        conf_list = []  # 置信度列表
        for s in subset:
            conf_list.append(s[18])
        idx_max = conf_list.index(max(conf_list))  # 最大置信度
        return subset[idx_max]

    def generate_skeleton(self):
        """
        骨骼点，列表
        :return: 骨骼点
        """
        # skeleton = [self.get_point(i) for i in self.parts]
        skeleton = []
        for i in self.parts:
            skeleton.append(self.get_point(i))
        return skeleton

    @staticmethod
    def generate_skeleton_v2(skt, skt1=None, skt2=None):
        o_skt = MatchstickSkeleton.generate_other_skeleton(skt1, skt2)
        skeleton = []
        for c, point in enumerate(skt):
            if skt[c][0] != 0:
                skeleton.append(skt[c])
            else:
                print('[Info] 补充点: {} - {}'.format(c, o_skt[c]))
                skeleton.append(o_skt[c])
        return skeleton

    @staticmethod
    def generate_other_skeleton(skt1, skt2):
        """
        生成其他的骨骼点
        :param skt1: 骨架1
        :param skt2: 骨架2
        :return: 平均骨骼
        """
        if not skt1:
            return skt2
        if not skt2:
            return skt1
        if not skt1 and not skt2:
            raise Exception('Error')
        o_skt = []
        for i in range(len(skt1)):
            s1 = skt1[i]
            s2 = skt2[i]
            if s1[0] != 0 and s2[0] != 0:
                x = int((s1[0] + s2[0]) // 2)
                y = int((s1[1] + s2[1]) // 2)
                o_skt.append((x, y))
            elif s1[0] == 0:
                o_skt.append(s2)
            elif s2[0] == 0:
                o_skt.append(s1)
            else:
                raise Exception('Error')

        return o_skt

    def get_point(self, idx):
        """
        candidate的索引点
        :param idx: 索引
        :return: xy坐标
        """
        if idx == -1:
            return 0, 0
        cx, cy = int(self.candidate[idx, 0]), int(self.candidate[idx, 1])
        return cx, cy

    @staticmethod
    def get_other_parameters(head_p, neck_p, rhip_p, lhip_p):
        """
        头的半径
        :param head_p: 头部的点
        :param neck_p: 颈部的点
        :param rhip_p: 右臀部的点
        :param lhip_p: 左臀部的点
        :return: 半径
        """
        ll = math.sqrt(math.pow((head_p[0] - neck_p[0]), 2) + math.pow((head_p[1] - neck_p[1]), 2))
        r = int(ll // 3 * 2)

        body_ex = int(min(rhip_p[0], lhip_p[0]) + abs(rhip_p[0] - lhip_p[0]) / 2)
        body_ey = int(min(rhip_p[1], lhip_p[1]) + abs(rhip_p[1] - lhip_p[1]) / 2)

        # ll = math.sqrt(math.pow((body_ex - neck_p[0]), 2) + math.pow((body_ey - neck_p[1]), 2))
        # r = int(ll // 2)

        pa, pb = int((r / ll) * (neck_p[0] - head_p[0])), int((r / ll) * (neck_p[1] - head_p[1]))
        body_sx, body_sy = int(head_p[0] + pa), int(head_p[1] + pb)  # 头和身的连接点

        return r, (body_sx, body_sy), (body_ex, body_ey)

    @staticmethod
    def resize_skt(skt, scale, off, f_skt=None):
        n_skt = []
        if f_skt:
            skt_min, _ = MatchstickSkeleton.get_skt_min_and_max(f_skt)
        else:
            skt_min = (0, 0)

        for s in skt:
            ns = int((s[0] - skt_min[0]) * scale + off[0]) + 1, int((s[1] - skt_min[1]) * scale + off[1]) + 1
            n_skt.append(ns)
        return n_skt

    @staticmethod
    def get_skt_min_and_max(skt):
        r, body_sp, body_ep = MatchstickSkeleton.get_other_parameters(skt[0], skt[1], skt[2], skt[3])
        head_min = (skt[0][0] - r, skt[0][1] - r)
        head_max = (skt[0][0] + r, skt[0][1] + r)
        skt_all = skt + [head_min, head_max]
        x_list = np.array([s[0] for s in skt_all])
        y_list = np.array([s[1] for s in skt_all])
        x_min, y_min = np.min(x_list), np.min(y_list)
        x_max, y_max = np.max(x_list), np.max(y_list)
        skt_min, skt_max = (x_min, y_min), (x_max, y_max)
        return skt_min, skt_max

    @staticmethod
    def draw(image_shape, skt, canvas=None):
        # if not canvas:
        #     canvas = np.ones(image_shape) * 255  # 白色背景

        black_color = (0, 0, 0)
        tn = 3

        r, body_sp, body_ep = MatchstickSkeleton.get_other_parameters(skt[0], skt[1], skt[2], skt[3])

        if skt[0][0] != 0 and r > 0:  # 头部
            cv2.circle(canvas, center=skt[0], radius=r, color=black_color, thickness=tn)  # 绘制头部
        else:
            print('[Warning] 头部缺失: 点 {}, 半径 {}'.format(skt[0], r))

        if skt[1][0] != 0 and skt[2][0] != 0 and skt[3][0] != 0:  # 绘制身躯，头到脖子，脖子到屁股
            cv2.line(canvas, pt1=body_sp, pt2=skt[1], color=black_color, thickness=tn)  # 绘制身体
            cv2.line(canvas, pt1=skt[1], pt2=body_ep, color=black_color, thickness=tn)  # 绘制身体
        else:
            print('[Warning] 身体缺失')

        if skt[1][0] != 0 and skt[4][0] != 0:  # 绘制右上臂
            cv2.line(canvas, pt1=skt[1], pt2=skt[4], color=black_color, thickness=tn)
        else:
            print('[Warning] 右上臂缺失')
        if skt[1][0] != 0 and skt[5][0] != 0:  # 绘制左上臂
            cv2.line(canvas, pt1=skt[1], pt2=skt[5], color=black_color, thickness=tn)
        else:
            print('[Warning] 左上臂缺失')

        if skt[4][0] != 0 and skt[6][0] != 0:  # 绘制右下臂
            cv2.line(canvas, pt1=skt[4], pt2=skt[6], color=black_color, thickness=tn)
        else:
            print('[Warning] 右下臂缺失')
        if skt[5][0] != 0 and skt[7][0] != 0:  # 绘制左下臂
            cv2.line(canvas, pt1=skt[5], pt2=skt[7], color=black_color, thickness=tn)
        else:
            print('[Warning] 左下臂缺失')

        if skt[2][0] != 0 and skt[3][0] != 0 and skt[8][0] != 0:  # 绘制右上腿
            cv2.line(canvas, pt1=body_ep, pt2=skt[8], color=black_color, thickness=tn)
        else:
            print('[Warning] 右上腿缺失')
        if skt[2][0] != 0 and skt[3][0] != 0 and skt[9][0] != 0:  # 绘制左上腿
            cv2.line(canvas, pt1=body_ep, pt2=skt[9], color=black_color, thickness=tn)
        else:
            print('[Warning] 左上腿缺失')

        if skt[8][0] != 0 and skt[10][0] != 0:  # 绘制右下腿
            cv2.line(canvas, pt1=skt[8], pt2=skt[10], color=black_color, thickness=tn)
        else:
            print('[Warning] 右下腿缺失')
        if skt[9][0] != 0 and skt[11][0] != 0:  # 绘制左下腿
            cv2.line(canvas, pt1=skt[9], pt2=skt[11], color=black_color, thickness=tn)
        else:
            print('[Warning] 左下腿缺失')

        canvas = canvas.astype(np.uint8)

        return canvas
