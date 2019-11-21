#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/11/8
"""

import math
import os

import cv2
import numpy as np
from root_dir import DATA_DIR


class MatchstickSkeleton(object):
    """
    火柴人的骨骼类
    """

    def __init__(self, subset, candidate, image):
        self.s = self.get_main_person(subset)
        self.candidate = candidate
        self.image = image
        # [鼻子0, 颈部1, 左胯8，右胯11, 右肘3,
        # 左肘6, 右腕4, 左腕7, 右膝盖9, 左膝盖12,
        # 右脚10, 左脚13]
        self.n_parts = 12
        self.parts = [int(self.s[i]) for i in [0, 1, 8, 11, 3, 6, 4, 7, 9, 12, 10, 13]]
        self.skt = None
        self.head_radius = 20  # 头部半径参数

    def get_skeleton(self):
        if not self.skt:
            self.skt = self.generate_skeleton()
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
        # r = int(ll // 3 * 2)
        # r = int(ll)
        r = 100

        body_ex = int(min(rhip_p[0], lhip_p[0]) + abs(rhip_p[0] - lhip_p[0]) / 2)
        body_ey = int(min(rhip_p[1], lhip_p[1]) + abs(rhip_p[1] - lhip_p[1]) / 2)

        # ll = math.sqrt(math.pow((body_ex - neck_p[0]), 2) + math.pow((body_ey - neck_p[1]), 2))
        # r = int(ll // 2)

        pa, pb = int((r / ll) * (neck_p[0] - head_p[0])), int((r / ll) * (neck_p[1] - head_p[1]))
        body_sx, body_sy = int(head_p[0] + pa), int(head_p[1] + pb)  # 头和身的连接点

        return r, (body_sx, body_sy), (body_ex, body_ey)

    @staticmethod
    def get_other_parameters_v2(head_p, neck_p, rhip_p, lhip_p):
        """
        根据颈部，生成头部的点
        :param head_p: 已有头部的点，未使用
        :param neck_p: 颈部的点
        :param rhip_p: 右臀部
        :param lhip_p: 左臀部
        :return: 头部半径, 头部中心点, 身体结束点
        """
        r = 50
        body_ex = int(min(rhip_p[0], lhip_p[0]) + abs(rhip_p[0] - lhip_p[0]) / 2)
        body_ey = int(min(rhip_p[1], lhip_p[1]) + abs(rhip_p[1] - lhip_p[1]) / 2)

        # ll = math.sqrt(math.pow((body_ex - neck_p[0]), 2) + math.pow((body_ey - neck_p[1]), 2))
        # r = int(ll // 2)

        # pa, pb = int((r / ll) * (neck_p[0] - head_p[0])), int((r / ll) * (neck_p[1] - head_p[1]))
        # body_sx, body_sy = int(head_p[0] + pa), int(head_p[1] + pb)  # 头和身的连接点

        head_p = neck_p[0], neck_p[1] - r

        return r, head_p, (body_ex, body_ey)

    @staticmethod
    def resize_skt(skt, scale, off, f_skt=None):
        """
        缩放火柴人的骨骼
        :param skt: 骨骼
        :param scale: 缩放尺寸
        :param off: 偏移
        :param f_skt: 首帧火柴人
        :return: 新的火柴人骨骼
        """
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
        """
        获取骨骼的小坐标和最大坐标
        :param skt: 骨骼
        :return: 最小坐标骨骼和最大坐标骨骼
        """
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
    def points2degree(point_a, point_b):
        """
        获取两个点之间的角度，point: （114, 108) (267, 324)
        :param point_a: 点a
        :param point_b: 点b
        :return 两个点之间的角度
        """
        x = point_a[0] - point_b[0]  # 水平值
        p_len = np.sqrt(np.power((point_a[0] - point_b[0]), 2) + np.power((point_a[1] - point_b[1]), 2))
        sin_data = x / p_len
        deg = np.rad2deg(np.arcsin(sin_data))  # arcsin(0.5) = 30°

        y = point_a[1] - point_b[1]
        if y > 0:
            deg = ((-1) * deg + 180) % 360

        return deg, p_len  # 角度

    @staticmethod
    def convert_transparent_png(image_4channel):
        """
        将PNG图像转换为白底图像
        :param image_4channel: 4通道的PNG图像
        :return: 白底图像
        """
        # image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)  # 读取PNG图像
        alpha_channel = image_4channel[:, :, 3]
        rgb_channels = image_4channel[:, :, :3]

        # White Background Image
        white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

        # Alpha factor
        alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
        alpha_factor = np.concatenate((alpha_factor, alpha_factor, alpha_factor), axis=2)

        # Transparent Image Rendered on White Background
        base = rgb_channels.astype(np.float32) * alpha_factor
        white = white_background_image.astype(np.float32) * (1 - alpha_factor)
        final_image = base + white
        return final_image.astype(np.uint8)

    @staticmethod
    def rotate_bound(image, angle):
        """
        旋转图像
        :param image: 图像
        :param angle: 角度
        :return: 旋转的图像
        """
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w / 2, h / 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    @staticmethod
    def generate_png_square(img_opencv, center, radius):
        """
        生成方形的PNG图像
        :param img_opencv: 图像
        :param center: 中心点
        :param radius: 半径
        :return: 经过偏移的起始点，调整过的图像
        """
        ih, iw, _ = img_opencv.shape
        # print('[Info] img shape: {}'.format(img_opencv.shape))

        scale_h = radius * 2 / ih
        scale_w = radius * 2 / iw

        img_opencv = cv2.resize(img_opencv, (int(iw * scale_w), int(ih * scale_h)))

        po_x, po_y = center[0] - radius, center[1] - radius
        return (po_x, po_y), img_opencv

    @staticmethod
    def generate_png_line(img_opencv, pa, pb, width=100):
        """
        生成PNG图像
        :param img_opencv: 图像
        :param pa: 起始点
        :param pb: 结束点
        :param width: 肢体resize的宽度
        :return: (偏移点)
        """

        ih, iw, _ = img_opencv.shape
        # print('[Info] img shape: {}'.format(img_opencv.shape))

        p_degree, p_len = MatchstickSkeleton.points2degree(pa, pb)  # 计算角度
        # print('[Info] 旋转角度: {}'.format(p_degree))

        scale = p_len / ih  # 两个点的长度 除以 图像的高

        img_opencv = cv2.resize(img_opencv, (width, int(ih * scale)))
        img_opencv_white = MatchstickSkeleton.convert_transparent_png(img_opencv)  # 透明图像转换为实体图像

        img_rotated = MatchstickSkeleton.rotate_bound(img_opencv_white, p_degree)  # 旋转图像

        img_ry, img_rx = img_rotated.shape[0], img_rotated.shape[1]

        # p_ly, p_lx = abs(pb[1] - pa[1]), abs(pb[0] - pa[0])
        # start_point = min(pa[0], pb[0]), min(pa[1], pb[1])

        p_ly, p_lx = pb[1] - pa[1], pb[0] - pa[0]
        start_point = pa[0], pa[1]

        po_x = start_point[0] - abs(p_lx - img_rx) // 2  # 偏移量x
        po_y = start_point[1] - abs(p_ly - img_ry) // 2  # 偏移量y

        img_rotated_rgba = MatchstickSkeleton.rotate_bound(img_opencv, p_degree)  # 旋转图像

        return (po_x, po_y), img_rotated_rgba

    @staticmethod
    def overlay_transparent(background, overlay, x, y):
        """
        将PNG图像贴入图片
        :param background: 背景
        :param overlay: 需要贴入的图像
        :param x: 起始点x
        :param y: 起始点y
        :return: 贴入后的图像
        """
        background_width = background.shape[1]
        background_height = background.shape[0]

        if x >= background_width or y >= background_height:
            return background

        h, w = overlay.shape[0], overlay.shape[1]

        if x + w > background_width:
            w = background_width - x
            overlay = overlay[:, :w]

        if y + h > background_height:
            h = background_height - y
            overlay = overlay[:h]

        if overlay.shape[2] < 4:
            overlay = np.concatenate(
                [
                    overlay,
                    np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
                ],
                axis=2,
            )

        overlay_image = overlay[..., :3]
        mask = overlay[..., 3:] / 255.0  #

        # 图像中对应的像素设置为0, 补充像素设置为1
        background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image

        return background

    @staticmethod
    def paste_png_on_bkg(background, overlay, x, y, mode=0):
        # print('[Info] draw_png shape: {}'.format(draw_png.shape))
        # print('[Info] bkg_png shape: {}'.format(bkg_png.shape))
        h, w, _ = overlay.shape

        alpha_mask = np.where(overlay[:, :, 3] == 255, 1, 0)
        alpha_mask = np.repeat(alpha_mask[:, :, np.newaxis], 4, axis=2)  # 将mask复制4次

        # print(background[y:y + h, x:x + w, :].shape)
        # print(((1.0 - alpha_mask) * background[y:y + h, x:x + w]).shape)
        # print(overlay.shape)
        background[y:y + h, x:x + w, :] = (1.0 - alpha_mask) * background[y:y + h, x:x + w] + alpha_mask * overlay

        return background

    @staticmethod
    def generate_png_canvas(frame_shape):
        """
        生成4通道的图像
        :param frame_shape: 四通道frame
        :return: 白色透明背景
        """
        h, w, _ = frame_shape
        canvas = np.ones((h, w, 4)) * 255
        canvas_alpha = np.zeros(frame_shape[0:2])
        canvas[:, :, 3] = canvas_alpha
        canvas = canvas.astype(np.uint8)
        return canvas

    @staticmethod
    def draw_png_line(canvas, img_png, pt1, pt2, color, thickness, mode=0):
        """
        绘制PNG直线，倾斜矩形
        """
        # width参数是生成的宽度
        (po_x, po_y), img_rotated_rgba = MatchstickSkeleton.generate_png_line(img_png, pt1, pt2, width=50)
        # 已经绘制在Canvas上
        # canvas = MatchstickSkeleton.overlay_transparent(canvas, img_rotated_rgba, po_x, po_y)
        canvas = MatchstickSkeleton.paste_png_on_bkg(canvas, img_rotated_rgba, po_x, po_y, mode=mode)
        return canvas

    @staticmethod
    def draw_png_square(canvas, img_png, center, radius, color, thickness):
        """
        绘制PNG方块
        """
        (po_x, po_y), img_rgba = MatchstickSkeleton.generate_png_square(img_png, center, radius)
        # canvas = MatchstickSkeleton.overlay_transparent(canvas, img_rgba, po_x, po_y)
        canvas = MatchstickSkeleton.paste_png_on_bkg(canvas, img_rgba, po_x, po_y)
        return canvas

    @staticmethod
    def draw_png(image_shape, skt, canvas, png_list):
        """
        绘制PNG图像
        :param image_shape:
        :param skt:
        :param canvas:
        :param png_list: [头0, 身体1, 左腿上2, 左腿下3, 右腿上4, 右腿下5, 左臂上6, 左臂下7, 右臂上8, 右臂下9]
        :return:
        """
        black_color = (0, 0, 0)
        tn = 3
        # img_part_path = os.path.join(DATA_DIR, 'custom', 'corn.png')
        # img_part_png = cv2.imread(img_part_path, cv2.IMREAD_UNCHANGED)  # 读取PNG图像
        # img_head_path = os.path.join(DATA_DIR, 'custom', 'watermelon.png')
        # img_head_png = cv2.imread(img_head_path, cv2.IMREAD_UNCHANGED)  # 读取PNG图像

        # r, body_sp, body_ep = MatchstickSkeleton.get_other_parameters(skt[0], skt[1], skt[2], skt[3])
        r, head_p, body_ep = MatchstickSkeleton.get_other_parameters_v2(skt[0], skt[1], skt[2], skt[3])

        if skt[0][0] != 0 and r > 0:  # 头部
            # img_head_path = os.path.join(DATA_DIR, 'parts', '2-head.out.png')
            # img_head_png = cv2.imread(img_head_path, cv2.IMREAD_UNCHANGED)  # 读取PNG图像
            MatchstickSkeleton.draw_png_square(canvas, img_png=png_list[0], center=head_p, radius=r,
                                               color=black_color, thickness=tn)  # 绘制头部
            # cv2.circle(canvas, center=skt[0], radius=r, color=black_color, thickness=tn)  # 绘制头部
        else:
            print('[Warning] 头部缺失: 点 {}, 半径 {}'.format(skt[0], r))

        if skt[1][0] != 0 and skt[2][0] != 0 and skt[3][0] != 0:  # 绘制身躯，头到脖子，脖子到屁股
            # MatchstickSkeleton.draw_png_line(canvas, img_png=img_part_png, pt1=body_sp, pt2=skt[1],
            #                                  color=black_color, thickness=tn)  # 绘制身体
            # img_part_path = os.path.join(DATA_DIR, 'parts', '2-body.out.png')
            # img_part_png = cv2.imread(img_part_path, cv2.IMREAD_UNCHANGED)  # 读取PNG图像
            MatchstickSkeleton.draw_png_line(canvas, img_png=png_list[1], pt1=skt[1], pt2=body_ep,
                                             color=black_color, thickness=tn)  # 绘制身体
            # cv2.line(canvas, pt1=skt[1], pt2=body_ep, color=black_color, thickness=tn)
        else:
            print('[Warning] 身体缺失')

        if skt[1][0] != 0 and skt[4][0] != 0:  # 绘制右上臂
            # img_part_path = os.path.join(DATA_DIR, 'parts', '2-right-arm-up.out.png')
            # img_part_png = cv2.imread(img_part_path, cv2.IMREAD_UNCHANGED)  # 读取PNG图像
            # x1 = (skt[1][0] - 55, skt[1][1])
            # x2 = (skt[4][0], skt[4][1] + 55)
            MatchstickSkeleton.draw_png_line(canvas, img_png=png_list[8], pt1=skt[1], pt2=skt[4],
                                             color=black_color, thickness=tn)
            # cv2.line(canvas, pt1=skt[1], pt2=skt[4], color=black_color, thickness=tn)
        else:
            print('[Warning] 右上臂缺失')

        if skt[1][0] != 0 and skt[5][0] != 0:  # 绘制左上臂
            # img_part_path = os.path.join(DATA_DIR, 'parts', '2-left-arm-up.out.png')
            # img_part_png = cv2.imread(img_part_path, cv2.IMREAD_UNCHANGED)  # 读取PNG图像
            # x1 = (skt[1][0] + 55, skt[1][1])
            # x2 = (skt[5][0], skt[5][1] + 55)
            MatchstickSkeleton.draw_png_line(canvas, img_png=png_list[6], pt1=skt[1], pt2=skt[5],
                                             color=black_color, thickness=tn)
            # cv2.line(canvas, pt1=skt[1], pt2=skt[5], color=black_color, thickness=tn)
        else:
            print('[Warning] 左上臂缺失')

        if skt[4][0] != 0 and skt[6][0] != 0:  # 绘制右下臂
            # img_part_path = os.path.join(DATA_DIR, 'parts', '2-left-arm-down.out.png')
            # img_part_png = cv2.imread(img_part_path, cv2.IMREAD_UNCHANGED)  # 读取PNG图像
            MatchstickSkeleton.draw_png_line(canvas, img_png=png_list[9], pt1=skt[4], pt2=skt[6],
                                             color=black_color, thickness=tn)
            # cv2.line(canvas, pt1=skt[4], pt2=skt[6], color=black_color, thickness=tn)
        else:
            print('[Warning] 右下臂缺失')

        if skt[5][0] != 0 and skt[7][0] != 0:  # 绘制左下臂
            # img_part_path = os.path.join(DATA_DIR, 'parts', '2-right-arm-down.out.png')
            # img_part_png = cv2.imread(img_part_path, cv2.IMREAD_UNCHANGED)  # 读取PNG图像
            MatchstickSkeleton.draw_png_line(canvas, img_png=png_list[7], pt1=skt[5], pt2=skt[7],
                                             color=black_color, thickness=tn)
            # cv2.line(canvas, pt1=skt[5], pt2=skt[7], color=black_color, thickness=tn)
        else:
            print('[Warning] 左下臂缺失')

        if skt[2][0] != 0 and skt[3][0] != 0 and skt[8][0] != 0:  # 绘制右上腿
            # img_part_path = os.path.join(DATA_DIR, 'parts', '2-right-leg-up.out.png')
            # img_part_png = cv2.imread(img_part_path, cv2.IMREAD_UNCHANGED)  # 读取PNG图像
            MatchstickSkeleton.draw_png_line(canvas, img_png=png_list[4], pt1=body_ep, pt2=skt[8],
                                             color=black_color, thickness=tn)
            # cv2.line(canvas, pt1=body_ep, pt2=skt[8], color=black_color, thickness=tn)
        else:
            print('[Warning] 右上腿缺失')

        if skt[2][0] != 0 and skt[3][0] != 0 and skt[9][0] != 0:  # 绘制左上腿
            # img_part_path = os.path.join(DATA_DIR, 'parts', '2-left-leg-up.out.png')
            # img_part_png = cv2.imread(img_part_path, cv2.IMREAD_UNCHANGED)  # 读取PNG图像
            MatchstickSkeleton.draw_png_line(canvas, img_png=png_list[2], pt1=body_ep, pt2=skt[9],
                                             color=black_color, thickness=tn)
            # cv2.line(canvas, pt1=body_ep, pt2=skt[9], color=black_color, thickness=tn)
        else:
            print('[Warning] 左上腿缺失')

        if skt[8][0] != 0 and skt[10][0] != 0:  # 绘制右下腿
            # img_part_path = os.path.join(DATA_DIR, 'parts', '2-left-leg-down.out.png')
            # img_part_png = cv2.imread(img_part_path, cv2.IMREAD_UNCHANGED)  # 读取PNG图像
            MatchstickSkeleton.draw_png_line(canvas, img_png=png_list[5], pt1=skt[8], pt2=skt[10],
                                             color=black_color, thickness=tn)
            # cv2.line(canvas, pt1=skt[8], pt2=skt[10], color=black_color, thickness=tn)
        else:
            print('[Warning] 右下腿缺失')

        if skt[9][0] != 0 and skt[11][0] != 0:  # 绘制左下腿
            # img_part_path = os.path.join(DATA_DIR, 'parts', '2-right-leg-down.out.png')
            # img_part_png = cv2.imread(img_part_path, cv2.IMREAD_UNCHANGED)  # 读取PNG图像
            MatchstickSkeleton.draw_png_line(canvas, img_png=png_list[3], pt1=skt[9], pt2=skt[11],
                                             color=black_color, thickness=tn)
            # cv2.line(canvas, pt1=skt[9], pt2=skt[11], color=black_color, thickness=tn)
        else:
            print('[Warning] 左下腿缺失')

        canvas = canvas.astype(np.uint8)

        return canvas

    @staticmethod
    def draw_png_demo(image_shape, skt, canvas, png_list):
        """
        绘制PNG图像
        :param image_shape:
        :param skt:
        :param canvas:
        :param png_list: [头0, 身体1, 左腿上2, 左腿下3, 右腿上4, 右腿下5, 左臂上6, 左臂下7, 右臂上8, 右臂下9]
        :return:
        """
        black_color = (0, 0, 0)
        tn = 3
        img_part_path = os.path.join(DATA_DIR, 'custom', 'corn.png')
        img_part_png = cv2.imread(img_part_path, cv2.IMREAD_UNCHANGED)  # 读取PNG图像
        img_head_path = os.path.join(DATA_DIR, 'custom', 'watermelon.png')
        img_head_png = cv2.imread(img_head_path, cv2.IMREAD_UNCHANGED)  # 读取PNG图像

        r, body_sp, body_ep = MatchstickSkeleton.get_other_parameters(skt[0], skt[1], skt[2], skt[3])

        if skt[0][0] != 0 and r > 0:  # 头部
            MatchstickSkeleton.draw_png_square(canvas, img_png=img_head_png, center=skt[0], radius=r,
                                               color=black_color, thickness=tn)  # 绘制头部
            cv2.circle(canvas, skt[0], 10, color=(0, 0, 255, 255), thickness=-1)
        else:
            print('[Warning] 头部缺失: 点 {}, 半径 {}'.format(skt[0], r))

        if skt[1][0] != 0 and skt[2][0] != 0 and skt[3][0] != 0:  # 绘制身躯，头到脖子，脖子到屁股
            MatchstickSkeleton.draw_png_line(canvas, img_png=img_part_png, pt1=skt[1], pt2=body_ep,
                                             color=black_color, thickness=tn)  # 绘制身体
        else:
            print('[Warning] 身体缺失')

        if skt[1][0] != 0 and skt[4][0] != 0:  # 绘制右上臂
            MatchstickSkeleton.draw_png_line(canvas, img_png=img_part_png, pt1=skt[1], pt2=skt[4],
                                             color=black_color, thickness=tn)
        else:
            print('[Warning] 右上臂缺失')

        if skt[1][0] != 0 and skt[5][0] != 0:  # 绘制左上臂
            MatchstickSkeleton.draw_png_line(canvas, img_png=img_part_png, pt1=skt[1], pt2=skt[5],
                                             color=black_color, thickness=tn)
        else:
            print('[Warning] 左上臂缺失')

        if skt[4][0] != 0 and skt[6][0] != 0:  # 绘制右下臂
            MatchstickSkeleton.draw_png_line(canvas, img_png=img_part_png, pt1=skt[4], pt2=skt[6],
                                             color=black_color, thickness=tn)
            cv2.circle(canvas, skt[4], 10, color=(0, 128, 128, 255), thickness=-1)
            cv2.circle(canvas, skt[6], 10, color=(128, 0, 128, 255), thickness=-1)
        else:
            print('[Warning] 右下臂缺失')

        if skt[5][0] != 0 and skt[7][0] != 0:  # 绘制左下臂
            MatchstickSkeleton.draw_png_line(canvas, img_png=img_part_png, pt1=skt[5], pt2=skt[7],
                                             color=black_color, thickness=tn)
            cv2.circle(canvas, skt[5], 10, color=(0, 128, 128, 255), thickness=-1)
            cv2.circle(canvas, skt[7], 10, color=(128, 0, 128, 255), thickness=-1)
        else:
            print('[Warning] 左下臂缺失')

        if skt[2][0] != 0 and skt[3][0] != 0 and skt[8][0] != 0:  # 绘制右上腿
            MatchstickSkeleton.draw_png_line(canvas, img_png=img_part_png, pt1=body_ep, pt2=skt[8],
                                             color=black_color, thickness=tn)
        else:
            print('[Warning] 右上腿缺失')

        if skt[2][0] != 0 and skt[3][0] != 0 and skt[9][0] != 0:  # 绘制左上腿
            MatchstickSkeleton.draw_png_line(canvas, img_png=img_part_png, pt1=body_ep, pt2=skt[9],
                                             color=black_color, thickness=tn)
        else:
            print('[Warning] 左上腿缺失')

        if skt[8][0] != 0 and skt[10][0] != 0:  # 绘制右下腿
            MatchstickSkeleton.draw_png_line(canvas, img_png=img_part_png, pt1=skt[8], pt2=skt[10],
                                             color=black_color, thickness=tn)
        else:
            print('[Warning] 右下腿缺失')

        if skt[9][0] != 0 and skt[11][0] != 0:  # 绘制左下腿
            MatchstickSkeleton.draw_png_line(canvas, img_png=img_part_png, pt1=skt[9], pt2=skt[11],
                                             color=black_color, thickness=tn)
        else:
            print('[Warning] 左下腿缺失')

        canvas = canvas.astype(np.uint8)

        return canvas

    @staticmethod
    def draw(image_shape, skt, canvas):
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
