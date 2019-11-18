#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/11/15
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from root_dir import DATA_DIR
from utils.project_utils import *


class PartsSegmentation(object):
    """
    身体部分的分割类
    """

    def __init__(self, img_png_path, img_config_path, img_draw_path):
        """
        待分割的图像，待分割的参数
        :param img_draw_path: 用户绘制的图像
        :param img_png_path: PNF路径
        :param img_config_path: PNG配置
        """
        self.img_png = self.read_png(img_png_path)
        self.img_config = self.read_config(img_config_path)
        self.img_draw = self.read_png(img_draw_path)

    def process(self):
        """
        根据身体配置，提取PNG各个部分
        :return: PNG部分
        """
        # 获取头部的PNG
        head_center, head_radius = self.img_config['head_point'], self.img_config['radius']
        head_crop, img_other = self.divide_circle_png(self.img_png, head_center, head_radius, self.img_draw)
        # self.show_png(head_crop)  # 测试
        # self.show_png(img_other)  # 测试

        # 获取身体的PNG
        body_start, body_end = self.img_config['body_start'], self.img_config['body_end']
        body_thick = self.img_config['body_thick']
        body_crop, img_other = self.divide_body_png(img_other, body_start, body_end, body_thick, self.img_draw)
        # self.show_png(body_crop)  # 测试
        # self.show_png(img_other)  # 测试

        # 获取双腿、左臂、右臂的PNG
        (legs_crop, legs_crop_draw), (l_arm_crop, l_arm_crop_draw), (r_arm_crop, r_arm_crop_draw) \
            = self.divide_arm_and_leg_png(img_other, self.img_draw)
        # self.show_png(legs_crop)  # 测试
        # self.show_png(legs_bc)  # 测试
        # self.show_png(l_arm_crop)  # 测试
        # self.show_png(l_arm_bc)  # 测试

        # 切分左腿和右腿
        (l_part, l_part_draw), (r_part, r_part_draw) = self.split_left_right_part(legs_crop, legs_crop_draw)
        # self.show_png(l_part)  # 测试
        # self.show_png(l_part_draw)  # 测试
        # self.show_png(r_part)  # 测试
        # self.show_png(r_part_draw)  # 测试

        # 旋转左腿和右腿
        l_leg_angle, r_leg_angle = self.img_config['left_leg_angle'], self.img_config['right_leg_angle']

        l_leg_crop = self.rotate_bound(l_part, l_leg_angle)  # 旋转角度
        l_leg_crop_draw = self.rotate_bound(l_part_draw, l_leg_angle)  # 旋转角度
        # self.show_png(l_leg_crop)  # 测试
        # self.show_png(l_leg_crop_draw)  # 测试
        l_leg_crop, l_leg_crop_draw = self.remove_png_draw_bkg(l_leg_crop, l_leg_crop_draw)
        # self.show_png(l_leg_crop)  # 测试
        # self.show_png(l_leg_crop_draw)  # 测试

        r_leg_crop = self.rotate_bound(r_part, r_leg_angle)
        r_leg_crop_draw = self.rotate_bound(r_part_draw, r_leg_angle)
        r_leg_crop, r_leg_crop_draw = self.remove_png_draw_bkg(r_leg_crop, r_leg_crop_draw)

        # self.show_png(r_leg_crop)  # 测试
        # self.show_png(r_leg_crop_draw)  # 测试

        # 旋转左臂和右臂
        left_arm_angle, right_arm_angle = self.img_config['left_arm_angle'], self.img_config['right_arm_angle']

        l_arm_crop = self.rotate_bound(l_arm_crop, left_arm_angle)
        l_arm_crop_draw = self.rotate_bound(l_arm_crop_draw, left_arm_angle)
        l_arm_crop, l_arm_crop_draw = self.remove_png_draw_bkg(l_arm_crop, l_arm_crop_draw)

        r_arm_crop = self.rotate_bound(r_arm_crop, right_arm_angle)
        r_arm_crop_draw = self.rotate_bound(r_arm_crop_draw, right_arm_angle)
        r_arm_crop, r_arm_crop_draw = self.remove_png_draw_bkg(r_arm_crop, r_arm_crop_draw)

        self.show_png(l_arm_crop_draw)  # 测试
        self.show_png(r_arm_crop_draw)  # 测试

        # 切分4个肢体
        (_, left_leg_up), (_, left_leg_down) = self.split_up_down_part(l_leg_crop, l_leg_crop_draw)
        (_, right_leg_up), (_, right_leg_down) = self.split_up_down_part(r_leg_crop, r_leg_crop_draw)
        (_, left_arm_up), (_, left_arm_down) = self.split_up_down_part(l_arm_crop, l_arm_crop_draw)
        (_, right_arm_up), (_, right_arm_down) = self.split_up_down_part(r_arm_crop, r_arm_crop_draw)
        # self.show_png(left_leg_up)  # 测试
        # self.show_png(left_leg_down)  # 测试
        # self.show_png(right_leg_up)  # 测试
        # self.show_png(right_leg_down)  # 测试
        # self.show_png(left_arm_up)  # 测试
        # self.show_png(left_arm_down)  # 测试
        # self.show_png(right_arm_up)  # 测试
        # self.show_png(right_arm_down)  # 测试

        # 拆解部分
        # [头, 身体, 左腿上, 左腿下, 右腿上, 右腿下, 左臂上, 左臂下, 右臂上, 右臂下]
        png_parts = [head_crop, body_crop, left_leg_up, left_leg_down, right_leg_up, right_leg_down,
                     left_arm_up, left_arm_down, right_arm_up, right_arm_down]

        return png_parts

    @staticmethod
    def read_png(img_png_path):
        """
        读取PNG图像
        :param img_png_path: 图像PNG路径
        :return: 图像PNG
        """
        img_png = cv2.imread(img_png_path, cv2.IMREAD_UNCHANGED)
        return img_png

    @staticmethod
    def read_config(img_config_path):
        """
        读取Json配置，包含图像的点信息
        :param img_config_path: 图像配置路径
        :return: 图像配置Dict
        """
        with open(img_config_path) as json_file:
            data = json.load(json_file)
        # print('[Info] head_point: {}'.format(data['head_point'])) # 测试
        return data

    @staticmethod
    def show_png(img_png):
        """
        展示PNG图像
        :param img_png: PNG图像
        :return: None
        """
        img_show = cv2.cvtColor(img_png, cv2.COLOR_BGRA2RGBA)
        plt.imshow(img_show)
        plt.show()

    @staticmethod
    def save_png(img_png):
        """
        存储图像，文件名以时间为主
        :param img_png: 图像
        :return: None
        """
        out_img_path = os.path.join(DATA_DIR, 'custom', 'img_png.{}.png'.format(get_current_time_str()))
        cv2.imwrite(out_img_path, img_png)

    @staticmethod
    def get_min_max_points(x_list, y_list):
        """
        获取像素点的边界，where的输出，左上角和右下角
        :param x_list: x序列
        :param y_list: y序列
        :return: 左上角和右下角
        """
        x_min = np.min(x_list)
        x_max = np.max(x_list)
        y_min = np.min(y_list)
        y_max = np.max(y_list)

        return (x_min, y_min), (x_max, y_max)

    @staticmethod
    def remove_list(a_list, b_list):
        """
        a_list中删除b_list
        :param a_list: a-b
        :param b_list: a-b
        :return: 删除之后的list
        """
        return [x for x in a_list if x not in b_list]

    @staticmethod
    def get_part_crop(img_png, part_mask, img_draw):
        """
        获取身体部分的切割
        :param img_png: 图像PNG
        :param part_mask: 分割的Mask
        :param img_draw: 用户绘制的部分
        :return: 分割和其他部分
        """
        img_part = img_draw.copy()
        # img_part[:, :, 3] = part_mask * 255

        yl, xl = np.where(part_mask == 1)
        (x_min, y_min), (x_max, y_max) = PartsSegmentation.get_min_max_points(xl, yl)

        # img_part[np.where(part_mask[:] == 0)] = (255, 255, 255, 0)  # 去除其他部分，避免干扰
        img_crop = img_part[y_min:y_max, x_min:x_max]  # 腿部分割

        img_other = img_png.copy()
        img_other[np.where(part_mask[:] == 1)] = (255, 255, 255, 0)  # 去除腿

        return img_crop, img_other

    @staticmethod
    def get_part_bkg_crop(img_png, part_mask, img_draw):
        """
        获取身体部分的切割，返回轮廓和用户绘制图
        :param img_png: 图像PNG
        :param part_mask: 分割的Mask
        :param img_draw: 用户绘制的部分
        :return: 分割和其他部分
        """
        img_part = img_draw.copy()
        # img_part[:, :, 3] = part_mask * 255

        yl, xl = np.where(part_mask == 1)
        (x_min, y_min), (x_max, y_max) = PartsSegmentation.get_min_max_points(xl, yl)

        # img_part[np.where(part_mask[:] == 0)] = (255, 255, 255, 0)  # 去除其他部分，避免干扰
        img_crop = img_part[y_min:y_max, x_min:x_max]  # 腿部分割

        img_bkg = img_png.copy()
        img_bkg[np.where(part_mask == 0)] = (255, 255, 255, 0)  # 去除腿
        img_bkg_crop = img_bkg[y_min:y_max, x_min:x_max]  # 腿部分割

        img_other = img_png.copy()
        img_other[np.where(part_mask == 1)] = (255, 255, 255, 0)  # 去除腿

        # PartsSegmentation.show_png(img_bkg_crop)
        # PartsSegmentation.show_png(img_png)
        # PartsSegmentation.show_png(img_draw)
        # PartsSegmentation.show_png(img_crop)
        # PartsSegmentation.show_png(img_bkg_crop)
        # PartsSegmentation.show_png(img_other)

        return (img_bkg_crop, img_crop), img_other

    @staticmethod
    def split_left_right_part(img_png, img_draw):
        """
        将图像拆分成左右两个部分
        :param img_png: PNG图像
        :param img_draw: 绘制图像
        :return: 左部分，右部分
        """
        h, w, _ = img_png.shape
        half = w // 2

        l_part_png, l_part_draw = img_png[:, 0:half], img_draw[:, 0:half]
        r_part_png, r_part_draw = img_png[:, half:w], img_draw[:, half:w]

        l_part, l_part_draw = PartsSegmentation.remove_png_draw_bkg(l_part_png, l_part_draw)  # 左半边部分
        r_part, r_part_draw = PartsSegmentation.remove_png_draw_bkg(r_part_png, r_part_draw)  # 右半边部分

        return (l_part, l_part_draw), (r_part, r_part_draw)

    @staticmethod
    def split_up_down_part(img_png, img_draw):
        """
        将图像拆分成上下两个部分
        :param img_png: PNG图像
        :param img_draw: 绘制图像
        :return: 上部分，下部分
        """
        h, w, _ = img_png.shape
        half = h // 2
        u_part_png, u_part_draw = img_png[0:half, :], img_draw[0:half, :]
        d_part_png, d_part_draw = img_png[half:h, :], img_draw[half:h, :]

        u_part_png, u_part_draw = PartsSegmentation.remove_png_draw_bkg(u_part_png, u_part_draw)  # 左半边部分
        d_part_png, d_part_draw = PartsSegmentation.remove_png_draw_bkg(d_part_png, d_part_draw)  # 右半边部分

        return (u_part_png, u_part_draw), (d_part_png, d_part_draw)

    @staticmethod
    def remove_png_bkg(img_png):
        """
        去掉PNG的透明边缘，规整图像
        :param img_png: 透明图像
        :return: 当前结果
        """
        h, w, _ = img_png.shape
        img_alpha = img_png[:, :, 3] // 255
        x_list, y_list = np.where(img_alpha == 1)
        (x_min, y_min), (x_max, y_max) = PartsSegmentation.get_min_max_points(x_list, y_list)
        img_crop = img_png[x_min:x_max, y_min:y_max]
        # margin_point = (x_min, y_min), (x_max, y_max)

        return img_crop

    @staticmethod
    def remove_png_draw_bkg(img_png, img_draw):
        """
        去掉PNG的透明边缘，规整图像
        :param img_png: 透明图像
        :param img_draw: 绘制图像
        :return: 当前结果
        """
        h, w, _ = img_png.shape
        img_alpha = img_png[:, :, 3] // 255
        x_list, y_list = np.where(img_alpha == 1)
        (x_min, y_min), (x_max, y_max) = PartsSegmentation.get_min_max_points(x_list, y_list)
        img_crop = img_png[x_min:x_max, y_min:y_max]
        img_draw_crop = img_draw[x_min:x_max, y_min:y_max]

        return img_crop, img_draw_crop

    @staticmethod
    def rotate_bound(img, angle):
        """
        旋转图像
        :param img: 图像
        :param angle: 角度
        :return: 旋转的图像
        """
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = img.shape[:2]
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
        return cv2.warpAffine(img, M, (nW, nH))

    @staticmethod
    def divide_arm_and_leg_png(img_png, img_draw):
        """
        分解手臂和腿的PNG
        :param img_png: 三个部分的PNG
        :param img_draw: 用户绘制的PNG
        :return: 两个腿部的PNG, 左臂的PNG, 右臂的PNG
        """

        image_alpha = img_png[:, :, 3]

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image_alpha, connectivity=4)
        sizes = stats[:, -1]

        part_indexes = np.argsort(sizes)[::-1]  # 排序
        part_indexes = part_indexes[1:4]  # 只选择前3个部分

        part_masks = []  # 身体部位的mask
        points_list = []  # 身体部分的边界
        y_max_list, x_min_list = [], []  # 用于选择点

        for i_part in part_indexes:
            img_tmp = np.zeros(output.shape)
            img_tmp[output == i_part] = 1
            part_masks.append(img_tmp)

            yl, xl = np.where(output == i_part)
            (x_min, y_min), (x_max, y_max) = PartsSegmentation.get_min_max_points(xl, yl)
            points_list.append([(x_min, y_min), (x_max, y_max)])
            y_max_list.append(y_max)
            x_min_list.append(x_min)

        # 最下面的是腿
        y_max_vals = np.array(y_max_list)
        legs_idx = int(np.argsort(y_max_vals)[::-1][0])  # 最下面的是腿
        legs_mask = part_masks[legs_idx]  # 腿部的Mask
        (legs_bc, legs_crop), _ = PartsSegmentation.get_part_bkg_crop(img_png, legs_mask, img_draw)
        # print('[Info] 腿部的索引: {}'.format(legs_idx))

        # 最左面除了腿的是左臂
        x_min_vals = np.array(x_min_list)
        x_min_sorted = np.argsort(x_min_vals)
        if x_min_sorted[0] != legs_idx:
            left_arm_idx = int(x_min_sorted[0])
        else:
            left_arm_idx = int(x_min_sorted[1])
        left_arm_mask = part_masks[left_arm_idx]
        (left_arm_bc, left_arm_crop), _ = PartsSegmentation.get_part_bkg_crop(img_png, left_arm_mask, img_draw)
        # print('[Info] 左臂的索引: {}'.format(left_arm_idx))

        # 最后的是右臂
        right_arm_idx = PartsSegmentation.remove_list([0, 1, 2], [left_arm_idx, legs_idx])[0]
        right_arm_mask = part_masks[right_arm_idx]
        (right_arm_bc, right_arm_crop), _ = PartsSegmentation.get_part_bkg_crop(img_png, right_arm_mask, img_draw)
        # print('[Info] 右臂的索引: {}'.format(right_arm_idx))

        # 返回源剪切图像，和背景剪切图像
        return (legs_bc, legs_crop), (left_arm_bc, left_arm_crop), (right_arm_bc, right_arm_crop)

    @staticmethod
    def divide_circle_png(img_png, center_point, radius, img_draw):
        """
        去除圆形部分
        :param img_png: 输入图像, 4个通道
        :param center_point: 中心点
        :param radius: 半径
        :param img_draw: 用户绘制
        :return: 剪切图像，剩余图像
        """
        center_point = tuple(center_point)
        h, w, _ = img_png.shape

        # 计算头部
        img_copy = img_draw.copy()  # 用户绘制的透明图
        img_mask = np.zeros((h, w), np.uint8)
        cv2.circle(img_mask, center=center_point, radius=radius, color=1, thickness=-1)  # 透明通道

        x, y = np.where(img_mask == 1)
        p_min = np.min(x), np.min(y)
        p_max = np.max(x), np.max(y)
        img_crop = img_copy[p_min[0]:p_max[0], p_min[1]:p_max[1]]

        # 计算其他
        img_other = img_png.copy()  # 模板的透明图
        # PartsSegmentation.show_png(img_other)  # 测试
        # img_no_mask = np.ones((h, w), np.uint8) - img_mask
        # img_other[:, :, 3] = img_no_mask * 255
        # PartsSegmentation.show_png(img_other)  # 测试
        img_other[np.where(img_mask[:, :] == 1)] = (255, 255, 255, 0)  # 去掉头部
        # PartsSegmentation.show_png(img_other)  # 测试

        return img_crop, img_other

    @staticmethod
    def divide_body_png(img_png, sp, ep, thick, img_draw):
        """
        去掉身体部分
        :param img_png: 输入图像, 4个通道
        :param sp: 起始点
        :param ep: 终止点
        :param thick: 宽度
        :param img_draw: 用户绘制的透明图像
        :return: 剪切图像，剩余图像
        """
        h, w, _ = img_png.shape

        img_copy = img_draw.copy()
        img_crop = img_copy[sp[1]:ep[1], sp[0] - thick:sp[0] + thick]  # 剪切图像

        img_other = img_png.copy()
        sp[1] = sp[1] - int((ep[1] - sp[1]) * 0.1)  # 去除额外的部分
        img_other[sp[1]:ep[1], sp[0] - thick:sp[0] + thick] = (255, 255, 255, 0)

        return img_crop, img_other


def paste_png_on_bkg(draw_png, bkg_png, offset):
    print('[Info] draw_png shape: {}'.format(draw_png.shape))
    print('[Info] bkg_png shape: {}'.format(bkg_png.shape))
    h, w, _ = draw_png.shape
    x, y = offset

    alpha_mask = np.where(draw_png[:, :, 3] == 255, 1, 0)
    alpha_mask = np.repeat(alpha_mask[:, :, np.newaxis], 4, axis=2)  # 将mask复制4次

    print(bkg_png[y:y + h, x:x + w, :].shape)
    print(((1.0 - alpha_mask) * bkg_png[y:y + h, x:x + w]).shape)
    print(draw_png.shape)
    # cv2.bitwise_and()
    bkg_png[y:y + h, x:x + w, :] = (1.0 - alpha_mask) * bkg_png[y:y + h, x:x + w] + alpha_mask * draw_png
    return bkg_png


def png_part_paste_test():
    """
    测试透明图像粘贴逻辑
    """
    frame_shape = (1024, 576, 4)
    canvas = np.ones(frame_shape) * 255
    canvas_alpha = np.zeros(frame_shape[0:2])
    canvas[:, :, 3] = canvas_alpha
    canvas = canvas.astype(np.uint8)

    img_png_path = os.path.join(DATA_DIR, 'custom', 'trans.png')
    img_draw_path = os.path.join(DATA_DIR, 'custom', 'x.png')
    img_config_path = os.path.join(DATA_DIR, 'configs', 'parts_config.json')
    ps = PartsSegmentation(img_png_path, img_config_path, img_draw_path)
    png_parts = ps.process()

    head_png = png_parts[3]
    print('[Info] head shape: {}'.format(head_png.shape))

    x = paste_png_on_bkg(draw_png=head_png, bkg_png=canvas, offset=(10, 10))

    plt.imshow(canvas)
    plt.show()

    plt.imshow(head_png)
    plt.show()

    img_show = cv2.cvtColor(x, cv2.COLOR_BGRA2RGBA)
    plt.imshow(img_show)
    plt.show()

    out_img_path = os.path.join(DATA_DIR, 'custom', 'xxx-head.png')
    cv2.imwrite(out_img_path, head_png)


def parts_segmentation_test():
    """
    测试PartsSegmentation类
    """
    img_png_path = os.path.join(DATA_DIR, 'custom', 'trans.png')
    img_draw_path = os.path.join(DATA_DIR, 'custom', 'x.png')
    img_config_path = os.path.join(DATA_DIR, 'configs', 'parts_config.json')
    ps = PartsSegmentation(img_png_path, img_config_path, img_draw_path)
    png_parts = ps.process()


def main():
    # parts_segmentation_test()
    png_part_paste_test()


if __name__ == '__main__':
    main()
