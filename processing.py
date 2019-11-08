#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/10/18
"""

import math

import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import util


def extract_parts(input_image, params, model, model_params):
    multiplier = [x * model_params['boxsize'] / input_image.shape[0] for x in params['scale_search']]

    # Body parts location heatmap, one per part (19)
    heatmap_avg = np.zeros((input_image.shape[0], input_image.shape[1], 19))
    # Part affinities, one per limb (38)
    paf_avg = np.zeros((input_image.shape[0], input_image.shape[1], 38))

    for scale in multiplier:
        image_to_test = cv2.resize(input_image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        image_to_test_padded, pad = util.pad_right_down_corner(image_to_test, model_params['stride'],
                                                               model_params['padValue'])

        # required shape (1, width, height, channels)
        input_img = np.transpose(np.float32(image_to_test_padded[:, :, :, np.newaxis]), (3, 0, 1, 2))

        output_blobs = model.predict(input_img)

        # extract outputs, resize, and remove padding
        heatmap = np.squeeze(output_blobs[1])  # output 1 is heatmaps
        heatmap = cv2.resize(heatmap, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                             interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap[:image_to_test_padded.shape[0] - pad[2], :image_to_test_padded.shape[1] - pad[3], :]
        heatmap = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_CUBIC)

        paf = np.squeeze(output_blobs[0])  # output 0 is PAFs
        paf = cv2.resize(paf, (0, 0), fx=model_params['stride'], fy=model_params['stride'],
                         interpolation=cv2.INTER_CUBIC)
        paf = paf[:image_to_test_padded.shape[0] - pad[2], :image_to_test_padded.shape[1] - pad[3], :]
        paf = cv2.resize(paf, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_CUBIC)

        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)

    all_peaks = []
    peak_counter = 0

    for part in range(18):
        hmap_ori = heatmap_avg[:, :, part]
        hmap = gaussian_filter(hmap_ori, sigma=3)

        # Find the pixel that has maximum value compared to those around it
        hmap_left = np.zeros(hmap.shape)
        hmap_left[1:, :] = hmap[:-1, :]
        hmap_right = np.zeros(hmap.shape)
        hmap_right[:-1, :] = hmap[1:, :]
        hmap_up = np.zeros(hmap.shape)
        hmap_up[:, 1:] = hmap[:, :-1]
        hmap_down = np.zeros(hmap.shape)
        hmap_down[:, :-1] = hmap[:, 1:]

        # reduce needed because there are > 2 arguments
        peaks_binary = np.logical_and.reduce(
            (hmap >= hmap_left, hmap >= hmap_right, hmap >= hmap_up, hmap >= hmap_down, hmap > params['thre1']))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))  # note reverse
        peaks_with_score = [x + (hmap_ori[x[1], x[0]],) for x in peaks]  # add a third element to tuple with score
        idx = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (idx[i],) for i in range(len(idx))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(util.hmapIdx)):
        score_mid = paf_avg[:, :, [x - 19 for x in util.hmapIdx[k]]]
        cand_a = all_peaks[util.limbSeq[k][0] - 1]
        cand_b = all_peaks[util.limbSeq[k][1] - 1]
        n_a = len(cand_a)  # 候选的点a，连接ab
        n_b = len(cand_b)  # 候选的点b，连接ab
        # index_a, index_b = util.limbSeq[k]
        if n_a != 0 and n_b != 0:
            connection_candidate = []
            for i in range(n_a):
                for j in range(n_b):
                    vec = np.subtract(cand_b[j][:2], cand_a[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    # failure case when 2 body parts overlaps
                    if norm == 0:
                        continue
                    vec = np.divide(vec, norm)  # cos

                    startend = list(zip(np.linspace(cand_a[i][0], cand_b[j][0], num=mid_num),
                                        np.linspace(cand_a[i][1], cand_b[j][1], num=mid_num)))
                    # 方向的置信度
                    vec_x = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0]
                         for I in range(len(startend))])
                    vec_y = np.array(
                        [score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1]
                         for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])  # a*cos
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(
                        0.5 * input_image.shape[0] / norm - 1, 0)  # 求均值，增加1个惩罚，连线过长
                    criterion1 = len(np.nonzero(score_midpts > params['thre2'])[0]) > (0.8 * len(
                        score_midpts))  # thre2=0.05，大于0.05的点超过8个
                    criterion2 = score_with_dist_prior > 0  # 均值大于0
                    if criterion1 and criterion2:  # 筛选连接点
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + cand_a[i][2] + cand_b[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)  # 距离优先级排序，从大到小
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if i not in connection[:, 3] and j not in connection[:, 4]:
                    connection = np.vstack([connection, [cand_a[i][3], cand_b[j][3], s, i, j]])
                    if len(connection) >= min(n_a, n_b):
                        break

            connection_all.append(connection)  # 18种连接，每种有多个
        else:
            special_k.append(k)
            connection_all.append([])

    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = np.empty((0, 20))  # 18维是身体，最后一维是身体部分，倒数第二维是身体概率
    candidate = np.array([item for sublist in all_peaks for item in sublist])  # 转换成矩阵

    for k in range(len(util.hmapIdx)):
        if k not in special_k:
            part_as = connection_all[k][:, 0]  # 第1个点的索引
            part_bs = connection_all[k][:, 1]  # 第2个点的索引
            index_a, index_b = np.array(util.limbSeq[k]) - 1

            for i in range(len(connection_all[k])):  # = 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # 1:size(subset,1):
                    if subset[j][index_a] == part_as[i] or subset[j][index_b] == part_bs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if subset[j][index_b] != part_bs[i]:
                        subset[j][index_b] = part_bs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1
                        subset[j1][index_b] = part_bs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[part_bs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[index_a] = part_as[i]
                    row[index_b] = part_bs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])

    # delete some rows of subset which has few parts occur
    delete_idx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:  # 小于4个部分，或者平均值小于0.4
            delete_idx.append(i)
    subset = np.delete(subset, delete_idx, axis=0)
    # all_peaks，全部点的序列，subset，每一个人的集合
    return all_peaks, subset, candidate


def draw_skeleton(input_image, subset, candidate):
    """
    通过已有的
    :param input_image:
    :param subset:
    :param candidate:
    :return:
    """
    canvas = np.ones(input_image.shape) * 255

    cx, cy, cr, nx, ny = 0, 0, 0, 0, 0  # 头部
    px, py, rhip_x, rhip_y, lhip_x, lhip_y, p_ax, p_ay = 0, 0, 0, 0, 0, 0, 0, 0  # 身体
    relb_x, relb_y, lelb_x, lelb_y, rwri_x, rwri_y, lwri_x, lwri_y = 0, 0, 0, 0, 0, 0, 0, 0  # 上身
    rkne_x, rkne_y, lkne_x, lkne_y, rank_x, rank_y, lank_x, lank_y = 0, 0, 0, 0, 0, 0, 0, 0  # 下身

    black_color = (0, 0, 0)  # 黑色
    # red_color = (0, 0, 255)  # 红色，用于排查Bug

    idx_nose, idx_neck = -1, -1
    idx_rhip, idx_lhip = -1, -1
    idx_relb, idx_lelb, idx_rwri, idx_lwri = -1, -1, -1, -1
    idx_rkne, idx_lkne, idx_rank, idx_lank = -1, -1, -1, -1

    for s in subset:
        # print('[Info] set: {}'.format(s))
        if s[19] < 15:
            continue

        # 绘制头部
        idx_nose = int(s[0])
        cx, cy = int(candidate[idx_nose, 0]), int(candidate[idx_nose, 1])

        idx_neck = int(s[1])
        nx, ny = int(candidate[idx_neck, 0]), int(candidate[idx_neck, 1])  # 脖子

        tmp_a = math.sqrt(math.pow((cx - nx), 2) + math.pow((cy - ny), 2))
        cr = int(tmp_a // 3 * 2)

        # 绘制身子
        pa, pb = int((cr / tmp_a) * (nx - cx)), int((cr / tmp_a) * (ny - cy))
        px, py = int(cx + pa), int(cy + pb)  # 头和身的连接点

        idx_rhip = int(s[8])
        rhip_x, rhip_y = int(candidate[idx_rhip, 0]), int(candidate[idx_rhip, 1])

        idx_lhip = int(s[11])
        lhip_x, lhip_y = int(candidate[idx_lhip, 0]), int(candidate[idx_lhip, 1])

        p_ax = int(min(rhip_x, lhip_x) + abs(rhip_x - lhip_x) / 2)
        p_ay = int(min(rhip_y, lhip_y) + abs(rhip_y - lhip_y) / 2)

        # 绘制上身
        idx_relb = int(s[3])  # 右肘部
        relb_x, relb_y = int(candidate[idx_relb, 0]), int(candidate[idx_relb, 1])

        idx_lelb = int(s[6])  # 左肘部
        lelb_x, lelb_y = int(candidate[idx_lelb, 0]), int(candidate[idx_lelb, 1])

        idx_rwri = int(s[4])  # 右腕
        rwri_x, rwri_y = int(candidate[idx_rwri, 0]), int(candidate[idx_rwri, 1])

        idx_lwri = int(s[7])  # 左腕
        lwri_x, lwri_y = int(candidate[idx_lwri, 0]), int(candidate[idx_lwri, 1])

        # 绘制下身
        idx_rkne = int(s[9])  # 右膝盖
        rkne_x, rkne_y = int(candidate[idx_rkne, 0]), int(candidate[idx_rkne, 1])

        idx_lkne = int(s[12])  # 左膝盖
        lkne_x, lkne_y = int(candidate[idx_lkne, 0]), int(candidate[idx_lkne, 1])

        idx_rank = int(s[10])  # 右脚
        rank_x, rank_y = int(candidate[idx_rank, 0]), int(candidate[idx_rank, 1])

        idx_lank = int(s[13])  # 左脚
        lank_x, lank_y = int(candidate[idx_lank, 0]), int(candidate[idx_lank, 1])

    if idx_nose != -1 and idx_neck != -1:
        cv2.circle(canvas, center=(cx, cy), radius=cr, color=black_color, thickness=4)  # 绘制头部

    if idx_neck != -1 and idx_rhip != -1 and idx_lhip != -1:
        cv2.line(canvas, pt1=(px, py), pt2=(nx, ny), color=black_color, thickness=4)  # 绘制身体
        cv2.line(canvas, pt1=(nx, ny), pt2=(p_ax, p_ay), color=black_color, thickness=4)  # 绘制身体

    if idx_neck != -1 and idx_relb != -1:
        cv2.line(canvas, pt1=(nx, ny), pt2=(relb_x, relb_y), color=black_color, thickness=4)  # 绘制右臂
    if idx_neck != -1 and idx_lelb != -1:
        cv2.line(canvas, pt1=(nx, ny), pt2=(lelb_x, lelb_y), color=black_color, thickness=4)  # 绘制左臂

    if idx_relb != -1 and idx_rwri != -1:
        cv2.line(canvas, pt1=(relb_x, relb_y), pt2=(rwri_x, rwri_y), color=black_color, thickness=4)  # 绘制右臂
    if idx_lelb != -1 and idx_lwri != -1:
        cv2.line(canvas, pt1=(lelb_x, lelb_y), pt2=(lwri_x, lwri_y), color=black_color, thickness=4)  # 绘制左臂

    if idx_rkne != -1:
        cv2.line(canvas, pt1=(p_ax, p_ay), pt2=(rkne_x, rkne_y), color=black_color, thickness=4)  # 绘制右腿
    if idx_lkne != -1:
        cv2.line(canvas, pt1=(p_ax, p_ay), pt2=(lkne_x, lkne_y), color=black_color, thickness=4)  # 绘制左腿

    if idx_rkne != -1 and idx_rank != -1:
        cv2.line(canvas, pt1=(rkne_x, rkne_y), pt2=(rank_x, rank_y), color=black_color, thickness=4)  # 绘制右下腿
    if idx_lkne != -1 and idx_lank != -1:
        cv2.line(canvas, pt1=(lkne_x, lkne_y), pt2=(lank_x, lank_y), color=black_color, thickness=4)  # 绘制左腿

    # canvas = np.where(canvas == 0, 255, canvas)
    # c_max, c_min = np.max(canvas), np.min(canvas)
    # print('[Info] 绘制图像max: {}, min: {}'.format(c_max, c_min))

    canvas = canvas.astype(np.uint8)

    return canvas


def draw(input_image, all_peaks, subset, candidate, resize_fac=1):
    canvas = input_image.copy()

    for i in range(18):
        for j in range(len(all_peaks[i])):
            a = all_peaks[i][j][0] * resize_fac
            b = all_peaks[i][j][1] * resize_fac
            cv2.circle(canvas, (a, b), 2, util.colors[i], thickness=-1)

    stick_width = 4

    for s in subset:
        if s[18] < 25:
            continue
        for i in range(17):
            index = s[np.array(util.limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            y = candidate[index.astype(int), 0]
            x = candidate[index.astype(int), 1]
            m_x = np.mean(x)
            m_y = np.mean(y)
            length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(x[0] - x[1], y[0] - y[1]))
            polygon = cv2.ellipse2Poly((int(m_y * resize_fac), int(m_x * resize_fac)),
                                       (int(length * resize_fac / 2), stick_width), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, util.colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas
