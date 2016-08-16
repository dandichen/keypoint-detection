import cv2
import numpy as np

import scipy.spatial.distance as sci_dis

import evaluation.form as form

__author__ = "Dandi Chen"

def read_img_pair(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    return img1, img2

def get_euclidean_dis(kp1, kp2, idx1, idx2, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1,
                 top_left_x2, top_left_y2, bom_right_x2, bom_right_y2):
    (x1, y1) = kp1[idx1].pt
    (x2, y2) = kp2[idx2].pt
    pos_x1, pos_y1 = form.normalize_coordinate_box(x1, y1, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1)
    pos_x2, pos_y2 = form.normalize_coordinate_box(x2, y2, top_left_x2, top_left_y2, bom_right_x2, bom_right_y2)

    pt1 = np.array([pos_x1, pos_y1])
    pt2 = np.array([pos_x2, pos_y2])
    dis = sci_dis.euclidean(pt1, pt2)
    return dis

def get_euclidean_vec_dis(kp1, kp2, matches, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1,
                          top_left_x2, top_left_y2, bom_right_x2, bom_right_y2):
    dis_vec = []
    for mat_idx in range(len(matches)):
        img1_idx = matches[mat_idx].queryIdx
        img2_idx = matches[mat_idx].trainIdx

        dis = get_euclidean_dis(kp1, kp2, img1_idx, img2_idx, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1,
                                top_left_x2, top_left_y2, bom_right_x2, bom_right_y2)
        dis_vec.append(dis)
    dis_vec_norm = form.normalize_len(dis_vec, 0, 1)
    return dis_vec_norm

# within bounding box
def get_weight_dis(kp1, kp2, good, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1,
                   top_left_x2, top_left_y2, bom_right_x2, bom_right_y2, weight=0.5):
    wgt_dis = []
    for mat_idx in range(len(good)):
        dis_pos = get_euclidean_vec_dis(kp1, kp2, good, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1,
                                        top_left_x2, top_left_y2, bom_right_x2, bom_right_y2)
        wgt_dis.append((1 - weight) * dis_pos[mat_idx] + weight * good[mat_idx].distance)
    return wgt_dis

# whether keypoint pair are neighbors
def get_neighbor(kp1, kp2, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1,
                 top_left_x2, top_left_y2, bom_right_x2, bom_right_y2, threshould=0.9):
    neighbor_mat = np.zeros((len(kp1), len(kp2)), dtype=bool)
    for idx2 in range(len(kp2)):
        for idx1 in range(len(kp1)):
            dis = get_euclidean_dis(kp1, kp2, idx1, idx2, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1,
                                    top_left_x2, top_left_y2, bom_right_x2, bom_right_y2)

            if dis <= threshould:
                neighbor_mat[idx1][idx2] = True
    return neighbor_mat

class Keypoint(object):
    def __init__(self, kp_num=200):
        self.kp_num = kp_num
        self.x = np.arange(self.kp_num)
        self.y = np.arange(self.kp_num)
        self.flow_X = np.arange(self.kp_num)
        self.flow_Y = np.arange(self.kp_num)