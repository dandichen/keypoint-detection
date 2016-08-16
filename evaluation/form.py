import cv2
import math
import numpy as np
from numpy import linalg as la
import scipy.spatial.distance as sci_dis


__author__ = "Dandi Chen"

def read_gt(gt_path):
    flow_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)

    # KITTI definition
    flow_X_gt = (np.float_(flow_gt[:, :, 2]) - 2 ** 15) / 64.0  # [-512..+512]
    flow_Y_gt = (np.float_(flow_gt[:, :, 1]) - 2 ** 15) / 64.0  # [-512..+512]
    flow_mask_gt = np.array(flow_gt[:, :, 0], dtype=bool)

    return flow_X_gt, flow_Y_gt, flow_mask_gt

def convert_gt_mask_box(flow_mask, top_left_x=0, top_right_y=0, bom_left_x=0, bom_right_y=0):
    new_flow_mask = np.zeros_like(flow_mask)
    new_flow_mask[int(top_right_y):int(bom_right_y), int(top_left_x):int(bom_left_x)] = \
        flow_mask[int(top_right_y):int(bom_right_y), int(top_left_x):int(bom_left_x)]
    return new_flow_mask

def normalize_len(old_arr, start, end):
    old_min = np.amin(old_arr)
    old_range = np.amax(old_arr) - old_min

    new_range = end - start
    new_arr = [(n - old_min) / old_range * new_range + start for n in old_arr]

    return new_arr

def normalize_mat(old_arr1, old_arr2):  # normalize to center
    height1, width1 = old_arr1.shape
    height2, width2 = old_arr2.shape

    min_height = np.min(height1, height2)
    min_width = np.min(width1, width2)

    new_arr1 = old_arr1[((height1 - min_height)/2):((height1 + min_height)/2),
               ((width1 - min_width)/2):((width1 + min_width)/2)]

    new_arr2 = old_arr2[((height2 - min_height) / 2):((height2 + min_height) / 2),
               ((width2 - min_width) / 2):((width2 + min_width) / 2)]

    return new_arr1, new_arr2, min_width, min_height

def convert(old_x, old_y, old_flow_X, old_flow_Y, flow_mask_gt):
    height, width = flow_mask_gt.shape
    flow_mask = np.zeros((height, width), dtype=bool)
    new_flow_X = np.zeros((height, width))
    new_flow_Y = np.zeros((height, width))

    for i in range(len(old_x)):
        flow_mask[old_y[i], old_x[i]] = True
        new_flow_X[old_y[i], old_x[i]] = old_flow_X[i]
        new_flow_Y[old_y[i], old_x[i]] = old_flow_Y[i]

    return new_flow_X, new_flow_Y, flow_mask

def normalize_coordinate_box(x, y, top_left_x, top_left_y, bom_right_x, bom_right_y):
    pos_x = (x - top_left_x) / (bom_right_x - top_left_x)
    pos_y = (y - top_left_y) / (bom_right_y - top_left_y)
    return pos_x, pos_y

def check_range(x, y, width, height):
    if x >= 0 and x < width and y >= 0 and y < height:
        return True
    else:
        return False

def get_vector_sim(flow_X, flow_Y, flow_X_gt, flow_Y_gt, flow_mask, flow_mask_gt, width, height, win_size=3, ang_thld=15, amp_thld=15):
    y = np.where(flow_mask == True)[0]
    x = np.where(flow_mask == True)[1]

    sim_mask = np.zeros_like(flow_mask_gt, dtype=bool)

    for j in y:
        for i in x:
            u = [flow_X[j, i], flow_Y[j, i]]
            ave_ang = []
            ave_amp = []
            for n in range((win_size - 1)/2, (win_size + 1)/2 + 2):
                for m in range((win_size - 1)/2, (win_size + 1)/2 + 2):
                    if check_range(i - 2 + m, j - 2 + n, width, height) and flow_mask_gt[j - 2 + n, i - 2 + m] == True:
                        v = [flow_X_gt[j - 2 + n, i - 2 + m], flow_Y_gt[j - 2 + n, i - 2 + m]]
                        if la.norm(u) * la.norm(v) != 0:
                            ang = math.acos(np.dot(u, v) / (la.norm(u) * la.norm(v)))
                            amp = sci_dis.euclidean(u, v)
                            ave_ang.append(ang)
                            ave_amp.append(amp)
                        else:
                            continue
                    else:
                        continue
            if len(ave_ang) != 0 and len(ave_amp) != 0 and np.mean(ave_ang) < ang_thld and np.mean(ave_amp) < amp_thld:
                sim_mask[j, i] = True
    return sim_mask

# overlap between flow and keypoint correspondence
def mask2vec_mask(flow_mask, matches, kp, matchesMask):
    mask_vector = np.zeros_like(matchesMask, dtype=bool)
    for match_idx in range(len(matches)):
        idx = matches[match_idx].queryIdx
        (x, y) = kp[idx].pt
        if flow_mask[int(y), int(x)] == True:
            mask_vector[match_idx] = matchesMask[match_idx]
    return mask_vector













