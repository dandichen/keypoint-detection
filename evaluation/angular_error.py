import math
import numpy as np
from numpy import linalg as la

__author__ = 'Dandi Chen'

def normalize(flow, max_val, min_val):
    return (flow - min_val)/(np.float_(max_val - min_val))

def normalize_flow(flow_X, flow_Y, flow_X_gt, flow_Y_gt):
    flow_X_nr = normalize(flow_X, np.max(flow_X), np.min(flow_X))
    flow_Y_nr = normalize(flow_Y, np.max(flow_Y), np.min(flow_Y))
    flow_X_gt_nr = normalize(flow_X_gt, np.max(flow_X_gt), np.min(flow_X_gt))
    flow_Y_gt_nr = normalize(flow_Y_gt, np.max(flow_Y_gt), np.min(flow_Y_gt))

    return flow_X_nr, flow_Y_nr, flow_X_gt_nr, flow_Y_gt_nr

def get_angular_err(flow_X, flow_Y, flow_X_gt, flow_Y_gt, flow_mask, flow_mask_gt):
    # [IJCV 2011] A Database and Evaluation Methodology for Optical Flow (section 4.1)
    valid_idx = np.logical_and(flow_mask, flow_mask_gt)

    dot_prod = 1.0 + np.dot(flow_X[valid_idx], flow_X_gt[valid_idx]) + np.dot(flow_Y[valid_idx], flow_Y_gt[valid_idx])
    norm_X = (1.0 + la.norm(flow_X[valid_idx]) ** 2 + la.norm(flow_X_gt[valid_idx]) ** 2) ** 0.5
    norm_Y = (1.0 + la.norm(flow_Y_gt[valid_idx]) ** 2 + la.norm(flow_Y_gt[valid_idx]) ** 2) ** 0.5

    if norm_X * norm_Y != 0:
        val = dot_prod/(norm_X * norm_Y)

        while (val > 1 or val < -1):
            print 'angle out of range'
            if val > 1:
                val -= 2
            else:
                val += 2

        ang_err_rad = math.acos(val)  # radius
        ang_err_deg = math.degrees(ang_err_rad)  # degree

    else:
        ang_err_rad = float('nan')
        ang_err_deg = float('nan')

    return ang_err_rad, ang_err_deg




