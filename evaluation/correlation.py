import numpy as np

__author__ = 'Dandi Chen'

def reshape(flow, width, height):
    flow_re = np.reshape(flow, width * height)
    return flow_re

def reshape_flow(flow_X, flow_Y, flow_X_gt, flow_Y_gt, width, height):
    flow_X_re = reshape(flow_X, width, height)
    flow_Y_re = reshape(flow_Y, width, height)

    flow_X_gt_re = reshape(flow_X_gt, width, height)
    flow_Y_gt_re = reshape(flow_Y_gt, width, height)

    return flow_X_re, flow_Y_re, flow_X_gt_re, flow_Y_gt_re

def get_correlation(flow_X, flow_Y, flow_X_gt, flow_Y_gt, flow_mask, flow_mask_gt):
    valid_idx = np.logical_and(flow_mask, flow_mask_gt)

    corr_X = np.corrcoef(flow_X[valid_idx], flow_X_gt[valid_idx])[1, 0]
    corr_Y = np.corrcoef(flow_Y[valid_idx], flow_Y_gt[valid_idx])[1, 0]
    return corr_X, corr_Y