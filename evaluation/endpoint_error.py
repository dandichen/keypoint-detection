import numpy as np

__author__ = 'Dandi Chen'


def get_endpoint_err(flow_X, flow_Y, flow_X_gt, flow_Y_gt, flow_mask, flow_mask_gt):
    # [IJCV 2011] A Database and Evaluation Methodology for Optical Flow (section 4.1)
    valid_idx = np.logical_and(flow_mask, flow_mask_gt)

    delta_X = flow_X[valid_idx] - flow_X_gt[valid_idx]
    delta_Y = flow_Y[valid_idx] - flow_Y_gt[valid_idx]

    err_amp = (delta_X**2 + delta_Y**2)**0.5

    return np.mean(err_amp)