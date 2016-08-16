import numpy as np

__author__ = "Dandi Chen"

def get_flow(kp1, kp2, width, height, matches, match_path, top_left_x1=0, top_left_y1=0, top_left_x2=0, top_left_y2=0):
    flow_x = np.zeros((height, width))
    flow_y = np.zeros((height, width))
    flow_mask = np.zeros((height, width), dtype=bool)

    for mat_idx in range(len(matches)):
        img1_idx = matches[mat_idx].queryIdx
        img2_idx = matches[mat_idx].trainIdx

        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        delta_x = (x2 + top_left_x2) - (x1 + top_left_x1)
        delta_y = (y2 + top_left_y2) - (y1 + top_left_y1)

        # flow has been defined in first frame
        flow_x[int(y1 + top_left_y1)][int(x1 + top_left_x1)] = delta_x
        flow_y[int(y1 + top_left_y1)][int(x1 + top_left_x1)] = delta_y
        flow_mask[int(y1 + top_left_y1)][int(x1 + top_left_x1)] = True

    # np.savez(match_path, x=int(x1), y=int(y1), flow_x=flow_x, flow_y=flow_y)

    return int(x1), int(y1), flow_x, flow_y, flow_mask

class Matcher(object):
    def __init__(self, match_num=200):
        self.match_num = match_num
