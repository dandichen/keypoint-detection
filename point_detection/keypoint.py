import cv2
import numpy as np

__author__ = "Dandi Chen"

def read_img_pair(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    return img1, img2


class Keypoint(object):
    def __init__(self, kp_num=200):
        self.kp_num = kp_num
        self.x = np.arange(self.kp_num)
        self.y = np.arange(self.kp_num)
        self.flow_X = np.arange(self.kp_num)
        self.flow_Y = np.arange(self.kp_num)