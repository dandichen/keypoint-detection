import cv2

from keypoint import Keypoint as Keypoint

__author__ = "Dandi Chen"

class SIFT_point(Keypoint):
    def __init__(self, kp_num):
        Keypoint.__init__(self, kp_num)

    def get_keypoint_single(self, img):
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        return kp, des
