import cv2

from keypoint import Keypoint as Keypoint

__author__ = "Dandi Chen"

class SURF_point(Keypoint):
    def __init__(self, kp_num):
        Keypoint.__init__(self, kp_num)

    def get_keypoint_single(self, img):
        surf = cv2.xfeatures2d.SURF_create()
        kp, des = surf.detectAndCompute(img, None)
        return kp, des
