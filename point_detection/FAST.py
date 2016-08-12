import cv2

from keypoint import Keypoint as Keypoint

__author__ = "Dandi Chen"

class FAST_point(Keypoint):
    def __init__(self, kp_num):
        Keypoint.__init__(self, kp_num)

    def get_keypoint_single(self, img):
        fast = cv2.FastFeatureDetector_create()
        kp = fast.detect(img, None)
        return kp
