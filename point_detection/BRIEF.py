import cv2

from keypoint import Keypoint as Keypoint

__author__ = "Dandi Chen"

class BRIEF_point(Keypoint):
    def __init__(self, kp_num):
        Keypoint.__init__(self, kp_num)

    def get_keypoint_single(self, img):
        # Initiate STAR detector
        star = cv2.xfeatures2d.StarDetector_create()

        # Initiate BRIEF extractor
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

        # find the keypoints with STAR
        kp = star.detect(img, None)

        # compute the descriptors with BRIEF
        kp, des = brief.compute(img, kp)
        return kp, des






