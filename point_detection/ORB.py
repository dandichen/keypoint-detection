import cv2

from keypoint import Keypoint as Keypoint

__author__ = "Dandi Chen"

class ORB_point(Keypoint):
    def __init__(self, kp_num):
        Keypoint.__init__(self, kp_num)

    def get_keypoint(self, img1, img2):
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # orb.setMaxFeatures(1200)
        # orb.setScaleFactor(1.25)
        # orb.setNLevels(6)
        # orb.setEdgeThreshold(10)
        # orb.setPatchSize(20)
        # orb.setFastThreshold(8)

        # find the keypoints with ORB
        kp1 = orb.detect(img1, None)
        kp2 = orb.detect(img2, None)

        # compute the descriptors with ORB
        kp1, des1 = orb.compute(img1, kp1)
        kp2, des2 = orb.compute(img2, kp2)

        return kp1, des1, kp2, des2

    def get_keypoint_single(self, img):
        # Initiate ORB detector
        orb = cv2.ORB_create()
        orb.setMaxFeatures(1200)
        orb.setScaleFactor(1.25)
        orb.setNLevels(6)
        orb.setEdgeThreshold(10)
        orb.setPatchSize(20)
        orb.setFastThreshold(8)

        # find the keypoints with ORB
        kp = orb.detect(img, None)

        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)

        return kp, des




