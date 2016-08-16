import cv2
import numpy as np

from matcher import Matcher as Matcher

from point_detection import keypoint

import evaluation.form as form


__author__ = "Dandi Chen"

class BFMatcher(Matcher):
    def __init__(self, match_num=200):
        Matcher.__init__(self, match_num)

    def get_matcher(self, des1, des2):
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        if des1 != None and des2 != None:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            return matches
        else:
            return None

    def get_good_matcher(self, matches, threshold=0.7):     # Lowe's ratio test for normalized matches distance
        good = []
        dis_norm = self.normalize_dis(matches)
        for idx in range(len(matches)):
            if matches[idx].distance < threshold * matches[-1].distance:
                matches[idx].distance = dis_norm[idx]
                good.append(matches[idx])
        return good

    def get_wgt_dis_matcher(self, kp1, kp2, good, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1,
                            top_left_x2, top_left_y2, bom_right_x2, bom_right_y2, weight=0.5):
        wgt_dis = keypoint.get_weight_dis(kp1, kp2, good, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1,
                                          top_left_x2, top_left_y2, bom_right_x2, bom_right_y2, weight)
        wgt_dis_matches = good
        for idx in range(len(good)):
            wgt_dis_matches[idx].distance = wgt_dis[idx]
        wgt_dis_matches = sorted(wgt_dis_matches, key=lambda x: x.distance)
        good_wgt_dis_matches = self.get_good_matcher(wgt_dis_matches)
        return good_wgt_dis_matches

    def get_homography(self, kp1, kp2, good, src=None, min_match_count=10, ransacReprojThreshold=3.0):
        if len(good) > min_match_count:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold)
            matchesMask = np.array(mask.ravel(), dtype=bool)

            if src != None:
                dst = cv2.perspectiveTransform(src, M)
            else:
                dst = None
        else:
            print "Not enough matches are found - %d/%d" % (len(good), min_match_count)
            matchesMask = None
            dst = None
        return matchesMask, dst

    def normalize_dis(self, matches, start=0, end=1):
        dis = []
        for match in matches:
            dis.append(match.distance)
        dis_norm = form.normalize_len(dis, start, end)
        return dis_norm










