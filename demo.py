import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import flow as vis_flow
import velocity_vector as vis_vel

print cv2.__version__

# img_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/image_2'
#
# img_num = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))]) - 1
# flow_num = img_num - 1  # continuous two frames

img1 = cv2.imread('./000000_10/000000_10.png', 0)
img2 = cv2.imread('./000000_11/000000_11.png', 0)
height, width = img1.shape

# Initiate FAST object with default values
# fast = cv2.FastFeatureDetector_create()

# # find and draw the keypoints
# kp1 = fast.detect(img1, None)
# kp2 = fast.detect(img2, None)
#
#
# # fast_img1 = img1
# fast_img2 = img2
#
# # fast_img1 = cv2.drawKeypoints(img1, kp1, fast_img1, color=(255, 0, 0))
# fast_img2 = cv2.drawKeypoints(img2, kp2, fast_img2, color=(255, 0, 0))
#
# # cv2.imwrite('000000_10_FAST.png', fast_img1)
# cv2.imwrite('000000_11_FAST.png', fast_img2)


# Print all default params
# print "Threshold: ", fast.getThreshold()
# print "nonmaxSuppression: ", fast.getNonmaxSuppression()
# print "neighborhood: ", fast.getType()
# print "Total Keypoints with nonmaxSuppression: ", len(kp1)
#
# cv2.imshow('fast_false_0.png', fast_img1)
# cv2.waitKey(0)
#
# # # Disable nonmaxSuppression
# fast.setNonmaxSuppression(0)
# # kp1_no = fast.detect(img1, None)
# kp2_no = fast.detect(img2, None)
# #
# # print "Total Keypoints without nonmaxSuppression: ", len(kp)
# #
# # fast_img_no1 = img1
# fast_img_no2 = img2
#
# # fast_img_no1 = cv2.drawKeypoints(img1, kp1, fast_img_no1, color=(255, 0, 0))
# fast_img_no2 = cv2.drawKeypoints(img2, kp2_no, fast_img_no2, color=(255, 0, 0))
# # cv2.imshow('fast_false_0.png', fast_img_no)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# cv2.imwrite('000000_11_FAST_no.png', fast_img_no2)


# Initiate ORB detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp1 = orb.detect(img1, None)
kp2 = orb.detect(img2, None)

# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img2, kp2)

ORB_img1 = img1
ORB_img2 = img2

# draw only keypoints location,not size and orientation
ORB_img1 = cv2.drawKeypoints(img1, kp1, ORB_img1, color=(0, 255, 0), flags=0)
cv2.imshow('image1', ORB_img1)
cv2.waitKey(0)

ORB_img2 = cv2.drawKeypoints(img2, kp2, ORB_img2, color=(0, 255, 0), flags=0)
cv2.imshow('image2', ORB_img2)
# cv2.waitKey(0)
#
# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
search_params = dict()      # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(np.float32(des1), k=2)
matches = flann.knnMatch(des1, des2, k=2)
matches = sorted(matches, key=lambda x:x.distance)

# Need to draw only good matches, so create a mask
# matchesMask = [[0, 0] for i in xrange(len(matches))]

# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.7 * n.distance:
#         matchesMask[i] = [1, 0]

# draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(0, 255, 0), matchesMask=matchesMask, flags=0)
# outImg = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

outImg = np.zeros((height, width*2))
outImg = cv2.drawMatches(img1, kp1, img2, kp2, matches, outImg, flags=2)
cv2.imshow('ORB-FLANN', outImg)
cv2.waitKey(0)
#
# cv2.imwrite('ORB-FLANN.png', outImg)
# cv2.destroyAllWindows()

# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# # Match descriptors
# matches = bf.match(des1, des2)
#
# # Sort them in the order of their distance
# matches = sorted(matches, key=lambda x:x.distance)
# # src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
# # dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
#
# flow_x, flow_y = vis_flow.get_flow(kp1, kp2, matches, width, height)



# src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
# dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
#
# flow = dst_pts - src_pts
# flow_X = flow[:, 0, 0]
# flow_Y = flow[:, 0, 1]
# x = src_pts[:, 0, 0]
# y = src_pts[:, 0, 1]

# vis_flow.flow2color(flow)
# vis_vel.plot_velocity_vector(flow, 30)
#
# np.savez('000000.npz', flow_X=flow_X, flow_Y=flow_Y, x=x, y=y)

# outImg = np.zeros((height, width*2))
# outImg = cv2.drawMatches(img1, kp1, img2, kp2, matches, outImg, flags=2)
# cv2.imshow('ORB matches', outImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# # Initiate STAR detector
# star = cv2.xfeatures2d.StarDetector_create()
#
# # Initiate BRIEF extractor
# brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
#
#
# # find the keypoints with STAR
# kp1 = star.detect(img1, None)
# kp2 = star.detect(img2, None)
#
# # compute the descriptors with BRIEF
# kp1, des1 = brief.compute(img1, kp1)
# kp2, des2 = brief.compute(img2, kp2)
#
# BRIEF_img1 = img1
# BRIEF_img2 = img2
#
# BRIEF_img1 = cv2.drawKeypoints(img1, kp1, BRIEF_img1, color=(0, 0, 255), flags=0)
# BRIEF_img2 = cv2.drawKeypoints(img2, kp2, BRIEF_img2, color=(0, 0, 255), flags=0)
#
# cv2.imshow('000000_10_BRIEF', BRIEF_img1)
# cv2.imwrite('000000_10_BRIEF.png', BRIEF_img1)
# cv2.waitKey(0)
#
# cv2.imshow('000000_11_BRIEF', BRIEF_img2)
# cv2.imwrite('000000_11_BRIEF.png', BRIEF_img2)
# cv2.waitKey(0)
#
# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# # Match descriptors
# matches = bf.match(des1, des2)
#
# # Sort them in the order of their distance
# matches = sorted(matches, key=lambda x:x.distance)
#
#
# outImg = np.zeros((height, width*2))
# outImg = cv2.drawMatches(img1, kp1, img2, kp2, matches, outImg, flags=2)
# cv2.imshow('BRIEF-BF', outImg)
# cv2.imwrite('BRIEF-BF.png', outImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# # Initiate SIFT detector
# sift = cv2.xfeatures2d.SIFT_create()
#
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)
# # draw only keypoints location,not size and orientation
#
# SIFT_img1 = img1
# SIFT_img2 = img2
#
# SIFT_img1 = cv2.drawKeypoints(img1, kp1, SIFT_img1, color=(0, 128, 255), flags=0)
# SIFT_img2 = cv2.drawKeypoints(img2, kp2, SIFT_img2, color=(0, 128, 255), flags=0)
# cv2.imshow('image1', SIFT_img1)
# cv2.imshow('image2', SIFT_img2)
# cv2.waitKey(0)
#
# cv2.imwrite('000000_10_SIFT.png', SIFT_img1)
# cv2.imwrite('000000_11_SIFT.png', SIFT_img2)
#
#
# # FLANN parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict() # or pass empty dictionary
#
# flann = cv2.FlannBasedMatcher(index_params, search_params)
#
# matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)
#
# # Need to draw only good matches, so create a mask
# matchesMask = [[0, 0] for i in xrange(len(matches))]
#
# # ratio test as per Lowe's paper
# for i,(m,n) in enumerate(matches):
#     if m.distance < 0.7 * n.distance:
#         matchesMask[i] = [1, 0]
#
# draw_params = dict(matchColor=(0, 128, 255), singlePointColor=(0, 128, 255), matchesMask=matchesMask, flags=0)
# outImg = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
#
# cv2.imshow('SIFT-FLANN', outImg)
# cv2.waitKey(0)
#
# cv2.imwrite('SIFT-FLANN.png', outImg)
# cv2.destroyAllWindows()