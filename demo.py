import os
import math
import cv2
import timeit
import numpy as np
import matplotlib.pyplot as plt

from point_detection import keypoint
from point_detection.ORB import ORB_point
from point_matching import matcher
from point_matching.brute_force import BFMatcher

import evaluation.angular_error as eval_ang
import evaluation.correlation as eval_corr
import evaluation.endpoint_error as eval_end
import evaluation.form as eval_form
import evaluation.outlier_error as eval_out
import evaluation.percentage as eval_per

import visualization.keypoint_pairs as vis_kp
import visualization.matchers as vis_matcher
import visualization.evaluation as vis_eval
import visualization.velocity_vector as vis_vel

__author__ = 'Dandi Chen'

img_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/image_2/'
flow_gt_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/flow_occ'

eval_path = '/mnt/scratch/DandiChen/keypoint/KITTI/pipeline/confidence/'
kp_path = eval_path + 'keypoint/'
match_path = eval_path + 'matches/000000_10_customized/box/weighted_dis/w-0.5/'
match_path_overlap = match_path + 'overlap/'
match_path_non_overlap = match_path + 'non_overlap/'
flow_path = eval_path + 'flow/'
velocity_path = eval_path + 'velocity/'


img_num = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])
flow_num = img_num - 1  # continuous two frames

# pair_num = img_num/2
# pair_num = 2
pair_num = 1

t_mat = []
per_mat = []
ol_num_mat = []
gt_num_mat = []

corr_X_mat = []
corr_Y_mat = []
err_ratio_mat = []
ang_err_mat = []
end_pt_err_mat = []


for img in range(pair_num):
    print ''
    print 'img number: ', img

    img_path1 = os.path.join(img_path, str(img).zfill(6) + '_10.png')
    img_path2 = os.path.join(img_path, str(img).zfill(6) + '_11.png')

    # start = timeit.default_timer()
    img1, img2 = keypoint.read_img_pair(img_path1, img_path2)
    height, width, _ = img1.shape

    """
        bounding box
            - get bounding box coordinates from detection
                - method 1: extract bounding box feature from two frames, then sort via similarity.
                            The larger similarity value, the better.
                - method 2: extract bounding boxes from two frames, then compute keypoint in two bounding boxes seperately.
                            The more keypoint matching, the better. (preferred)
                - method 3: extract bonding box from first frame, then compute keypoint in bounding box. Compute keypoint
                            in second frame within neighborhood correspoinding to keypoint in first frame.
                - method 4: draw corresponding bounding boxes by hand from detection results. (current)
            - get bounding box affine transformation using good matching perspectiveTransformation
                - good keypoints come from overlap between detection and affine transformation
                - failed
    """
    # start = timeit.default_timer()

    # bounding box coordinates
    top_left_x1, top_left_y1, bom_right_x1, bom_right_y1 = 154.07749939, 181.342102051, 405.574401855, 305.924407959
    top_left_x2, top_left_y2, bom_right_x2, bom_right_y2 = 0.0, 156.604873657, 353.453063965, 351.0

    # # bounding box affine transformation
    # src = np.float32([[top_left_x2, top_left_y2], [top_left_x2, bom_right_y2], [bom_right_x2, bom_right_y2],
    #                   [bom_right_x2, top_left_y2]]).reshape(-1, 1, 2)
    #
    # matchesMask, dst = bfm.get_homography(kp1, kp2, good, src)
    #
    # out_img = img2
    # out_img = cv2.polylines(out_img, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
    #
    # cv2.imwrite('bbox 2.png', out_img)
    # cv2.imshow('affine transformation', out_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    """
        keypoint extraction
            - compute keypoint with proper parameters
                - SIFT
                - SURF
                - FAST
                - ORB (current)
                - BRIEF
                - LBP
                - Shi-Tomasi Corner Detector(opencv/Haoyi)
    """
    # ORB keypoint
    orb = ORB_point(200)
    kp1, des1, kp2, des2 = orb.get_keypoint(img1[int(top_left_y1):int(bom_right_y1), int(top_left_x1):int(bom_right_x1)],
                                            img2[int(top_left_y2):int(bom_right_y2), int(top_left_x2):int(bom_right_x2)])
    print 'keypoint number:', len(kp1), len(kp2)
    # vis_kp.vis_pt_pairs(img1, img2, kp1, kp2, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1)
    # vis_kp.write_pt_pairs(img1, img2, kp1, kp2, os.path.join(kp_path, str(img).zfill(6) + '_10.png'),
    #                       os.path.join(kp_path, str(img).zfill(6) + '_11.png'),
    #                       top_left_x1, top_left_y1, bom_right_x1, bom_right_y1)

    """
        keypoint matching
            - methods
                - brute force matching (current)
                - FLANN
            - pre processing
                - keypoints are within bounding box
                    - rectangle bounding box (current)
                    - Gaussian bounding box
                - keypoint pairs are within neighbor area (prepared for RANSAC)
                - Lowe's good feature threshould (current)
            - post processing
                - RANSAC between matches (current)
            - confidence
                - matcher similarity (Hamming distance)
                - keypoint position constraint (Euclidean distance)
                - similarity and position and linear weighted (current)
    """
    # BFMatcher
    bfm = BFMatcher()
    matches = bfm.get_matcher(des1, des2)
    print 'matcher number:', len(matches)

    if matches == None:
        continue

    # find homography
    good = bfm.get_good_matcher(matches)           # Lowe's good feature threshold criteria(feature similarity distance)
    print 'good matcher number:', len(good)
    print 'good ratio:', len(good) / float(len(matches))

    # confidence evaluation
    good_wgt_dis_matches = bfm.get_wgt_dis_matcher(kp1, kp2, good, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1,
                                                   top_left_x2, top_left_y2, bom_right_x2, bom_right_y2, 0.5)
    matchesMask, _ = bfm.get_homography(kp1, kp2, good_wgt_dis_matches)

    # vis_matcher.vis_matches(img1, img2, kp1, kp2, good, matchesMask, 1, 0, len(good), top_left_x1, top_left_y1,
    #                         bom_right_x1, bom_right_y1, top_left_x2, top_left_y2, bom_right_x2, bom_right_y2)
    # vis_matcher.write_matches(img1, img2, kp1, kp2, good_wgt_dis_matches, matchesMask,
    #                           os.path.join(match_path, str(img).zfill(6) + '_10_non_overlap_match.png'), 1,
    #                           0, len(good_wgt_dis_matches), top_left_x1, top_left_y1, bom_right_x1, bom_right_y1,
    #                           top_left_x2, top_left_y2, bom_right_x2, bom_right_y2)
    # vis_matcher.write_matches_overlap(img1, kp1, kp2, good_wgt_dis_matches, matchesMask,
    #                           os.path.join(match_path, str(img).zfill(6) + '_10_overlap_match.png'), 1,
    #                           0, len(good_wgt_dis_matches), top_left_x1, top_left_y1, bom_right_x1, bom_right_y1,
    #                           top_left_x2, top_left_y2, bom_right_x2, bom_right_y2)
    print 'good weight distance matcher number:', len(good_wgt_dis_matches)
    print 'good weighted ratio:', len(good_wgt_dis_matches)/float(len(matches))

    dis = []
    for match_idx in range(len(good_wgt_dis_matches) - 1):
        print 'idx = ', match_idx, 'distance = ', good_wgt_dis_matches[match_idx].distance
        dis.append([good_wgt_dis_matches[match_idx].distance])
        # vis_matcher.vis_matches(img1, img2, kp1, kp2, good_wgt_dis_matches, matchesMask, 1,
        #                         match_idx, match_idx + 1, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1,
        #                         top_left_x2, top_left_y2, bom_right_x2, bom_right_y2)
        # vis_matcher.write_matches(img1, img2, kp1, kp2, good_wgt_dis_matches, matchesMask,
        #                           os.path.join(match_path_non_overlap, str(match_idx).zfill(4) + '_10.png'), 1,
        #                           match_idx, match_idx + 1, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1,
        #                           top_left_x2, top_left_y2, bom_right_x2, bom_right_y2)
        # vis_matcher.write_matches_overlap(img1, kp1, kp2, good_wgt_dis_matches, matchesMask,
        #                                   os.path.join(match_path_overlap, str(match_idx).zfill(4) + '_10.png'), 1,
        #                                   match_idx, match_idx + 1, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1,
        #                                   top_left_x2, top_left_y2, bom_right_x2, bom_right_y2)

    # plt.figure()
    # plt.plot(np.arange(len(dis)), dis)
    # plt.title('matches distance')
    # plt.show()

    # # show all good matches with confidences
    # plt.figure()
    # dis_shown = np.zeros((len(good_wgt_dis_matches), 1))

    # 000000_10.png pair in original DMatch feature distance
    # success_match = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    #                           24, 25, 28, 29, 30, 32, 33, 34, 35, 36, 41, 42, 45, 46, 47, 48, 49, 51, 57, 58, 65,
    #                           66, 67, 68, 69, 81, 82, 92, 93, 94, 102, 104, 105, 110, 121, 122, 123, 124, 125, 126,
    #                           138, 153, 154, 155, 156, 157, 159, 179, 180, 190, 191, 192, 193, 195, 196, 198, 199,
    #                           212, 213, 225, 228, 229, 240, 241, 242, 258, 259, 260, 261, 270, 271, 272, 289, 291,
    #                           292, 294, 296, 307, 308, 309, 323, 324, 330, 331, 332, 358, 359, 367, 380, 381, 405, 406])
    # for shown_idx in range(len(good_wgt_dis_matches)):
    #     if shown_idx in success_match:
    #         dis_shown[shown_idx] = good_wgt_dis_matches[shown_idx].distance
    #         print 'idx = ', shown_idx, 'distance = ', good_wgt_dis_matches[shown_idx].distance
    # plt.scatter(np.arange(len(good_wgt_dis_matches)), dis_shown)
    # plt.show()

    """
        keypoint correspondence evaluation with flow ground truth
            - velocity vector visualization (current)
            - vanishing point (done)
            - time for correspondence computation (current)
            - quantitative criteria between correspondence and flow
                - overlap percentage
                    - pixel level (current)
                    - neighborhood level
                - correlation (current)
                - outlier error (current)
                - angular error (current)
                - endpoint error (current)
    """

    # _, _, flow_X, flow_Y, flow_mask = matcher.get_flow(kp1, kp2, width, height, good_wgt_dis_matches,
    #                                                    os.path.join(flow_path, str(img).zfill(6) + '.npz'),
    #                                                    top_left_x1, top_left_y1, top_left_x2, top_left_y2)

#
#     # vis_matcher.vis_matches(img1, img2, kp1, kp2, matches, 1, 100, 125, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1)
#     # vis_matcher.write_matches(img1, img2, kp1, kp2, matches, matchesMask,
#     #                           os.path.join(match_path, str(img).zfill(6) + '_10.png'),
#     #                           1, 100, 125, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1)
#
    # end = timeit.default_timer()
    # t = end - start
    # print 'time = ', t
    # t_mat.append([t])

    # # flow evaluation
    # flow_X_gt, flow_Y_gt, flow_mask_gt = eval_form.read_gt(os.path.join(flow_gt_path, str(img).zfill(6) + '_10.png'))
    # flow_mask_gt_box = eval_form.convert_gt_mask_box(flow_mask_gt, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1)
    #
    # # similarity evaluation-successful correspondence
    # sim_mask = eval_form.get_vector_sim(flow_X, flow_Y, flow_X_gt, flow_Y_gt, flow_mask, flow_mask_gt, width, height)
    # print 'good flow number:', len(np.where(sim_mask ==True)[0])
    # vis_matcher.write_flow2match_mask(img1, img2, flow_X, flow_Y, sim_mask, width, height,
    #                                           os.path.join(match_path, str(img).zfill(6) + '_10.png'), 1,
    #                                           top_left_x1, top_left_y1, bom_right_x1, bom_right_y1,
    #                                           top_left_x2, top_left_y2, bom_right_x2, bom_right_y2)

    # vis_matcher.write_flow2match_overlap_mask(img1, flow_X, flow_Y, sim_mask, width, height,
    #                                           os.path.join(match_path, str(img).zfill(6) + '_10.png'), 1,
    #                                           top_left_x1, top_left_y1, bom_right_x1, bom_right_y1,
    #                                           top_left_x2, top_left_y2, bom_right_x2, bom_right_y2)

    # # flow comparison visualization
    # vis_vel.write_velocity_vector_compare(flow_X, flow_Y, flow_X_gt, flow_Y_gt, img1, width, height,
    #                                       os.path.join(velocity_path, str(img).zfill(6) + '_10_line.png'), 20, 3)

    # # positive/negative flow visualization
    # flow_X_pos_pos = np.zeros_like(flow_X)
    # flow_Y_pos_pos = np.zeros_like(flow_Y)
    # flow_X_pos_pos[np.where(np.logical_and(flow_X > 0, flow_Y > 0))] = flow_X[np.where(np.logical_and(flow_X > 0,
    #                                                                                                   flow_Y > 0))]
    # flow_Y_pos_pos[np.where(np.logical_and(flow_X > 0, flow_Y > 0))] = flow_Y[np.where(np.logical_and(flow_X > 0,
    #                                                                                                   flow_Y > 0))]
    # vis_vel.write_velocity_vector_compare(flow_X_pos_pos, flow_Y_pos_pos, flow_X_gt, flow_Y_gt, img1, width, height,
    #                                       os.path.join(velocity_path, str(img).zfill(6) + '_10_pos_pos.png'), 20, 3)
    #
    # flow_X_pos_neg = np.zeros_like(flow_X)
    # flow_Y_pos_neg = np.zeros_like(flow_Y)
    # flow_X_pos_neg[np.where(np.logical_and(flow_X > 0, flow_Y < 0))] = flow_X[np.where(np.logical_and(flow_X > 0,
    #                                                                                                   flow_Y < 0))]
    # flow_Y_pos_neg[np.where(np.logical_and(flow_X > 0, flow_Y < 0))] = flow_Y[np.where(np.logical_and(flow_X > 0,
    #                                                                                                   flow_Y < 0))]
    # vis_vel.write_velocity_vector_compare(flow_X_pos_neg, flow_Y_pos_neg, flow_X_gt, flow_Y_gt, img1, width, height,
    #                                       os.path.join(velocity_path, str(img).zfill(6) + '_10_pos_neg.png'), 20, 3)
    #
    # flow_X_neg_pos = np.zeros_like(flow_X)
    # flow_Y_neg_pos = np.zeros_like(flow_Y)
    # flow_X_neg_pos[np.where(np.logical_and(flow_X < 0, flow_Y > 0))] = flow_X[np.where(np.logical_and(flow_X < 0,
    #                                                                                                   flow_Y > 0))]
    # flow_Y_neg_pos[np.where(np.logical_and(flow_X < 0, flow_Y > 0))] = flow_Y[np.where(np.logical_and(flow_X < 0,
    #                                                                                                   flow_Y > 0))]
    # vis_vel.write_velocity_vector_compare(flow_X_neg_pos, flow_Y_neg_pos, flow_X_gt, flow_Y_gt, img1, width, height,
    #                                       os.path.join(velocity_path, str(img).zfill(6) + '_10_neg_pos.png'), 20, 3)
    #
    # flow_X_neg_neg = np.zeros_like(flow_X)
    # flow_Y_neg_neg = np.zeros_like(flow_Y)
    # flow_X_neg_neg[np.where(np.logical_and(flow_X < 0, flow_Y < 0))] = flow_X[np.where(np.logical_and(flow_X < 0,
    #                                                                                                   flow_Y < 0))]
    # flow_Y_neg_neg[np.where(np.logical_and(flow_X < 0, flow_Y < 0))] = flow_Y[np.where(np.logical_and(flow_X < 0,
    #                                                                                                   flow_Y < 0))]
    # vis_vel.write_velocity_vector_compare(flow_X_neg_neg, flow_Y_neg_neg, flow_X_gt, flow_Y_gt, img1, width, height,
    #                                       os.path.join(velocity_path, str(img).zfill(6) + '_10_neg_neg.png'), 20, 3)



#     per, ol_num, gt_num = eval_per.get_overlap_per(flow_mask, flow_mask_gt_box)
#     print 'percentage = ', per
#     per_mat.append([per])
#     ol_num_mat.append([ol_num])
#     gt_num_mat.append([gt_num])
#
#     corr_X, corr_Y = eval_corr.get_correlation(flow_X, flow_Y, flow_X_gt, flow_Y_gt, flow_mask, flow_mask_gt_box)
#     if math.isnan(corr_X) or math.isnan(corr_Y):
#         continue
#     print 'corr_X = ', corr_X, 'corr_Y = ', corr_Y
#     corr_X_mat.append([corr_X])
#     corr_Y_mat.append([corr_Y])
#
#     err_ratio = eval_out.get_outlier_err(flow_X, flow_Y, flow_X_gt, flow_Y_gt, flow_mask, flow_mask_gt_box)
#     if math.isnan(err_ratio):
#         continue
#     print 'err_ratio = ', err_ratio
#     err_ratio_mat.append([err_ratio])
#
    # ang_err_rad, ang_err_deg = eval_ang.get_angular_err(flow_X, flow_Y, flow_X_gt, flow_Y_gt, flow_mask, flow_mask_gt_box)
    # if math.isnan(ang_err_deg):
    #     continue
    # print 'ang_err_deg = ', ang_err_deg
    # ang_err_mat.append([ang_err_deg])
#
#     end_pt_err = eval_end.get_endpoint_err(flow_X, flow_Y, flow_X_gt, flow_Y_gt, flow_mask, flow_mask_gt_box)
#     print 'end_pt_err = ', end_pt_err
#     end_pt_err_mat.append([end_pt_err])
#
# print ''
# print 'ave time = ', np.mean(t_mat)
# print 'ave percentage = ', np.mean(per_mat)
# print 'ave overlap num = ', np.mean(ol_num_mat)
# print 'ave ground truth num = ', np.mean(gt_num_mat)
# print 'ave corr_X = ', np.mean(corr_X_mat), 'aver corr_Y = ', np.mean(corr_Y_mat)
# print 'ave err_ratio = ', np.mean(err_ratio_mat)
# print 'ave ang_err = ', np.mean(ang_err_mat)
# print 'ave end_pt_err = ', np.mean(end_pt_err_mat)
#
# vis_eval.vis_evaluation(per_mat, corr_X_mat, corr_Y_mat, err_ratio_mat, ang_err_mat, end_pt_err_mat)
# vis_eval.write_evaluation(per_mat, corr_X_mat, corr_Y_mat, err_ratio_mat, ang_err_mat, end_pt_err_mat, eval_path)