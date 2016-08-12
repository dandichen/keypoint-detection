import os
import cv2

from point_detection import keypoint
from point_detection import boundingbox as bbox
from point_detection.SIFT import SIFT_point
from point_detection.SURF import SURF_point
from point_detection.FAST import FAST_point
from point_detection.ORB import ORB_point
from point_detection.BRIEF import BRIEF_point



import visualization.keypoint_pairs as vis_kp
import visualization.bounding_box as vis_box

img_path = '/mnt/scratch/haoyiliang/KITTI/SceneFLow2015/data_scene_flow/training/image_2/'

eval_path = '/mnt/scratch/DandiChen/keypoint/KITTI/pt_distribution/default_BRIEF/'

img_num = len([name for name in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, name))])

pair_num = img_num/2


for img in range(pair_num):
    print ''
    print 'img number: ', img

    img_path1 = os.path.join(img_path, str(img).zfill(6) + '_10.png')
    img_path2 = os.path.join(img_path, str(img).zfill(6) + '_11.png')

    img1, img2 = keypoint.read_img_pair(img_path1, img_path2)
    height, width, _ = img1.shape

    # bounding box coordinates
    top_left_x1, top_left_y1, bom_right_x1, bom_right_y1, bbox_num1 = bbox.get_box_image(img1, width, height)

    # vis_box.vis_boxes(img1, top_left_x1, top_left_y1, bom_right_x1, bom_right_y1)
    vis_box.write_boxes(img1, os.path.join(eval_path, str(img).zfill(6) + '_10.png'),
                      top_left_x1, top_left_y1, bom_right_x1, bom_right_y1)

    top_left_x2, top_left_y2, bom_right_x2, bom_right_y2, bbox_num2 = bbox.get_box_image(img2, width, height)
    # vis_box.vis_boxes(img2, top_left_x2, top_left_y2, bom_right_x2, bom_right_y2)
    vis_box.write_boxes(img2, os.path.join(eval_path, str(img).zfill(6) + '_11.png'),
                      top_left_x2, top_left_y2, bom_right_x2, bom_right_y2)

    # SIFT keypoint
    # sift = SIFT_point(1)

    # SURF keypoint
    # surf = SURF_point(1)

    # FAST keypoint
    # fast = FAST_point(1)

    # ORB keypoint
    # orb = ORB_point(1)

    # BRIEF keypoint
    brief = BRIEF_point(1)


    for idx1 in range(bbox_num1):
        kp1, _ = brief.get_keypoint_single(img1[int(top_left_y1[idx1]):int(bom_right_y1[idx1]),
                                          int(top_left_x1[idx1]):int(bom_right_x1[idx1])])
        # vis_kp.vis_pt(out_img1, kp1, top_left_x1[idx1], top_left_y1[idx1])
        vis_kp.write_pt(img1, kp1, os.path.join(eval_path, str(img).zfill(6) + '_10_' +
                                                    str(idx1).zfill(2) + '.png'), top_left_x1[idx1], top_left_y1[idx1])
        print 'box_dix:', idx1, 'keypoint num:', len(kp1)


    for idx2 in range(bbox_num2):
        kp2, _ = brief.get_keypoint_single(img2[int(top_left_y2[idx2]):int(bom_right_y2[idx2]), int(top_left_x2[idx2]):int(bom_right_x2[idx2])])
        # vis_kp.vis_pt(out_img2, kp2, top_left_x2[idx2], top_left_y2[idx2])
        vis_kp.write_pt(img2, kp2, os.path.join(eval_path, str(img).zfill(6) + '_11_' +
                                                    str(idx2).zfill(2) + '.png'), top_left_x2[idx2], top_left_y2[idx2])
        print 'box_dix:', idx2, 'keypoint num:', len(kp2)
