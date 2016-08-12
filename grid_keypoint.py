import cv2
import math
import numpy as np

__author__ = "Dandi Chen"

def _grid_img(img, width, height, x_blk_size=100, y_blk_size=100):
    x_num = int(math.ceil((width - x_blk_size) / x_blk_size)) + 2
    y_num = int(math.ceil((height - y_blk_size) / y_blk_size)) + 2

    img_patch = []
    x_trans = []
    y_trans = []
    patch_x_idx = []
    patch_y_idx = []
    for i in range(y_num):
        for j in range(x_num):
            if i != y_num - 1 and j != x_num - 1:
                block = img[i*y_blk_size:(i+1)*y_blk_size, j*x_blk_size:(j+1)*x_blk_size]
                img_patch.append(np.array(block))
                x_trans.append(j*x_blk_size)
                y_trans.append(i*y_blk_size)
                patch_x_idx.append(j)
                patch_y_idx.append(i)
    return img_patch, x_trans, y_trans, patch_x_idx, patch_y_idx, x_num, y_num

def _get_ORB_pt(img, max_kp_num):
    # Initiate ORB detector
    orb = cv2.ORB_create(max_kp_num)

    # find the keypoints with ORB
    kp = orb.detect(img, None)

    return kp

def _convert_trans(kp, x_trans, y_trans):
    x_mat = []
    y_mat = []
    for idx in range(len(kp)):
        (x, y) = kp[idx].pt
        x_mat.extend([int(x) + x_trans])
        y_mat.extend([int(y) + y_trans])
    return x_mat, y_mat

def draw_Keypoints(img, img_x, img_y):
    for idx in range(len(img_x)):
        cv2.circle(img, (img_x[idx], img_y[idx]), 4, (255, 0, 0), 2)

    cv2.imshow('ORB grid keypoint', img)
    cv2.waitKey(0)

def grid_keypoint(img, x_blk_size=100, y_blk_size=100, max_kp_num = 500):
    height, width, _ = img.shape
    img_patch, x_trans, y_trans, _, _, _, _ = _grid_img(img, width, height, x_blk_size, y_blk_size)

    img_x = []
    img_y = []

    for patch_idx in range(len(img_patch)):
        print ''
        print 'patch index: ', patch_idx

        # keypoints in each image patch
        kp = _get_ORB_pt(np.array(img_patch[patch_idx]), max_kp_num)

        # convert keypoint coordinates
        patch_x, patch_y = _convert_trans(kp, x_trans[patch_idx], y_trans[patch_idx])

        img_x.extend(patch_x)
        img_y.extend(patch_y)

    return img_x, img_y

img = cv2.imread('000000_10.png')
x, y = grid_keypoint(img, 150, 150, 3)
draw_Keypoints(img, x, y)