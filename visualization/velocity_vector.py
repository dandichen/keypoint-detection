import cv2
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'Dandi Chen'

def plot_velocity_vector(opt_flow, step, trans = 0):

    if trans == 0:
        flow = opt_flow
    else:
        flow = opt_flow - trans
    img = np.ones(flow.shape[:2] + (3,))
    for i in range(0, img.shape[0] - step, step):
        for j in range(0, img.shape[1] - step, step):
            try:
                # opencv 3.1.0
                if flow.shape[-1] == 2:
                    cv2.arrowedLine(img, (j, i), (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))),
                                    (150, 0, 0), 2)
                else:
                    cv2.arrowedLine(img, (j, i), (j + int(round(flow[i, j, 0])), i), (150, 0, 0), 2)

            except AttributeError:
                # opencv 2.4.8
                if flow.shape[-1] == 2:
                    cv2.line(img, (j, i), (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))),
                             (150, 0, 0), 2)
                else:
                    cv2.line(img, pt1=(j, i), pt2=(j + int(round(flow[i, j])), i), color=(150, 0, 0), thickness=1)

    plt.figure()
    plt.imshow(img)
    plt.title('velocity vector')
    plt.waitforbuttonpress()

def plot_velocity_vector_mask(opt_flow, mask, step, trans = 0):

    if trans == 0:
        flow = opt_flow
    else:
        flow = opt_flow - trans

    img = np.ones(flow.shape[:2] + (3,))
    for i in range(0, img.shape[0] - step, step):
        for j in range(0, img.shape[1] - step, step):
            if mask[i, j] != 0:
                try:
                    # opencv 3.1.0
                    if flow.shape[-1] == 2:
                        cv2.arrowedLine(img, (j, i), (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))),
                                        (150, 0, 0), 2)
                    else:
                        cv2.arrowedLine(img, (j, i), (j + int(round(flow[i, j, 0])), i), (150, 0, 0), 2)

                except AttributeError:
                    # opencv 2.4.8
                    if flow.shape[-1] == 2:
                        cv2.line(img, (j, i), (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))),
                                 (150, 0, 0), 2)
                    else:
                        cv2.line(img, pt1=(j, i), pt2=(j + int(round(flow[i, j])), i), color=(150, 0, 0), thickness=1)

    plt.figure()
    plt.imshow(img)
    plt.title('velocity vector')
    plt.waitforbuttonpress()

def write_velocity_vector_compare(flow_x, flow_y, flow_x_gt, flow_y_gt, img, width, height, vel_path, step1=10, step2=10):
    # white background
    vel_img = np.ones((height, width, 3), dtype=np.float64)*255
    for j in range(0, width - step1, step1):
        for i in range(0, height - step1, step1):
            # cv2.arrowedLine(vel_img, (j, i), (j + int(round(flow_x[i, j])), i + int(round(flow_y[i, j]))), (255, 0, 0), 2)
            cv2.arrowedLine(vel_img, (j, i), (j + int(round(flow_x_gt[i, j])), i + int(round(flow_y_gt[i, j]))), (0, 0, 150), 2)

            # cv2.arrowedLine(img, (j, i), (j + int(round(flow_x[i, j])), i + int(round(flow_y[i, j]))), (255, 0, 0), 2)
            # cv2.arrowedLine(img, (j, i), (j + int(round(flow_x_gt[i, j])), i + int(round(flow_y_gt[i, j]))), (0, 0, 150), 2)
    for j in range(0, width - step2, step2):
        for i in range(0, height - step2, step2):
            cv2.arrowedLine(vel_img, (j, i), (j + int(round(flow_x[i, j])), i + int(round(flow_y[i, j]))), (255, 0, 0), 2)
            # cv2.arrowedLine(vel_img, (j, i), (j + int(round(flow_x_gt[i, j])), i + int(round(flow_y_gt[i, j]))), (0, 0, 150), 2)
    cv2.imwrite(vel_path, vel_img)

def write_velocity_vector_compare_mask(flow_x, flow_y, flow_mask, flow_x_gt, flow_y_gt, width, height,
                                       vel_path, step1=10, step2=10):
    # white background
    vel_img = np.ones((height, width, 3), dtype=np.float64)*255
    for j in range(0, width - step1, step1):
        for i in range(0, height - step1, step1):
                cv2.arrowedLine(vel_img, (j, i), (j + int(round(flow_x_gt[i, j])), i + int(round(flow_y_gt[i, j]))),
                                (0, 0, 150), 2)

    for j in range(0, width - step2, step2):
        for i in range(0, height - step2, step2):
            if flow_mask[i, j] == True:
                cv2.arrowedLine(vel_img, (j, i), (j + int(round(flow_x[i, j])), i + int(round(flow_y[i, j]))),
                                (255, 0, 0), 2)
    cv2.imwrite(vel_path, vel_img)