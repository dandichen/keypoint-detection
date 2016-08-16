import cv2
import numpy as np

import bbox.grid as grid
import visualization.bounding_box as vis_box

__author__ = 'Dandi Chen'

def vis_matches(img1, img2, kp1, kp2, matches, matchesMask, flag=0, show_start=0, show_end=50,
                top_left_x1=0, top_left_y1=0, bom_right_x1=0, bom_right_y1=0,
                top_left_x2=0, top_left_y2=0, bom_right_x2=0, bom_right_y2=0):
    valid_idx = np.where(matchesMask == True)[0]

    if top_left_x1 == 0 and top_left_y1 == 0 and top_left_x2 == 0 and top_left_y2 == 0:
        height, width, _ = img1.shape
        outImg = np.zeros((width * 2, height))
        outImg = cv2.drawMatches(img1, kp1, img2, kp2, matches[show_start:show_end], outImg, flags=2)
        cv2.imshow('ORB matches', outImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        rows1, cols1, _ = img1.shape
        rows2, cols2, _ = img2.shape

        if flag == 0:   # horizontal visualization
            out_img = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
            out_img[:, 0:cols1, :] = img1
            out_img[:, cols1:cols1 + cols2, :] = img2

        else:           # vertical visualization
            out_img = np.zeros((rows1 + rows2, max([cols1, cols2]), 3), dtype='uint8')
            out_img[0:rows1, :, :] = img1
            out_img[rows1:rows1 + rows2, :, :] = img2

        for mat_idx in range(len(matches[show_start:show_end])):
            if mat_idx in valid_idx or show_end - show_start == 1:
                img1_idx = matches[show_start + mat_idx].queryIdx
                img2_idx = matches[show_start + mat_idx].trainIdx

                (x1, y1) = kp1[img1_idx].pt
                (x2, y2) = kp2[img2_idx].pt

                cv2.circle(out_img, (int(x1 + top_left_x1), int(y1 + top_left_y1)), 3, (255, 0, 0), 1)

                if flag == 0:   # horizontal visualization
                    cv2.circle(out_img, (int(x2 + top_left_x2) + cols1, int(y2 + top_left_y2)), 3, (255, 0, 0), 1)
                else:           # vertical visualization
                    cv2.circle(out_img, (int(x2 + top_left_x2), int(y2 + top_left_y2) + rows1), 3, (255, 0, 0), 1)

                color = np.random.randint(0, 255, (100, 3))

                if flag == 0:  # horizontal visualization
                    cv2.line(out_img, (int(x1 + top_left_x1), int(y1 + top_left_y1)), (int(x2 + top_left_x2) + cols1,
                             int(y2 + top_left_y2)), color[np.mod(mat_idx, 100)].tolist(), 1)
                else:          # vertical visualization
                    cv2.line(out_img, (int(x1 + top_left_x1), int(y1 + top_left_y1)), (int(x2 + top_left_x2),
                             int(y2 + top_left_y2) + rows1), color[np.mod(mat_idx, 100)].tolist(), 1)
            else:
                continue

        # draw bounding box
        cv2.rectangle(out_img, (int(top_left_x1), int(top_left_y1)), (int(bom_right_x1), int(bom_right_y1)),
                      (0, 255, 0), 4)
        if flag == 0:  # horizontal visualization
            cv2.rectangle(out_img, (int(top_left_x2) + cols1, int(top_left_y2)),
                          (int(bom_right_x2), int(bom_right_y2) + rows1), (0, 255, 0), 4)
        else:          # vertical visualization
            cv2.rectangle(out_img, (int(top_left_x2), int(top_left_y2) + rows1),
                              (int(bom_right_x2), int(bom_right_y2) + rows1), (0, 255, 0), 4)

        # draw bounding box grid
        x_blk_size = 32
        y_blk_size = 32
        _, x_trans1, y_trans1, _, _, x_num1, y_num1 = grid.grid_img(img1[int(top_left_y1):int(bom_right_y1),
                                                                        int(top_left_x1):int(bom_right_x1)],
                                                                        int(bom_right_x1 - top_left_x1),
                                                                        int(bom_right_y1 - top_left_y1),
                                                                        x_blk_size, y_blk_size)

        vis_box.vis_box_grid(out_img, x_trans1, y_trans1, x_num1, y_num1, x_blk_size, y_blk_size,
                             int(bom_right_x1 - top_left_x1), int(bom_right_y1 - top_left_y1),
                             0, 0, int(top_left_x1), int(top_left_y1))

        _, x_trans2, y_trans2, _, _, x_num2, y_num2 = grid.grid_img(img2[int(top_left_y2):int(bom_right_y2),
                                                                        int(top_left_x2):int(bom_right_x2)],
                                                                        int(bom_right_x2 - top_left_x2),
                                                                        int(bom_right_y2 - top_left_y2),
                                                                        x_blk_size, y_blk_size)
        if flag == 0:  # horizontal visualization
            vis_box.vis_box_grid(out_img, x_trans2, y_trans2, x_num2, y_num2, x_blk_size, y_blk_size,
                                 int(bom_right_x2 - top_left_x2), int(bom_right_y2 - top_left_y2), cols1, 0,
                                 int(top_left_x2), int(top_left_y2))
        else:           # vertical visualization
            vis_box.vis_box_grid(out_img, x_trans2, y_trans2, x_num2, y_num2, x_blk_size, y_blk_size,
                                 int(bom_right_x2 - top_left_x2), int(bom_right_y2 - top_left_y2), 0, rows1,
                                 int(top_left_x2), int(top_left_y2))
        cv2.imshow('ORB matches', out_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def write_matches_overlap(img1, kp1, kp2, matches, matchesMask, match_path, flag=0, show_start=0, show_end=50,
                  top_left_x1=0, top_left_y1=0, bom_right_x1=0, bom_right_y1=0,
                  top_left_x2=0, top_left_y2=0, bom_right_x2=0, bom_right_y2=0):
    valid_idx = np.where(matchesMask == True)[0]

    # 000000_10.png pair in original DMatch feature distance(weight = 0.5)
    # success_match = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    #                           24, 25, 28, 29, 30, 32, 33, 34, 35, 36, 41, 42, 45, 46, 47, 48, 49, 51, 57, 58, 65,
    #                           66, 67, 68, 69, 81, 82, 92, 93, 94, 102, 104, 105, 110, 121, 122, 123, 124, 125, 126,
    #                           138, 153, 154, 155, 156, 157, 159, 179, 180, 190, 191, 192, 193, 195, 196, 198, 199,
    #                           212, 213, 225, 228, 229, 240, 241, 242, 258, 259, 260, 261, 270, 271, 272, 289, 291,
    #                           292, 294, 296, 307, 308, 309, 323, 324, 330, 331, 332, 358, 359, 367, 380, 381, 405, 406])

    out_img = img1.copy()
    for mat_idx in range(len(matches[show_start:show_end])):
        if mat_idx in valid_idx or show_end - show_start == 1:
        # if mat_idx in success_match:
            img1_idx = matches[show_start + mat_idx].queryIdx
            img2_idx = matches[show_start + mat_idx].trainIdx

            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt

            cv2.circle(out_img, (int(x1 + top_left_x1), int(y1 + top_left_y1)), 3, (255, 0, 0), 1)
            if flag == 0:  # horizontal visualization
                cv2.circle(out_img, (int(x2 + top_left_x2), int(y2 + top_left_y2)), 3, (255, 0, 0), 1)
            else:          # vertical visualization
                cv2.circle(out_img, (int(x2 + top_left_x2), int(y2 + top_left_y2)), 3, (255, 0, 0), 1)

            color = np.random.randint(0, 255, (100, 3))
            if flag == 0:  # horizontal visualization
                cv2.line(out_img, (int(x1 + top_left_x1), int(y1 + top_left_y1)),
                         (int(x2 + top_left_x2), int(y2 + top_left_y2)), color[np.mod(mat_idx, 100)].tolist(), 1)
            else:          # vertical visualization
                cv2.line(out_img, (int(x1 + top_left_x1), int(y1 + top_left_y1)),
                         (int(x2 + top_left_x2), int(y2 + top_left_y2)), color[np.mod(mat_idx, 100)].tolist(), 1)
            font = cv2.FONT_HERSHEY_PLAIN
            if np.mod(mat_idx, 2) == 0:
                cv2.putText(out_img, str(mat_idx), (int(x1 + top_left_x1), int(y1 + top_left_y1)), font, 1,
                            color[np.mod(mat_idx, 100)], 1, cv2.LINE_AA)
            else:
                cv2.putText(out_img, str(mat_idx), (int(x2 + top_left_x2), int(y2 + top_left_y2)), font, 1,
                            color[np.mod(mat_idx, 100)], 1, cv2.LINE_AA)
        else:
            continue

    # draw bounding box
    cv2.rectangle(out_img, (int(top_left_x1), int(top_left_y1)), (int(bom_right_x1), int(bom_right_y1)), (0, 255, 0), 4)
    if flag == 0:  # horizontal visualization
        cv2.rectangle(out_img, (int(top_left_x2), int(top_left_y2)), (int(bom_right_x2), int(bom_right_y2)), (0, 255, 0), 4)
    else:          # vertical visualization
        cv2.rectangle(out_img, (int(top_left_x2), int(top_left_y2)), (int(bom_right_x2), int(bom_right_y2)), (0, 255, 0), 4)

        # # draw bounding box grid
        # x_blk_size = 32
        # y_blk_size = 32
        # _, x_trans1, y_trans1, _, _, x_num1, y_num1 = grid.grid_img(img1[int(top_left_y1):int(bom_right_y1),
        #                                                                 int(top_left_x1):int(bom_right_x1)],
        #                                                                 int(bom_right_x1 - top_left_x1),
        #                                                                 int(bom_right_y1 - top_left_y1),
        #                                                                 x_blk_size, y_blk_size)
        #
        # vis_box.write_box_grid(out_img, x_trans1, y_trans1, x_num1, y_num1, x_blk_size, y_blk_size,
        #                        int(bom_right_x1 - top_left_x1), int(bom_right_y1 - top_left_y1),
        #                        0, 0, int(top_left_x1), int(top_left_y1))
        #
        # _, x_trans2, y_trans2, _, _, x_num2, y_num2 = grid.grid_img(img2[int(top_left_y2):int(bom_right_y2),
        #                                                                 int(top_left_x2):int(bom_right_x2)],
        #                                                                 int(bom_right_x2 - top_left_x2),
        #                                                                 int(bom_right_y2 - top_left_y2),
        #                                                                 x_blk_size, y_blk_size)
        # if flag == 0:  # horizontal visualization
        #     vis_box.write_box_grid(out_img, x_trans2, y_trans2, x_num2, y_num2, x_blk_size, y_blk_size,
        #                            int(bom_right_x2 - top_left_x2), int(bom_right_y2 - top_left_y2), cols1, 0,
        #                            int(top_left_x2), int(top_left_y2))
        # else:  # vertical visualization
        #     vis_box.write_box_grid(out_img, x_trans2, y_trans2, x_num2, y_num2, x_blk_size, y_blk_size,
        #                            int(bom_right_x2 - top_left_x2), int(bom_right_y2 - top_left_y2), 0, rows1,
        #                            int(top_left_x2), int(top_left_y2))

    cv2.imwrite(match_path, out_img)


def write_matches(img1, img2, kp1, kp2, matches, matchesMask, match_path, flag=0, show_start=0, show_end=50,
                  top_left_x1=0, top_left_y1=0, bom_right_x1=0, bom_right_y1=0,
                  top_left_x2=0, top_left_y2=0, bom_right_x2=0, bom_right_y2=0):
    valid_idx = np.where(matchesMask == True)[0]

    # 000000_10.png pair in original DMatch feature distance(weight = 0.5)
    # success_match = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    #                           24, 25, 28, 29, 30, 32, 33, 34, 35, 36, 41, 42, 45, 46, 47, 48, 49, 51, 57, 58, 65,
    #                           66, 67, 68, 69, 81, 82, 92, 93, 94, 102, 104, 105, 110, 121, 122, 123, 124, 125, 126,
    #                           138, 153, 154, 155, 156, 157, 159, 179, 180, 190, 191, 192, 193, 195, 196, 198, 199,
    #                           212, 213, 225, 228, 229, 240, 241, 242, 258, 259, 260, 261, 270, 271, 272, 289, 291,
    #                           292, 294, 296, 307, 308, 309, 323, 324, 330, 331, 332, 358, 359, 367, 380, 381, 405, 406])
    if top_left_x1 == 0 and top_left_y1 == 0 and top_left_x2 == 0 and top_left_y2 == 0:
        height, width, _ = img1.shape
        outImg = np.zeros((width * 2, height))
        outImg = cv2.drawMatches(img1, kp1, img2, kp2, matches, outImg, flags=2)
        cv2.imwrite(match_path, outImg)
    else:
        rows1, cols1, _ = img1.shape
        rows2, cols2, _ = img2.shape

        if flag == 0:  # horizontal visualization
            out_img = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
            out_img[:, 0:cols1, :] = img1
            out_img[:, cols1:cols1 + cols2, :] = img2
        else:          # vertical visualization
            out_img = np.zeros((rows1 + rows2, max([cols1, cols2]), 3), dtype='uint8')
            out_img[0:rows1, :, :] = img1
            out_img[rows1:rows1 + rows2, :, :] = img2

        for mat_idx in range(len(matches[show_start:show_end])):
            if mat_idx in valid_idx or show_end - show_start == 1:
            # if mat_idx in success_match:
                img1_idx = matches[show_start + mat_idx].queryIdx
                img2_idx = matches[show_start + mat_idx].trainIdx

                (x1, y1) = kp1[img1_idx].pt
                (x2, y2) = kp2[img2_idx].pt

                cv2.circle(out_img, (int(x1 + top_left_x1), int(y1 + top_left_y1)), 3, (255, 0, 0), 1)
                if flag == 0:  # horizontal visualization
                    cv2.circle(out_img, (int(x2 + top_left_x2) + cols1, int(y2 + top_left_y2)), 3, (255, 0, 0), 1)
                else:          # vertical visualization
                    cv2.circle(out_img, (int(x2 + top_left_x2), int(y2 + top_left_y2) + rows1), 3, (255, 0, 0), 1)

                color = np.random.randint(0, 255, (100, 3))
                if flag == 0:  # horizontal visualization
                    cv2.line(out_img, (int(x1 + top_left_x1), int(y1 + top_left_y1)), (int(x2 + top_left_x2) + cols1,
                             int(y2 + top_left_y2)), color[np.mod(mat_idx, 100)].tolist(), 1)
                else:          # vertical visualization
                    cv2.line(out_img, (int(x1 + top_left_x1), int(y1 + top_left_y1)), (int(x2 + top_left_x2),
                             int(y2 + top_left_y2) + rows1), color[np.mod(mat_idx, 100)].tolist(), 1)
                    font = cv2.FONT_HERSHEY_PLAIN
                    if np.mod(mat_idx, 2) == 0:
                        cv2.putText(out_img, str(mat_idx), (int(x1 + top_left_x1), int(y1 + top_left_y1)),
                                    font, 1, color[np.mod(mat_idx, 100)], 1, cv2.LINE_AA)
                    else:
                        cv2.putText(out_img, str(mat_idx), (int(x2 + top_left_x2), int(y2 + top_left_y2) + rows1),
                                    font, 1, color[np.mod(mat_idx, 100)], 1, cv2.LINE_AA)
            else:
                continue

        # draw bounding box
        cv2.rectangle(out_img, (int(top_left_x1), int(top_left_y1)), (int(bom_right_x1), int(bom_right_y1)),
                      (0, 255, 0), 4)
        if flag == 0:  # horizontal visualization
            cv2.rectangle(out_img, (int(top_left_x2) + cols1, int(top_left_y2)),
                          (int(bom_right_x2), int(bom_right_y2) + rows1), (0, 255, 0), 4)
        else:          # vertical visualization
            cv2.rectangle(out_img, (int(top_left_x2), int(top_left_y2) + rows1),
                              (int(bom_right_x2), int(bom_right_y2) + rows1), (0, 255, 0), 4)

        # # draw bounding box grid
        # x_blk_size = 32
        # y_blk_size = 32
        # _, x_trans1, y_trans1, _, _, x_num1, y_num1 = grid.grid_img(img1[int(top_left_y1):int(bom_right_y1),
        #                                                                 int(top_left_x1):int(bom_right_x1)],
        #                                                                 int(bom_right_x1 - top_left_x1),
        #                                                                 int(bom_right_y1 - top_left_y1),
        #                                                                 x_blk_size, y_blk_size)
        #
        # vis_box.write_box_grid(out_img, x_trans1, y_trans1, x_num1, y_num1, x_blk_size, y_blk_size,
        #                        int(bom_right_x1 - top_left_x1), int(bom_right_y1 - top_left_y1),
        #                        0, 0, int(top_left_x1), int(top_left_y1))
        #
        # _, x_trans2, y_trans2, _, _, x_num2, y_num2 = grid.grid_img(img2[int(top_left_y2):int(bom_right_y2),
        #                                                                 int(top_left_x2):int(bom_right_x2)],
        #                                                                 int(bom_right_x2 - top_left_x2),
        #                                                                 int(bom_right_y2 - top_left_y2),
        #                                                                 x_blk_size, y_blk_size)
        # if flag == 0:  # horizontal visualization
        #     vis_box.write_box_grid(out_img, x_trans2, y_trans2, x_num2, y_num2, x_blk_size, y_blk_size,
        #                            int(bom_right_x2 - top_left_x2), int(bom_right_y2 - top_left_y2), cols1, 0,
        #                            int(top_left_x2), int(top_left_y2))
        # else:  # vertical visualization
        #     vis_box.write_box_grid(out_img, x_trans2, y_trans2, x_num2, y_num2, x_blk_size, y_blk_size,
        #                            int(bom_right_x2 - top_left_x2), int(bom_right_y2 - top_left_y2), 0, rows1,
        #                            int(top_left_x2), int(top_left_y2))

        cv2.imwrite(match_path, out_img)


def write_flow2match_mask(img1, img2, flow_x, flow_y, flow_mask, width, height, vel_path, step=3,
                          top_left_x1=0, top_left_y1=0, bom_right_x1=0, bom_right_y1=0,
                          top_left_x2=0, top_left_y2=0, bom_right_x2=0, bom_right_y2=0):
    rows1, cols1, _ = img1.shape
    rows2, cols2, _ = img2.shape

    out_img = np.zeros((rows1 + rows2, max([cols1, cols2]), 3), dtype='uint8')
    out_img[0:rows1, :, :] = img1
    out_img[rows1:rows1 + rows2, :, :] = img2

    # draw bounding box
    cv2.rectangle(out_img, (int(top_left_x1), int(top_left_y1)), (int(bom_right_x1), int(bom_right_y1)), (0, 255, 0), 4)
    cv2.rectangle(out_img, (int(top_left_x2), int(top_left_y2) + rows1),
                  (int(bom_right_x2), int(bom_right_y2) + rows1), (0, 255, 0), 4)

    for j in range(0, width - step, step):
        for i in range(0, height - step, step):
            if flow_mask[i, j] == True:
                cv2.circle(out_img, (j, i), 3, (255, 0, 0), 1)
                cv2.circle(out_img, (j + int(round(flow_x[i, j])),
                           i + int(round(flow_y[i, j])) + rows1), 3, (255, 0, 0), 1)

                color = np.random.randint(0, 255, (100, 3))
                cv2.line(out_img, (j, i), (j + int(round(flow_x[i, j])),
                         i + int(round(flow_y[i, j])) + rows1),
                         color[np.mod(i + j, 100)].tolist(), 1)
    cv2.imwrite(vel_path, out_img)


def write_flow2match_overlap_mask(img, flow_x, flow_y, flow_mask, width, height, vel_path, step=3,
                                  top_left_x1=0, top_left_y1=0, bom_right_x1=0, bom_right_y1=0,
                                  top_left_x2=0, top_left_y2=0, bom_right_x2=0, bom_right_y2=0):
    out_img = img.copy()

    # draw bounding box
    cv2.rectangle(out_img, (int(top_left_x1), int(top_left_y1)), (int(bom_right_x1), int(bom_right_y1)), (0, 255, 0), 4)
    cv2.rectangle(out_img, (int(top_left_x2), int(top_left_y2)), (int(bom_right_x2), int(bom_right_y2)), (0, 255, 0), 4)

    for j in range(0, width - step, step):
        for i in range(0, height - step, step):
            if flow_mask[i, j] == True:
                cv2.circle(out_img, (j, i), 3, (255, 0, 0), 1)
                cv2.circle(out_img, (j + int(round(flow_x[i, j])), i + int(round(flow_y[i, j]))), 3, (255, 0, 0), 1)

                color = np.random.randint(0, 255, (100, 3))
                cv2.line(out_img, (j, i), (j + int(round(flow_x[i, j])), i + int(round(flow_y[i, j]))),
                                color[np.mod(i + j, 100)].tolist(), 1)
    cv2.imwrite(vel_path, out_img)