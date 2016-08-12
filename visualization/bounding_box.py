import cv2

def vis_boxes(img, top_left_x_mat, top_right_y_mat, bom_left_x_mat, bom_right_y_mat):
    for idx in range(len(top_left_x_mat)):
        cv2.rectangle(img, (int(top_left_x_mat[idx]), int(top_right_y_mat[idx])),
                      (int(bom_left_x_mat[idx]), int(bom_right_y_mat[idx])), (0, 255, 0), 4)
    cv2.imshow('box grid', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def write_boxes(img, box_path, top_left_x_mat, top_right_y_mat, bom_left_x_mat, bom_right_y_mat):
    for idx in range(len(top_left_x_mat)):
        cv2.rectangle(img, (int(top_left_x_mat[idx]), int(top_right_y_mat[idx])),
                      (int(bom_left_x_mat[idx]), int(bom_right_y_mat[idx])), (0, 255, 0), 4)
    cv2.imwrite(box_path, img)

def vis_box_grid(img, x_trans, y_trans, x_num, y_num, x_blk_size, y_blk_size, width, height,
                  img_x_trans=0, img_y_trans=0, box_x_trans=0, box_y_trans=0):
    for i in range(x_num - 1):
        cv2.line(img, (x_trans[i] + img_x_trans + box_x_trans, y_trans[0] + img_y_trans + box_y_trans),
                 (x_trans[i] + img_x_trans + box_x_trans, height + img_y_trans + box_y_trans), (0, 255, 0))
    cv2.line(img, (x_trans[x_num - 2] + x_blk_size + img_x_trans + box_x_trans, y_trans[0] + img_y_trans + box_y_trans),
             (x_trans[x_num - 2] + x_blk_size + img_x_trans + box_x_trans, height + img_y_trans + box_y_trans), (0, 255, 0))

    for j in range(y_num - 1):
        cv2.line(img, (x_trans[0] + img_x_trans + box_x_trans, y_trans[j * (x_num - 1)] + img_y_trans + box_y_trans),
                 (width + img_x_trans + box_x_trans, y_trans[j * (x_num - 1)] + img_y_trans + box_y_trans), (0, 255, 0))
    cv2.line(img, (x_trans[0] + img_x_trans + box_x_trans, y_trans[(y_num - 2) * (x_num - 1)] + y_blk_size + img_y_trans + box_y_trans),
             (width + img_x_trans + box_x_trans, y_trans[(y_num - 2) * (x_num - 1)] + y_blk_size + img_y_trans + box_y_trans), (0, 255, 0))

    cv2.imshow('box grid', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_box_grid(img, x_trans, y_trans, x_num, y_num, x_blk_size, y_blk_size, width, height,
                  img_x_trans=0, img_y_trans=0, box_x_trans=0, box_y_trans=0):
    for i in range(x_num - 1):
        cv2.line(img, (x_trans[i] + img_x_trans + box_x_trans, y_trans[0] + img_y_trans + box_y_trans),
                 (x_trans[i] + img_x_trans + box_x_trans, height + img_y_trans + box_y_trans), (0, 255, 0))
    cv2.line(img, (x_trans[x_num - 2] + x_blk_size + img_x_trans + box_x_trans, y_trans[0] + img_y_trans + box_y_trans),
             (x_trans[x_num - 2] + x_blk_size + img_x_trans + box_x_trans, height + img_y_trans + box_y_trans), (0, 255, 0))

    for j in range(y_num - 1):
        cv2.line(img, (x_trans[0] + img_x_trans + box_x_trans, y_trans[j * (x_num - 1)] + img_y_trans + box_y_trans),
                 (width + img_x_trans + box_x_trans, y_trans[j * (x_num - 1)] + img_y_trans + box_y_trans), (0, 255, 0))
    cv2.line(img, (x_trans[0] + img_x_trans + box_x_trans, y_trans[(y_num - 2) * (x_num - 1)] + y_blk_size + img_y_trans + box_y_trans),
             (width + img_x_trans + box_x_trans, y_trans[(y_num - 2) * (x_num - 1)] + y_blk_size + img_y_trans + box_y_trans), (0, 255, 0))
