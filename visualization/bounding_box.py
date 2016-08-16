import cv2

__author__ = 'Dandi Chen'

def vis_box(bboxes, output, writer, width=1920, height=1080, flag=0):
    for i in range(len(bboxes)):
        # draw bounding box
        bbox_type = bboxes[i].get('type', '')
        if bbox_type == 'true_positive':
            color = (0, 255, 0)
        elif bbox_type == 'false_positive':
            color = (0, 0, 255)
        elif bbox_type == 'false_negative':
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)

        # top_left_x, top_left_y, bom_right_x, bom_right_y
        cv2.rectangle(output, (int(width*bboxes[i]['left']), int(height*bboxes[i]['top'])),
                      (int(width*bboxes[i]['right']), int(height*bboxes[i]['bottom'])), color, 4)

        # # # draw bounding box ID
        # text = str(bboxes[i]['id'])
        # position = (bboxes[i]['left'], bboxes[i]['top'])
        # cv2.putText(output, text, position, cv2.FONT_HERSHEY_PLAIN, 4, color, 4)
        print width*bboxes[i]['left'], height*bboxes[i]['top'], width*bboxes[i]['right'], height*bboxes[i]['bottom']

        cv2.imshow('box', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # write video
    if flag == 0:
        writer.write(output)
    else:
        cv2.imwrite(writer, output)

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