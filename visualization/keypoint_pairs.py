import cv2

__author__ = 'Dandi Chen'

def vis_pt_pairs(img1, img2, kp1, kp2, top_left_x1=0, top_left_y1=0, top_left_x2=0, top_left_y2=0):
    shown_img1 = img1
    shown_img2 = img2

    if top_left_x1 == 0 and top_left_y1 == 0 and top_left_x2 == 0 and top_left_y2 == 0:
        shown_img1 = cv2.drawKeypoints(img1, kp1, shown_img1, color=(0, 255, 0), flags=0)
        cv2.imshow('image1', shown_img1)
        cv2.waitKey(0)

        shown_img2 = cv2.drawKeypoints(img2, kp2, shown_img2, color=(0, 255, 0), flags=0)
        cv2.imshow('image2', shown_img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        if top_left_x1 != 0 or top_left_y1 != 0:
            for kp_idx1 in kp1:
                (x, y) = kp_idx1.pt
                x += top_left_x1
                y += top_left_y1
                cv2.circle(shown_img1, (int(x), int(y)), 3, color=(0, 255, 0))
            cv2.imshow('image1', shown_img1)
            cv2.waitKey(0)

        if top_left_x2 != 0 or top_left_y2 != 0:
            for kp_idx2 in kp2:
                (x, y) = kp_idx2.pt
                x += top_left_x2
                y += top_left_y2
                cv2.circle(shown_img2, (int(x), int(y)), 3, color=(0, 255, 0))
            cv2.imshow('image2', shown_img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def write_pt_pairs(img1, img2, kp1, kp2, kp_path1, kp_path2, top_left_x1=0, top_left_y1=0,
                   top_left_x2=0, top_left_y2=0):
    shown_img1 = img1
    shown_img2 = img2

    if top_left_x1 == 0 and top_left_y1 == 0 and top_left_x2 == 0 and top_left_y2 == 0:
        shown_img1 = cv2.drawKeypoints(img1, kp1, shown_img1, color=(0, 255, 0), flags=0)
        cv2.imwrite(kp_path1, shown_img1)

        shown_img2 = cv2.drawKeypoints(img2, kp2, shown_img2, color=(0, 255, 0), flags=0)
        cv2.imwrite(kp_path2, shown_img2)
    else:
        if top_left_x1 != 0 or top_left_y1 != 0:
            for kp_idx1 in kp1:
                (x, y) = kp_idx1.pt
                x += top_left_x1
                y += top_left_y1
                cv2.circle(shown_img1, (int(x), int(y)), 2, color=(0, 255, 0))
            cv2.imwrite(kp_path1, shown_img1)

        if top_left_x2 != 0 or top_left_y2 != 0:
            for kp_idx2 in kp2:
                (x, y) = kp_idx2.pt
                x += top_left_x2
                y += top_left_y2
                cv2.circle(shown_img2, (int(x), int(y)), 2, color=(0, 255, 0))
            cv2.imwrite(kp_path2, shown_img2)