import cv2
import requests
import base64
import json

server = "http://detection.app.tusimple.sd/v1/analyzer/objdetect"

def box_cordinates(bboxes, width=1920, height=1080):
    top_left_x_mat = []
    top_right_y_mat = []
    bom_left_x_mat = []
    bom_right_y_mat = []

    for i in range(len(bboxes)):
        top_left_x = width*bboxes[i]['left']
        top_right_y = height*bboxes[i]['top']
        bom_left_x = width*bboxes[i]['right']
        bom_right_y = height*bboxes[i]['bottom']

        print top_left_x, top_right_y, bom_left_x, bom_right_y

        top_left_x_mat.append(top_left_x)
        top_right_y_mat.append(top_right_y)
        bom_left_x_mat.append(bom_left_x)
        bom_right_y_mat.append(bom_right_y)

    return top_left_x_mat, top_right_y_mat, bom_left_x_mat, bom_right_y_mat

def get_box_video(video_path):
    video = cv2.VideoCapture(video_path)
    num_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_count in range(num_of_frames):
        try:
            okay, frame = video.read()
            if not okay:
                break
            binary = cv2.imencode('.jpg', frame)[1].tostring()
            encoded_string = base64.b64encode(binary)
            payload = {'image_base64': encoded_string, 'trim_detect': 0.8}
            response = requests.post(server, json=payload)
            result = json.loads(response.text)
            print result
            detection_bboxes = result['objs']
            top_left_x_mat, top_right_y_mat, bom_left_x_mat, bom_right_y_mat = box_cordinates(detection_bboxes)
            return top_left_x_mat, top_right_y_mat, bom_left_x_mat, bom_right_y_mat, len(top_left_x_mat)
        except KeyboardInterrupt:
            print "can not read video " + video_path
            exit()

def get_box_image(img, width, height):
    binary = cv2.imencode('.png', img)[1].tostring()
    encoded_string = base64.b64encode(binary)
    payload = {'image_base64': encoded_string, 'trim_detect': 0.8}
    response = requests.post(server, json=payload)
    result = json.loads(response.text)
    print result
    detection_bboxes = result['objs']
    top_left_x_mat, top_right_y_mat, bom_left_x_mat, bom_right_y_mat = box_cordinates(detection_bboxes, width, height)
    return top_left_x_mat, top_right_y_mat, bom_left_x_mat, bom_right_y_mat, len(top_left_x_mat)







