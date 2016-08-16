import cv2
import requests
import base64
import json

import visualization.bounding_box as vis_box

__author__ = 'Dandi Chen'

server = "http://detection.app.tusimple.sd/v1/analyzer/objdetect"

def get_box_video(video_path, output_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(output_path, fourcc, 30, (1920, 1080))

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
            output = frame.copy()
            vis_box.vis_box(detection_bboxes, output, writer)
            # cv2.imshow('frame', output)
            # cv2.waitKey(0)
        except KeyboardInterrupt:
            print "can not read video " + video_path
            exit()

def get_box_image(img_path, writer):
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    binary = cv2.imencode('.png', img)[1].tostring()
    encoded_string = base64.b64encode(binary)
    payload = {'image_base64': encoded_string, 'trim_detect': 0.8}
    response = requests.post(server, json=payload)
    result = json.loads(response.text)
    print result
    detection_bboxes = result['objs']
    output = img.copy()
    vis_box.vis_box(detection_bboxes, output, writer, width, height, 1)

    box_edge = []
    for i in range(len(detection_bboxes)):
        # top_left_x, top_left_y, bom_right_x, bom_right_y
        box_edge.append([width * detection_bboxes[i]['left'], height * detection_bboxes[i]['top'],
            width * detection_bboxes[i]['right'], height * detection_bboxes[i]['bottom']])
    return box_edge




