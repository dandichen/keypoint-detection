import cv2
import requests
import base64
import json

import numpy as np
import matplotlib.pyplot as plt

server = "http://192.168.1.13:32795/v1/analyzer/objdetect"

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
            draw_box(detection_bboxes, output, writer)
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
    draw_box(detection_bboxes, output, writer, width, height, 1)


def draw_box(bboxes, output, writer, width=1920, height=1080, flag=0):
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

img_path = './000000_11/000000_11.png'
writer = './box/000000_11.png'
get_box_image(img_path, writer)






