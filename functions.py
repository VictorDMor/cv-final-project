import cv2
import argparse
import numpy as np
import pdb
from utils import detect_scoreboard, detect_skin, color_to_image_proportion, show_image

'''
TODO: Imagens open da torcida não funciona
'''

def camera_detection(img, debug=False):
    cut_percentage = 0.175
    fan_cut_percentage = 0.2
    img_width, img_height = img.shape[1], img.shape[0]
    fan_cut = img[:int(img_height * fan_cut_percentage),:]
    proportion_of_green_fans = round(color_to_image_proportion(fan_cut, 'green') * 100, 2)
    focus_cut_x = int(img_width * cut_percentage)
    focus_cut_y = int(img_height * cut_percentage)

    if proportion_of_green_fans < 10:
        focus_cut_y_top = int(img_width * (cut_percentage + (fan_cut_percentage/2)))
    else:
        focus_cut_y_top = focus_cut_y

    focus_region = img[focus_cut_y_top:img_height - focus_cut_y, focus_cut_x:img_width - focus_cut_x]

    proportion_of_green = round(color_to_image_proportion(focus_region, 'green') * 100, 2)
    if debug: print('Proportion of green in the image is of {}%.'.format(proportion_of_green))

    proportion_of_skin = round(detect_skin(focus_region) * 100, 2)
    if debug: print('Proportion of skin color in the image is of {}%.'.format(proportion_of_skin))

    if debug: 
        print('=' * 80)
    if proportion_of_green < 65 or proportion_of_skin >= 1.8:
        close_camera = True
    else:
        close_camera = False

    return close_camera

def replay_detection(img):
    has_scoreboard = detect_scoreboard(img, debug=True)
    if not has_scoreboard:
        return True
    return False

def player_detection(img, net, classes, colors, layer_names):
    # Load names of classes and get random colors
    img = cv2.resize(img, (1280, 704))
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (img.shape[1], img.shape[0]), swapRB=True, crop=False)

    net.setInput(blob)
    outputs = net.forward(layer_names)

    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.8:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            area = (x+w)*(y+h)
            if area > 200000:
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    return cv2.resize(img, (1280, 720))