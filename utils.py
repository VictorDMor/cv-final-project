from constants import SCOREBOARD_MEAN, REPLAY_TRANSCRIPTION, TRANSCRIPTION, OPEN_CLOSE_FRAMES
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def color_to_image_proportion(img, color):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color == 'green':
        lower = np.array([30, 40, 40])
        higher = np.array([70, 255, 255])
    elif color == 'skin':
        bgr_1 = np.uint8([[[104, 139, 177]]])
        hsv_bgr_1 = cv2.cvtColor(bgr_1, cv2.COLOR_BGR2HSV)
        lower = np.array([hsv_bgr_1[0][0][0]-10, 100, 100])
        higher = np.array([hsv_bgr_1[0][0][0]+10, 255, 255])
    mask = cv2.inRange(hsv, lower, higher)
    proportion = cv2.countNonZero(mask)/(img.shape[0] * img.shape[1])
    return proportion
    
def show_image(img, window_title='Window'):
    cv2.imshow(window_title, img)
    cv2.waitKey(0)

def evaluate_hit(result, frame_number):
    for event in TRANSCRIPTION:
        if result == False:
            if frame_number in range(event['from'], event['to']+1):
                outcome = 'hit'
                break
            outcome = 'miss'
        else:
            if frame_number in range(event['from'], event['to']+1):
                outcome = 'miss'
                break
            outcome = 'hit'
    return outcome

def evaluate_replay_hit(result, frame_number):
    for replay in REPLAY_TRANSCRIPTION:
        if result == True:
            if frame_number in range(replay['from'], replay['to']+1):
                outcome = 'hit'
                break
            outcome = 'miss'
    return outcome

def count_open_close_frames():
    open_frames = OPEN_CLOSE_FRAMES.count(False)
    close_frames = OPEN_CLOSE_FRAMES.count(True)
    return open_frames, close_frames

def detect_scoreboard(image, debug=False):
    # 1. Cut frames to the top left
    scoreboard = image[25:140, 130:280]
    scoreboard_mean = np.array(SCOREBOARD_MEAN)
    # 2. Get color mean
    offset = 15
    mean = np.array(cv2.mean(scoreboard)[0:3]).astype(np.uint8)
    if np.linalg.norm(mean - scoreboard_mean) < offset:
        return True
    return False

def skin_lines(axis):
    '''return a list of lines for a give axis'''
    line1 = 1.5862  * axis + 20
    line2 = 0.3448  * axis + 76.2069
    line3 = -1.005 * axis + 234.5652
    line4 = -1.15   * axis + 301.75
    line5 = -2.2857 * axis + 432.85
    return [line1,line2,line3,line4,line5]

def detect_skin(img, test=False):
    ''' Source: https://github.com/Harmouch101/Face-Recogntion-Detection/blob/master/skin_seg.py '''
    b_frame, g_frame, r_frame = cv2.split(img)
    bgr_max = np.maximum.reduce([b_frame, g_frame, r_frame])
    bgr_min = np.minimum.reduce([b_frame, g_frame, r_frame])

    # Uniform daylight rule
    rule_1 = np.logical_and.reduce([
        r_frame > 95, g_frame > 40, b_frame > 20,
        bgr_max - bgr_min > 15,
        abs(r_frame - g_frame) > 15,
        r_frame > g_frame, r_frame > b_frame
    ])

    # Under flashlight or lateral daylight
    rule_2 = np.logical_and.reduce([
        r_frame > 220, g_frame > 210, b_frame > 170,
        abs(r_frame - g_frame) <= 15,
        r_frame > b_frame, g_frame > b_frame
    ])

    rgb_rule = np.logical_or(rule_1, rule_2)

    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_frame, cr_frame, cb_frame = [ycrcb_img[..., ycrcb] for ycrcb in range(3)]

    line1, line2, line3, line4, line5 = skin_lines(cb_frame)

    ycrcb_rule = np.logical_and.reduce([
        line1 - cr_frame >= 0,
        line2 - cr_frame <= 0,
        line3 - cr_frame <= 0,
        line4 - cr_frame >= 0,
        line5 - cr_frame >= 0
    ])

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = [hsv_img[...,i] for i in range(3)]
    hsv_rule = np.logical_or(hue < 50, hue > 150)

    skin_mask = np.logical_and.reduce([
        rgb_rule,
        ycrcb_rule,
        hsv_rule
    ])

    binary = skin_mask.astype(np.uint8)
    total_area = binary.shape[0] * binary.shape[1]
    ratio = round(cv2.countNonZero(binary)/total_area, 3)

    if test:
        import pdb; pdb.set_trace()
    
    return ratio

def load_yolo():
    classes = open('coco.names').read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, colors, ln