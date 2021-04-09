from constants import COLOR_BOUNDARIES
import cv2
import numpy as np

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
    pixels_amount = img[mask > 0].shape[0]
    proportion = pixels_amount/(img.shape[0] * img.shape[1])
    return proportion

def convert_to_hsv(color):
    bgr = np.uint8([[color]])
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[0][0]
    
def show_image(img, window_title='Window'):
    cv2.imshow(window_title, img)
    cv2.waitKey(0)

def detect_scoreboard(image):
    # 1. Cut frames to the top left
    width = image.shape[1]
    height = image.shape[0]
    x_cut_percentage = 0.40
    y_cut_percentage = 0.125
    image = image[10:int(height*y_cut_percentage), 10:int(width*x_cut_percentage)]
    scoreboard = False

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)

    lines = cv2.HoughLines(edges,1,np.pi/180,100)
    scoreboard = True if lines is not None else False
    image = cv2.putText(image, '{}'.format(scoreboard), (int(image.shape[1] * 0.85), int(image.shape[0] * 0.92)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), thickness=2)

    # blur = cv2.GaussianBlur(image,(5, 5),0)
    # edges = cv2.Canny(blur, int(255/3), 255)
    # images = np.hstack((image, edges))
    show_image(image)

def apply_mask(image, color):
    return cv2.inRange(image, np.array(COLOR_BOUNDARIES[color][0]), np.array(COLOR_BOUNDARIES[color][1]))