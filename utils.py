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
    
def show_image(img, window_title='Window'):
    cv2.imshow(window_title, img)
    cv2.waitKey(0)