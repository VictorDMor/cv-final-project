import argparse
import cv2
from functions import camera_detection, player_movement
from utils import show_image, detect_scoreboard, detect_skin

parser = argparse.ArgumentParser(description='Camera type detection')
parser.add_argument('function', type=str, help='Choose which function to test')
parser.add_argument('--image_path', type=str, help='Path for a custom image that you might want to use')
args = parser.parse_args()

if args.image_path:
    path = args.image_path
    print('Image path was defined: {}'.format(path))
else:
    path = 'images/corner_kick_1.png'
    print('No image path defined, using the default: {}'.format(path))

img = cv2.imread(path)
if args.function == 'detect_scoreboard':
    detect_scoreboard(img)

if args.function == 'detect_skin':
    detect_skin(img, test=True)

if args.function == 'player_movement':
    player_movement(img1, img2)

if args.function == 'camera_detection':
    camera_detection(img, debug=True)