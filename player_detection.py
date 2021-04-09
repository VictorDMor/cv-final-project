import cv2
import argparse
import numpy as np
from utils import show_image, color_to_image_proportion, apply_mask

parser = argparse.ArgumentParser(description='Player detetction')
parser.add_argument('--image_path', type=str, help='Path for a custom image that you might want to use')
parser.add_argument('--show_images', help='Show whether to show the images during this script\'s execution', action='store_true')
parser.add_argument('--team_1_color', type=str, help='Team 1 jersey color')
parser.add_argument('--team_2_color', type=str, help='Team 2 jersey color')
args = parser.parse_args()

if args.image_path:
    path = args.image_path
    print('Image path was defined: {}'.format(path))
else:
    path = 'images/penalty_1.png'
    team_1_color = 'yellow'
    team_2_color = 'red'
    print('No image path defined, using the default: {}'.format(path))

print('Reading and showing the original image...')
img = cv2.imread(path)

if args.show_images:
    show_image(img,'Football play image - ORIGINAL')

print('=' * 80)
cut_percentage = 0.30
print('Let\'s focus just on the field part\nFor this, we will cut off the top part of the image, down to a percentage of {}%'.format(int(cut_percentage * 100)))

img_width, img_height = img.shape[1], img.shape[0]
print('Image size is {} x {} pixels'.format(img_width, img_height))

field_height_cut = int(img_height * cut_percentage)
field_region = img[field_height_cut:][:]

if args.show_images:
    show_image(field_region, 'Field region')

print('The possible focus of this image is concentrated at the center of it')
print('=' * 80)
print('Let\'s use a mask to identify the players by their jersey color')

team_1_mask = apply_mask(field_region, team_1_color)
team_2_mask = apply_mask(field_region, team_2_color)

masks = cv2.bitwise_or(team_1_mask, team_2_mask)

print('Masks are done! Now, we are going to create a window so we can calculate the ratio of masks in each window, \
    the higher ratio,\n the higher a chance of being a player!')

gray = cv2.cvtColor(field_region, cv2.COLOR_BGR2GRAY)
ret, threshed_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Window has a size of 70x120
y_window = 70
x_window = 120

for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    # draw a green rectangle to visualize the bounding rect
    if (w * h) >= (y_window * x_window * 0.4) and w * h < (y_window * x_window * 2):
        mask_cut = masks[y:y+h, x:x+w]
        if np.sum(mask_cut) > 0:
            cv2.rectangle(field_region, (x, y), (x+w, y+h), (0, 255, 0), 2)

print(len(contours))
show_image(field_region, "contours")



# for i in range(len(masks)):
#     for j in range(len(masks[i])):
#         color
