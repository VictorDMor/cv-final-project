import cv2
import argparse
import numpy as np
import pytesseract
from utils import show_image, color_to_image_proportion

parser = argparse.ArgumentParser(description='Camera type detection')
parser.add_argument('--image_path', type=str, help='Path for a custom image that you might want to use')
args = parser.parse_args()

if args.image_path:
    path = args.image_path
    print('Image path was defined: {}'.format(path))
else:
    path = 'images/corner_kick_1.png'
    print('No image path defined, using the default: {}'.format(path))

print('Reading and showing the original image...')
img = cv2.imread(path)
show_image(img,'Football play image - ORIGINAL')

print('=' * 80)
cut_percentage = 0.30
print('Let\'s focus just on the field part\nFor this, we will cut off the top part of the image, down to a percentage of {}%'.format(int(cut_percentage * 100)))

img_width, img_height = img.shape[1], img.shape[0]
print('Image size is {} x {} pixels'.format(img_width, img_height))

field_height_cut = int(img_height * cut_percentage)
field_region = img[field_height_cut:][:]

show_image(field_region, 'Field region')

print('=' * 80)
print('Now, we are going to check the amount of green color in the image, it is likely to be an open camera shot if the green color predominates.')
proportion_of_green = round(color_to_image_proportion(field_region, 'green') * 100, 2)
print('Proportion of green in the image is of {}%.'.format(proportion_of_green))

print('=' * 80)
print('To be sure, let\'s also check the amount of skin color in the image, it is likely to be a CLOSED camera shot if there is a good amount of skin color in the image')
proportion_of_skin = round(color_to_image_proportion(field_region, 'skin') * 100, 2)
print('Proportion of skin color in the image is of {}%.'.format(proportion_of_skin))

print('=' * 80)
if proportion_of_green < 50 and proportion_of_skin > 5:
    print('It\'s a closed camera shot')
else:
    print('It\'s an open camera shot')