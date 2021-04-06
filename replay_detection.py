import cv2
import argparse
import numpy as np
from utils import detect_scoreboard, show_image, color_to_image_proportion

# Next steps: MASK!

parser = argparse.ArgumentParser(description='Camera type detection')
parser.add_argument('--video_path', type=str, help='Path for a custom video that you might want to use')
args = parser.parse_args()

if args.video_path:
    path = args.video_path
    print('Video path was defined: {}'.format(path))
else:
    path = 'videos/torino_2-2_juventus.mp4'
    print('No video path defined, using the default: {}'.format(path))

video = cv2.VideoCapture(path)
fps = int(video.get(cv2.CAP_PROP_FPS))+1
number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frames = []
i = 0
replays = 0
replay_images = []
while(video.isOpened()):
    try:
        ret, frame = video.read()
        if ret == True:
            i += 1
            green_proportion = color_to_image_proportion(frame, 'green')
            if len(frames) > 0:
                if np.sum(cv2.subtract(frames[0], frame)) == 0:
                    replays += 1
                    replay_images.append([frames[0], frame])
                else:
                    frames = []
            frames.append(frame)
            # Detection of scoreboard in a given region of the screen, probably top-left
            # if i % 30 == 0:
                # detect_scoreboard(frame)
        else:
            break
    except cv2.error as e:
        print('Error: {}'.format(e))

print('Total of {} replays'.format(replays))
video.release()
cv2.destroyAllWindows()