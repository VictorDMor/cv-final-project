# cv-final-project
Computer Vision's subject final project: Detection of features in a football match using computer vision: A study case using the 2016 Rio Olympic Games gold medal match

## Run instructions

There are some test functions used by debugging, but the main.py is all you need. The video path is hard-coded but you can change it there and it will work as well.

Many arguments are used to debug, such as:

* --save_video: Save the output as a .mp4 file
* --playback: Display the playback while the program runs
* --break_on_frame: Break execution on given frame, good for debugging
* --verbose: Enable some debugging prints
* --start_from: Frame number to start from, for debugging
* --end_at: Frame number to end at, for debugging
* --track_players: Enable player tracking with YoloV3 and OpenCV DNN

You will need OpenCV, Matplotlib and Numpy installed in your machine. If you want to track the players, you are going to need
the yolov3.weights file, with the network's weights: https://pjreddie.com/media/files/yolov3.weights

For saving the video, you might need a video codec so OpenCV can write to it. For Windows, you can download the openh264-1.8.0-win64.dll at https://github.com/cisco/openh264/tree/v1.8.0

For Mac, no codecs are needed. I don't know about Linux.