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
