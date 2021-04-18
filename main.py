from functions import camera_detection, player_detection, replay_detection
from utils import detect_skin, load_yolo, evaluate_hit, count_misses
import argparse
import random
import cv2
import pdb

'''
TODO se possível: 
    Tirar pessoas de fora do jogo da detecção de pessoas
    Fazer bounding boxes diferentes para cada equipe e o árbitro
    Consertar a visão da torcida
    Otimizar o tempo para os frames
'''

parser = argparse.ArgumentParser(description='Camera type detection')
parser.add_argument('--save_video', help='Save the output as a .mp4 file', action='store_true')
parser.add_argument('--playback', help='Display the playback', action='store_true')
parser.add_argument('--break_on_frame', type=int, help='Break execution on given frame, good for debugging')
parser.add_argument('--verbose', help='Enable some debugging prints', action='store_true')
parser.add_argument('--start_from', type=int, help='Frame number to start from, for debugging', default=0)
parser.add_argument('--end_at', type=int, help='Frame number to end at, for debugging')
parser.add_argument('--track_players', help='Enable player tracking with YoloV3 and OpenCV DNN', action='store_true')
args = parser.parse_args()

save_video = True if args.save_video else False
playback = True if args.playback else False
frame_sequences = [1, 139, 1358, 1721, 2026, 2510]
detect_movement = False

if __name__ == '__main__':
    video = cv2.VideoCapture('videos/cv-brager2016_Trim.mp4')
    frames = []
    last_frame = []
    number_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    net, classes, colors, layer_names = load_yolo()
    i = 0
    open_hits = 0
    close_hits = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret == True:
            i += 1
            if i >= args.start_from:
                if args.verbose: print('Playing video and detecting specifics... Frame {} - {}%'.format(i, round(i/number_of_frames * 100, 2)))
                
                if args.track_players:
                    if i in frame_sequences:
                        detect_movement = not detect_movement
                    
                    if detect_movement:
                        frame = player_detection(frame, net, classes, colors, layer_names)
                
                close_camera = camera_detection(frame, debug=False)
                if close_camera:
                    final_frame = cv2.putText(frame, 'Close camera shot: Frame {}'.format(i), (int(frame.shape[1] * 0.65), int(frame.shape[0] * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)
                    close_hits += evaluate_hit(close_camera, i-1)
                else:
                    final_frame = cv2.putText(frame, 'Open camera shot: Frame {}'.format(i), (int(frame.shape[1] * 0.65), int(frame.shape[0] * 0.9)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), thickness=2)
                    open_hits += evaluate_hit(close_camera, i-1)
                
                replay = replay_detection(frame)
                if replay:
                    final_frame = cv2.putText(final_frame, 'Replay', (int(frame.shape[1] * 0.65), int(frame.shape[0] * 0.85)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), thickness=2)

                if args.break_on_frame:
                    if i == args.break_on_frame: break
                
                if args.end_at:
                    if i == args.end_at:
                        break

                if playback:
                    cv2.imshow('Video', final_frame)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    if key == ord('p'):
                        cut_percentage = 0.175
                        img_width, img_height = final_frame.shape[1], final_frame.shape[0]
                        focus_cut_y = int(img_height * cut_percentage)
                        focus_cut_x = int(img_width * cut_percentage)
                        focus_region = frame[focus_cut_y:img_height - focus_cut_y, focus_cut_x:img_width - focus_cut_x]
                        cv2.imshow('adasd', focus_region)
                        cv2.waitKey(-1)
                    if args.end_at:
                        if i == args.end_at:
                            cut_percentage = 0.175
                            img_width, img_height = final_frame.shape[1], final_frame.shape[0]
                            focus_cut_y = int(img_height * cut_percentage)
                            focus_cut_x = int(img_width * cut_percentage)
                            focus_region = frame[focus_cut_y:img_height - focus_cut_y, focus_cut_x:img_width - focus_cut_x]
                            cv2.imshow('adasd', focus_region)
                            cv2.waitKey(-1)
                            break

                if save_video: frames.append(final_frame)
        else:
            break
    if save_video:
        if args.end_at:
            final_video = cv2.VideoWriter('tmp/brager2016_opencv_{}-{}.mp4'.format(args.start_from, args.end_at), cv2.VideoWriter_fourcc(*'avc1'), video.get(cv2.CAP_PROP_FPS), (frames[0].shape[1], frames[0].shape[0]))
        else:
            final_video = cv2.VideoWriter('tmp/brager2016_opencv_{}.mp4'.format(args.start_from), cv2.VideoWriter_fourcc(*'avc1'), video.get(cv2.CAP_PROP_FPS), (frames[0].shape[1], frames[0].shape[0]))
        for video_frame in frames:
            final_video.write(video_frame)
        video.release()
    if args.verbose:
        open_misses, close_misses = count_misses(open_hits, close_hits)
        open_precision = round(open_hits/(open_hits + open_misses) * 100, 2)
        close_precision = round(close_hits/(close_hits + close_misses) * 100, 2)
        print('Open hits: {}'.format(open_hits))
        print('Close hits: {}'.format(close_hits))
        print('Open misses: {}'.format(open_misses))
        print('Close misses: {}'.format(close_misses))
        print('Open hit precision: {}'.format(open_precision))
        print('Close hit precision: {}'.format(close_precision))
        print('Overall precision: {}'.format(round((open_precision + close_precision)/2, 2)))