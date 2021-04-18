from sort import *
import argparse
import cv2
import time

parser = argparse.ArgumentParser(description='Camera type detection')
parser.add_argument('--image_path', type=str, help='Path for a custom image that you might want to use')
args = parser.parse_args()

if args.image_path:
    path = args.image_path
    print('Image path was defined: {}'.format(path))
else:
    path = 'images/open_2.png'
    print('No image path defined, using the default: {}'.format(path))

# Load names of classes and get random colors
classes = open('coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

img = cv2.resize(cv2.imread(path), (1280, 704))
blob = cv2.dnn.blobFromImage(img, 1/255.0, (img.shape[1], img.shape[0]), swapRB=True, crop=False)

net.setInput(blob)
t0 = time.time()
outputs = net.forward(ln)
t = time.time()

print('Time elapsed of training: {} seconds'.format(round(t-t0, 2)))

print(len(outputs))
for out in outputs:
    print(out.shape)

boxes = []
confidences = []
classIDs = []
h, w = img.shape[:2]

for output in outputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > 0.8:
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            box = [x, y, int(width), int(height)]
            boxes.append(box)
            confidences.append(float(confidence))
            classIDs.append(classID)

indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
if len(indices) > 0:
    for i in indices.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in colors[classIDs[i]]]
        import pdb; pdb.set_trace()
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

cv2.imshow('window', img)
cv2.waitKey(0)