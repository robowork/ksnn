# import required packages
import cv2
import argparse
import numpy as np
import os
import time
import sys
from ksnn.api import KSNN
from ksnn.types import *
import cv2 as cv
import time

GRID0 = 19
GRID1 = 38
GRID2 = 76
LISTSIZE = 21  # NUM_CLS+5
SPAN = 3
NUM_CLS = 16
MAX_BOXES = 500
OBJ_THRESH = 0.5
NMS_THRESH = 0.6

# handle command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
parser.add_argument("--library", help="Path to C static library file")
parser.add_argument("--model", help="Path to nbg file")
parser.add_argument("--picture", help="Path to input picture")
parser.add_argument("--level", help="Information printer level: 0/1/2")

args = parser.parse_args()
if args.model :
    if os.path.exists(args.model) == False:
        sys.exit('Model \'{}\' not exist'.format(args.model))
    model = args.model
else :
    sys.exit("NBG file not found !!! Please use format: --model")
if args.picture :
    if os.path.exists(args.picture) == False:
        sys.exit('Input picture \'{}\' not exist'.format(args.picture))
    picture = args.picture
else :
    sys.exit("Input picture not found !!! Please use format: --picture")
if args.library :
    if os.path.exists(args.library) == False:
        sys.exit('C static library \'{}\' not exist'.format(args.library))
    library = args.library
else :
    sys.exit("C static library not found !!! Please use format: --library")
if args.level == '1' or args.level == '2' :
    level = int(args.level)
else :
    level = 0

yolov3 = KSNN('VIM3')
print(' |---+ KSNN Version: {} +---| '.format(yolov3.get_nn_version()))

print('Start init neural network ...')
yolov3.nn_init(library=library, model=model, level=level)
print('Done.')

print('Get input data ...')
cv_img =  list()
img = cv.imread(picture, cv.IMREAD_COLOR)
cv_img.append(img)
print('Done.')

print('Start inference ...')
start = time.time()

data = yolov3.nn_inference(cv_img, platform='DARKNET', reorder='2 1 0', output_tensor=3, output_format=output_format.OUT_FORMAT_FLOAT32)
end = time.time()
print('Done. inference time: ', end - start)

input0_data = data[0]
input1_data = data[1]
input2_data = data[2]
print("data[0] output of network len = {}".format(len(input0_data)))
print("data[1] output of network len = {}".format(len(input1_data)))
print("data[2] output of network len = {}".format(len(input2_data)))

print("data[0] output of network shape = {}".format(len(input0_data.shape)))
print("data[1] output of network shape = {}".format(len(input1_data.shape)))
print("data[2] output of network shape = {}".format(len(input2_data.shape)))
#                                   3     85       13      13
input0_data = input0_data.reshape(SPAN, LISTSIZE, GRID0, GRID0)
input1_data = input1_data.reshape(SPAN, LISTSIZE, GRID1, GRID1)
input2_data = input2_data.reshape(SPAN, LISTSIZE, GRID2, GRID2)

input_data = list()
input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))
# read class names from text file
classes_list = None
with open(args.classes, 'r') as f:
    classes_list = [line.strip() for line in f.readlines()]
# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes_list), 3))

# function to get the output layer names
# in the architecture


# def get_output_layers(net):

#     layer_names = net.getLayerNames()

#     output_layers = [layer_names[i[0] - 1]
#                      for i in net.getUnconnectedOutLayers()]

#     return output_layers

# function to draw bounding box on the detected object with class name

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])
    box_wh = np.exp(input[..., 2:4])
    box_wh = box_wh * anchors

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (416, 416)
    box_xy -= (box_wh / 2.)
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs):

    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= OBJ_THRESH)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):

    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov3_post_process(input_data):

    masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
            [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input,mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, boxes, scores, classes):

    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        print('class: {}, score: {}'.format(classes_list[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(x, y, x+w, y+h))
        x *= image.shape[1]
        y *= image.shape[0]
        w *= image.shape[1]
        h *= image.shape[0]
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv.putText(image, '{0} {1:.2f}'.format(classes_list[cl], score),
                    (top, left - 6),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)




boxes, classes, scores = yolov3_post_process(input_data)

if boxes is not None:
    draw(img, boxes, scores, classes)

cv.imshow("results", img)
cv.waitKey(0)

# class yoloInference:
#     def __init__(self):
#         pass

#     def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):

#         label = str(classes[class_id])

#         color = COLORS[class_id]

#         cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

#         cv2.putText(img, label, (x-10, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


#     def detect_and_draw(self, image, net):
#         Width = image.shape[1]
#         Height = image.shape[0]
#         scale = 0.00392

#         # create input blob
#         blob = cv2.dnn.blobFromImage(
#             image, scale, (416, 416), (0, 0, 0), True, crop=False)

#         # set input blob for the network
#         net.setInput(blob)

#         # run inference through the network
#         # and gather predictions from output layers
#         outs = net.forward(get_output_layers(net))

#         # initialization
#         class_ids = []
#         confidences = []
#         boxes = []
#         conf_threshold = 0.5
#         nms_threshold = 0.4

#         # for each detetion from each output layer
#         # get the confidence, class id, bounding box params
#         # and ignore weak detections (confidence < 0.5)
#         for out in outs:
#             for detection in out:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]
#                 if confidence > 0.5:
#                     center_x = int(detection[0] * Width)
#                     center_y = int(detection[1] * Height)
#                     w = int(detection[2] * Width)
#                     h = int(detection[3] * Height)
#                     x = center_x - w / 2
#                     y = center_y - h / 2
#                     class_ids.append(class_id)
#                     confidences.append(float(confidence))
#                     boxes.append([x, y, w, h])

#         # apply non-max suppression
#         indices = cv2.dnn.NMSBoxes(
#             boxes, confidences, conf_threshold, nms_threshold)

#         # go through the detections remaining
#         # after nms and draw bounding box
#         for i in indices:
#             i = i[0]
#             box = boxes[i]
#             x = box[0]
#             y = box[1]
#             w = box[2]
#             h = box[3]

#             draw_bounding_box(image, class_ids[i], confidences[i], round(
#                 x), round(y), round(x+w), round(y+h))


# # read pre-trained model and config file
# net = cv2.dnn.readNet(args.weights, args.config)

# # read input image
# image = cv2.imread(args.image)

# start_time = time.time()
# detect_and_draw(image, net)
# use_time = time.time()-start_time
# print('cost %f seconds' % use_time)
# # display output image
# # cv2.imshow("object detection", image)

# # wait until any key is pressed
# # cv2.waitKey()

# # save output image to disk
# cv2.imwrite("output.jpg", image)

# # release resources
# cv2.destroyAllWindows()
