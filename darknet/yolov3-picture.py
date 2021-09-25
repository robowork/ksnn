import numpy as np
import os
import urllib.request
from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image
import argparse
import sys
import math
from ksnn.api import KSNN
from ksnn.types import *
import cv2 as cv
import time

GRID0 = 13
GRID1 = 26
GRID2 = 52
LISTSIZE = 85
SPAN = 3
NUM_CLS = 80
MAX_BOXES = 500
OBJ_THRESH = 0.5
NMS_THRESH = 0.6

CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

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
		print('class: {}, score: {}'.format(CLASSES[cl], score))
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
		cv.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
					(top, left - 6),
					cv.FONT_HERSHEY_SIMPLEX,
					0.6, (0, 0, 255), 2)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--nb-file", help="the path for nb file")
	parser.add_argument("--so-lib", help="the path for so lib")
	parser.add_argument("--input-picture-path", help="the path for input picture")
	args = parser.parse_args()
	if args.nb_file :
		nbfile = args.nb_file
	else :
		sys.exit("nb-file not found !!! Please use format :--nb-file")
	if args.input_picture_path :
		if os.path.exists(args.input_picture_path) == False:
			sys.exit(args.input_picture_path + ' not exist')
		inputpicturepath = bytes(args.input_picture_path,encoding='utf-8')
	else :
		sys.exit(" input-picture-path not found !!! Please use format :--input-picture-path ")
	if args.so_lib :
		solib = args.so_lib
	else :
		sys.exit(" so lib not found !!! Please use format :--so-lib ")

	yolov3 = KSNN('VIM3')
	print(' |---+ KSNN Version: {} +---| '.format(yolov3.get_nn_version()))

	print('Start init neural network ...')
	yolov3.nn_init(c_lib_p = solib, nb_p = nbfile,level=0)
	print('Done.')

	print('Get input data ...')
	cv_img =  []
	img = cv.imread( args.input_picture_path, cv.IMREAD_COLOR )
	cv_img.append(img)
	print('Done.')

	print('Start inference ...')
	start = time.time()
	data = yolov3.nn_inference(cv_img, platform='DARKNET', reorder='2 1 0', output_num=3)
	end = time.time()
	print('Done. inference time: ', end - start)

	input0_data = data[0]
	input1_data = data[1]
	input2_data = data[2]

	input0_data = input0_data.reshape(SPAN, LISTSIZE, GRID0, GRID0)
	input1_data = input1_data.reshape(SPAN, LISTSIZE, GRID1, GRID1)
	input2_data = input2_data.reshape(SPAN, LISTSIZE, GRID2, GRID2)

	input_data = list()
	input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
	input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
	input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

	boxes, classes, scores = yolov3_post_process(input_data)

	if boxes is not None:
		draw(img, boxes, scores, classes)

	cv.imshow("results", img)
	cv.waitKey(0)
