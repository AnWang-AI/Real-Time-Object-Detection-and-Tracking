import sys
from os.path import realpath, dirname, join

import cv2
import numpy as np
import tensorflow as tf
import torch

from net import SiamRPNvot
from nets import ssd_vgg_300, np_methods
from preprocessing import ssd_vgg_preprocessing
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import cxy_wh_2_rect

# -------------------------------  #


'''
        classes:
        1.Aeroplanes     2.Bicycles   3.Birds       4.Boats           5.Bottles
        6.Buses          7.Cars       8.Cats        9.Chairs          10.Cows
        11.Dining tables 12.Dogs      13.Horses     14.Motorbikes     15.People
        16.Potted plants 17.Sheep     18.Sofas      19.Trains         20.TV/Monitors
    '''
detect_class = 15

slim = tf.contrib.slim

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
# ckpt_filename = 'checkpoints/ssd_300_vgg.ckpt'
ckpt_filename = 'SSD/checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'

isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


def get_bboxes(rclasses, rbboxes):
    # get center location of object

    number_classes = rclasses.shape[0]
    object_bboxes = []
    for i in range(number_classes):
        object_bbox = dict()
        object_bbox['i'] = i
        object_bbox['class'] = rclasses[i]
        object_bbox['y_min'] = rbboxes[i, 0]
        object_bbox['x_min'] = rbboxes[i, 1]
        object_bbox['y_max'] = rbboxes[i, 2]
        object_bbox['x_max'] = rbboxes[i, 3]
        object_bboxes.append(object_bbox)
    return object_bboxes


# load net
net = SiamRPNvot()
net.load_state_dict(torch.load(join(realpath(dirname(__file__)), 'DaSiamRPN-master/code/SiamRPNVOT.model')))

net.eval()

# open video capture
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Could not open video")
    sys.exit()

index = True
while index:
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Define an initial bounding box
    height = frame.shape[0]
    width = frame.shape[1]
    rclasses, rscores, rbboxes = process_image(frame)
    bboxes = get_bboxes(rclasses, rbboxes)
    for bbox in bboxes:
        if bbox['class'] == detect_class:
            print(bbox)
            ymin = int(bbox['y_min'] * height)
            xmin = int((bbox['x_min']) * width)
            ymax = int(bbox['y_max'] * height)
            xmax = int((bbox['x_max']) * width)
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            h = ymax - ymin
            w = xmax - xmin
            new_bbox = (cx, cy, w, h)
            print(new_bbox)
            index = False
            break

# tracker init
target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
state = SiamRPN_init(frame, target_pos, target_sz, net)

# tracking and visualization
toc = 0
count_number = 0

while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Start timer
    tic = cv2.getTickCount()

    # Update tracker
    state = SiamRPN_track(state, frame)  # track
    # print(state)

    toc += cv2.getTickCount() - tic

    if state:

        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        res = [int(l) for l in res]
        cv2.rectangle(frame, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
        count_number += 1

        if (not state) or count_number % 40 == 3:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            index = True
            while index:
                ok, frame = video.read()
                rclasses, rscores, rbboxes = process_image(frame)
                bboxes = get_bboxes(rclasses, rbboxes)
                for bbox in bboxes:
                    if bbox['class'] == detect_class:
                        ymin = int(bbox['y_min'] * height)
                        xmin = int(bbox['x_min'] * width)
                        ymax = int(bbox['y_max'] * height)
                        xmax = int(bbox['x_max'] * width)
                        cx = (xmin + xmax) / 2
                        cy = (ymin + ymax) / 2
                        h = ymax - ymin
                        w = xmax - xmin
                        new_bbox = (cx, cy, w, h)
                        target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
                        state = SiamRPN_init(frame, target_pos, target_sz, net)

                        p1 = (int(xmin), int(ymin))
                        p2 = (int(xmax), int(ymax))
                        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

                        index = 0

                        break

    cv2.imshow('SSD+SiamRPN', frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
