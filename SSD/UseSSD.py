import cv2
import tensorflow as tf

slim = tf.contrib.slim

from nets import ssd_vgg_300, np_methods
from preprocessing import ssd_vgg_preprocessing

import multiprocessing


def set_centers():

    print("开启线程：将object_centers放入queue")

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
    ckpt_filename = 'checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'

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

    def get_centers(rclasses, rbboxes):
        # get center location of object

        number_classes = rclasses.shape[0]
        object_centers = []
        for i in range(number_classes):
            object_center = dict()
            object_center['i'] = i
            object_center['class'] = rclasses[i]
            object_center['x'] = (rbboxes[i, 1] + rbboxes[i, 3]) / 2  # 对象中心的坐标x
            object_center['y'] = (rbboxes[i, 0] + rbboxes[i, 2]) / 2  # 对象中心的坐标y
            object_centers.append(object_center)
        return object_centers

    count = 0
    cap = cv2.VideoCapture(0)

    while count < 100:
        # 打开摄像头
        ret, img = cap.read()
        rclasses, rscores, rbboxes = process_image(img)

        '''
        classes:
        1.Aeroplanes     2.Bicycles   3.Birds       4.Boats           5.Bottles
        6.Buses          7.Cars       8.Cats        9.Chairs          10.Cows
        11.Dining tables 12.Dogs      13.Horses     14.Motorbikes     15.People
        16.Potted plants 17.Sheep     18.Sofas      19.Trains         20.TV/Monitors
        '''
        object_centers = get_centers(rclasses, rbboxes)
        # print("put object centers: " + str(object_centers))
        for object_center in object_centers:
            if object_center['class'] == 5 or object_center['class'] == 7:
                new_object_center = object_center
                q.put(new_object_center)
                count += 1
                break
    print("完成输入")
    cap.release()




def print_centers():


    print("开启线程：将object_center打印出来")
    while True:
        if q:
            print("get object center:" + str(q.get(True)))

    print("完成输出")


q = multiprocessing.Queue()

set_process = multiprocessing.Process(target=set_centers)
print_process = multiprocessing.Process(target=print_centers)

set_process.start()
print_process.start()

set_process.join()
print_process.terminate()

print("退出主线程")

