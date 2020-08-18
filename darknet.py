#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
Darknet pytno interface
"""

import argparse
from ctypes import c_char_p, c_float, c_int, c_void_p, pointer
from ctypes import CDLL, POINTER, RTLD_GLOBAL, Structure
import math
import random


import cv2
import numpy as np
from PIL import Image


from yolo_result import YoloResult


def sample(probs):
    """
    sample function
    """

    probs_sum = sum(probs)
    probs = [a/probs_sum for a in probs]
    rand = random.uniform(0, 1)
    for idx, prob in enumerate(probs):
        rand = rand - prob
        if rand <= 0:
            return idx
    return len(probs)-1


def c_array(ctype, values):
    """
    convert to carray from value
    """

    arr = (ctype*len(values))()
    arr[:] = values
    return arr


class BOX(Structure):
    """
    Structure definision of BBOX
    """

    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    """
    Structure definision of DETECTION
    """

    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    """
    Structure definision of IMAGE
    """

    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    """
    Structure definision of META DATA
    """

    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


class Darknet(object):
    """
    Darknet class
    """

    def __init__(self,
                 libfilepath,
                 cfgfilepath,
                 datafilepath,
                 weightsfilepath,
                 logger=None):
        """
        Initialize metod
        """

        self.libfilepath = libfilepath
        self.cfgfilepath = cfgfilepath
        self.datafilepath = datafilepath
        self.weightsfilepath = weightsfilepath
        self.logger = logger
        self.net = None
        self.meta = None

        self.colors = [
                       [1, 0, 1], [0, 0, 1],
                       [0, 1, 1], [0, 1, 0],
                       [1, 1, 0], [1, 0, 0]
                      ]

        self.lib = CDLL(self.libfilepath, RTLD_GLOBAL)
        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int

        self.predict = self.lib.network_predict
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        self.set_gpu = self.lib.cuda_set_device
        self.set_gpu.argtypes = [c_int]

        self.make_image = self.lib.make_image
        self.make_image.argtypes = [c_int, c_int, c_int]
        self.make_image.restype = IMAGE

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [
            c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.make_network_boxes = self.lib.make_network_boxes
        self.make_network_boxes.argtypes = [c_void_p]
        self.make_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.free_ptrs = self.lib.free_ptrs
        self.free_ptrs.argtypes = [POINTER(c_void_p), c_int]

        self.network_predict = self.lib.network_predict
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.reset_rnn = self.lib.reset_rnn
        self.reset_rnn.argtypes = [c_void_p]

        self.load_net = self.lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.do_nms_obj = self.lib.do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = self.lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.letterbox_image = self.lib.letterbox_image
        self.letterbox_image.argtypes = [IMAGE, c_int, c_int]
        self.letterbox_image.restype = IMAGE

        self.load_meta = self.lib.get_metadata
        self.lib.get_metadata.argtypes = [c_char_p]
        self.lib.get_metadata.restype = METADATA

        self.load_image = self.lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.rgbgr_image = self.lib.rgbgr_image
        self.rgbgr_image.argtypes = [IMAGE]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

    def load_conf(self):
        """
        loading network from weights file
        """
        # net = load_net(net_file.encode('utf-8'), weight_file.encode('utf-8'), 0)
        # meta = load_meta(data_file.encode('utf-8'))
        print(self.cfgfilepath.decode().encode('utf-8'))
        self.net = self.load_net(self.cfgfilepath,
                                 self.weightsfilepath,
                                 0)
        self.meta = self.load_meta(self.datafilepath)

    def load_image(imagefilepath):
        """
        loading image
        """
        image = self.load_image(imagefilepath, 0, 0)
        return image

    def convert_to_yolo_img(self, img):
        """
        converting from rgb(PIL) image class to yolo image class
        """

        img = img / 255.0
        h, w, c = img.shape
        img = img.transpose(2, 0, 1)
        x = img.shape
        img = img.reshape((w*h*c))

        outimg = self.make_image(w, h, c)
        data = c_array(c_float, img)
        outimg.data = data
        self.rgbgr_image(outimg)
        return outimg


    def get_color(self, c, x, max_num):
        """
        Getting color based on yolo src
        """

        ratio = 5*(float(x)/max_num)
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio -= i
        r = (1 - ratio) * self.colors[i][c] + ratio*self.colors[j][c]
        return int(255*r)

    def detect_orgin(self, image_path, thresh=.5, hier_thresh=.5, nms=.45):
        im = self.load_image(image_path, 0, 0)
        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(self.net, im)
        dets = self.get_network_boxes(self.net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if (nms): self.do_nms_obj(dets, num, self.meta.classes, nms)

        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((self.meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(im)
        self.free_detections(dets, num)
        return res

    def detect_v1(self, image_path, thresh=.5, hier_thresh=.5, nms=.45):
        im = self.load_image(image_path, 0, 0)
        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(self.net, im)
        dets = self.get_network_boxes(self.net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if (nms): self.do_nms_obj(dets, num, self.meta.classes, nms)

        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((self.meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(im)
        self.free_detections(dets, num)
        return res

    def detect(self, image, thresh=.5, hier_thresh=.5, nms=.45):
        """
        detecting
        """

        image = self.convert_to_yolo_img(image)

        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(self.net, image)
        if self.logger is not None:
            self.logger.info("=========classify_img===========")
            self.logger.info(self.classify_img(image))
        dets = self.get_network_boxes(
            self.net, image.w, image.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if nms:
            self.do_nms_obj(dets, num, self.meta.classes, nms)

        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if dets[j].prob[i] > 0:
                    bbox = dets[j].bbox
                    res.append(
                        YoloResult(
                                   i,self.meta.names[i],
                                   dets[j].prob[i],
                                   (
                                    bbox.x, bbox.y,
                                    bbox.w, bbox.h
                                   )
                                  )
                              )
        res = sorted(res, key=lambda x: x.score, reverse=True)
        # self.free_image(image)
        self.free_detections(dets, num)
        return res


    def draw_detections(self, img, yolo_results):
        """
        drawing result of yolo
        """

        _, height, _ = img.shape
        for yolo_result in yolo_results:
            class_index = yolo_result.class_index
            obj_name = yolo_result.obj_name
            x = yolo_result.x_min
            y = yolo_result.y_min
            w = yolo_result.width
            h = yolo_result.height

            offset = class_index * 123457 % self.meta.classes

            red = self.get_color(2, offset, self.meta.classes)
            green = self.get_color(1, offset, self.meta.classes)
            blue = self.get_color(0, offset, self.meta.classes)
            box_width = int(height * 0.006)
            cv2.rectangle(img, (int(x), int(y)), (int(x+w)+1, int(y+h)+1), (red, green, blue), box_width)
            cv2.putText(
                        img, obj_name,
                        (int(x) -2, int(y) -5),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1.2, (red, green, blue),
                        2, cv2.LINE_AA
                       )

        return img

    def classify(self, imagefilepath):
        """
        classify
        """

        image = self.load_image(imagefilepath)
        return self.classify_img(image)

    def classify_img(self, image):
        """
        classify
        """

        out = self.predict_image(self.net, image)
        res = []
        for i in range(self.meta.classes):
            res.append((self.meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res


def importargs():
    """
    Get arguments
    """

    parser = argparse.ArgumentParser("This Darknet python sample")
    parser.add_argument("--libfilepath", "-lf",
                        default="/mnt/workspcae/pycharm/Git/data-augmentation/libdarknet.so",
                        type=str,
                        help="filepath of libdarknet.default:./libdarknet.so")

    parser.add_argument("--cfgfilepath", "-cf",
                        default="/mnt/workspcae/caffe/test_model/2020-8-17/cfg/yolov3-voc.cfg",
                        type=str,
                        help="cfgfilepath.default ./cfg/yolov3.cfg")

    parser.add_argument("--datafilepath", "-df",
                        default="/mnt/workspcae/caffe/test_model/2020-8-17/cfg/voc.data",
                        type=str,
                        help="datafilepath.default: ./cfg/coco.data")

    parser.add_argument("--weightsfilepath", "-wf",
                        default="/mnt/workspcae/caffe/test_model/2020-8-17/cfg/yolov3-voc_final.weights",
                        type=str,
                        help="weightsfilepath.default: ./yolov3.weights")

    parser.add_argument("--imagefilepath", "-if",
                        default="/mnt/share/test/0.jpg",
                        type=str,
                        help="imagefilepath.default: ./data/dog.jpg")

    args = parser.parse_args()

    return args.libfilepath, args.cfgfilepath, \
        args.datafilepath, args.weightsfilepath, args.imagefilepath


def save_pred_img(img, outputfilepath):
    """
    saving yolo result image
    img: numpy.ndarray bgr(cv2 format) image
    outputfilepath: str outputting filepath
    """
    cv2.imwrite(outputfilepath, img)


def predict_from_cv2(yolo, inputfilepath, outputfilepath):
    """
    Predicting from cv2 format
    yolo: Yolo class
    inputfilepath: filepath of image
    """

    print("call func of predict_from_cv2")
    print("image: %s" % inputfilepath)
    cv2_img = cv2.imread(inputfilepath)
    img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    yolo_results = yolo.detect(cv2_img)
    for yolo_result in yolo_results:
        yolo_result.show()
    pred_img = yolo.draw_detections(cv2_img, yolo_results)
    save_pred_img(pred_img, outputfilepath)


def predict_from_pil(yolo, inputfilepath, outputfilepath):
    """
    Predicting from PIL format
    yolo: Yolo class
    inputfilepath: filepath of image
    """

    print("call func of predict_from_pil")
    img = np.array(Image.open(inputfilepath))

    yolo_results = yolo.detect(img)
    for yolo_result in yolo_results:
        yolo_result.show()

    cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pred_img = yolo.draw_detections(cv2_img, yolo_results)
    save_pred_img(pred_img, outputfilepath)


import os
def drawbbox(res, image_path):
    """

    :param res: inflence result
    :param image_path: iamge path
    :return:
    """
    if len(res) == 0:
        print("Not bbox found!")
        return

    if not os.path.exists(image_path):
        print("% not found!" % image_path)
        return
    image = cv2.imread(image_path)

    for bbox in res:
        claeese = bbox[0].decode()
        conf = bbox[1]
        xmin = int(bbox[2][0] - bbox[2][2] / 2)
        ymin = int(bbox[2][1] - bbox[2][3] / 2)
        xmax = int(bbox[2][2] / 2 + bbox[2][0])
        ymax = int(bbox[2][3] / 2 + bbox[2][1])
        # cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color=(0,255,0))
        # cv2.putText(image, str(claeese), (xmin, ymin-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        # cv2.putText(image, str(conf), (xmin, ymin -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0),1)

        cls_cof = "{}:{:.2f}".format(claeese, conf)

        color = tuple(map(int, np.uint8(np.random.uniform(0, 255, 3))))

        cv2.rectangle(image, tuple((xmin, ymin)), tuple((xmax, ymax)), color, 1)
        t_size = cv2.getTextSize(cls_cof, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        pt2 = xmin + t_size[0] + 3, ymin + t_size[1] + 4
        cv2.rectangle(image, tuple((xmin, ymin)), tuple(pt2), color, -1)
        cv2.putText(image, cls_cof, (xmin, t_size[1] + 4 + ymin), cv2.FONT_HERSHEY_PLAIN,
                    cv2.FONT_HERSHEY_PLAIN, 1, 1, 2)

        save_path = '/mnt/share/test/Test_' + os.path.basename(image_path)
        # cv2.imshow("screen_title", image)

        # cv2.imwrite(save_path, image)
    return image

def main():
    """
    main
    """

    libfilepath, cfgfilepath, \
        datafilepath, weightsfilepath, imgfilepath = importargs()

    print("libfilepath: {}".format(libfilepath))
    darknet = Darknet(libfilepath=libfilepath,
                      cfgfilepath=cfgfilepath.encode(),
                      weightsfilepath=weightsfilepath.encode(),
                      datafilepath=datafilepath.encode())

    darknet.load_conf()

    print("======================================")
    predict_from_cv2(darknet, imgfilepath, '/mnt/share/test/pred_cv2_0.jpg')


    print("======================================")
    predict_from_pil(darknet, imgfilepath, '/mnt/share/test/pred_pil0.jpg')

    r = darknet.detect_orgin(imgfilepath.encode('utf-8'))
    images=drawbbox(r,imgfilepath)
    cv2.imshow("test",images)
    cv2.waitKey(0)
    print(r)

if __name__ == "__main__":
    main()
