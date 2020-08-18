from ctypes import *
import math
import os
import random
import cv2
import numpy as np

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    

#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("/home/xu/PycharmProjects/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.3, hier_thresh=.3, nms=.45):
    #im = load_image(image, 0, 0)
    im = image
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

def detect_by_camera(net, meta, thresh=.3, hier_thresh=.3, nms=.45):
    # 引入库
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        res = detect(net, meta, frame)
        cv2.imshow("Video", drawbbox(res, frame, False))
        # 读取内容
        if cv2.waitKey(10) == ord("q"):
            break

    # 随时准备按q退出
    cap.release()
    cv2.destroyAllWindows()




def Iou(box1, box2, wh=False):
    if wh == False:
	    xmin1, ymin1, xmax1, ymax1 = box1
	    xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0]-box1[2]/2.0), int(box1[1]-box1[3]/2.0)
        xmax1, ymax1 = int(box1[0]+box1[2]/2.0), int(box1[1]+box1[3]/2.0)
        xmin2, ymin2 = int(box2[0]-box2[2]/2.0), int(box2[1]-box2[3]/2.0)
        xmax2, ymax2 = int(box2[0]+box2[2]/2.0), int(box2[1]+box2[3]/2.0)
    # 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    tlx = np.max([xmin1, xmin2])
    tly = np.max([ymin1, ymin2])
    brx = np.min([xmax1, xmax2])
    bry = np.min([ymax1, ymax2])
    # 计算两个矩形框面积
    area1 = (xmax1-xmin1) * (ymax1-ymin1)
    area2 = (xmax2-xmin2) * (ymax2-ymin2)
    inter_area=(np.max([0,brx-tlx]))*(np.max(bry-tly)) #计算交集面积

    iou=inter_area/(area1+area2-inter_area+1e-6)#计算交并比

    return iou


def drawbbox(res,image_path,is_save=True):
    if len(res)==0:
        print("Not bbox found!")
        return

    # if not os.path.exists(image_path):
    #     print("% not found!" %image_path )
    #     return
    if is_save:
        image=cv2.imread(image_path)
    else:
        image = image_path

    for bbox in res:

        claeese=bbox[0]
        conf=bbox[1]
        xmin=int(bbox[2][0]-bbox[2][2]/2)
        ymin=int(bbox[2][1]-bbox[2][3]/2)
        xmax=int(bbox[2][2]/2 + bbox[2][0])
        ymax=int(bbox[2][3]/2 + bbox[2][1])
        cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color=(0,255,0))
        cv2.putText(image, str(claeese), (xmin, ymin-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(image, str(conf), (xmin, ymin -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0),1)



        if is_save:
            save_path = '../test/outputs/' + os.path.basename(image_path)
            cv2.imwrite(save_path,image)
    return image






import glob
if __name__ == "__main__":

    test_img="../000034.jpg"
    net = load_net("../cfg/yolov3-voc.cfg".encode('utf-8'), "../backup/yolov3-voc_30000.weights".encode('utf-8'), 0)
    meta = load_meta("../cfg/voc_python.data".encode('utf-8'))




    total_counter=0
    target_counter=0

    detect_by_camera(net, meta)
    img_list=glob.glob(os.path.join('../test/', "*.jpg"))
    for img in  img_list:
        r = detect(net, meta, img.encode('utf-8'))
        drawbbox(r, img)
        print(r)
        total_counter+=1
        if len(r):
            target_counter+=1


    print("Test image %d pic, found target %d pic" %(total_counter,target_counter))





    

