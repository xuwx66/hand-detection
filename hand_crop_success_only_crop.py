from ctypes import *
import math
import random
import cv2
import numpy as np

import os

from PIL import Image

from Sort_correlation import Sort

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
lib = CDLL("/tmp/darknet/libdarknet.so", RTLD_GLOBAL)
# lib = CDLL(b"/Users/dandancui/Desktop/darknet/libdarknet.so", RTLD_GLOBAL)
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

def get_spaced_colors(n):
    max_value = 16581375
    interval = int(max_value/n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    res = []
    try:
        im = load_image(image, 0, 0)
        num = c_int(0)
        pnum = pointer(num)
        predict_image(net, im)
        dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
        num = pnum[0]
        if (nms): do_nms_obj(dets, num, meta.classes, nms);

        for j in range(num):
            for i in range(meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        free_image(im)
        free_detections(dets, num)
        print("Detect completed successfuly")

    except Exception as e:
        res = []
        print("Detect failed")
        print(e.args)
    return res

if __name__ == "__main__":
    
    dirname = 'crop'
    os.mkdir(dirname)
    
    tracker =  Sort()

    net = load_net(b"/tmp/darknet/cfg/yolov3-voc_test.cfg", b"/tmp/darknet/yolov3-voc_40000.weights", 0)
    meta = load_meta(b"/tmp/darknet/cfg/voc.data")

    frameC = 0
    cap = cv2.VideoCapture('/tmp/darknet/hand.mov')
#     cap = cv2.VideoCapture('/tmp/darknet/hand.mp4')
    out = None
    colors = get_spaced_colors(1000)
    total_frames = 100
    if(cap.isOpened()):
        width = cap.get(3)
        height = cap.get(4)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(width)
        print(height)
        #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        #out = cv2.VideoWriter('processed.mp4', fourcc, 25.0, (int(width),int(height)))
    while(cap.isOpened()):
        ret, frame = cap.read()
        cv2.imwrite("tmp.jpg", frame)
        r = detect(net, meta, b"tmp.jpg")
        print(r)
        #for each class run tracking
        detections = []
#         cv2.rectangle(frame,(0,535),(10,545),(255,255,0),1)
#         cv2.rectangle(frame,(715,1070),(725,1080),(0,255,255),1)
#         cv2.rectangle(frame,(1910,0),(1920,10),(255,0,255),1)
#         cv2.rectangle(frame,(1435,0),(1445,10),(0,255,0),1)
        #cv2.imwrite("d-tmp"+str(frameC).zfill(4)+".jpg",frame)

        for e in r:
            x = int(e[2][0]-(e[2][2]/2))
            y = int(e[2][1]-(e[2][3]/2))
            w = int(e[2][2])
            h = int(e[2][3])
            cl = e[0].decode("utf-8")
            conf = e[1]
            detections.append([x,y,x+w,y+h,conf])

            print ("test2")

        trackers = tracker.update(np.array(detections),frame)
        for trk in trackers:
            x = int(trk[0])
            y = int(trk[1])
            w = int(trk[2]-trk[0])
            h = int(trk[3]-trk[1])
            id = int(trk[4])
            print("test3")
#             cv2.rectangle(frame,(x,y), (x+w,y+h),(0,0,255),1)
            print(x,y,x+w,y+h)
#             cv2.imwrite("Original"+str(frameC).zfill(4)+".jpg",frame)
            cropped_frame = frame[y:y+h, x:x+w]
#             cv2.imwrite("cropped"+str(frameC).zfill(4)+".jpg",cropped_frame)
#             cv2.imwrite(os.path.join(dirname, face_file_name), image)
            cv2.imwrite(os.path.join(dirname, "ok_cropped"+str(frameC).zfill(4)+".jpg"), cropped_frame)
        
            print("test4")
        frameC = frameC+1
        #if(out is not None):
            #out.write(frame)
        print(frameC)
        if(frameC>total_frames):
#         if(frameC>10):
            break

    cap.release()
    #out.release()
    cv2.destroyAllWindows()
    print(r)
