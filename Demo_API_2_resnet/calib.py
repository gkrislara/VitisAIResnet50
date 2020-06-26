import cv2
import os
import numpy as np

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    ih, iw, _ = image.shape
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_AREA)
    new_image = np.ones((h,w,3), np.uint8) * 128
    h_start = (h-nh)//2
    w_start = (w-nw)//2
    new_image[h_start:h_start+nh, w_start:w_start+nw, :] = image
    return new_image

def mps(npi):
    mean =[104,117,124]
    pimg= npi[:,:,[2,1,0]]
    pimg[:,:,0]-=mean[0]
    pimg[:,:,1]-=mean[1]
    pimg[:,:,2]-=mean[2]
    return pimg

def preprocess(image):
    im=letterbox_image(image,tuple(reversed((224,224))))
    img=mps(im)
    return img

calib_image_dir = "/workspace/images/calib/image_net/cat/"
calib_image_list = "/workspace/images/calib/image_net/calib_list.txt"
calib_batch_size = 50

def calib_input(iter):
    images = []
    line = open(calib_image_list).readlines()
    for index in range(0, calib_batch_size):
        curline = line[iter * calib_batch_size + index]
        calib_image_name = curline.strip()
        filename = calib_image_dir + calib_image_name
        im = cv2.imread(filename)
        image = preprocess(im)
        images.append(image.tolist())
    return {"input": images}
