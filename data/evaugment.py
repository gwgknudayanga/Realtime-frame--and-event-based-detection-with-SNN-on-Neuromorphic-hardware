#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py

import math
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    '''Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio.'''
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates
 
 
def random_affine(img, labels=(), degrees=10, translate=.1, scale=.1, shear=10,
                  new_shape=(640, 640)):
    '''Applies Random affine transformation.'''
    n = len(labels)
    if isinstance(new_shape, int):
        height = width = new_shape
    else:
        height, width = new_shape
 
    M, s = get_transform_matrix(img.shape[:2], (height, width), degrees, scale, shear, translate)
    if (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(0, 0, 0))
 
    # Transform label coordinates
    if n:
        new = np.zeros((n, 4))
 
        xy = np.ones((n * 4, 3))
        xy[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine
 
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
 
        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
 
        # filter candidates
        i = box_candidates(box1=labels[:, 1:5].T * s, box2=new.T, area_thr=0.1)
        labels = labels[i]
        labels[:, 1:5] = new[i]
 
    return img, labels
 
 
def get_transform_matrix(img_shape, new_shape, degrees, scale, shear, translate):
    new_height, new_width = new_shape
    # Center
    C = np.eye(3)
    C[0, 2] = -img_shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img_shape[0] / 2  # y translation (pixels)
 
    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
 
    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)
 
    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_height  # y transla ion (pixels)
 
    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT
    return M, s
 
"""if __name__ == "__main__":
 
    img = np.random.randint(0, 2, (15,25,2,3))
    img = img.reshape(img.shape[0],img.shape[1],-1)
    img = img.astype(np.uint8)
 
    labels = np.array([0,2,3,10,12]).reshape(-1,5)
 
    img2, label2 = random_affine(img, labels=labels, degrees=10, translate=.1, scale=.1, shear=10,
                  new_shape=(15,25))
   
    img2 = img2.reshape(15,25,2,3)
   
    print("label2 ",img2.shape)"""


def rotate90antclk(im,anns,img_width):

    #rotating bboxes anticlockwise
    x_min,y_min,x_max,y_max = anns[:,1],anns[:,2],anns[:,3],anns[:,4]

    new_xmin = y_min
    new_ymin = img_width-x_max
    new_xmax = y_max
    new_ymax = img_width-x_min

    boxes_cpy = anns[:,1:].copy()

    boxes_cpy[:,0] = new_xmin
    boxes_cpy[:,1] = new_ymin
    boxes_cpy[:,2] = new_xmax
    boxes_cpy[:,3] = new_ymax

    if not isinstance(im,torch.Tensor):
        im = torch.from_numpy(im)
    anns[:,1:] = boxes_cpy
    
    return torch.rot90(im,k = 1,dims=[-2,-1]),anns #rotate anticlockwise

def rotate90clk(im,anns,img_height):
    x_min,y_min,x_max,y_max = anns[:,1],anns[:,2],anns[:,3],anns[:,4]

    new_xmin = img_height - y_max
    new_ymin = x_min
    new_xmax = img_height - y_min
    new_ymax = x_max

    boxes_cpy = anns[:,1:].copy()

    boxes_cpy[:,0] = new_xmin
    boxes_cpy[:,1] = new_ymin
    boxes_cpy[:,2] = new_xmax
    boxes_cpy[:,3] = new_ymax

    if not isinstance(im,torch.Tensor):
        im = torch.from_numpy(im)

    anns[:,1:] = boxes_cpy
    return torch.rot90(im,k = -1,dims=[-2,-1]),anns

def evaugment_rawevents(event,anns):
    x_shift = 10 #20
    y_shift = 5 #10
    theta = 10 #10
    xjitter = np.random.randint(2*x_shift) - x_shift
    yjitter = np.random.randint(2*y_shift) - y_shift
    ajitter = (np.random.rand() - 0.5) * theta / 180 * 3.141592654
    sin_theta = np.sin(ajitter)
    cos_theta = np.cos(ajitter)

    event[:,1] = event[:,1] * cos_theta - event[:,2] * sin_theta + xjitter
    event[:,2] = event[:,1] * sin_theta + event[:,2] * cos_theta + yjitter
    event[:,1] = np.clip(event[:,1],0,345)
    event[:,2] = np.clip(event[:,2],0,259)

    bboxes = anns[:,1:].copy()
    bboxes[:,[0,2]] = anns[:,[1,3]] * cos_theta - anns[:,[2,4]] * sin_theta + xjitter
    bboxes[:,[1,3]] = bboxes[:,[0,2]] * sin_theta + anns[:,[2,4]] * cos_theta + yjitter
    bboxes[:,[0,2]] = np.clip(bboxes[:,[0,2]],0,345)
    bboxes[:,[1,3]] = np.clip(bboxes[:,[1,3]],0,259)
    anns[:,1:] = bboxes
    
    return event,anns


def evaugment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    '''HSV color-space augmentation.'''
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


#vertical flip
def evvflip(im,anns,img_height):
    
    boxes_cpy = anns[:,1:].copy()
    boxes_cpy[:,[1,3]] = img_height - anns[:,[2,4]]
    boxes_cpy[:,[1,3]] = boxes_cpy[:,[3,1]]

    if not isinstance(im, torch.Tensor):
        im = torch.from_numpy(im)
        
    anns[:,1:] = boxes_cpy
    return torch.flip(im,dims=[-2]),anns


def evhflip(im,anns,img_width):

    boxes_cpy = anns[:,1:].copy()
    boxes_cpy[:,[0,2]] = img_width - anns[:,[1,3]]
    boxes_cpy[:,[0,2]] = boxes_cpy[:,[2,0]]

    if not isinstance(im,torch.Tensor):
        im = torch.from_numpy(im)

    anns[:,1:] = boxes_cpy
    
    return torch.flip(im,dims=[-1]),anns

def evrotate90antclk(im,anns,img_width):

    #rotating bboxes anticlockwise
    x_min,y_min,x_max,y_max = anns[:,1],anns[:,2],anns[:,3],anns[:,4]

    new_xmin = y_min
    new_ymin = img_width-x_max
    new_xmax = y_max
    new_ymax = img_width-x_min

    boxes_cpy = anns[:,1:].copy()

    boxes_cpy[:,0] = new_xmin
    boxes_cpy[:,1] = new_ymin
    boxes_cpy[:,2] = new_xmax
    boxes_cpy[:,3] = new_ymax

    if not isinstance(im,torch.Tensor):
        im = torch.from_numpy(im)
    anns[:,1:] = boxes_cpy
    
    return torch.rot90(im,k = 1,dims=[-2,-1]),anns #rotate anticlockwise

def evrotate90clk(im,anns,img_height):
    x_min,y_min,x_max,y_max = anns[:,1],anns[:,2],anns[:,3],anns[:,4]

    new_xmin = img_height - y_max
    new_ymin = x_min
    new_xmax = img_height - y_min
    new_ymax = x_max

    boxes_cpy = anns[:,1:].copy()

    boxes_cpy[:,0] = new_xmin
    boxes_cpy[:,1] = new_ymin
    boxes_cpy[:,2] = new_xmax
    boxes_cpy[:,3] = new_ymax

    if not isinstance(im,torch.Tensor):
        im = torch.from_numpy(im)

    anns[:,1:] = boxes_cpy
    return torch.rot90(im,k = -1,dims=[-2,-1]),anns

def evletterbox(im, new_shape=(640, 640), auto=True, scaleup=True, stride=32,rect_mode = False):
    
    '''Resize and pad image while meeting stride-multiple constraints.'''

    shape = im.shape[-2:]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    elif isinstance(new_shape, list) and len(new_shape) == 1:
       new_shape = (new_shape[0], new_shape[0])

    # Scale ratio (new / old)

    r = max(new_shape) / max(shape[0], shape[1])
    
    #r = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) #new_shape[0] should be for height because we know shape[0] is for height.

    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = (192,256)  #int(shape[0] * r), int(shape[1] * r)
    dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]  # wh padding
    #padding1!
    #print("new unpad issss ",new_unpad)

    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padd ing into 2 sides
    dh /= 2

    im = T.Resize((new_unpad[0],new_unpad[1]),T.InterpolationMode.NEAREST)(im)  # T.Resize((int(shape[0] * r),int(shape[1] * r)),T.InterpolationMode.NEAREST)(im)  #Here we resize based on unpadded stuff.
    #The padded stuff may come later.

    """if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)"""
    
    if rect_mode:

        pad2d = (0,0,0,0)
        top,bottom = 0,0
        left,right = 0,0
    else:
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        pad2d = (left,right,top,bottom)  #(last,last,next_to_last,next_to_last)
        im = F.pad(im, pad2d, "constant", 0)   #Assuming im is in (T,X,H,W) forma

    #im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    #im.shape == tr.Resize()" "

    #Now we are passing ratio and pad information so that the annotations can be resized later.

    return im, r, (left, top)


def evmixup(im, labels, im2, labels2):
    '''Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf.'''
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    '''Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio.'''
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


"""def evrandom_affine(img, labels=(), degrees=10, translate=.1, scale=.1, shear=10,
                  new_shape=(640, 640)):
    '''Applies Random affine transformation.'''
    n = len(labels)
    if isinstance(new_shape, int):
        height = width = new_shape
    else:
        height, width = new_shape

    M, s = get_transform_matrix(img.shape[:2], (height, width), degrees, scale, shear, translate)
    if (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    if n:
        new = np.zeros((n, 4))

        xy = np.ones((n * 4, 3))
        xy[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=labels[:, 1:5].T * s, box2=new.T, area_thr=0.1)
        labels = labels[i]
        labels[:, 1:5] = new[i]

    return img, labels"""


def get_transform_matrix(img_shape, new_shape, degrees, scale, shear, translate):
    new_height, new_width = new_shape
    # Center
    C = np.eye(3)
    C[0, 2] = -img_shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img_shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_height  # y transla ion (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT
    return M, s


def evmosaic_augmentation(shape, imgs, hs, ws, labels, hyp, specific_shape = False, target_height=640, target_width=640):
    '''Applies Mosaic augmentation.'''
    assert len(imgs) == 4, "Mosaic augmentation of current version only supports 4 images."
    labels4 = []
    if not specific_shape:
        if isinstance(shape, list) or isinstance(shape, np.ndarray):
            target_height, target_width = shape
        else:
            target_height = target_width = shape

    yc, xc = (int(random.uniform(x//2, 3*x//2)) for x in (target_height, target_width) )  # mosaic center x, y

    for i in range(len(imgs)):
        # Load image
        img, h, w = imgs[i], hs[i], ws[i]
        # place img in img4
        if i == 0:  # top left
            img4 = np.full((target_height * 2, target_width * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles

            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, target_width * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(target_height * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, target_width * 2), min(target_height * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels_per_img = labels[i].copy()
        if labels_per_img.size:
            boxes = np.copy(labels_per_img[:, 1:])
            boxes[:, 0] = w * (labels_per_img[:, 1] - labels_per_img[:, 3] / 2) + padw  # top left x
            boxes[:, 1] = h * (labels_per_img[:, 2] - labels_per_img[:, 4] / 2) + padh  # top left y
            boxes[:, 2] = w * (labels_per_img[:, 1] + labels_per_img[:, 3] / 2) + padw  # bottom right x
            boxes[:, 3] = h * (labels_per_img[:, 2] + labels_per_img[:, 4] / 2) + padh  # bottom right y
            labels_per_img[:, 1:] = boxes

        labels4.append(labels_per_img)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    # for x in (labels4[:, 1:]):
    #     np.clip(x, 0, 2 * s, out=x)
    labels4[:, 1::2] = np.clip(labels4[:, 1::2], 0, 2 * target_width)
    labels4[:, 2::2] = np.clip(labels4[:, 2::2], 0, 2 * target_height)

    # Augment
    img4, labels4 = random_affine(img4, labels4,
                                  degrees=hyp['degrees'],
                                  translate=hyp['translate'],
                                  scale=hyp['scale'],
                                  shear=hyp['shear'],
                                  new_shape=(target_height, target_width))

    return img4, labels4