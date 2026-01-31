import torch
import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional
import time
import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset
import json

from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM

from .utils import HELP_URL, IMG_FORMATS

data_visualize_output_path = "./visualize_data"

def letterbox_2dim(im,anns,new_shape= (320,320)):

    import torchvision.transforms  as T
    import torch.nn.functional as F

    #new_shape should be (h,w)
    
    #Expected tensor input format --> TCHW
    #Expected annotation format --> (category_id,x_min,y_min,x_max,y_max)
    
    '''Resize and pad image while meeting stride-multiple constraints.'''
    im #this is a tensor
    h0,w0 = im.shape[-2:] # this is the original shape

    #Here for example, the new shape = (320,256) as both are divided by 64 and 32
    #However when 346 is resized to 320, then to preserve the same aspect ratio, unpad_height should be (260 * (320/346)) = 240
    #from 240 to 256 there should be a padding.

    ratio = max(new_shape) / max(h0, w0) # new_shape is the shape to be resized. Also we assume that the new resize shape is also
                                    # aspect ratio preserved. hence when max(h0,w0) if h0 max that means h of the new shape is 
                                    # max from both(h,w). 

    im = T.Resize((int(ratio*h0),int(ratio*w0)),T.InterpolationMode.NEAREST)(im)

    # Compute padding
    new_unpad = int(ratio*h0),int(ratio*w0)
    h,w = new_unpad
    dh, dw = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]  # wh padding
 
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    pad2d = (left,right,top,bottom)  #(last,last,next_to_last,next_to_last)
    im = F.pad(im, pad2d, "constant", 0)   #Assuming im is in (T,X,H,W) format

    ratio = (h/h0,w/w0)
    pad = (top,left)

    bboxes = np.copy(anns[:, 1:])
    bboxes[:, 0] = ratio[1] * anns[:, 1] + pad[1] # top left x
    bboxes[:, 1] = ratio[0] * anns[:, 2] + pad[0] # top left y
    bboxes[:, 2] = ratio[1] * anns[:, 3] + pad[1] # bottom right x
    bboxes[:, 3] = ratio[0] * anns[:, 4] + pad[0] # bottom right y
    anns[:, 1:] = bboxes
    
    return im,anns,ratio,pad #Here we send the ratio and pad for future rescaling to original size 
                            #and then calculate metrics and visualization of detections

from .evaugment import (
    evaugment_hsv,
    evletterbox,
    evmixup,
    evmosaic_augmentation,
    evaugment_rawevents,
    evvflip,
    evhflip,
    evrotate90antclk,
    evrotate90clk,
)
#from .visualize_data import dump_image_with_labels,make_dvs_frame

#from yolov6.utils.events import LOGGER
import copy
import psutil
from multiprocessing.pool import ThreadPool

       
def COCO2YOLO(anns,img_width,img_height):
    anns[:,1] = (anns[:,1] + anns[:,3]/2)/img_width
    anns[:,2] = (anns[:,2] + anns[:,4]/2)/img_height
    anns[:,3] /= img_width  
    anns[:,4] /= img_height
    return anns

def yolo2bbox(ann,img_width,img_height):

    x1 = (ann[:,1] - (ann[:,3] / 2)) * img_width
    y1 = (ann[:,2] - (ann[:,4] / 2)) * img_height
    x2 = (ann[:,1] + (ann[:,3] / 2)) * img_width
    y2 = (ann[:,2] + (ann[:,4] / 2)) * img_height
    ann[:,1] = x1
    ann[:,2] = y1
    ann[:,3] = x2
    ann[:,4] = y2

def bbox2yolo(ann,img_width,img_height):

    normalized_mid_x = (ann[:,1] + ann[:,3])/(2.0*img_width)
    normalized_mid_y = (ann[:,2] + ann[:,4])/(2.0*img_height)
    normalized_box_width = (ann[:,3] - ann[:,1])/img_width
    normalized_box_height = (ann[:,4] - ann[:,2])/img_height

    ann[:,1] = normalized_mid_x
    ann[:,2] = normalized_mid_y
    ann[:,3] = normalized_box_width
    ann[:,4] = normalized_box_height


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

def round_to_nearest_1e3(x):
    return int(round(x / 1e3) * 1e3)


def make_dvs_frame(events, height=None, width=None, color=True, clip=3,forDisplay = False):
    """Create a single frame.

    Mainly for visualization purposes

    # Arguments
    events : np.ndarray
        (t, x, y, p)
    x_pos : np.ndarray
        x positions
    """
    if height is None or width is None:
        height = events[:, 2].max()+1
        width = events[:, 1].max()+1

    histrange = [(0, v) for v in (height, width)]

    pol_on = (events[:, 3] == 1)
    pol_off = np.logical_not(pol_on)
    img_on, _, _ = np.histogram2d(
            events[pol_on, 2], events[pol_on, 1],
            bins=(height, width), range=histrange)
    img_off, _, _ = np.histogram2d(
            events[pol_off, 2], events[pol_off, 1],
            bins=(height, width), range=histrange)

    on_non_zero_img = img_on.flatten()[img_on.flatten() > 0]
    on_mean_activation = np.mean(on_non_zero_img)
    off_non_zero_img = img_off.flatten()[img_off.flatten() > 0]
    off_mean_activation = np.mean(off_non_zero_img)

    # on clip
    if clip is None:
        on_std_activation = np.std(on_non_zero_img)
        img_on = np.clip(
            img_on, on_mean_activation-3*on_std_activation,
            on_mean_activation+3*on_std_activation)
    else:
        img_on = np.clip(
            img_on, -clip, clip)

    # off clip
    
    if clip is None:
        off_std_activation = np.std(off_non_zero_img)
        img_off = np.clip(
            img_off, off_mean_activation-3*off_std_activation,
            off_mean_activation+3*off_std_activation)
    else:
        img_off = np.clip(
            img_off, -clip, clip)

    if color:

        frame = np.zeros((height, width, 2))
        #img_on /= img_on.max()
        frame[..., 0] = img_on
        """img_on -= img_on.min()
        img_on /= img_on.max()"""

        #img_off /= img_off.max()
        frame[..., 1] = img_off
        """img_off -= img_off.min()
        img_off /= img_off.max()"""

        #print("absolute max and min = ",np.abs(frame).max())
        frame /= np.abs(frame).max()
        if forDisplay:
            third_channel = np.zeros((height,width,1))
            frame = np.concatenate((frame,third_channel),axis=2)

    else:
        frame = img_on - img_off
        #frame -= frame.min()
        #frame /= frame.max()
        frame /= np.abs(frame).max()

    return frame

def get_event_cube(events1,param_dict = {"TSteps" : 5, "tbins" : 1,"quantized_h" : 260 ,"quantized_w" : 346} ):
    
    tbin = param_dict["tbins"]
    C = 2 * tbin
    quantized_h = param_dict["quantized_h"]
    quantized_w = param_dict["quantized_w"]
    T = param_dict["TSteps"]
    
    #sample = file_path #"/dtu/eumcaerotrain/data/latest_dataset/dset_1/npz_files_event_based/crack/crack_20/crack_20_7572871.npz"
    #data = np.load(file_path)

    events = events1 #data["events"]
    events[:,0] -= events[0,0]
    sample_size = int(round_to_nearest_1e3(events[:,0].max()))
    #print("sample size is ",sample_size)
    quantization_size = [np.ceil(sample_size / T), 1, 1]

    events = events[events[:,0] < sample_size,:]
    coords1 = events[:,:3]
    coords = torch.floor(coords1 / torch.tensor(quantization_size))
    coords[:,0] = coords[:,0].clamp(min = 0)
    coords[:,1] = coords[:,1].clamp(min = 0,max = quantized_w - 1)
    coords[:,2] = coords[:,2].clamp(min = 0, max = quantized_h - 1)
    tbin_size = quantization_size[0] / tbin
    tbin_coords = (events[:,0] % quantization_size[0]) // tbin_size
    tbin_feats = (2 * tbin_coords) + events[:,3]
    feats = torch.nn.functional.one_hot(torch.from_numpy(tbin_feats).to(torch.long), 2*tbin)

    """
    ##Start : New code for dense tensor
    dense_tensor = torch.zeros((T, quantized_w, quantized_h, 2 * tbin), dtype=torch.float32)
    # Vectorized operation to fill the dense tensor

    indices = coords.to(torch.int32)
    print("T max ",indices[:, 0].max())
    print("x max ",indices[:, 1].max())
    print("y max ",indices[:, 2].max())


    if indices[:, 0].max() > 6:
        print("hiiiii ",indices[:, 0].max(),"  ",coords[:,0].max())

    dense_tensor.index_put_((indices[:, 0], indices[:, 1], indices[:, 2]), feats.float(), accumulate=True)
    dense_tensor = dense_tensor.to(bool)
    dense_tensor = dense_tensor.to(torch.float32)
    ## End: New code for dense tensor"""
     
    sparse_tensor = torch.sparse_coo_tensor(
    coords.t().to(torch.int32),
    feats,
    size=(T,quantized_w,quantized_h,C),
    ).coalesce()
    sparse_tensor = sparse_tensor.to(bool)
    sparse_tensor = sparse_tensor.to(torch.float32)
    #return shape [T,w,h,C]
    #if not is_sparse:
    #    sparse_tensor.to_dense().permute(0,3,1,2)
    #return shape [T,w,h,C]     
    return sparse_tensor.to_dense()



class TrainValDataset(Dataset):
    '''YOLOv6 train_loader/val_loader, loads images and labels for training and validation.'''

    def __init__(
        self,
        img_data_csv,
        ann_data_csv,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        check_images=False,
        check_labels=False,
        stride=32,
        pad=0.0,
        rank=-1,
        data_dict=None,
        task="train",
        specific_shape = False,
        height=-1,
        width=-1,
        cache_ram=False,
        dataset_parent_folder="",
        input_img_type = 0
    ):
        assert task.lower() in ("train", "val", "test", "speed"), f"Not supported task: {task}"
        tik = time.time()
        self.__dict__.update(locals())
        self.main_process = self.rank in (-1, 0)
        self.task = self.task.capitalize()
        self.class_names = ["crack","spalling"]
        self.target_ann_save_path = "./gt.json"
        self.img_paths, self.labels = self.get_imgs_labels(img_data_csv,ann_data_csv,dataset_parent_folder)
        self.rect = rect
        self.specific_shape = specific_shape
        self.target_height = height
        self.target_width = width
        self.cache_ram = cache_ram
        if self.cache_ram:
            self.num_imgs = len(self.img_paths)
            self.imgs = [None] * self.num_imgs
            self.cache_images(num_imgs=self.num_imgs)

        self.need_remove_this_param_later = False
        self.root = dataset_parent_folder
        self.augment = augment
        self.input_img_type = input_img_type
        self.mode = task

        tok = time.time()

        #print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS ",self.augment)

    def __del__(self):
        if self.cache_ram:
            del self.imgs

    def __len__(self):
        """Get the length of dataset"""
        return len(self.img_paths)
 
    def __getitem__(self, index):
        """Fetching a data sample for a given key.
        This function applies mosaic and mixup augments during training.
        During validation, letterbox augment is applied.
        """
        target_shape = (self.target_height, self.target_width)
        
        path = self.img_paths[index]
        events = np.load(path)["events"]

        labels = self.labels[index].copy()

        if self.augment:

            if random.random() <= 0.8:
                #print("augmenting 1111 ")
                yolo2bbox(labels,346,260)
                events,labels = evaugment_rawevents(events,labels) #this is like affine transformation
                labels[:,[1,3]] = np.clip(labels[:,[1,3]] ,0,346) #since we converted from COCO2YOLO at get_images_and_labels function
                labels[:,[2,4]] = np.clip(labels[:,[2,4]] ,0,260)
                #bbox2yolo(labels,346,260)
                #Now the tensor is in TCHW format

        event_cube = get_event_cube(events)  #,self.param_dict)
        event_cube = event_cube.permute(0,3,2,1) #event_cube.to_dense().permute(0,3,2,1)
        #Now the tensor is in TCHW format

        #START : 
        #event_cube = make_dvs_frame(events, height=260, width=346, color=True, clip=None,forDisplay = False)
        
        """event_cube = np.load(path)["ev_color_img"]
        event_cube = torch.from_numpy(event_cube) #H,W,C format
        event_cube = event_cube.permute(2,0,1)
        event_cube = event_cube.unsqueeze(0)
        event_cube = event_cube.repeat(1, 1, 1, 1)"""

        #END: 
        
        h0,w0 = event_cube.shape[-2:] #here h0 and w0 are original event_cube shape before resizing.

        event_cube,ann_array,ratio_tuple, pad_tuple = letterbox_2dim(event_cube,labels,new_shape = target_shape)

        #print("event cube shapeeeeeeeeeeee ",event_cube.shape, " ",ratio_tuple," ",pad_tuple)

        new_height,new_width = event_cube.shape[-2:]

        if self.augment:
            if random.random() <= 0.6:
                event_cube,ann_array = evvflip(event_cube,ann_array,target_shape[0] - 1) #height related 
            if random.random() <= 0.5:
                event_cube,ann_array = evhflip(event_cube,ann_array,target_shape[1] - 1) #width related
            """if random.random() <= 0.5:
                event_cube,ann_array = rotate90antclk(event_cube,ann_array,self.target_spatial_size)
            if random.random() <= 0.5:
                event_cube,ann_array = rotate90clk(event_cube,ann_array,self.target_spatial_size)"""
        
        #In this case the ann array should be in bbox format
        ann_array = ann_array.reshape(-1,5)
        temp = ann_array[:,1:].copy()
        temp[:,0] = (ann_array[:,1] + ann_array[:,3]) / 2
        temp[:,1] = (ann_array[:,2] + ann_array[:,4]) / 2
        temp[:,2] = (ann_array[:,3] - ann_array[:,1])
        temp[:,3] = (ann_array[:,4] - ann_array[:,2])

        if len(target_shape) == 1:
            temp /= target_shape
        else:
            temp[:,0] /= target_shape[1] #box mid x
            temp[:,2] /= target_shape[1] #box width
            temp[:,1] /= target_shape[0] #box mid y
            temp[:,3] /= target_shape[0] #box height
        
        temp = np.clip(temp,0,0.999)

        #Here the event_cube is in TCHW format.

        labels[:,1:] = temp

        labels_out = torch.zeros((len(labels), 6))

        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)


        desired_local_npz_path_idx = path.find(self.root) + len(self.root)
        local_npz_path = path[desired_local_npz_path_idx:]

        #if "dtu_4_crack_7_2238804" in local_npz_path:
        #    print("local npz_path ",local_npz_path)

        #print("local_npz_path ",local_npz_path)
        #print("sssss ",labels_out)
        #dump_image_with_labels(event_cube,labels,target_shape,data_visualize_output_path,(local_npz_path.rsplit(".",1)[0]).rsplit("/",1)[1],create_histo_frames = True)
        data_dict = {}

        data_dict['im_file'] = path
        data_dict['img'] = event_cube
        data_dict['cls'] = labels_out[:,1].reshape(-1,1)
        data_dict['bboxes'] = labels_out[:,2:].reshape(-1,4)
        #print("data_dict 11111 ... ",data_dict)
        data_dict['ori_shape'] = [h0,w0]
        data_dict['resized_shape'] = [new_height,new_width]
        data_dict['batch_idx'] = torch.tensor([0] * data_dict['bboxes'].shape[0])
        #print("data_dict 22222222 ... ",data_dict)
        if self.mode == "val":
            data_dict['ratio_pad'] = [[ratio_tuple[0],ratio_tuple[1]], [pad_tuple[0],pad_tuple[1]]] 

        return data_dict    #event_cube, labels_out, self.img_paths[index], shapes
    

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        #print("valuessssssssssssssssss ",values)
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img':
                value = torch.stack(value, 0)
                value = value.permute(1,0,2,3,4)
            if k in ['masks', 'keypoints', 'bboxes', 'cls']:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        #print("new batch ",new_batch)
        return new_batch
 

    @staticmethod
    def generate_coco_format_labels_custom(save_path,img_paths_list,anns_for_images_list,class_names,input_img_type = 0):

        # for evaluation with pycocotools
        dataset = {"categories": [], "annotations": [], "images": []}
        dataset_size = len(img_paths_list)

        for i, class_name in enumerate(class_names):
            dataset["categories"].append(
                {"id": i, "name": class_name, "supercategory": ""}
            )

        ann_id = 0
        #LOGGER.info(f"Convert to COCO format")
        #print(f"Dataset size: {dataset_size}")

        for i in range(len(img_paths_list)):  # (img_path, info)
            print(i)

            #_,labels, name,width,height = self.get_element(i,is_event_frame)
            path = Path(img_paths_list[i])
            if input_img_type == 0:
                im = cv2.imread(img_paths_list[i])
                width = im.shape[1]
                height = im.shape[0]
            elif input_img_type == 1:
                im = np.load(img_paths_list[i])["frame_img"]
                width = im.shape[1]
                height = im.shape[0]
            elif input_img_type == 2:
                im = np.load(img_paths_list[i])["ev_color_img"]
                width = im.shape[1]
                height = im.shape[0]
                
            name = path.stem
            labels = anns_for_images_list[i]

            dataset["images"].append(
                {
                    "file_name": name,
                    "id": name,
                    "width": width,
                    "height": height,
                }
            )
            if list(labels):
                for label in labels:
                    c, x, y, w, h = label[:5]
                    # convert x,y,w,h to x1,y1,x2,y2
                    x1 = (x - w / 2) * width
                    y1 = (y - h / 2) * height
                    x2 = (x + w / 2) * width
                    y2 = (y + h / 2) * height
                    # cls_id starts from 0
                    cls_id = int(c)
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    dataset["annotations"].append(
                        {
                            "area": h * w,
                            "bbox": [x1, y1, w, h],
                            "category_id": cls_id,
                            "id": ann_id,
                            "image_id": name,
                            "iscrowd": 0,
                            # mask
                            "segmentation": [],
                        }
                    )
                    ann_id += 1

        with open(save_path, "w") as f:
            json.dump(dataset, f)
            #LOGGER.info(
            #    f"Convert to COCO format finished. Results saved in {save_path}")

    
    def get_imgs_labels(self, img_data_csv,ann_data_csv,dataset_parent_folder = ""):

        img_file_names = []
        anns_list_for_images = []
        #print("KKKKKKKKKKKKKKKKKKK IMG CSV ",img_data_csv)
        with open(img_data_csv,"r") as f1:
            img_file_names = f1.readlines()
            img_file_names = [os.path.join(dataset_parent_folder,file.rstrip()) for file in img_file_names]
            if self.input_img_type > 0:
                for file_full_name in img_file_names:
                    #file_full_name = os.path.join(args.dataset_parent_folder,file.strip())
                    npz_file = np.load(file_full_name)
                    ann_list_per_image = npz_file["ann_array"]
                    if self.input_img_type == 1:
                        img_width,img_height = npz_file["frame_img"].shape[1],npz_file["frame_img"].shape[0]
                    else:
                        img_width,img_height = npz_file["ev_color_img"].shape[1],npz_file["ev_color_img"].shape[0]
                    ann_list_per_image = COCO2YOLO(ann_list_per_image,img_width,img_height)
                    anns_list_for_images.append(ann_list_per_image)
                    #print("loading ev annotation ")
                    
        if self.input_img_type == 0: 
            with open(ann_data_csv,"r") as f2:
                ann_file_names = f2.readlines()
                for ann_file in ann_file_names:
                    ann_file = os.path.join(dataset_parent_folder,ann_file.rstrip())
                    ann_list_per_image = np.loadtxt(ann_file)
                    ann_list_per_image = ann_list_per_image.reshape(-1,5)
                    anns_list_for_images.append(ann_list_per_image)
  
        """if self.task.lower() == "val":
            TrainValDataset.generate_coco_format_labels_custom(self.target_ann_save_path,img_file_names,anns_list_for_images,self.class_names,self.input_img_type)"""
        
        return img_file_names,anns_list_for_images
        

    def general_augment(self, img, labels):

        #Since here in this function we expect coco labels.
        """Gets images and labels after general augment
        This function applies hsv, random ud-flip and random lr-flips augments.
        """
        nl = len(labels)

        # HSV color-space
        """augment_hsv(
            img,
            hgain=self.hyp["hsv_h"],
            sgain=self.hyp["hsv_s"],
            vgain=self.hyp["hsv_v"],
        )"""

        # Flip up-down
        if random.random() < self.hyp["flipud"]:
            img = np.flipud(img)
            if nl:
                labels[:, 2] = 1 - labels[:, 2]

        # Flip left-right
        if random.random() < self.hyp["fliplr"]:
            img = np.fliplr(img)
            if nl:
                labels[:, 1] = 1 - labels[:, 1]

        return img, labels


    @staticmethod
    def generate_coco_format_labels(img_info, class_names, save_path):
        # for evaluation with pycocotools
        dataset = {"categories": [], "annotations": [], "images": []}
        for i, class_name in enumerate(class_names):
            dataset["categories"].append(
                {"id": i, "name": class_name, "supercategory": ""}
            )

        ann_id = 0
        LOGGER.info(f"Convert to COCO format")
        for i, (img_path, info) in enumerate(tqdm(img_info.items())):
            labels = info["labels"] if info["labels"] else []
            img_id = osp.splitext(osp.basename(img_path))[0]
            img_h, img_w = info["shape"]
            dataset["images"].append(
                {
                    "file_name": os.path.basename(img_path),
                    "id": img_id,
                    "width": img_w,
                    "height": img_h,
                }
            )
            if labels:
                for label in labels:
                    c, x, y, w, h = label[:5]
                    # convert x,y,w,h to x1,y1,x2,y2
                    x1 = (x - w / 2) * img_w
                    y1 = (y - h / 2) * img_h
                    x2 = (x + w / 2) * img_w
                    y2 = (y + h / 2) * img_h
                    # cls_id starts from 0
                    cls_id = int(c)
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    dataset["annotations"].append(
                        {
                            "area": h * w,
                            "bbox": [x1, y1, w, h],
                            "category_id": cls_id,
                            "id": ann_id,
                            "image_id": img_id,
                            "iscrowd": 0,
                            # mask
                            "segmentation": [],
                        }
                    )
                    ann_id += 1

        with open(save_path, "w") as f:
            json.dump(dataset, f)
            LOGGER.info(
                f"Convert to COCO format finished. Resutls saved in {save_path}"
            )

    @staticmethod
    def get_hash(paths):
        """Get the hash value of paths"""
        assert isinstance(paths, list), "Only support list currently."
        h = hashlib.md5("".join(paths).encode())
        return h.hexdigest()