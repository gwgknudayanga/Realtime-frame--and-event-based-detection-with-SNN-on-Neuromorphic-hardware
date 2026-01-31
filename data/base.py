import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import psutil
from torch.utils.data import Dataset

from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM

from .utils import HELP_URL, IMG_FORMATS

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
        pad=0.4,
        rank=-1,
        data_dict=None,
        task="train",
        specific_shape = False,
        height=-1,
        width=-1,
        cache_ram=False,
        dataset_parent_folder="",
        input_img_type = 0,
        param_dict = {"TSteps" : 7, "tbins" : 1,"quantized_h" : 260 ,"quantized_w" : 346},
        dataset_name = "evCIVIL"
    ):
        assert task.lower() in ("train", "val", "test", "speed"), f"Not supported task: {task}"
        tik = time.time()
        self.__dict__.update(locals())
        self.main_process = self.rank in (-1, 0)
        self.task = self.task.capitalize()
        self.class_names = ["crack","spalling"]
        self.target_ann_save_path = "./gt.json"

        with open(img_data_csv,"r") as file: #when events and anns are on the same npz
            self.img_paths = file.readlines()
        
        self.image_data_root = dataset_parent_folder

        #self.img_paths, self.labels = self.get_imgs_labels(img_data_csv,ann_data_csv,dataset_parent_folder)
        
        self.rect = rect
        self.specific_shape = specific_shape
        self.target_height = height
        self.target_width = width
        self.imgsz = max(self.target_height,self.target_width)

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
        self.pad = pad
        self.is_direct_spike = False

        self.param_dict = param_dict
        self.dataset_name = dataset_name

        if self.rect:
            arr = np.array([260,346]) #Need to change based on dataset
            self.shapes = np.tile(arr, (len(self.img_paths), 1)) 
            assert self.batch_size is not None
            self.set_rectangle()

        tok = time.time()

        #print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS ",self.augment)

    def __del__(self):
        if self.cache_ram:
            del self.imgs

    def __len__(self):
        """Get the length of dataset"""
        return len(self.img_paths)
    

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""

        print('')
        bi = np.floor(np.arange(len(self.shapes)) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = self.shapes #np.array([x.pop('shape') for x in self.shapes])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.round(np.array(shapes) * self.imgsz / self.stride.numpy() + self.pad).astype(int) * self.stride.numpy() #np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        
        self.batch = bi  # batch index of image
 
    def __getitem__(self, index):
        """Fetching a data sample for a given key.
        This function applies mosaic and mixup augments during training.
        During validation, letterbox augment is applied.
        """
        target_shape = (self.target_height, self.target_width)
        #print("self.task self.augment self.rect",self.task," ",self.augment," ",self.rect)

        """if self.rect:
            target_shape = self.batch_shapes[self.batch[index]]"""
            #print("rrrrrrrrrrrrrrrrrrrr ",target_shape)
        
        #print("new shape ",target_shape)

        local_event_npz_path = self.img_paths[index % len(self.img_paths)].rstrip()
        path = os.path.join(self.image_data_root,local_event_npz_path)

        #print("pathhhhhhh ",path)

        data_file = np.load(path)

        events = data_file["events"]
        labels = data_file["ann_array"]
        
        if self.dataset_name == "GEN1":
            labels = BBOX2YOLO(labels,self.param_dict["quantized_w"],self.param_dict["quantized_h"])
        else:
            labels = COCO2YOLO(labels,self.param_dict["quantized_w"],self.param_dict["quantized_h"])

        #events = np.load(path)["events"]
        #labels = self.labels[index].copy()

        #if self.augment:

        """if random.random() <= 0.7:
                #print("augmenting 1111 ")
                yolo2bbox(labels,346,260)
                events,labels = evaugment_rawevents(events,labels) #this is like affine transformation
                labels[:,[1,3]] = np.clip(labels[:,[1,3]] ,0,346) #since we converted from COCO2YOLO at get_images_and_labels function
                labels[:,[2,4]] = np.clip(labels[:,[2,4]] ,0,260)
                bbox2yolo(labels,346,260)
        """

        #event_cube = get_event_cube(events,self.param_dict)
        #event_cube = event_cube.permute(0,3,2,1) #event_cube.to_dense().permute(0,3,2,1)
        #Now the tensor is in TCHW format

        event_cube = make_dvs_frame(events, height=self.param_dict["quantized_h"], width=self.param_dict["quantized_w"], color=True, clip=3,forDisplay = False)
        event_cube = torch.from_numpy(event_cube).float().permute(2,0,1)
        

        #START : 
        
        """event_cube = np.load(path)["ev_color_img"]
        event_cube = torch.from_numpy(event_cube) #H,W,C format
        event_cube = event_cube.permute(2,0,1)
        event_cube = event_cube.unsqueeze(0)
        event_cube = event_cube.repeat(1, 1, 1, 1)"""

        #END: 
        
        h0,w0 = event_cube.shape[-2:] #here h0 and w0 are original event_cube shape before resizing.

        # letterbox
        event_cube, ratio, pad = evletterbox(event_cube, target_shape, auto=False,scaleup=True,rect_mode=self.rect)

        #print("event cube shape ",event_cube.shape)

        #print("event cube  ratio pad ",event_cube.shape," ",ratio,"  ",pad)

        shapes = (h0, w0), ((ratio,ratio), pad)  # for COCO mAP rescaling
            #print("ratiooooooooooooooooooooooooooooo and pad ",ratio, "  ",pad)
        
        if labels.size:

            w = int(w0 * ratio) #these w,h are the resized stuff to preserve the aspect ratio (346,260) --> (640,480)
            h = int(h0 * ratio)

            #print("w and h ..... ",w,"  ",h,"  ",w0,"  ",h0,"  ",ratio)
            # new boxes
            boxes = np.copy(labels[:, 1:])
            boxes[:, 0] = (
                    w * (labels[:, 1] - labels[:, 3] / 2) + pad[0]
                )  # top left x
            boxes[:, 1] = (
                    h * (labels[:, 2] - labels[:, 4] / 2) + pad[1]
                )  # top left y
            boxes[:, 2] = (
                    w * (labels[:, 1] + labels[:, 3] / 2) + pad[0]
                )  # bottom right x
            boxes[:, 3] = (
                    h * (labels[:, 2] + labels[:, 4] / 2) + pad[1]
                )  # bottom right y
            labels[:, 1:] = boxes
        
        new_height,new_width = event_cube.shape[-2:] # This is the final shape after resizing (to preserve aspect ratio) and padding after that --> (640,640) 

        if self.augment:
            
            #Random affine start

            #print("1111111111111111111111111")

            if self.dataset_name != "GEN1" and random.random() < 0.8:

                event_cube = event_cube.permute(2,3,1,0)
                #now h,w,c,T
                temp_t,temp_c = event_cube.shape[3],event_cube.shape[2]

                event_cube = event_cube.reshape(event_cube.shape[0],event_cube.shape[1],-1)

                event_cube = event_cube.numpy().astype(np.uint8)

                event_cube, labels = random_affine(event_cube, labels=labels, degrees=5, translate=.1, scale=.1, shear=5,
                    new_shape=(event_cube.shape[0],event_cube.shape[1]))
                
                event_cube = event_cube.reshape(event_cube.shape[0],event_cube.shape[1],temp_c,temp_t)

                event_cube = torch.from_numpy(event_cube)

                event_cube = event_cube.permute(3,2,0,1)

            else:
                if random.random() < 0.5:
                    event_cube = event_cube.permute(1,2,0)
                    event_cube = event_cube.numpy().astype(np.uint8)
                    #print("event cube shape 1111 ",event_cube.shape)
                    event_cube, labels = random_affine(event_cube, labels=labels, degrees=5, translate=.1, scale=.1, shear=5,
                        new_shape=(event_cube.shape[0],event_cube.shape[1]))
                    event_cube = torch.from_numpy(event_cube)
                    event_cube = event_cube.permute(2,0,1)

            #Random affine end
            
            #print("augment 222 ")
            if random.random() <=0.6:
                event_cube,labels = evhflip(event_cube,labels,new_width - 1)
            if random.random() <= 0.5:
                event_cube,labels = evvflip(event_cube,labels,new_height - 1)
            """if random.random() <= 0.5:
                event_cube,labels = rotate90antclk(event_cube,labels,new_width)
            if random.random() <= 0.5:
                event_cube,labels = rotate90clk(event_cube,labels,self.target_spatial_size)"""

        #In this case the ann array should be in bbox format

        labels = labels.reshape(-1,5)

        #Briging for bbox to yolo

        temp = labels[:,1:].copy()
        temp[:,0] = (labels[:,1] + labels[:,3]) / 2
        temp[:,1] = (labels[:,2] + labels[:,4]) / 2
        temp[:,2] = (labels[:,3] - labels[:,1])
        temp[:,3] = (labels[:,4] - labels[:,2])
        
        temp[:,0] /= new_width
        temp[:,1] /= new_height
        temp[:,2] /= new_width
        temp[:,3] /= new_height

        #Now it is in yolo format

        temp = np.clip(temp,0,0.999) 

        labels[:,1:] = temp

        labels_out = torch.zeros((len(labels), 6))

        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)
            #labels_out = labels_out[labels_out[:,1] == 0].reshape(-1,6) #ignore spalling

        #Here the event_cube is in TCHW format.

        desired_local_npz_path_idx = path.find(self.root) + len(self.root)
        local_npz_path = path[desired_local_npz_path_idx:]

        #if "dtu_4_crack_7_2238804" in local_npz_path:
        #    print("local npz_path ",local_npz_path)

        #print("local_npz_path ",local_npz_path)
        #print("sssss ",labels_out)

        #dump_image_with_labels(event_cube.permute(0,2,3,1),labels_out[:,1:],target_shape,data_visualize_output_path,index,create_histo_frames = True)
        
        #dvs_frame = dump_image_with_labels(event_cube.permute(0,2,3,1),labels_out[:,1:],target_shape,data_visualize_output_path,index,create_histo_frames = True)

        data_dict = {}


        #print("labels out issssssssssss ",labels_out)
        #print("event cube shape ... ",event_cube.shape)

        #print("dvs frame shapeeeeeeeeeeee ",dvs_frame.shape)

        data_dict['im_file'] = path
        data_dict['img'] = event_cube #torch.from_numpy(dvs_frame).float().permute(2,0,1)        #event_cube
        #data_dict['distill_frame'] = torch.from_numpy(dvs_frame).float().permute(2,0,1)
        data_dict['cls'] = labels_out[:,1].reshape(-1,1)
        data_dict['bboxes'] = labels_out[:,2:].reshape(-1,4)
        #print("data_dict 11111 ... ",data_dict)
        data_dict['ori_shape'] = [h0,w0]
        data_dict['resized_shape'] = [new_height,new_width]
        data_dict['batch_idx'] = torch.tensor([0] * data_dict['bboxes'].shape[0])
        #print("data_dict 22222222 ... ",data_dict)
        if self.mode == "val":
            #print("sssssssssssssssssss ")
            data_dict['ratio_pad'] = [[ratio,ratio], [pad[0],pad[1]]]

        #print("data dict ............... ",data_dict)
        #print("ratio pad is ",pad[0],"   ",pad[1],"   ",event_cube.shape,"  ",ratio)
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
                #print("valueeeeee  ",value.shape,"  ",value[0].shape)
                #value = value.permute(1,0,2,3,4)
            if k == 'distill_frame':
                value = torch.stack(value, 0)
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
    def generate_coco_format_labels_custom(save_path,img_paths_list,anns_for_images_list,class_names,input_img_type = 0,param_dict = None):

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
                #im = np.load(img_paths_list[i])["ev_color_img"]
                width = param_dict.quantized_w    #im.shape[1]
                height = param_dict.quantized_w #im.shape[0]
                
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
                        img_width,img_height = self.param_dict["quantized_w"],self.param_dict["quantized_h"] #npz_file["ev_color_img"].shape[1],npz_file["ev_color_img"].shape[0]
                    if self.dataset_name == "GEN1":
                        ann_list_per_image = BBOX2YOLO(ann_list_per_image,img_width,img_height)
                    else:
                        ann_list_per_image = COCO2YOLO(ann_list_per_image,img_width,img_height)

                    anns_list_for_images.append(ann_list_per_image)

                    print("11111111111111111111111111 ")
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


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    Args:
        img_path (str): Path to the folder containing images.
        imgsz (int, optional): Image size. Defaults to 640.
        cache (bool, optional): Cache images to RAM or disk during training. Defaults to False.
        augment (bool, optional): If True, data augmentation is applied. Defaults to True.
        hyp (dict, optional): Hyperparameters to apply data augmentation. Defaults to None.
        prefix (str, optional): Prefix to print in log messages. Defaults to ''.
        rect (bool, optional): If True, rectangular training is used. Defaults to False.
        batch_size (int, optional): Size of batches. Defaults to None.
        stride (int, optional): Stride. Defaults to 32.
        pad (float, optional): Padding. Defaults to 0.0.
        single_cls (bool, optional): If True, single class training is used. Defaults to False.
        classes (list): List of included classes. Default is None.
        fraction (float): Fraction of dataset to utilize. Default is 1.0 (use all data).

    Attributes:
        im_files (list): List of image file paths.
        labels (list): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        ims (list): List of loaded images.
        npy_files (list): List of numpy file paths.
        transforms (callable): Image transformation function.
    """

    def __init__(self,
                 img_path,
                 imgsz=640,
                 cache=False,
                 augment=True,
                 hyp=DEFAULT_CFG,
                 prefix='',
                 rect=False,
                 batch_size=16,
                 stride=32,
                 pad=0.0,       #0.5,
                 single_cls=False,
                 classes=None,
                 fraction=1.0):
        super().__init__()
        """Initialize BaseDataset with given configuration and options."""
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache stuff
        if cache == 'ram' and not self.check_cache_ram():
            cache = False
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.im_files]
        if cache:
            self.cache_images(cache)

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{self.prefix}{p} does not exist')
            im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f'{self.prefix}No images found in {img_path}'
        except Exception as e:
            raise FileNotFoundError(f'{self.prefix}Error loading data from {img_path}\n{HELP_URL}') from e
        if self.fraction < 1:
            im_files = im_files[:round(len(im_files) * self.fraction)]
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """include_class, filter labels to include only these classes (optional)."""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]['cls']
                bboxes = self.labels[i]['bboxes']
                segments = self.labels[i]['segments']
                keypoints = self.labels[i]['keypoints']
                j = (cls == include_class_array).any(1)
                self.labels[i]['cls'] = cls[j]
                self.labels[i]['bboxes'] = bboxes[j]
                if segments:
                    self.labels[i]['segments'] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]['keypoints'] = keypoints[j]
            if self.single_cls:
                self.labels[i]['cls'][:, 0] = 0

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]

        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f'Image Not Found {f}')
            h0, w0 = im.shape[:2]  # orig hw
            
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(round(w0 * r), self.imgsz), min(round(h0 * r), self.imgsz)) #(min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self, cache):
        """Cache images to memory or disk."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn = self.cache_images_to_disk if cache == 'disk' else self.load_image
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if cache == 'disk':
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f'{self.prefix}Caching images ({b / gb:.1f}GB {cache})'
            pbar.close()

    def cache_images_to_disk(self, i):
        """Saves an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)

    def check_cache_ram(self, safety_margin=0.5):
        """Check image caching requirements vs available memory."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = cv2.imread(random.choice(self.im_files))  # sample image
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio ** 2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = psutil.virtual_memory()
        cache = mem_required < mem.available  # to cache or not to cache, that is the question
        if not cache:
            LOGGER.info(f'{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images '
                        f'with {int(safety_margin * 100)}% safety margin but only '
                        f'{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, '
                        f"{'caching images ✅' if cache else 'not caching images ⚠️'}")
        return cache

    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop('shape') for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.round(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride #np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        
        self.batch = bi  # batch index of image

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):

        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop('shape', None)  # shape is for rect, remove it
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)  #这里还是初始图片的比例
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                              label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
        if self.rect:
            label['rect_shape'] = self.batch_shapes[self.batch[index]]
            
        return self.update_labels_info(label)

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label):
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """Users can custom augmentations here
        like:
            if self.augment:
                # Training transforms
                return Compose([])
            else:
                # Val transforms
                return Compose([])
        """
        raise NotImplementedError

    def get_labels(self):
        """Users can custom their own format here.
        Make sure your output is a list with each element like below:
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
        """
        raise NotImplementedError
