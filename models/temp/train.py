
import torch
import numpy as np
import os
import argparse
import random

from spikingjelly.activation_based import functional
from obd.utils.loss import v8DetectionLoss
from obd.models.postprocess import DetectionValidator
from obd.utils.utils import colorstr,check_det_dataset,TQDM
#from obd.models import ann_model
from obd.models import ann_repvgg #import CustomModel
#from obd.models.snn_model import CustomModel
#from obd.models.spjelly_snn_model import CustomModel
from obd.models import lava_snn_model2 #import CustomModel
from obd.models import snn_repvgg

from obd.data.dataset import YOLODataset
from torch.utils.data import dataloader
from namespace_loader import load_yaml_to_namespace
from pathlib import Path
from PIL import Image


def load_model_for_available_keys(model,checkpoint,device):
    
    #We added this because we need to initialize the event-histogram detector 
    #with the weights of image-based detector. Here the histogram detector is 2 channel and 
    # the image-based detector is 3 channel input.
    
    for key in model.state_dict().keys():
        temp = key
        if  temp in checkpoint.keys():
            #print("key is ",key)
            if checkpoint[temp].shape != model.state_dict()[key].shape:
                print("continuing as tensor shape difference ")
                continue
            model.state_dict()[key] -= model.state_dict()[key]
            model.state_dict()[key] += checkpoint[temp].to(device)

def load_model_for_synapse_weights(model,ckpt_state_dict_path,device,from_snn = False):
     
    checkpoint = torch.load(ckpt_state_dict_path)["model"]
    
    for key in model.state_dict().keys():
        if "synapse" not in key:
            continue

        a,b,c,d = key.rsplit(".",3)
        print('')
        #if  temp in checkpoint.keys():
        g = a
        if from_snn:
            h = int(b)
        else:
            h = int(b) - 1
        i = c
        j = d
        if c == 'synapse1':
            if from_snn:
                i = "conv3x3.conv"
            else:
                i = "conv3x3"
        elif c == 'synapse2':
            if from_snn:
                i = "conv1x1.conv"
            else:
                i = "conv1x1"
        else:
            i = "conv"

        temp = g + "." + str(h) + "." + i + "." + j

        if temp in checkpoint.keys():
            if checkpoint[temp].shape != model.state_dict()[key].shape:
                print("continuing as tensor shape difference ")
                continue
            model.state_dict()[key] -= model.state_dict()[key]
            model.state_dict()[key] += checkpoint[temp]


class InfiniteDataLoader(dataloader.DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """
        Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)

def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def preprocess_train_batch(batch,device):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(device, non_blocking=True).float() / 255
        return batch

def get_dataloader(mode="train", args = None,data = None,cfg = None,stridee = 32):

        if mode == "train":
             img_path = os.path.join(data['path'],data['train'])
             shuffle = True
        else:
             shuffle = False
             img_path = os.path.join(data['path'],data['val'])

        dataset = YOLODataset(
                img_path=img_path,
                imgsz=args.imgsz,
                batch_size=args.batch_size,
                augment=mode == 'train',  # augmentation
                hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
                rect=False if mode == 'train' else True,  # rectangular batches, True for val and False for train
                cache=cfg.cache or None,
                single_cls=cfg.single_cls or False,
                stride=int(stridee),
                pad=0.0 if mode == 'train' else 0.4,
                prefix=colorstr(f'{mode}: '),
                use_segments=cfg.task == 'segment',
                use_keypoints=cfg.task == 'pose',
                classes=cfg.classes,
                data=data,
                fraction=cfg.fraction if mode == 'train' else 1.0)
     
        
        nd = torch.cuda.device_count()  # number of CUDA devices
        nw = 4
        sampler = None
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205)
        return InfiniteDataLoader(dataset=dataset,
                                batch_size=args.batch_size,
                                shuffle=shuffle and sampler is None,
                                num_workers=nw,
                                sampler=sampler,
                                pin_memory=True,
                                collate_fn=getattr(dataset, 'collate_fn', None),
                                worker_init_fn=seed_worker,
                              generator=generator)

        """return DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=getattr(dataset, 'collate_fn', None),
                        num_workers=args.num_workers,
                        pin_memory=True,drop_last=True)"""
        

def label_loss_items(self, loss_items=None, prefix='train'):
        """Returns a loss dict with labelled training loss items tensor."""
        # Not needed for classification but necessary for segmentation & detection
        return {'loss': loss_items} if loss_items is not None else ['loss']

def save_metrics(metrics,result_csv_file,epoch):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = '' if os.path.exists(result_csv_file) else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header
        with open(result_csv_file, 'a') as f:
            f.write(s + ('%23.5g,' * n % tuple([epoch + 1] + vals)).rstrip(',') + '\n')


def label_loss_items(loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        loss_names = 'box_loss', 'cls_loss', 'dfl_loss'

        keys = [f'{prefix}/{x}' for x in loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

def save_checkpoint(save_parent_path,current_mAP50_val,current_epoch,model_state_dict,optimizer_state_dict,scheduler_state_dict,is_last = False):

    if not os.path.exists(save_parent_path):
        os.makedirs(save_parent_path)

    if is_last:
        checkpoint_path = "last.pt" #f"{current_epoch}_{current_mAP50_val:.5f}.pt"
    else:
        checkpoint_path = "best.pt"

    checkpoint_path = os.path.join(save_parent_path,checkpoint_path)

    # Save the checkpoint
    full_ckpt = {
        "model": model_state_dict,
        "epoch": current_epoch,
        "optim": optimizer_state_dict,
        "schedule": scheduler_state_dict
    }

    torch.save(full_ckpt, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def main(args,default_cfg_file = "./default.yaml"):
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data_info = check_det_dataset(args.data_yaml_file)
    csv_file_path = args.result_csv_path
    cfg = load_yaml_to_namespace(default_cfg_file)

    #teacher net

    if args.distill:

        teacher_net = snn_repvgg.CustomModel(in_channels=args.in_channels,num_classes=data_info['nc'],T = 1, for_distillation = args.distill)

        teacher_net.to(device=device)

        teacher_net.detect.stride = torch.tensor([args.imgsz / x.shape[-2] for x in teacher_net(torch.zeros(1,args.in_channels,args.imgsz,args.imgsz).cuda())[0]])

        teacher_net.load_state_dict(torch.load(args.teacher)["model"])

        for module in teacher_net.modules():

            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = False
                print('')                
        
        for param in teacher_net.parameters():

            param.requires_grad = False

        #teacher_net.eval()

    #teacher net end

    #initialize the model
    
    net = lava_snn_model2.CustomModel(in_channels=args.in_channels,num_classes=data_info['nc'],T=7,distillation_required = args.distill)

    #net = ann_repvgg.CustomModel(in_channels = args.in_channels,num_classes = data_info['nc']) #for_distillation = False)

    #net = ann_repvgg.CustomModel(in_channels = args.in_channels,num_classes = data_info['nc'])  #,T = 7) 

    functional.set_step_mode(net, step_mode='m')

    net = net.to(device=device)

    print(net)
    
    #temp = torch.zeros((8,3,256,256),dtype=torch.float).to(device=device)

    temp_height, temp_width = (args.imgsz if isinstance(args.imgsz, tuple) else (args.imgsz, args.imgsz))

    net.detect.stride = torch.tensor([temp_height / x.shape[-2] for x in net(torch.zeros(1,args.in_channels,temp_height,temp_width).cuda())[0]])

    #net.detect.stride = torch.tensor([args.imgsz / x.shape[-2] for x in net(torch.zeros(1,args.in_channels,args.imgsz,args.imgsz).cuda())[0]])

    net.initialize_model_weights()
 
    #sss = torch.load(args.load)["model"]

    load_model_for_synapse_weights(net,args.teacher,device,args.from_snn_distill)

    #load_model_for_available_keys(net,torch.load(args.load)["model"],device)

    #net.load_state_dict(torch.load(args.teacher)["model"])

    functional.reset_net(net)

    #oo = net(temp)

    train_loader = get_dataloader(mode="train", args = args,data = data_info,cfg = cfg,stridee = net.detect.stride)
    test_loader = get_dataloader(mode="val", args = args,data = data_info,cfg = cfg,stridee = net.detect.stride)

    #initialize the optimizers and scheduler

    current_epoch = 0

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr,weight_decay=5e-4)
    """optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay = 0.0005,
                                nesterov = True)"""

    print('Creating Optimizer')
    """optimizer = torch.optim.Adam(net.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.wd)"""
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.epoch,
    )

    loss_criterion = v8DetectionLoss(net,cfg,max_epoch=args.epoch)
    validator = DetectionValidator(model=net,data=data_info,cfg=cfg,device=device,save_dir=Path("./runs"),imgsz = args.imgsz,is_train=False)

    # Define learning rate s-heduler
    def lf(x):
        return (min(x / args.warmup, 1)
                * ((1 + np.cos(x * np.pi / args.epoch)) / 2)
                * (1 - args.lrf)
                + args.lrf)

    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    
    tloss = 0
    train_loss_items = None
    current_max_val = 0.0
    
    

    for epoch in range(current_epoch,args.epoch):

        print(f'{epoch=}')
        net.train()
        train_pbar = TQDM(enumerate(train_loader), total=len(train_loader))
        
        for i,batch in train_pbar:
                
                #For lava,lava_dl, the input shape is [N,C,H,W,T]
                #For spikingjelly, the input shape is [T,N,C,H,W] 
                  
                #inputs = inputs.permute(4,0,1,2,3)
                #inputs = inputs.to(device)

                idx = random.randint(0, 7)

                #weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 3, 1, 1)
                #gray_batch = (batch['img'].float() * weights).sum(dim=1).to(torch.uint8) 
                #batch['img'] = gray_batch.unsqueeze(1)

                batch = preprocess_train_batch(batch,device)
                preds,feature_maps = net(batch['img'])
                
                if args.distill:

                    with torch.no_grad():

                        t_preds,t_feature_maps = teacher_net(batch['img'])

                    loss,train_loss_items = loss_criterion(preds,batch,with_distillation =args.distill,epoch=epoch,teacher_preds=t_preds,s_feature_maps = feature_maps,t_feature_maps = t_feature_maps,from_snn_distill = args.from_snn_distill)
                
                else:
                    loss,train_loss_items = loss_criterion(preds,batch)

                if torch.isnan(loss):
                    functional.reset_net(net)
                    print("loss is nan, continuing")
                    continue

                tloss = (tloss * i + train_loss_items) / (i + 1) if tloss is not None \
                        else train_loss_items

                clip = 1.
                optimizer.zero_grad()
                loss.backward()
                #net.validate_gradients()
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
                optimizer.step()

                functional.reset_net(net)

        net.eval()

        validator.init_metrics()

        temp = torch.tensor([2.1803, 1.9852, 1.3019], device='cuda:0')

        with torch.no_grad():
            
            val_loss = torch.zeros_like(temp, device=device)

            val_pbar = TQDM(test_loader, desc=validator.get_desc(), total=len(test_loader))
            
            for i,batch in enumerate(val_pbar):

                validator.batch_i = i

                #start
                #inputs = inputs.permute(4,0,1,2,3)
                #inputs = inputs.to(device)

                """weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 3, 1, 1)
                gray_batch = (batch['img'].float() * weights).sum(dim=1).to(torch.uint8)
                batch['img'] = gray_batch.unsqueeze(1)"""

                batch = validator.preprocess(batch)

                preds,feature_maps = net(batch['img'])
                #preds = teacher_net(batch['img'])

                val_loss += loss_criterion(preds,batch)[1]
                preds = validator.postprocess(preds)
                validator.update_metrics(preds, batch)
                
                functional.reset_net(net)
            
            stats = validator.get_stats()
            #validator.check_stats(stats)
  
            validator.finalize_metrics()
            validator.print_results()

            results = {**stats, **label_loss_items(val_loss.cpu() / len(test_loader), prefix='val')}
            metrics = {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
           
            lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(optimizer.param_groups)}  # for loggers
            save_metrics(metrics={**label_loss_items(tloss), **metrics, **lr},result_csv_file=csv_file_path,epoch=epoch)

            if results['metrics/mAP50(B)'] > current_max_val + 0.008:
                
                current_max_val = results['metrics/mAP50(B)']
                save_checkpoint(os.path.join(args.output_dir,"checkpoints"),current_max_val,epoch,net.state_dict(),optimizer.state_dict(),scheduler.state_dict())

            if epoch > 5:
                save_checkpoint(os.path.join(args.output_dir,"checkpoints"),results['metrics/mAP50(B)'],epoch,net.state_dict(),optimizer.state_dict(),scheduler.state_dict(),is_last = True)
            
        stats.update()
        scheduler.step()

        #torch.cuda.empty_cache()


def get_args():

    #For SNN model initialization is very important... 

    parser = argparse.ArgumentParser()

    parser.add_argument('-load',type=str,default="/media/atiye/Data/Udaya_Research_stuff/ultralytic_compatible_ANN_SNN/checkpoints/evCIVIL_snn_img/last.pt")    #"/media/atiye/Data/Udaya_Research_stuff/patch_based/checkpoints/fused_models_mean_only/mean_only_fuse_temp2.pt")   
  
    parser.add_argument('-epoch',  type=int, default=300, help='number of epochs to run')

    #parser.add_argument('-output_dir', type=str, default='/media/atiye/Data/Udaya_Research_stuff/spikingjelly/yolov3_2_snn_head1_model/', help='directory in which to put log folders')
    parser.add_argument('-output_dir', type=str, default='/media/atiye/Data/Udaya_Research_stuff/ultralytic_compatible_ANN_SNN', help='directory in which to put log folders')
    parser.add_argument('-num_workers', type=int, default=8, help='number of dataloader workers')

    #parser.add_argument("-train_data_path",type=str,default="/media/atiye/Data/Udaya_Research_stuff/PASCALVOC_FINAL/trainval",help="csv file...") #night_outdoor_and_daytime_train_files_event_based.txt
    #parser.add_argument("-test_data_path",type=str,default="/media/atiye/Data/Udaya_Research_stuff/PASCALVOC_FINAL/test",help="csv file...")

    parser.add_argument('-batch_size', type=int, default=8, help='batch_size')
    parser.add_argument('-data_yaml_file',type=str,default="./voc.yaml")
    parser.add_argument('-result_csv_path',type=str,default="./results.csv")
    parser.add_argument('-imgsz',type=int,default=256,help="reisized img size")
    parser.add_argument('-in_channels',type=int,default=3,help="number of in channels ")
    parser.add_argument('-lr',type=float,default=0.001,help="learning rate")
    parser.add_argument("-distill",action="store_true", default=True, help="Enable distillation for training")
    parser.add_argument("-from_snn_distill",action="store_true", default=True, help="Enable distillation for training")
    #parser.add_argument("-teacher",type=str,default="/media/atiye/Data/Udaya_Research_stuff/ultralytic_compatible_ANN_SNN/checkpoints/ann_40_regmax_repvgg_backbone_256/best.pt",help="path for the teacher model")
    parser.add_argument("-teacher",type=str,default='/media/atiye/Data/Udaya_Research_stuff/ultralytic_compatible_ANN_SNN/checkpoints/snn_repvgg_voc_quantized_3max/last2.pt',help="path for the teacher model")
    args = parser.parse_args()
    #snn_repvgg_voc_quantized_3max/last.pt
    return args

if __name__ == "__main__":
        args = get_args()
        main(args)