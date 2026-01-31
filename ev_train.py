import numpy as np
import os
import argparse
import random
import torch

from spikingjelly.activation_based import functional
from obd.utils.loss import v8DetectionLoss
from obd.models.postprocess import DetectionValidator
from obd.utils.utils import colorstr,check_det_dataset,TQDM
#from obd.models.ann_model import CustomModel
from obd.models import ann_repvgg #import CustomModel
#from obd.models.snn_model import CustomModel
#from obd.models.spjelly_snn_model import CustomModel
from obd.models import lava_snn_model2_sota,lava_snn_model2,snn_repvgg_sota,lava_snn_model2_feedforward,lava_snn_model2_feedforward_mean #impot CustomModel
from obd.models import snn_repvgg
from obd.data.visualize_data import dump_image_with_labels

from obd.data.ev_dataset import TrainValDataset
from torch.utils.data import dataloader
from namespace_loader import load_yaml_to_namespace
from pathlib import Path
import torch.nn.functional as F

def compute_channel_attention(feature_maps):

    """Computes channel attention weights by applying Global Average Pooling (GAP)."""
    B, C, H, W = feature_maps.shape  # Get batch, channels, height, width
    attention = F.adaptive_avg_pool2d(feature_maps, (1, 1))  # Global Average Pooling -> (B, C, 1, 1)
    attention = attention.view(B, C)  # Reshape to (B, C) for KL divergence
    return attention

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
        if "synapse"  in key:

            if "projection" in key:
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

        """elif "running_mean" in key:

            if key == "layers.1.neuron.norm1.running_mean":

                temp = "layers.1.conv3x3.bn.running_mean"

            elif key == "layers.1.neuron.norm2.running_mean":

                temp = "layers.1.conv1x1.bn.running_mean"

            elif key == "layers.2.neuron.norm1.running_mean":

                temp = "layers.2.conv3x3.bn.running_mean"

            elif key == "layers.2.neuron.norm2.running_mean":

                temp = "layers.2.conv1x1.bn.running_mean"

            elif key == "layers.3.neuron.norm.running_mean":

                temp = 'layers.3.bn.running_mean'

            elif key == "layers.4.neuron.norm.running_mean":
                
                temp = 'layers.4.bn.running_mean'

            elif key == "layers.5.neuron.norm.running_mean":
            
                temp = 'layers.5.bn.running_mean'

            elif key == "layers.6.neuron.norm.running_mean":

                temp = 'layers.6.bn.running_mean'

            elif key == "layers.7.neuron.norm.running_mean":

                temp = 'layers.7.bn.running_mean'
            
            elif key == "layers.8.neuron.norm.running_mean":

                temp = 'layers.8.bn.running_mean'
            
            elif key == "layers.9.neuron.norm.running_mean":

                temp = 'layers.9.bn.running_mean'
            
            elif key == "layers.10.neuron.norm.running_mean":

                temp = 'layers.10.bn.running_mean'

            elif key == 'detect.head_backend.0.neuron.norm.running_mean':

                temp = "detect.cv2.0.0.bn.running_mean"

            elif key == 'detect.head_backend.1.neuron.norm.running_mean':

                temp = "detect.cv2.0.1.bn.running_mean"
                
            
            if temp in checkpoint.keys():
                if checkpoint[temp].shape != model.state_dict()[key].shape:
                    print("continuing as tensor shape difference ")
                    continue
                model.state_dict()[key] -= model.state_dict()[key]
                model.state_dict()[key] += checkpoint[temp]

        elif "running_var" in key:

            if key == "layers.1.neuron.norm1.running_var":

                temp = "layers.1.conv3x3.bn.running_var"

            elif key == "layers.1.neuron.norm2.running_var":

                temp = "layers.1.conv1x1.bn.running_var"

            elif key == "layers.2.neuron.norm1.running_var":

                temp = "layers.2.conv3x3.bn.running_var"

            elif key == "layers.2.neuron.norm2.running_var":

                temp = "layers.2.conv1x1.bn.running_var"

            elif key == "layers.3.neuron.norm.running_var":

                temp = 'layers.3.bn.running_var'

            elif key == "layers.4.neuron.norm.running_var":
                
                temp = 'layers.4.bn.running_var'

            elif key == "layers.5.neuron.norm.running_var":
            
                temp = 'layers.5.bn.running_var'

            elif key == "layers.6.neuron.norm.running_var":

                temp = 'layers.6.bn.running_var'

            elif key == "layers.7.neuron.norm.running_var":

                temp = 'layers.7.bn.running_var'
            
            elif key == "layers.8.neuron.norm.running_var":

                temp = 'layers.8.bn.running_var'
            
            elif key == "layers.9.neuron.norm.running_var":

                temp = 'layers.9.bn.running_var'
            
            elif key == "layers.10.neuron.norm.running_var":

                temp = 'layers.10.bn.running_var'

            elif key == 'detect.head_backend.0.neuron.norm.running_var':

                temp = "detect.cv2.0.0.bn.running_var"

            elif key == 'detect.head_backend.1.neuron.norm.running_var':

                temp = "detect.cv2.0.1.bn.running_var"
            
            if temp in checkpoint.keys():
                if checkpoint[temp].shape != model.state_dict()[key].shape:
                    print("continuing as tensor shape difference ")
                    continue
                model.state_dict()[key] -= model.state_dict()[key]
                model.state_dict()[key] += checkpoint[temp]"""

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


def preprocess_train_batch(batch,device,normalize = True):
        """Preprocesses a batch of images by scaling and converting to float."""

        batch['img'] = batch['img'].to(device, non_blocking=True).float()
        if normalize:
            batch['img'] = batch['img'] / 255
        return batch


def get_dataloader(args,mode ="train",stridee = 32):

    height, width = (args.imgsz if isinstance(args.imgsz, tuple) else (args.imgsz, args.imgsz))
    data_parent_folder = args.ev_data_parent
    input_img_type = args.input_img_type
    batch_size = args.batch_size

    
    dataset_name = args.dataset_name

    if dataset_name == "GEN1":
        original_width = 304
        original_height = 240
    else:
        original_width = 346
        original_height = 260


    param_dict = {"TSteps" : args.TSteps, "tbins" : 1,"quantized_h" : original_height ,"quantized_w" : original_width}

    if mode == "train":
        shuffle = True
        rect = False
        img_data_csv = args.train_img_data_csv    
    else:
        shuffle = False
        rect = True
        img_data_csv = args.val_img_data_csv

    dataset = TrainValDataset(
        img_data_csv = img_data_csv,
        ann_data_csv = "",
        batch_size=batch_size,
        augment=mode=="train",
        hyp=None,
        rect=rect,
        check_images=False,
        check_labels=False,
        stride=stridee,
        pad=0.4,
        rank=-1,
        data_dict=None,
        task=mode,
        specific_shape = False,
        height=height,
        width=width,
        cache_ram=False,
        dataset_parent_folder=data_parent_folder,
        input_img_type = input_img_type,
        param_dict = param_dict,
        dataset_name = dataset_name
    )

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

def save_checkpoint(save_parent_path,current_mAP50_val,current_epoch,model_state_dict,optimizer_state_dict,scheduler_state_dict,is_last = False, epoch = -1):

    if not os.path.exists(save_parent_path):
        os.makedirs(save_parent_path)

    if is_last:
        if epoch > 0:
            checkpoint_path = "epoch_" + str(epoch) + ".pt"
        else:
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


    ##ADVICE : Uncomment and enable  detect in loss.py "the m = model #.detect  # Detect() module"
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data_info = check_det_dataset(args.data_yaml_file)
    csv_file_path = args.result_csv_path
    cfg = load_yaml_to_namespace(default_cfg_file)

    #teacher net

    if args.distill:

        #teacher_net = ann_repvgg.CustomModel(in_channels=args.in_channels,num_classes=data_info['nc'],for_distillation = args.distill)

        teacher_net = snn_repvgg.CustomModel(in_channels=args.in_channels,num_classes=data_info['nc'],T = 1, for_distillation = args.distill)

        teacher_net.to(device=device)

        #teacher_net.detect.stride = torch.tensor([args.imgsz / x.shape[-2] for x in teacher_net(torch.zeros(1,args.in_channels,args.imgsz,args.imgsz).cuda())[0]])

        teacher_net.detect.stride = torch.tensor([args.imgsz[0] / x.shape[-2] for x in teacher_net(torch.zeros(1,args.in_channels,args.imgsz[0],args.imgsz[1]).cuda())[0]]) #assume x.shape[-2] is height

        teacher_net.load_state_dict(torch.load(args.load)["model"])

        for module in teacher_net.modules():

            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = False
                print('')                
        
        for param in teacher_net.parameters():

            param.requires_grad = False

        #teacher_net.eval()

    #teacher net end

    #initialize the model
    
    #net = lava_snn_model2_sota.CustomModel(in_channels=args.in_channels,num_classes=data_info['nc'],T=7,distillation_required=True)

    net = lava_snn_model2.CustomModel(in_channels=args.in_channels,num_classes=data_info['nc'],T=7,distillation_required=True)

    #net = lava_snn_model2_feedforward_mean.CustomModel(in_channels=args.in_channels,num_classes=data_info['nc'],T=7,distillation_required = args.distill)

    #net = ann_repvgg.CustomModel(in_channels=args.in_channels,num_classes=data_info['nc'])

    #net = snn_repvgg_sota.CustomModel(in_channels = args.in_channels ,num_classes = data_info['nc'] ,for_distillation = False,T=1)

    functional.set_step_mode(net, step_mode='m')

    net = net.to(device=device)

    print(net)
    
    #temp = torch.zeros((8,3,256,256),dtype=torch.float).to(device=device)
    
    temp_height, temp_width = (args.imgsz if isinstance(args.imgsz, tuple) else (args.imgsz, args.imgsz))

    #net.detect.stride 
    net.detect.stride = torch.tensor([temp_height / x.shape[-2] for x in net(torch.zeros(1,args.in_channels,temp_height,temp_width).cuda())[0]])

    net.initialize_model_weights()

    #sss = torch.load(args.load)["model"]

    #load_model_for_synapse_weights(net,args.load,device,args.from_snn_distill)

    load_model_for_available_keys(net,torch.load(args.load)["model"],device)

    #net.load_state_dict(torch.load(args.load)["model"])

    functional.reset_net(net)

    #oo = net(temp)

    #train_loader = get_dataloader(mode="train", args = args,data = data_info,cfg = cfg,stridee = net.detect.stride)
    #test_loader = get_dataloader(mode="val", args = args,data = data_info,cfg = cfg,stridee = net.detect.stride)

    train_loader = get_dataloader(args,mode ="train",stridee = net.detect.stride)
    test_loader = get_dataloader(args,mode ="val",stridee = net.detect.stride)
                                 

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

    if args.sparsity:
        sparsity_montior = lava_snn_model2_feedforward_mean.SparsityChecker(
            max_rate=args.sp_rate, lam=args.sp_lam)
    else:
        sparsity_montior = None
    
    for epoch in range(current_epoch,args.epoch):


        """print(f'{epoch=}')
        net.train()
        train_pbar = TQDM(enumerate(train_loader), total=len(train_loader))

        #For lava,lava_dl, the input shape is [N,C,H,W,T]
        #For spikingjelly, the input shape is [T,N,C,H,W] 
        
        for i,batch in train_pbar:
                
                #For lava,lava_dl, the input shape is [N,C,H,W,T]
                #For spikingjelly, the input shape is [T,N,C,H,W] 
                  
                #inputs = inputs.permute(4,0,1,2,3)
                #inputs = inputs.to(device)
                
                if args.distill:

                    with torch.no_grad():


                        #dvs_frame = dump_image_with_labels(batch['img'].permute(0,2,3,1),batch['cls'],batch['resized_shape'] ,"./temp",0,create_histo_frames = True)
                        #dvs_frame = dvs_frame.to(device, non_blocking=True).float()
                        #batch['distill_frame']
                        t_preds,t_feature_maps = teacher_net(batch['img'].to(device, non_blocking=True).float())

                    
                    #batch['img'] = batch['img'].permute(1,0,2,3,4)

                    batch = preprocess_train_batch(batch,device,normalize=False)

                    preds,feature_maps,attention_maps = net(batch['img'])
                    
                    student_attentions = []
                    teacher_attentions = []

                    #if epoch > -1:

                    #    #teacher_channel_attention1 = compute_channel_attention(t_feature_maps[1].mean(0))
                    #    #student_channel_attention1 = compute_channel_attention(attention_maps[0])

                    #    teacher_channel_attention2 = compute_channel_attention(t_feature_maps[2].mean(0))
                    #    student_channel_attention2 = compute_channel_attention(attention_maps[1])
                        
                        
                    #    #teacher_attentions.append(teacher_channel_attention1)
                    #    teacher_attentions.append(teacher_channel_attention2)

                    #    #student_attentions.append(student_channel_attention1)
                    #    student_attentions.append(student_channel_attention2)


                    #loss,train_loss_items = loss_criterion(preds,batch,with_distillation =args.distill,epoch=epoch,teacher_preds=t_preds,s_feature_maps = feature_maps,t_feature_maps = t_feature_maps)
                    loss,train_loss_items = loss_criterion(preds,batch,with_distillation =args.distill,epoch=epoch,teacher_preds=t_preds,s_feature_maps = feature_maps,t_feature_maps = t_feature_maps,s_attention = student_attentions,t_attention = teacher_attentions,from_snn_distill = args.from_snn_distill)
                
                else:
                    
                    batch['img'] = batch['img'].permute(1,0,2,3,4) #only when direct spikes

                    batch = preprocess_train_batch(batch,device,normalize=False)

                    preds,feature_maps = net(batch['img'])
                
                    loss,train_loss_items = loss_criterion(preds,batch)

                if torch.isnan(loss):
                    functional.reset_net(net)
                    print("loss is nan, continuing")
                    continue
                
                if sparsity_montior is not None:

                        loss += sparsity_montior.loss
                        sparsity_montior.clear()

                tloss = (tloss * i + train_loss_items) / (i + 1) if tloss is not None \
                        else train_loss_items

                clip = 1.
                optimizer.zero_grad()
                loss.backward()
                #net.validate_gradients()
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
                optimizer.step()

                functional.reset_net(net)"""

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
                #batch['img'] = batch['img'].permute(1,0,2,3,4) #only when direct spikes

                batch = validator.preprocess(batch,normalize=False)

                preds,feature_maps = net(batch['img'])
                #preds = teacher_net(batch['img'])

                val_loss += loss_criterion(preds,batch)[1]
                preds = validator.postprocess(preds)
                validator.update_metrics(preds, batch)
                
                #visualize start
                temp_ann_arr = torch.concatenate((preds[0][:,5].reshape(-1,1),preds[0][:,:4].reshape(-1,4),preds[0][:,4].reshape(-1,1)),axis = 1)
                #idx = random.randint(1, 1000)
                #distill_frame
                idx = batch["im_file"][0].rsplit("/",1)[1]
                idx = idx.rsplit(".",1)[0]
                ev_save_path = "/home/nrthpc/Udaya_stuff/evCIVIL_direct_train/ultralytic_compatible_ANN_SNN_snn_repvgg/visualize_data_folder/model2_GEN1"
                dump_image_with_labels(batch['distill_frame'].permute(0,2,3,1)[0],temp_ann_arr,[192,256],ev_save_path,idx,create_histo_frames = False,annotation_format="bbox")
                #visualize end
                
                functional.reset_net(net)
            
            stats = validator.get_stats()
            #validator.check_stats(stats)
  
            validator.finalize_metrics()
            validator.print_results()

            results = {**stats, **label_loss_items(val_loss.cpu() / len(test_loader), prefix='val')}
            metrics = {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
           
            lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(optimizer.param_groups)}  # for loggers
            save_metrics(metrics={**label_loss_items(tloss), **metrics, **lr},result_csv_file=csv_file_path,epoch=epoch)

            if results['metrics/mAP50(B)'] > current_max_val + 0.005:
                
                current_max_val = results['metrics/mAP50(B)']
                save_checkpoint(os.path.join(args.output_dir,"checkpoints"),current_max_val,epoch,net.state_dict(),optimizer.state_dict(),scheduler.state_dict())

            if epoch > 5:
                save_checkpoint(os.path.join(args.output_dir,"checkpoints"),results['metrics/mAP50(B)'],epoch,net.state_dict(),optimizer.state_dict(),scheduler.state_dict(),is_last = True)
            
            if epoch % 10 == 0:  
                save_checkpoint(os.path.join(args.output_dir,"checkpoints"),results['metrics/mAP50(B)'],epoch,net.state_dict(),optimizer.state_dict(),scheduler.state_dict(),is_last = True)
            
        stats.update()
        scheduler.step()

        #torch.cuda.empty_cache()

def get_args():

    #For SNN model initialization is very important... 

    parser = argparse.ArgumentParser()

    #parser.add_argument('-load',type=str,default="/home/nrthpc/Udaya_stuff/evCIVIL_direct_train/ultralytic_compatible_ANN_SNN_snn_repvgg/out_dir/snnrepvgg_gen1_1T_32_maxval/best.pt")    #"/media/atiye/Data/Udaya_Research_stuff/patch_based/checkpoints/fused_models_mean_only/mean_only_fuse_temp2.pt")   
    
    #parser.add_argument('-load',type=str,default="/home/nrthpc/Udaya_stuff/Result_Collector/ann_mini/evCIVIL_event/best.pt")   #"/media/atiye/Data/Udaya_Research_stuff/patch_based/checkpoints/fused_models_mean_only/mean_only_fuse_temp2.pt")   
    parser.add_argument('-epoch',  type=int, default=300, help='number of epochs to run')
    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/From_server_ultralytic_compatible_ANN_SNN_event_based_to_slayer/checkpoints/evCIVIL_lava_event_based/direct_trained_38_16_mAP/best.pt")
    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/From_server_ultralytic_compatible_ANN_SNN_event_based_to_slayer/./checkpoints/evCIVIL_lava_img_based/evCIVIL_lava_3.42_after_trained/2.94_undecayed_best_results/best.pt")
    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/Result_Collector/snn_mini/evCIVIL_img/last_399.pt")
    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/Result_Collector/ann_mini/evCIVIL_img/custom/last_307.pt")
    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/Result_Collector/snn_mini/evCIVIL_img/last_399.pt")
    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/Result_Collector/snn_mini/evCIVIL_event/best.pt")
    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/Result_Collector/snn_mini/evCIVIL_event/best.pt")
    #parser.add_argument("-load",default="/home/nrthpc/Udaya_stuff/Result_Collector/lava_mini_direct/gen1/best_initialized_with_snn.pt",type=str)
    #parser.add_argument("-load",default="/home/nrthpc/Udaya_stuff/Result_Collector/lava_mini_direct/gen1/best_normal_init.pt",type=str)
    #parser.add_argument("-load",default="/home/nrthpc/Udaya_stuff/From_server_ultralytic_compatible_ANN_SNN_event_based_to_slayer/mmm.pt")
    
    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/evCIVIL_direct_train/ultralytic_compatible_ANN_SNN_snn_repvgg/out_dir/bbb/checkpoints/best.pt.old")
    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/Result_Collector/SOTA_Anchorfree_ann/evCIVIL_event/best.pt")
    #parser.add_argument("-load",type)
    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/From_server_ultralytic_compatible_ANN_SNN_event_based_to_slayer/checkpoints/SOTA_ANCHORFREE/evCIVIL_lava/feedforward_best.pt")
    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/Result_Collector/lava_mini_distilled/evCIVIL_event/best.pt")
    #parser.add_argument("-load",default="/home/nrthpc/Udaya_stuff/From_server_ultralytic_compatible_ANN_SNN_event_based_to_slayer/checkpoints/SOTA_ANCHORFREE/GEN1_lava/gen1_lava_direct_sota_anchorfree.pt")
    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/Result_Collector/SOTA_Anchorfree_ann/evCIVIL_event/best.pt")
    #parser.add_argument("-load",default="/home/nrthpc/Udaya_stuff/evCIVIL_direct_train/ultralytic_compatible_ANN_SNN_snn_repvgg/out_dir/bbb/checkpoints/best.pt")
    #parser.add_argument('-output_dir', type=str, default='/media/atiye/Data/Udaya_Research_stuff/spikingjelly/yolov3_2_snn_head1_model/', help='directory in which to put log folders')
    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/From_server_ultralytic_compatible_ANN_SNN_event_based_to_slayer/checkpoints/SOTA_ANCHORFREE/evCIVIL_lava/feedforward_best.pt")
    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/From_server_ultralytic_compatible_ANN_SNN_event_based_to_slayer/checkpoints/SOTA_ANCHORFREE/evCIVIL_lava/direct_spike/best.pt")
    #parser.add_argument("-load",default="/home/nrthpc/Udaya_stuff/From_server_ultralytic_compatible_ANN_SNN_event_based_to_slayer/checkpoints/SOTA_ANCHORFREE/GEN1_lava/gen1_lava_direct_sota_anchorfree.pt")
    #parser.add_argument("-load",type=str,deault="/home/nrthpc/Udaya_stuff/From_server_ultralytic_compatible_ANN_SNN_event_based_to_slayer/checkpoints/SOTA_ANCHORFREE/evCIVIL_lava/last.pt")
    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/Result_Collector/lava_mini_distilled/evCIVIL_event/best.pt")
    parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/Result_Collector/lava_mini_distilled/gen1/best_initialized_with_snn.pt")
    parser.add_argument('-output_dir', type=str, default='/home/nrthpc/Udaya_stuff/evCIVIL_direct_train/ultralytic_compatible_ANN_SNN_snn_repvgg/out_dir/bbb', help='directory in which to put log folders')

    parser.add_argument('-num_workers', type=int, default=1, help='number of dataloader workers')
    
    #parser.add_argument("-train_data_path",type=str,default="/media/atiye/Data/Udaya_Research_stuff/PASCALVOC_FINAL/trainval",help="csv file...") #night_outdoor_and_daytime_train_files_event_based.txt
    #parser.add_argument("-test_data_path",type=str,default="/media/atiye/Data/Udaya_Research_stuff/PASCALVOC_FINAL/test",help="csv file...")

    parser.add_argument('-batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('-result_csv_path',type=str,default="./result_ev_lava_gen1_standalone_7T.csv")
    parser.add_argument('-imgsz',type=int,default=(192,256),help="reisized img size")
    parser.add_argument('-lr',type=float,default=0.001,help="learning rate")

    parser.add_argument("-distill",action="store_true", default=False, help="Enable distillation for training")
    parser.add_argument("-from_snn_distill",action="store_true", default=True, help="Enable distillation for training")
    
    #parser.add_argument("-teacher",type=str,default="/media/atiye/Data/Udaya_Research_stuff/ultralytic_compatible_ANN_SNN/checkpoints/evCIVIL_ann_repvgg_img/best.pt",help="path for the teacher model")
    #parser.add_argument("-teacher",type=str,default="/media/atiye/Data/Udaya_Research_stuff/ultralytic_compatible_ANN_SNN/checkpoints/evCIVIL_img_grayscale_60_nms_15mAP/last.pt",help="path for the teacher model")
    #parser.add_argument("-teacher",type=str,default="/work3/kniud/object_detection/evCIVIL_ultralytic_direct_train/ultralytic_compatible_ANN_SNN/out_dir_img/checkpoints/last_307.pt")
    #parser.add_argument("-teacher",type=str,default="/work3/kniud/object_detection/evCIVIL_ultralytic_direct_train/ultralytic_compatible_ANN_SNN/distilled_img_ckpts/lava_2.94_undecayed/best.pt")
    parser.add_argument("-teacher",type=str,default="/work3/kniud/object_detection/evCIVIL_ultralytic_direct_train/ultralytic_compatible_ANN_SNN/out_dir_snn/checkpoints/best.pt")
    parser.add_argument("-input_img_type",type=int, default=3.0, help="")
    parser.add_argument("-in_channels",type=int,default=2,help="number of in channels")


    """parser.add_argument("-ev_data_parent",type=str,default="/home/nrthpc/Udaya_stuff/latest_dataset",help="parent folder that holds the ev dataset")
    parser.add_argument("-train_img_data_csv",type=str,default="/home/nrthpc/Udaya_stuff/latest_dataset/night_outdoor_and_daytime_train_files_event_based.txt",help="")
    parser.add_argument("-val_img_data_csv",type=str,default="/home/nrthpc/Udaya_stuff/latest_dataset/night_outdoor_test_files_event_based.txt",help="")
    #parser.add_argument("-val_img_data_csv",type=str,default="/home/nrthpc/Udaya_stuff/latest_dataset/test_files_event_based.txt")
    parser.add_argument("-TSteps",type=int,default=7,help="Tsteps for the event cube bins")
    parser.add_argument("--dataset_name",type=str,default="evCIVIL",help="GEN1,evCIVIL")
    parser.add_argument('-data_yaml_file',type=str,default="./defect.yaml")"""

    parser.add_argument("-ev_data_parent",type=str,default="/home/nrthpc/Yan_SC_Stuff/prophesee_processed_dataset",help="parent folder that holds the ev dataset")
    parser.add_argument("-train_img_data_csv",type=str,default="/home/nrthpc/Yan_SC_Stuff/prophesee_processed_dataset/gen1_train_a.csv",help="")
    parser.add_argument("-val_img_data_csv",type=str,default="/home/nrthpc/Yan_SC_Stuff/prophesee_processed_dataset/gen1_test_a.csv",help="")
    parser.add_argument("-TSteps",type=int,default=7,help="Tsteps for the event cube bins")
    parser.add_argument("--dataset_name",type=str,default="GEN1",help="GEN1,evCIVIL")
    parser.add_argument('-data_yaml_file',type=str,default="./defect.yaml")

    parser.add_argument('-sparsity', action ='store_true', default=False, help='enable sparsity loss')
    parser.add_argument('-sp_lam',   type=float, default=0.1, help='sparsity loss mixture ratio')
    parser.add_argument('-sp_rate',  type=float, default=0.01, help='minimum rate for sparsity penalization')


    args = parser.parse_args()

    return args

if __name__ == "__main__":
        args = get_args()
        main(args)