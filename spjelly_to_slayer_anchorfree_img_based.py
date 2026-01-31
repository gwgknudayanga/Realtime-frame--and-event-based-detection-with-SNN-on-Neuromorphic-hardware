
import torch
from spikingjelly.activation_based import functional
from obd.models import lava_snn_model2,lava_snn_model2_feedforward,lava_snn_model2_feedforward_mean,lava_snn_model2_sota_mean

from obd.utils.utils import colorstr,check_det_dataset,TQDM
from namespace_loader import load_yaml_to_namespace
from obd.data.dataset import YOLODataset
from torch.utils.data import dataloader
import os
import argparse
import random
import torch.nn as nn
from obd.models.postprocess import DetectionValidator
from pathlib import Path
from ultralytics.utils.tal import TORCH_1_10, dist2bbox, make_anchors
import h5py

import numpy as np


from lava.lib.dl import slayer
from lava.lib.dl import netx

dequantizer = netx.modules.Dequantize(exp=6,num_raw_bits=24)
#quantizer = netx.modules.Quantize(exp= 6)



class SpikeDFL2_lava(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)  #[0,1,2,...,15]
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1)) #这里不是脉冲驱动的，但是是整数乘法
        self.c1 = c1  #本质上就是个加权和。输入是每个格子的概率(小数)，权重是每个格子的位置(整数)

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)  # 原版

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

def get_dataloader(mode="train", args = None,data = None,cfg = None,stridee = 32,no_rect = True,strict_no_rect = False):

        if mode == "train":
             img_path = os.path.join(data['path'],data['train'])
             shuffle = True
        else:
            shuffle = True
            img_path = os.path.join(data['path'],data['val'])

            if strict_no_rect:
                mode = "train"

        imgsz = 256 if isinstance(args.imgsz, tuple) else args.imgsz

        dataset = YOLODataset(
                img_path=img_path,
                imgsz=imgsz,
                batch_size=args.batch_size,
                augment=False,  # augmentation
                hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
                rect=False if mode == "train" else True,  # rectangular batches, True for val and False for train
                cache=cfg.cache or None,
                single_cls=cfg.single_cls or False,
                stride=int(stridee),
                pad=0.0 if mode == 'train' else 0.4,
                prefix=colorstr(f'{mode}: '),
                use_segments=cfg.task == 'segment',
                use_keypoints=cfg.task == 'pose',
                classes=cfg.classes,
                data=data,
                fraction=1.0)
     
        
        #nd = torch.cuda.device_count()  # number of CUDA devices
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
        checkpoint_path = "last_" + str(current_epoch) + ".pt"  #f"{current_epoch}_{current_mAP50_val:.5f}.pt"
    else:
        checkpoint_path = "best_" + str(current_epoch) + ".pt"

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

def get_config_parameters():
    return

def _quantize_8bit(x: torch.tensor,
                            scale: int = (1 << 6),
                            descale: bool = False) -> torch.tensor:
                
                #to quantize integer representation
                temp = torch.clamp(slayer.utils.quantize_hook_fx(x, scale=scale,
                                                    num_bits=8, descale=True),-254,254)
                #to fixed-point representation
                temp2 = dequantizer(temp.cpu().detach().numpy())
                #temp2 = torch.from_numpy(temp2).float().cuda()

                if descale:
                    #return slayer.utils.quantize_hook_fx(torch.from_numpy(temp2), scale=scale,
                    #                           num_bits=8, descale=True)
                
                    #quantizer = netx.modules.Quantize(exp= 6)
                    return temp
                else:
                    """return slayer.utils.quantize_hook_fx(x, scale=scale,
                                                    num_bits=8, descale=False)"""
                    return torch.from_numpy(temp2).float().cuda()


def export_hdf5(model,filename : str):
    
    h = h5py.File(filename,'w')
    layer = h.create_group('layer')
    offset = 0

    for i, b in enumerate(model):
        b.export_hdf5(layer.create_group(f'{i + offset}'))

class AnchorFreePredDecoder():
     
    def __init__(self,reg_max,num_classes,stride,device):
        super(AnchorFreePredDecoder,self).__init__()

        self.nc = num_classes
        self.reg_max = reg_max
        self.no = 4 * self.reg_max + self.nc
        self.stride = stride

        self.anchors = []
        self.strides = []
        self.shape = None

        self.dfl = SpikeDFL2_lava(self.reg_max) if self.reg_max > 1 else nn.Identity()
        #self.dfl.to(device=device)
    
    def decode_predictions(self,x,shape):

        #x : prediction
        #shape : shape for 24 out channel equivalent
        x = [x]

        if self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1) #box: [B,reg_max * 4,anchors]
        
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return (y, x)


def main(args,default_cfg_file = "./default.yaml"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_info = check_det_dataset(args.data_yaml_file)
    csv_file_path = args.result_csv_path
    cfg = load_yaml_to_namespace(default_cfg_file)

    classes_output = {'evCIVIL': 2}
    print('Creating Network')

    net = lava_snn_model2_feedforward_mean.CustomModel(in_channels=args.in_channels,num_classes=data_info['nc'],T=args.TSteps,distillation_required = args.distill)
    #lava_snn_model2_sota_mean
    #net = lava_snn_model2_sota_mean.CustomModel(in_channels=args.in_channels,num_classes=data_info['nc'],T=7,distillation_required = args.distill)

    functional.set_step_mode(net, step_mode='m')

    net = net.to(device=device)

    print(net)

    temp_height, temp_width = (args.imgsz if isinstance(args.imgsz, tuple) else (args.imgsz, args.imgsz))

    net.stride = torch.tensor([temp_height / x.shape[-2] for x in net(torch.zeros(1,args.in_channels,temp_height,temp_width).to(device=device))[0]])

    net.load_state_dict(torch.load(args.load)["model"])

    functional.reset_net(net)


    nett = net.to_slayer().to(device)
    nett.eval()

    neuron_conv_params = {
                'threshold'     : 1.25,
                'current_decay' : 1.0,
                'voltage_decay' : 0.25,
                'requires_grad' : False,
    }

    neuron_output_params = {
                'threshold'     : 2048.0,
                'current_decay' : 1.0,
                'voltage_decay' : 0.25,
                'requires_grad' : False,
                'persistent_state' : False,
    }

    quantizer = _quantize_8bit

    neuron_conv_kwargs = {**neuron_conv_params, 'norm': slayer.neuron.norm.MeanOnlyBatchNorm}
        
    block_kwargs = dict(weight_norm=False, delay_shift=False ,pre_hook_fx=quantizer)

    in_channels = args.in_channels
    reg_max = 5
    num_output = 4 * reg_max + data_info['nc']

    anchorfree_decoder  = AnchorFreePredDecoder(reg_max=reg_max,num_classes=data_info['nc'],stride=net.stride,device=device)

    net_ladl = nn.Sequential(
        slayer.block.cuba.Conv(neuron_conv_kwargs,in_channels,4,kernel_size=3, stride=2, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,4,16,kernel_size=3, stride=2, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,16,64,kernel_size=3, stride=1, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,64,128,kernel_size=3, stride=2, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,128,256,kernel_size=3, stride=1, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,256,256,kernel_size=3, stride=2, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,256,512,kernel_size=3, stride=1, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,512,256,kernel_size=3, stride=2, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,256,128,kernel_size=3, stride=1, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,128,24,kernel_size=3, stride=1, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,24,128,kernel_size=3, stride=1, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,128,128,kernel_size=3, stride=1, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_output_params,128,num_output,kernel_size=1, stride=1, padding=0,**block_kwargs), #Need to be changed...
    )


    """net_ladl = nn.Sequential(
         
        slayer.block.cuba.Conv(neuron_conv_kwargs,in_channels,16,kernel_size=3, stride=2, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,16,32,kernel_size=3, stride=2, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,32,64,kernel_size=3, stride=2, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,64,128,kernel_size=3, stride=2, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,128,256,kernel_size=3, stride=1, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,256,256,kernel_size=3, stride=2, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,256,512,kernel_size=3, stride=1, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,512,256,kernel_size=1, stride=1, padding=0,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,256,512,kernel_size=3, stride=1, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,512,128,kernel_size=3, stride=1, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_conv_kwargs,128,128,kernel_size=3, stride=1, padding=1,**block_kwargs),
        slayer.block.cuba.Conv(neuron_output_params,128,num_output,kernel_size=1, stride=1, padding=0,**block_kwargs), #Need to be changed...
    )"""
    
    H, W = (192,256)
    N, C, T = 8, args.in_channels, 7
    input = torch.rand(N, C, H, W, T).float().cuda()
    net_ladl = net_ladl.to(device)
    net_ladl(input)


    """net_ladl.eval()
    with torch.no_grad():
        net_ladl(input)"""

    net_ladl.load_state_dict(nett.state_dict())

    """net_ladl = nn.Sequential(
        slayer.block.cuba.Input(neuron_conv_kwargs,delay_shift=False),
        *net_ladl,
    )

    net_ladl = net_ladl.to(device)
    net_ladl(input)"""

    net_ladl[0].neuron.norm.pre_hook_fx = quantizer
    net_ladl[1].neuron.norm.pre_hook_fx = quantizer
    net_ladl[2].neuron.norm.pre_hook_fx = quantizer
    net_ladl[3].neuron.norm.pre_hook_fx = quantizer
    net_ladl[4].neuron.norm.pre_hook_fx = quantizer
    net_ladl[5].neuron.norm.pre_hook_fx = quantizer
    net_ladl[6].neuron.norm.pre_hook_fx = quantizer
    net_ladl[7].neuron.norm.pre_hook_fx = quantizer
    net_ladl[8].neuron.norm.pre_hook_fx = quantizer
    net_ladl[9].neuron.norm.pre_hook_fx = quantizer
    net_ladl[10].neuron.norm.pre_hook_fx = quantizer
    #net_ladl[11].neuron.norm.pre_hook_fx = quantizer


    #net_ladl[12].neuron.norm.pre_hook_fx = quantizer

    net_ladl.eval()

    anchorfree_decoder  = AnchorFreePredDecoder(reg_max=reg_max,num_classes=data_info['nc'],stride=net.stride,device=device)
    anchorfree_decoder.dfl = anchorfree_decoder.dfl.to(device)

    #train_loader = get_dataloader(mode="train", args = args,data = data_info,cfg = cfg,stridee = net.stride)
    test_loader = get_dataloader(mode="val", args = args,data = data_info,cfg = cfg,stridee = net.stride,strict_no_rect = args.strict_no_rect)

    validator = DetectionValidator(model=net,data=data_info,cfg=cfg,device=device,save_dir=Path("./runs"),imgsz = temp_width,is_train=False)

    validator.init_metrics()

    temp = torch.tensor([2.1803, 1.9852, 1.3019], device=device)

    epoch = 0

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr,weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.epoch,
    )

    current_max_val = 0.0

    count_collected = 0

    dest_numpy_data_samples_path = "/media/atiye/Data/Udaya_Research_stuff/Numpy_Datasets/img_voc_224_224_numpy"

    #if not os.path.exists(dest_numpy_data_samples_path):
    #    os.makedirs(dest_numpy_data_samples_path)


    min_sparsity = 100
    max_sparsity = 0

    full_tot = 0
    full_non_zeros = 0
    full_synapse_non_zeros = 0

    with torch.no_grad():
            
            val_loss = torch.zeros_like(temp, device=device)

            val_pbar = TQDM(test_loader, desc=validator.get_desc(), total=len(test_loader))
            
            for i,batch in enumerate(val_pbar):

                tot_neurons = 0
                tot_non_zeros = 0
                tot_synapse_non_zeros = 0

                validator.batch_i = i

                """"if "crack_100_1005749" in batch["im_file"][0]: #"crack_100_1005749" "000004.jpg"
                    print("")
                else:
                    continue"""

                #start
                #inputs = inputs.permute(4,0,1,2,3)
                #inputs = inputs.to(device)

                weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 3, 1, 1)
                gray_batch = (batch['img'].float() * weights).sum(dim=1).to(torch.uint8)
                batch['img'] = gray_batch.unsqueeze(1)

                batch['img'] = (batch['img'].unsqueeze(0)).repeat(args.TSteps, 1, 1, 1, 1)

                
                #Since slayer need to be (N,C,H,W,T)

                batch = validator.preprocess(batch)
                idx = 0
                
                temp = batch['img'].permute(1,2,3,4,0)

                ### TO SAVE AT NUMPY FOR EVALUATION

                """if count_collected < 500:

                    for fr_idx,frame in enumerate(temp[:,:,:,:,0]):
                        count_collected += 1

                        file_name = batch['im_file'][fr_idx].rsplit('/',1)[1]
                        full_name = os.path.join(dest_numpy_data_samples_path,file_name)

                        img_arr = frame.clone()
                        img_arr = img_arr.cpu().numpy()
                        gt = torch.concatenate((batch['cls'],batch['bboxes']),axis=1)[fr_idx].reshape(-1,5)
                        gt =gt.cpu().numpy()

                        np.savez(full_name, img=img_arr, ann=gt)"""

                #####################

                shape_for_anchor_free = None 

                for block in net_ladl:

                    if idx == 11:  #12:
                        temp = block.synapse(temp)
                        tot_synapse_non_zeros += (temp != 0).sum().item()
                        current,out_voltage = block.neuron.dynamics(temp)
                        #temp = block.neuron.spike(out_voltage)
                    else:
                        #temp = block(temp)

                        temp = block.synapse(temp)
                        tot_synapse_non_zeros += (temp != 0).sum().item()

                        current,out_voltage = block.neuron.dynamics(temp)
                        temp = block.neuron.spike(out_voltage)

                        tot_neurons += temp.numel()
                        tot_non_zeros += (temp != 0).sum().item()

                        """temp = block.synapse(temp)
                        current,out_voltage = block.neuron.dynamics(temp)
                        temp = block.neuron.spike(out_voltage)
                        out_voltage[torch.where(temp > 0)] = 0"""

                        if idx == 8:   #9:
                            shape_for_anchor_free = temp.mean(-1).shape
                        #temp = block(temp)
                    idx += 1
                
                
                sparsity = ((tot_neurons - tot_non_zeros)/tot_neurons)*100.0

                ideal_neurons = ((tot_neurons - tot_synapse_non_zeros)/tot_neurons)*100.0

                full_tot += tot_neurons
                full_non_zeros += tot_non_zeros
                full_synapse_non_zeros += tot_synapse_non_zeros

                print("sparsity is ",sparsity,"  ",ideal_neurons)

                preds = out_voltage[...,-1]

                preds = anchorfree_decoder.decode_predictions(x=preds,shape=shape_for_anchor_free)

                preds = validator.postprocess(preds)
                validator.update_metrics(preds, batch)

                

                if sparsity < min_sparsity:
                    min_sparsity = sparsity
                if sparsity > max_sparsity:
                     max_sparsity = sparsity

                
            stats = validator.get_stats()
            validator.finalize_metrics()
            validator.print_results()

            results = {**stats, **label_loss_items(val_loss.cpu() / len(test_loader), prefix='val')}
            #metrics = {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
           
            #lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(optimizer.param_groups)}  # for loggers
            #save_metrics(metrics={**label_loss_items(tloss), **metrics, **lr},result_csv_file=csv_file_path,epoch=epoch)

            #if results['metrics/mAP50(B)'] > current_max_val + 0.005:
            current_max_val = results['metrics/mAP50(B)']

            save_checkpoint(os.path.join(args.output_dir,"checkpoints"),current_max_val,epoch,net.state_dict(),optimizer.state_dict(),scheduler.state_dict(),is_last=False)

   
    all_tensors = torch.cat([v.view(-1) for v in net_ladl.state_dict().values()])

    print("all tensors min ",all_tensors.min())
    print("all tensors max ",all_tensors.max())

    print("max sparsity is ",max_sparsity)
    print("min sparsity is ",min_sparsity)
    print("test set sparsity ",((full_tot - full_non_zeros)/full_tot)*100.0)

    print("test set ideal neurons ",((full_tot - full_synapse_non_zeros)/full_tot)*100.0)

    for key in net_ladl.state_dict().keys():
        if torch.any(net_ladl.state_dict()[key] < -10):
            print(key)

    print("export hdf5 ")
    device_cpu = torch.device('cpu')
    net_ladl = net_ladl.to(device_cpu)
    output_path = os.path.join("./","custom_img_evCIVIL_256_192.net")
    export_hdf5(net_ladl,output_path)


def get_args():

    #For SNN model initialization is very important... 

    parser = argparse.ArgumentParser()

    #parser.add_argument("-load",type=str,default="/media/atiye/Data/Udaya_Research_stuff/ultralytic_compatible_ANN_SNN/checkpoints/evCIVIL_lava_img_based/evCIVIL_lava_3.42_after_trained/fused_model/finetune_feedforward.pt")
    #parser.add_argument("-load",type=str,default="/media/atiye/Data/Udaya_Research_stuff/From_server/ultralytic_compatible_ANN_SNN/checkpoints/evCIVIL_lava_img_based/evCIVIL_lava_3.42_after_trained/fused_model/3_kernel_feedforward/last_196.pt")
    #parser.add_argument("-load",type=str,default="/media/atiye/Data/Udaya_Research_stuff/ultralytic_compatible_ANN_SNN/checkpoints/fused_feedforward_checkpoint_lava/absorbed_bnorm_init_342.pt")
    #parser.add_argument("-load",type=str,default="/media/atiye/Data/Udaya_Research_stuff/From_server/ultralytic_compatible_ANN_SNN/checkpoints/evCIVIL_lava_img_based/evCIVIL_lava_3.42_after_trained/fused_model/finetune_meanonly_last_28_3kernel.pt")

    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/From_server_ultralytic_compatible_ANN_SNN_event_based_to_slayer/checkpoints/evCIVIL_lava_img_based/evCIVIL_lava_3.42_after_trained/fused_model/fine_tune_mean_only_best_135itr.pt")

    #parser.add_argument("-load",type=str,default="/media/atiye/Data/Udaya_Research_stuff/ultralytic_compatible_ANN_SNN/out_dir/checkpoints/last_130.pt")  #135

    #parser.add_argument("-load",type=str,default="/media/atiye/Data/Udaya_Research_stuff/ultralytic_compatible_ANN_SNN/checkpoints/fused_feedforward_checkpoint_lava/fused_2.94_anchorfree.pt")
    #parser.add_argument("-load",type=str,default="/media/atiye/Data/Udaya_Research_stuff/From_server/ultralytic_compatible_ANN_SNN/best_1.pt")    #last_198.pt")

    parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/From_server_ultralytic_compatible_ANN_SNN_event_based_to_slayer/checkpoints/evCIVIL_lava_img_based/evCIVIL_lava_3.42_after_trained/fused_model/finetune_meanonly_last_129.pt")

    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/From_server_ultralytic_compatible_ANN_SNN_event_based_to_slayer/checkpoints/SOTA_ANCHORFREE/evCIVIL_img_lava/mean_only_fine_tuned.pt")

    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/From_server_ultralytic_compatible_ANN_SNN_event_based_to_slayer/checkpoints/VOC_ckpts/lava/voc_lava_3.42quant_based_distilled/with_channel_distilled_0.29max/mean_only_VOC_fine_tuned.pt",help="")
    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/From_server_ultralytic_compatible_ANN_SNN_event_based_to_slayer/checkpoints/SOTA_ANCHORFREE/VOC_lava/mean_only_finetune_best.pt")
    #parser.add_argument("-load",type=str,default="/home/nrthpc/Udaya_stuff/From_server_ultralytic_compatible_ANN_SNN_event_based_to_slayer/checkpoints/gen1_lava_event_based/lava_train/snn_to_lava_frame_distilled/mean_only_finetune_best.pt")
    parser.add_argument('-epoch',  type=int, default=150, help='number of epochs to run')
    

    parser.add_argument('-output_dir', type=str, default='./out_dir', help='directory in which to put log folders')
    #parser.add_argument('-num_workers', type=int, default=8, help='number of dataloader workers')

    #parser.add_argument("-train_data_path",type=str,default="/media/atiye/Data/Udaya_Research_stuff/PASCALVOC_FINAL/trainval",help="csv file...") #night_outdoor_and_daytime_train_files_event_based.txt
    #parser.add_argument("-test_data_path",type=str,default="/media/atiye/Data/Udaya_Research_stuff/PASCALVOC_FINAL/test",help="csv file...")

    parser.add_argument('-batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('-data_yaml_file',type=str,default="./defect.yaml")
    parser.add_argument('-result_csv_path',type=str,default="./results_feedforward.csv") 
    parser.add_argument('-imgsz',type=int,default=(256,192),help="reisized img size")
    parser.add_argument('-in_channels',type=int,default=1,help="number of in channels ")
    parser.add_argument('-lr',type=float,default=0.0002,help="learning rate")
    parser.add_argument("-distill",action="store_true", default=False, help="Enable distillation for training")
    parser.add_argument("-from_snn_distill",action="store_true", default=True, help="Enable distillation for training")
    parser.add_argument('-sparsity', action ='store_true', default=False, help='enable sparsity loss')
    parser.add_argument('-sp_lam',type=float, default=0.01, help='sparsity loss mixture ratio')
    parser.add_argument('-sp_rate',type=float, default=0.01, help='minimum rate for sparsity penalization')
    parser.add_argument("-TSteps",type=str,default=7,help="ssss")
    parser.add_argument("--strict_no_rect",action='store_true',default=False,help="force to ensure no rectangular mode in inferencing ")

    args = parser.parse_args()
   
    #snn_repvgg_voc_quantized_3max/last.pt
    return args

if __name__ == "__main__":

    args = get_args()
    main(args)









