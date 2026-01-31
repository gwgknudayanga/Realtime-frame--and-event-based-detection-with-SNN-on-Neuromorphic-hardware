import torch
import torch.nn as nn
import math

from spikingjelly.activation_based import layer,surrogate
from ultralytics.utils.tal import TORCH_1_10, dist2bbox, make_anchors

from obd.models import lava_exchange

from lava.lib.dl import slayer


#may be we need to change the voltage_decay parameter.
#Need to load the last conv weights
#Check for both MSE and Distribution.


neuron_stride_params = {
                    'v_threshold'     : 1.0,
                    'current_decay' : 1.0,
                    'voltage_decay' : 0.25,
                    'surrogate_function' : surrogate.ATan(),                             #surrogate.ATan(),
                    'requires_grad' : False,
        }

neuron_output_params = {
                'v_threshold'     : 2048.0,
                'current_decay' : 1.0,
                'voltage_decay' : 0.25,
                'surrogate_function' : surrogate.ATan(),                           #surrogate.ATan(),
                'requires_grad' : False,
        }

def _quantize_8bit(x: torch.tensor,
                            scale: int = (1 << 6),
                            descale: bool = False) -> torch.tensor:
                return slayer.utils.quantize_hook_fx(x, scale=scale,
                                                    num_bits=8, descale=descale)

class Lava_GetT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=1):
        super().__init__()
        
        print("in channels ",in_channels)
        print("out channels ",out_channels)
        print("time steps ",T)
        
        self.T = T
        self.in_channels = in_channels

    def forward(self, x):
        
        #If len(x.shape) == 5, it already contains the time dimension

        if len(x.shape) == 5: #Already time is in the first dimension
            return x
        else:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
            return x


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

class LavaDetect(nn.Module):

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self,nc = 20,ch = ()):

        super().__init__() 

        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers - Since we have only one head
        self.reg_max = 5  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        self.dfl = SpikeDFL2_lava(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self.head_backend = torch.nn.ModuleList([

            lava_exchange.BlockContainer(
                layer.Conv2d(24,128, kernel_size=3, padding=1, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(128),detach_reset=True),
            ),

            lava_exchange.BlockContainer(
                layer.Conv2d(128,128, kernel_size=3, padding=1, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(128),detach_reset=True),
            ),

            lava_exchange.BlockContainer(
                layer.Conv2d(128, self.no, kernel_size=1, padding=0, stride=1, bias=False),
                #layer.Conv2d(128, self.no, kernel_size=1, padding=0, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_output_params,norm = None,detach_reset=True),
            ),

            #lava_exchange.BlockContainer(
            #    layer.Conv2d(128, self.no, kernel_size=1, padding=0, stride=1, bias=False),
            #    lava_exchange.CubaLIFNode(**neuron_output_params,norm = None ,detach_reset=True),
            #),
        ])





        #self.num_head_backend_blocks = len(self.head_backend)
    
    def forward(self,x):

        x = [x]

        shape = x[0].mean(0).shape

        for i in range(self.nl):
            temp = x[i]
            for m in self.head_backend:
                temp = m(temp)
            #x[i] = temp    #self.head_backend(x[i])  #torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 2)
            x[i] = self.head_backend[-1].neuron.voltage_state   #x[i].mean(0)  #[2，144，32，684]  #这个地方有时候全是1.之后debug看看

        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1) #box: [B,reg_max * 4,anchors]
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)
    
    
    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        return


class CustomModel(nn.Module):

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self,in_channels = 3,num_classes = 20,T=5,distillation_required = False):

        super(CustomModel, self).__init__()

        ch=[24]
        self.nc = num_classes  # number of classes
        self.nl = len(ch)  # number of detection layers - Since we have only one head
        self.reg_max = 5  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = num_classes + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build


        net = torch.nn.ModuleList([

            Lava_GetT(in_channels = in_channels,T=T),

            lava_exchange.BlockContainer(
                layer.Conv2d(in_channels, 4, kernel_size=3, padding=1, stride=2, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(4),detach_reset = True),
            ),

            #layer.Dropout2d(p=0.1), 
            
             lava_exchange.BlockContainer(
                layer.Conv2d(4, 16, kernel_size=3, padding=1, stride=2, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(16), detach_reset = True),
            ),

            #layer.Dropout2d(p=0.15),

            lava_exchange.BlockContainer(
                layer.Conv2d(16, 64, kernel_size=3, padding=1, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(64),detach_reset = True),
            ),

            lava_exchange.BlockContainer(
                layer.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(128),detach_reset = True),
            ),

            #layer.Dropout2d(p=0.15),

            lava_exchange.BlockContainer(
                layer.Conv2d(128, 256, kernel_size=3, padding=1, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(256),detach_reset = True),
            ),

            #layer.Dropout2d(p=0.2),

            lava_exchange.BlockContainer(
                layer.Conv2d(256,256,kernel_size=3, padding=1, stride=2, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(256),detach_reset = True),
            ),

           #layer.Dropout2d(p=0.15), 

           lava_exchange.BlockContainer(
                layer.Conv2d(256,512,kernel_size=3, padding=1, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(512),detach_reset = True),
            ),

            #layer.Dropout2d(p=0.2), 

            lava_exchange.BlockContainer(
                layer.Conv2d(512,256,kernel_size=3, padding=1, stride=2, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(256),detach_reset = True),
            ),

            lava_exchange.BlockContainer(
                layer.Conv2d(256,128, kernel_size=3, padding=1, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(128),detach_reset = True),
            ),

            lava_exchange.BlockContainer(
                layer.Conv2d(128,24, kernel_size=3, padding=1, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(24), detach_reset = True),
            ),

        ])

        self.head_backend = torch.nn.ModuleList([

            lava_exchange.BlockContainer(
                layer.Conv2d(24,128, kernel_size=3, padding=1, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(128),detach_reset=True),
            ),

            lava_exchange.BlockContainer(
                layer.Conv2d(128,128, kernel_size=3, padding=1, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(128),detach_reset=True),
            ),

            lava_exchange.BlockContainer(
                layer.Conv2d(128, self.no, kernel_size=1, padding=0, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_output_params,norm = None,detach_reset=True),
            ),

        ])

        self.dfl = SpikeDFL2_lava(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self.layers = nn.Sequential(*net)
        
        #self.detect = LavaDetect(ch=[24],nc=num_classes)

        quantizer = _quantize_8bit

        for i in range(len(self.layers)):

            if i == 0: #continue if module is Lava_GetT
                continue

            self.layers[i].neuron.norm.pre_hook_fx = quantizer
        
        for i in range(len(self.head_backend)):

            if i < 2:
                self.head_backend[i].neuron.norm.pre_hook_fx = quantizer


    def initialize_model_weights(self) -> None:

        for i in range(len(self.layers)):

            if i == 0: #continue if module is Lava_GetT
                continue
            
            torch.nn.init.kaiming_normal_(self.layers[i].synapse.weight.data)

        for i in range(len(self.head_backend)):

            torch.nn.init.kaiming_normal_(self.head_backend[i].synapse.weight.data)


    def forward(self, x):
        
        temp = x

        feature_maps = []

        for i in range(len(self.layers)):
            temp = self.layers[i](temp)

        #x = self.layers(x)

        x = [temp]

        shape = x[0].mean(0).shape

        for i in range(self.nl):
            temp = x[i]
            for m in self.head_backend:
                temp = m(temp)
            #x[i] = temp    #self.head_backend(x[i])  #torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 2)
            x[i] = self.head_backend[-1].neuron.voltage_state   #x[i].mean(0)  #[2，144，32，684]  #这个地方有时候全是1.之后debug看看

        if self.training:

            return x,feature_maps
        
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1) #box: [B,reg_max * 4,anchors]
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return (y, x),feature_maps