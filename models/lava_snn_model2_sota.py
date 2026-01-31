
import torch
import torch.nn as nn
import math
from obd.models import lava_exchange

from spikingjelly.activation_based import layer,surrogate
from ultralytics.utils.tal import TORCH_1_10, dist2bbox, make_anchors

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
                layer.Conv2d(512,128, kernel_size=3, padding=1, stride=1, bias=False),
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

    def __init__(self,in_channels = 3,num_classes = 20,T=5,distillation_required = False):

        super(CustomModel, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(2.2)) 

        net = torch.nn.ModuleList([

            Lava_GetT(in_channels = in_channels,T=T),

            lava_exchange.BlockContainer(
                layer.Conv2d(in_channels,16, kernel_size=3, padding=1, stride=2, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(16),detach_reset = True),
            ),

            lava_exchange.BlockContainer(
                layer.Conv2d(16,32, kernel_size=3, padding=1, stride=2, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(32),detach_reset = True),
            ),

            lava_exchange.BlockContainer(
                layer.Conv2d(32,64, kernel_size=3, padding=1, stride=2, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(64),detach_reset = True),
            ),

            lava_exchange.BlockContainer(
                layer.Conv2d(64,128, kernel_size=3, padding=1, stride=2, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(128),detach_reset = True),
            ),

            lava_exchange.BlockContainer(
                layer.Conv2d(128,256, kernel_size=3, padding=1, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(256)),
            ),

            lava_exchange.BlockContainer(
                layer.Conv2d(256,256, kernel_size=3, padding=1, stride=2, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(256)),
            ),


            lava_exchange.BlockContainer(
                layer.Conv2d(256,512, kernel_size=3, padding=1, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(512)),
            ),


            lava_exchange.BlockContainer(
                layer.Conv2d(512,256,kernel_size=1, padding=0, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(256)),
            ),

            lava_exchange.BlockContainer(
                layer.Conv2d(256,512,kernel_size=3, padding=1, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(512)),
            ),

        ])

        self.layers = nn.Sequential(*net)
        
        self.detect = LavaDetect(ch=[512],nc=num_classes)

        """quantizer = _quantize_8bit

        for i in range(len(self.layers)):

            if i == 0: #continue if module is Lava_GetT
                continue

            if i == 1 or i == 2 or i == 3 or i == 4: #continue if module is Lava_GetT
                self.layers[i].neuron.norm1.pre_hook_fx = quantizer
                self.layers[i].neuron.norm2.pre_hook_fx = quantizer
                continue
                
            self.layers[i].neuron.norm.pre_hook_fx = quantizer
        
        for i in range(len(self.detect.head_backend)):

            if i < 2:

                self.detect.head_backend[i].neuron.norm.pre_hook_fx = quantizer"""

        
        #projections for distillation

        self.distillation_required = distillation_required

        if distillation_required:

            #self.projection1 = layer.Conv2d(self.layers[3].synapse.out_channels, self.layers[3].synapse.out_channels, kernel_size=1,padding = 0)
            #self.projection2 = layer.Conv2d(self.layers[7].synapse.out_channels, self.layers[7].synapse.out_channels, kernel_size=1,padding = 0)
            #self.projection3 = layer.Conv2d(self.layers[10].synapse.out_channels, self.layers[10].synapse.out_channels, kernel_size=1, padding = 0)

            self.projection1  = lava_exchange.BlockContainer(
                layer.Conv2d(self.layers[3].synapse.out_channels, self.layers[3].synapse.out_channels,kernel_size=1, padding=0, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_output_params,norm = None,detach_reset = True),
            )    
            self.projection2  = lava_exchange.BlockContainer(
                layer.Conv2d(self.layers[7].synapse.out_channels, self.layers[7].synapse.out_channels,kernel_size=1, padding=0, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_output_params,norm = None,detach_reset = True),
            )
    
            self.projection3  = lava_exchange.BlockContainer(
                layer.Conv2d(self.layers[9].synapse.out_channels, self.layers[9].synapse.out_channels,kernel_size=1, padding=0, stride=1, bias=False),
                lava_exchange.CubaLIFNode(**neuron_output_params,norm = None,detach_reset = True),
            )

            ## Channel attention ...

            

            ## Spatial attention

    def initialize_model_weights(self) -> None:

        for i in range(len(self.layers)):

            if i == 0: #continue if module is Lava_GetT
                continue

            """if i == 1 or i == 2 or i == 3 or i == 4:
                torch.nn.init.kaiming_normal_(self.layers[i].synapse1.weight.data)
                torch.nn.init.kaiming_normal_(self.layers[i].synapse2.weight.data)
                continue """      
            
            torch.nn.init.kaiming_normal_(self.layers[i].synapse.weight.data)

        for i in range(len(self.detect.head_backend)):
            
            """if i == (len(self.detect.head_backend) - 1):
                torch.nn.init.kaiming_normal_(self.detect.head_backend[i].synapse1.weight.data)
                torch.nn.init.kaiming_normal_(self.detect.head_backend[i].synapse2.weight.data)
            else:"""
            torch.nn.init.kaiming_normal_(self.detect.head_backend[i].synapse.weight.data)

        
        if self.distillation_required:

            torch.nn.init.kaiming_normal_(self.projection1.synapse.weight.data)
            torch.nn.init.kaiming_normal_(self.projection2.synapse.weight.data)
            torch.nn.init.kaiming_normal_(self.projection3.synapse.weight.data)



    def forward(self, x):

        alpha = torch.clamp(torch.sigmoid(self.alpha), min=0.1, max=0.99)
        
        temp = x

        feature_maps = []
        for_attention_maps = []

        for i in range(len(self.layers)):
            temp = self.layers[i](temp)

            if self.distillation_required:

                if i == 3:
                    feature_maps.append(self.projection1(temp))

                elif i == 7:

                    feature_maps.append(self.projection2(temp))

                    ema = temp[0]
                    for t in range(1,temp.shape[0]):
                        ema = alpha * temp[t] + (1 - alpha) * ema 
                    for_attention_maps.append(ema)

                if i == 9: #10

                    feature_maps.append(self.projection3(temp))
                    ema = temp[0]
                    for t in range(1,temp.shape[0]):
                        ema = alpha * temp[t] + (1 - alpha) * ema 
                    for_attention_maps.append(ema)
 

        #x = self.layers(x)
        detection_output = self.detect(temp)
        return detection_output,feature_maps
