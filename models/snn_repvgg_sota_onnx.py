
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from obd.utils.tal import dist2bbox, make_anchors

class MS_GetT(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, T=1):
        super().__init__()
        
        print("in channels ",in_channels)
        print("out channels ",out_channels)
        print("time steps ",T)
        
        self.T = T
        self.in_channels = in_channels

    def forward(self, x):

        return x
        

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class QuantizeCustom(nn.Module):

    class quant2custom(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):

            ctx.save_for_backward(input)

            #quant_levels = torch.tensor([0, 0.57, 1.14, 1.71, 2.28, 2.85, 3.42]).to(input.device) 
            quant_levels = torch.tensor([0, 0.49, 0.98, 1.47, 1.96, 2.45, 2.94]).to(input.device)
        
            # Clamp the input between 0 and 3
            clamped_input = torch.clamp(input, min=0, max=2.94)

            # Scale the input to map into the quantization set range
            scaled_input = clamped_input / 2.94 * (len(quant_levels) - 1)
            
            # Round to the nearest index and clamp to valid indices
            indices = torch.round(scaled_input).long()
            indices = torch.clamp(indices, min=0, max=len(quant_levels) - 1)

            # Use indices to get the corresponding quantized values
            quantized_output = quant_levels[indices]

            return quantized_output

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 3.42] = 0
            return grad_input
        
    def forward(self, x):
        return self.quant2custom.apply(x)
    


class MultiSpike4(nn.Module):

    class quant4(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return torch.round(torch.clamp(input, min=0, max=4))

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            #             print("grad_input:",grad_input)
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input

    def forward(self, x):
        return self.quant4.apply(x)

    
class mem_update(nn.Module):
    def __init__(self, act=False):
        super(mem_update, self).__init__()
        # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)

        self.act = act
        self.qtrick = nn.ReLU6() #QuantizeCustom() #MultiSpike4()  # change the max value

    def forward(self, x):

        output = self.qtrick(x)
        #output = output.unsqueeze(0)
        return output


class SpikeDFL2(nn.Module):
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
        #self.lif = mem_update()


    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)  # 原版


class SpikeDetect2(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):

        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 5  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        
        self.cv2 = nn.Sequential(
            SpikeConv2(512, 128, k=3, s=1, p=1),
            SpikeConv2(128, 128, k=3, s=1, p=1),
            SpikeConvWithoutBN2(128, self.no, k=1, s=1, p=0)
        )

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        """x = [x]
        shape = x[0].mean(0).shape  # BCHW  推理：[1，2，64，32，84]  这里必须mean0，否则推理时用到shape会导致报错
        for i in range(self.nl):
            x[i] = self.cv2(x[i]) #torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 2)
            #x[i] = x[i].mean(0)  #[2，144，32，684]  #这个地方有时候全是1.之后debug看看
        """
        return self.cv2(x)

    
class SpikeConvWithoutBN2(nn.Module): #no lif too
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):

        x = self.conv(x)
        return x

class SpikeConv2(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.lif = nn.ReLU6()
        self.bn = nn.BatchNorm2d(c2,affine=False)
        self.s = s
        self.act = act #self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()'

        self.hook_handle = None

    def activation_hook(self, module, input, output):

        """Hook function to capture activations."""
        act_min = output.min().item()
        act_max = output.max().item()
        act_mean = output.mean().item()
        print(f"LIF Neuron | Min: {act_min:.4f}, Max: {act_max:.4f}, Mean: {act_mean:.4f}")
    
    def forward(self, x):

        #if not self.training and self.hook_handle is None:
        #    self.hook_handle = self.lif.register_forward_hook(self.activation_hook)

        x = self.bn(self.conv(x))
        return self.lif(x)


class DownSampling2(nn.Module):

    def __init__(self, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)
        self.s = s

    def forward(self,x):

        return self.m(x)


class CustomModel(nn.Module):

    def __init__(self,in_channels = 3,num_classes = 20,for_distillation = False,T=5):

        super(CustomModel, self).__init__()
        
        self.layers = nn.Sequential(
            MS_GetT(in_channels=in_channels,T=T) ,
            SpikeConv2(in_channels, 16, k=3, s=2, p=1),
            SpikeConv2(16, 32, k=3, s=2, p=1),
            SpikeConv2(32, 64, k=3, s=2, p=1),
            SpikeConv2(64, 128, k=3, s=2, p=1),
            SpikeConv2(128, 256, k=3, s=1, p=1),
            SpikeConv2(256, 256, k=3, s=2, p=1),
            SpikeConv2(256, 512, k=3, s=1, p=1),
            SpikeConv2(512, 256, k=1, s=1, p=0),
            SpikeConv2(256, 512, k=3, s=1, p=1),
        )

        self.detect = SpikeDetect2(nc=num_classes,ch=[512])

        self.for_distillation = for_distillation

    
    def forward(self, x,sparsity_monitor = None):
        
        temp = x

        for i in range(len(self.layers)):
            temp = self.layers[i](temp)

            """if self.for_distillation:
                if i == 9:
                    feature_maps.append(temp)"""
            
        detection_output = self.detect(temp)

        return detection_output

decay = 0.25  # 0.25 # decay constants
