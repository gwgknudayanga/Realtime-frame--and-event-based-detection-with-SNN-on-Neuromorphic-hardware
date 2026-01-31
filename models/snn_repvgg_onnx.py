
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

        #x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
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

            #quant_levels = torch.tensor([0, 0.57, 1.14, 1.71, 2.28, 2.85, 3.42]).to(input.device) #torch.tensor([0, 0.49, 0.98, 1.47, 1.96, 2.45, 2.94]).to(input.device)
            #
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
            grad_input[input > 2.94] = 0
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
        self.qtrick = torch.nn.ReLU6() #QuantizeCustom() #MultiSpike4()  # change the max value

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
        self.lif = mem_update()


    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)  # 原版

"""class SpikeDetect2(nn.Module):

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):

        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 5  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        
        self.cv2 = nn.Sequential(
            SpikeConv2(24, 128, k=3, s=1, p=1),
            SpikeConv2(128, 128, k=3, s=1, p=1),
            SpikeConvWithoutBN2(128, self.no, k=1, s=1, p=0)
        )

    def forward(self, x):

        return self.cv2(x)"""

class SpikeDetect2(nn.Module):

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):

        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 5  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = 128,128 #max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels

        self.cv2 = nn.ModuleList([
            nn.Sequential(
                    SpikeConv2(24, 128, 3),
                    SpikeConv2(128, 128, 3),
                    SpikeConvWithoutBN2(128, self.no, 1)
            )]
        )

    def forward(self, x):

        return self.cv2[0](x)

class SpikeRepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizee=3, stride=1, padding=1, groups=1):
        super(SpikeRepVGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        """if kernel_sizee == 1:
            new_padding = 0
        else:"""
        new_padding = (1 - (kernel_sizee//2))

        # Training-time branches
        self.conv3x3 = SpikeConv2_actFalse(in_channels, out_channels, k=kernel_sizee, 
                                 s=stride, p=padding,act=False)

        self.conv1x1 = SpikeConv2_actFalse(in_channels, out_channels, k=1, 
                                 s=stride, p=new_padding,act=False)

        # Identity branch (used if in_channels == out_channels and stride == 1)
        #self.identity = nn.BatchNorm2d(in_channels) if in_channels == out_channels and stride == 1 else None

        self.act = nn.ReLU6() #mem_update()  #nn.SiLU(inplace=True)

    def forward(self, x):
        # Add outputs of all branches
        out = self.conv3x3(x) + self.conv1x1(x)

        """if self.identity is not None:
            out += self.identity(x)"""
        
        return self.act(out)


class SpikeConvWithoutBN2(nn.Module): #no lif too
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)"""
        #x = self.conv(x.flatten(0, 1)).reshape(T, B, -1, H_new, W_new)
        x = self.conv(x)
        return x

class SpikeConv2_actFalse(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=False):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.lif = nn.ReLU6() #mem_update()
        self.bn = nn.BatchNorm2d(c2,affine=False)
        self.s = s
        self.act = act #self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)"""
        #x = self.bn(self.conv(x.flatten(0, 1))).reshape(T, B, -1, H_new, W_new)
        x = self.bn(self.conv(x))
        return x
        
class SpikeConv2(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.lif = nn.ReLU6() #mem_update()
        self.bn = nn.BatchNorm2d(c2,affine=False)
        self.s = s
        self.act = act #self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)"""
        #x = self.bn(self.conv(x.flatten(0, 1))).reshape(T, B, -1, H_new, W_new)
        x = self.bn(self.conv(x))

        return self.lif(x)


class DownSampling2(nn.Module):

    def __init__(self, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)
        self.s = s

    def forward(self,x):

        """T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)

        return self.m(x.flatten(0, 1)).reshape(T, B, -1, H_new, W_new)"""
        return self.m(x)

class CustomModel(nn.Module):

    def __init__(self,in_channels = 3,num_classes = 20,for_distillation = False,T=5):

        super(CustomModel, self).__init__()
        
        self.layers = nn.Sequential(
            MS_GetT(in_channels=in_channels,T=T) ,
            SpikeRepVGGBlock(in_channels,4,3,2),
            SpikeRepVGGBlock(4,16,3,2),
            SpikeConv2(16, 64, k=3, s=1, p=1),
            SpikeConv2(64, 128, k=3, s=2, p=1),
            SpikeConv2(128, 256, k=3, s=1, p=1), #256
            SpikeConv2(256, 256, k=3, s=2, p=1), #256
            SpikeConv2(256, 512, k=3, s=1, p=1),
            SpikeConv2(512, 256, k=3, s=2, p=1),
            SpikeConv2(256, 128, k=3, s=1, p=1),
            SpikeConv2(128, 24, k=3, s=1, p=1),
        )

        self.detect = SpikeDetect2(nc=num_classes,ch=[24])

        #self.for_distillation = for_distillation

    
    def initialize_model_weights(self) -> None:

        for i in range(len(self.layers)):

            if i == 0: #continue if module is Lava_GetT
                continue
            if i == 1 or i == 2:
                torch.nn.init.kaiming_normal_(self.layers[i].conv3x3.conv.weight.data)
                torch.nn.init.kaiming_normal_(self.layers[i].conv1x1.conv.weight.data)
                continue

            torch.nn.init.kaiming_normal_(self.layers[i].conv.weight.data)
        
        for i in range(len(self.detect.cv2)):

            torch.nn.init.kaiming_normal_(self.detect.cv2[0][i].conv.weight.data)
    
    def forward(self, x,sparsity_monitor = None):
        
        temp = x

        feature_maps = []

        for i in range(len(self.layers)):
            temp = self.layers[i](temp)
            
        detection_output = self.detect(temp)

        return detection_output #[0]  #,feature_maps,None

decay = 0.25  # 0.25 # decay constants