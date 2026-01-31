
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
        #return x
    
        if len(x.shape) == 5: #Already time is in the first dimension
            return x
        else:
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
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
        self.qtrick = QuantizeCustom() #MultiSpike4()  # change the max value

    def forward(self, x):

        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) * decay + x[i]

            else:
                mem = x[i]
            spike = self.qtrick(mem)    #/(time_window + 0.0)

            mem_old = mem.clone()
            output[i] = spike
        # print(output[0][0][0][0])
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
        c2, c3 = 128,128 #max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(SpikeConv2(x, c2, 3), SpikeConv2(c2, c2, 3), SpikeConvWithoutBN2(c2,self.no, 1)) for x in ch)
        #self.cv3 = nn.ModuleList(nn.Sequential(SpikeConv2(x, c3, 3), SpikeConv2(c3, c3, 3), SpikeConvWithoutBN2(c3, self.nc, 1)) for x in ch)
        self.dfl = SpikeDFL2(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        x = [x]
        shape = x[0].mean(0).shape  # BCHW  推理：[1，2，64，32，84]  这里必须mean0，否则推理时用到shape会导致报错
        for i in range(self.nl):
            x[i] = self.cv2[i](x[i]) #torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 2)
            x[i] = x[i].mean(0)  #[2，144，32，684]  #这个地方有时候全是1.之后debug看看
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

    """def bias_init(self):
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].conv.bias.data[:] = 1.0  # box
            b[-1].conv.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)"""


class SpikeConvWithoutBN2(nn.Module): #no lif too
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.s = s
        # self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.conv(x.flatten(0, 1)).reshape(T, B, -1, H_new, W_new)
        return x

class SpikeConv2(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.lif = mem_update()
        self.bn = nn.BatchNorm2d(c2,affine=False)
        self.s = s
        self.act = act #self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)
        x = self.bn(self.conv(x.flatten(0, 1))).reshape(T, B, -1, H_new, W_new)
        if self.act:
            return self.lif(x)
        else:
            return x


class DownSampling2(nn.Module):

    def __init__(self, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)
        self.s = s

    def forward(self,x):

        T, B, C, H, W = x.shape
        H_new = int(H / self.s)
        W_new = int(W / self.s)

        return self.m(x.flatten(0, 1)).reshape(T, B, -1, H_new, W_new)

class SpikeRepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizee=3, stride=1, padding=1, groups=1):
        super(SpikeRepVGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if kernel_sizee == 1:
            new_padding = 0
        else:
            new_padding = (1 - (kernel_sizee//2))

        # Training-time branches
        self.conv3x3 = SpikeConv2(in_channels, out_channels, k=kernel_sizee, 
                                 s=stride, p=padding,act=False)

        self.conv1x1 = SpikeConv2(in_channels, out_channels, k=1, 
                                 s=stride, p=new_padding,act=False)

        # Identity branch (used if in_channels == out_channels and stride == 1)
        #self.identity = nn.BatchNorm2d(in_channels) if in_channels == out_channels and stride == 1 else None

        self.act = mem_update()  #nn.SiLU(inplace=True)

    def forward(self, x):
        # Add outputs of all branches
        out = self.conv3x3(x) + self.conv1x1(x)

        """if self.identity is not None:
            out += self.identity(x)"""
        
        return self.act(out)
    

class CustomModel(nn.Module):

    def __init__(self,in_channels = 3,num_classes = 20,for_distillation = False,T=5):

        super(CustomModel, self).__init__()
        
        self.layers = nn.Sequential(
            MS_GetT(in_channels=in_channels,T=T) ,
            SpikeRepVGGBlock(in_channels,4,3,2),
            SpikeRepVGGBlock(4,16,3,2),
            SpikeConv2(16, 64, k=3, s=1, p=1),
            SpikeConv2(64, 128, k=3, s=2, p=1),
            SpikeConv2(128, 128, k=3, s=1, p=1), #256
            SpikeConv2(128, 256, k=3, s=2, p=1), #256
            SpikeConv2(256, 512, k=3, s=1, p=1),
            SpikeConv2(512, 256, k=3, s=2, p=1),
            SpikeConv2(256, 128, k=3, s=1, p=1),
            SpikeConv2(128, 24, k=3, s=1, p=1),
        )

        self.detect = SpikeDetect2(nc=num_classes,ch=[24])

        self.for_distillation = for_distillation

    
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
    
    def forward(self, x):
        
        temp = x

        feature_maps = []

        for i in range(len(self.layers)):
            temp = self.layers[i](temp)

            if self.for_distillation:
                if i == 3:
                    feature_maps.append(temp)

                if i == 7:
                    feature_maps.append(temp)

                if i == 10:
                    feature_maps.append(temp)
            
        detection_output = self.detect(temp)

        return detection_output,feature_maps,None

decay = 0.25  # 0.25 # decay constants