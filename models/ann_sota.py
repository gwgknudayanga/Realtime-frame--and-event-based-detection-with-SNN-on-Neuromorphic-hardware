import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from obd.utils.tal import dist2bbox, make_anchors

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

class Detect(nn.Module):
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
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        #self.cv2 = nn.ModuleList(
            #nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
         #   nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), RepVGGBlock(c2,self.reg_max * 4,1,1,0),) for x in ch)
        #self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, 128, 3), Conv(128, 128, 3), nn.Conv2d(128, self.no, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        x = [x]
        shape = x[0].shape  # BCHW , since x is not a list

        for i in range(self.nl):
            x[i] = self.cv2[i](x[i]) #torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
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
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
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

        if self.export:
            return y
        else:
            return (y, x)
        #return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding = 1, bias=False):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

        self.hook_handle = None

    def activation_hook(self, module, input, output):
        """Hook function to capture activations."""
        act_min = output.min().item()
        act_max = output.max().item()
        act_mean = output.mean().item()
        print(f"LIF Neuron | Min: {act_min:.4f}, Max: {act_max:.4f}, Mean: {act_mean:.4f}")

    def forward(self, x):

        #if not self.training and self.hook_handle is None:
        #    self.hook_handle = self.act.register_forward_hook(self.activation_hook)

        return self.act(self.bn(self.conv(x)))

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizee=3, stride=1, padding=1, groups=1):
        super(RepVGGBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if kernel_sizee == 1:
            new_padding = 0
        else:
            new_padding = (1 - (kernel_sizee//2))

        # Training-time branches
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizee, 
                                 stride=stride, padding=padding, groups=groups, bias=False)
        self.bn3x3 = nn.BatchNorm2d(out_channels)

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                 stride=stride, padding=new_padding, groups=groups, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

        # Identity branch (used if in_channels == out_channels and stride == 1)
        self.identity = nn.BatchNorm2d(in_channels) if in_channels == out_channels and stride == 1 else None

        self.act = nn.SiLU(inplace=True)

        self.hook_handle = None

    def activation_hook(self, module, input, output):
        """Hook function to capture activations."""
        act_min = output.min().item()
        act_max = output.max().item()
        act_mean = output.mean().item()
        print(f"LIF Neuron | Min: {act_min:.4f}, Max: {act_max:.4f}, Mean: {act_mean:.4f}")
        
    def forward(self, x):
        # Add outputs of all branches

        #if not self.training and self.hook_handle is None:
        #    self.hook_handle = self.act.register_forward_hook(self.activation_hook)

        out = self.bn3x3(self.conv3x3(x)) + self.bn1x1(self.conv1x1(x))
        if self.identity is not None:
            out += self.identity(x)
        return self.act(out)
    

class CustomModel(nn.Module):

    def __init__(self,in_channels = 3,num_classes = 20,for_distillation = False):

        super(CustomModel, self).__init__()
        
        self.layers = nn.Sequential(
            Conv(in_channels, 16, kernel_size=3, stride=2, padding=1),
            Conv(16,32, kernel_size=3, stride=2, padding=1),
            Conv(32, 64, kernel_size=3, stride=2, padding=1),
            Conv(64, 128, kernel_size=3, stride=2, padding=1),
            Conv(128, 256, kernel_size=3, stride=1, padding=1),
            Conv(256, 256, kernel_size=3, stride=2, padding=1),
            Conv(256, 512, kernel_size=3, stride=1, padding=1),
            Conv(512, 256, kernel_size=1, stride=1, padding=0),
            Conv(256, 512, kernel_size=3, stride=1, padding=1),
        )

        self.detect = Detect(nc=num_classes,ch=[512])

        self.for_distillation = for_distillation

    def forward(self, x):
        
        temp = x

        feature_maps = []

        for i in range(len(self.layers)):
            temp = self.layers[i](temp)

            if self.for_distillation:
                if i == 2:
                    feature_maps.append(temp)
                if i == 6:
                    feature_maps.append(temp)
                if i == 9:
                    feature_maps.append(temp)
            
        detection_output = self.detect(temp)

        return detection_output,feature_maps,None
