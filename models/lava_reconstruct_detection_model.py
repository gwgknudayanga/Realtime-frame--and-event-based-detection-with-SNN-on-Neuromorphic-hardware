import torch.nn as nn
import torch
from model.model import BaseE2VID
from model import lava_exchange
from spikingjelly.activation_based import layer,surrogate

neuron_stride_params = {
                    'v_threshold'     : 1.0,
                    'current_decay' : 1.0,
                    'voltage_decay' : 0.25,     #0.03,
                    'surrogate_function' : surrogate.ATan(),                             #surrogate.ATan(),
                    'requires_grad' : False,
        }


neuron_output_params = {
                'v_threshold'     : 2048.0,
                'current_decay' : 1.0,
                'voltage_decay' : 0.25,    #0.03,
                'surrogate_function' : surrogate.ATan(),                           #surrogate.ATan(),
                'requires_grad' : False,
}


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

    

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        return


class E2VID(nn.Module):
    def __init__(self,encoder_kernel_size,with_detection = False,num_classes = 2,reg_max = 16,num_input_channels = 5):
        super(E2VID, self).__init__()

        self.with_detection = with_detection
        self.nc = num_classes
        self.reg_max = reg_max

        self.no = self.nc + self.reg_max * 4

        self.stride = torch.zeros(1)

        self.dfl = SpikeDFL2_lava(self.reg_max)

        self.encoder_kernel_size = encoder_kernel_size #5 or #3

        self.reconstruct_backbone = torch.nn.ModuleList([

        lava_exchange.BlockContainer(
                layer.Conv2d(num_input_channels,32, kernel_size=self.encoder_kernel_size, padding=2, stride=2, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(32),detach_reset=True),
            ),

        lava_exchange.BlockContainer(
                layer.Conv2d(32,64, kernel_size=self.encoder_kernel_size, padding=2, stride=2, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(64),detach_reset=True),
            ),

        lava_exchange.BlockContainer(
                layer.Conv2d(64,128, kernel_size=self.encoder_kernel_size, padding=2, stride=2, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(128),detach_reset=True),
            ),

        lava_exchange.BlockContainer(
                layer.Conv2d(128,256, kernel_size=self.encoder_kernel_size, padding=2, stride=2, bias=False),
                lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(256),detach_reset=True),
            ),

        lava_exchange.BlockRepContainer(
                layer.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False),
                layer.Conv2d(256, 256, kernel_size=1, padding=(1 - 3//2), stride=1, bias=False),
                lava_exchange.CubaRepLIFNode(**neuron_stride_params,norm1 = lava_exchange.BatchNorm2d(256),norm2 = lava_exchange.BatchNorm2d(256),detach_reset = True),
            ),

        lava_exchange.BlockRepContainer(
                layer.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False),
                layer.Conv2d(256, 256, kernel_size=1, padding=(1 - 3//2), stride=1, bias=False),
                lava_exchange.CubaRepLIFNode(**neuron_stride_params,norm1 = lava_exchange.BatchNorm2d(256),norm2 = lava_exchange.BatchNorm2d(256),detach_reset = True),
            ),

        ])

        self.detect_backbone = torch.nn.ModuleList([

                lava_exchange.BlockContainer(
                    layer.Conv2d(256,512,kernel_size=3, padding=1, stride=2, bias=False),
                    lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(512),detach_reset = True),
                ),
                lava_exchange.BlockContainer(
                    layer.Conv2d(512,256,kernel_size=1, padding=0, stride=1, bias=False),
                    lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(256),detach_reset = True),
                ),
                lava_exchange.BlockContainer(
                    layer.Conv2d(256,512,kernel_size=3, padding=1, stride=1, bias=False),
                    lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(512),detach_reset = True),
                ),
            ])

        self.detect = torch.nn.ModuleList([

                lava_exchange.BlockContainer(
                    layer.Conv2d(512,128,kernel_size=3, padding=1, stride=1, bias=False),
                    lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(128),detach_reset = True),
                ),
                lava_exchange.BlockContainer(
                    layer.Conv2d(128,128,kernel_size=3, padding=1, stride=1, bias=False),
                    lava_exchange.CubaLIFNode(**neuron_stride_params,norm = lava_exchange.BatchNorm2d(128),detach_reset = True),
                ),
                lava_exchange.BlockContainer(
                    layer.Conv2d(128, self.no, kernel_size=1, padding=0, stride=1, bias=False),
                    #layer.Conv2d(128, self.no, kernel_size=1, padding=0, stride=1, bias=False),
                    lava_exchange.CubaLIFNode(**neuron_output_params,norm = None,detach_reset=True),
                ),

            ])

    def initialize_model_weights(self):

        for i in range(len(self.reconstruct_backbone)):

            if i == 4 or i == 5:

                torch.nn.init.kaiming_normal_(self.reconstruct_backbone[i].synapse1.weight.data)
                torch.nn.init.kaiming_normal_(self.reconstruct_backbone[i].synapse2.weight.data)

            else:
                torch.nn.init.kaiming_normal_(self.reconstruct_backbone[i].synapse.weight.data)

        for i in range(len(self.detect_backbone)):

            torch.nn.init.kaiming_normal_(self.detect_backbone[i].synapse.weight.data)
            
        for i in range(len(self.detect)):

            torch.nn.init.kaiming_normal_(self.detect[i].synapse.weight.data)
    
    def forward(self, event_tensor):
        """
        :param event_tensor: N x num_bins x H x W
        :return: a predicted image of size N x 1 x H x W, taking values in [0,1].
        """

        temp = event_tensor

        for block in self.reconstruct_backbone:
            temp = block(temp)

        for block in self.detect_backbone:
            temp = block(temp)

        for block in self.detect:
            temp = block(temp)

        detector_output = self.detect[-1].neuron.voltage_state

        return detector_output