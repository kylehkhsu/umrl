import torch
from torch import nn


class MiniAlexNet(nn.Module):
    r"""
    A variant of AlexNet.
    The changes with respect to the original AlexNet are:
        - LRN (local response normalization) layers are not included
        - The Fully Connected (FC) layers (fc6 and fc7) have smaller dimensions
          due to the lower resolution of mini-places images (128x128) compared
          with ImageNet images (usually resized to 256x256)
    """ 
    def __init__(self, num_classes, sobel):
        super(MiniAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 1024, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        self.top_layer = nn.Sequential(
            nn.Linear(1024, num_classes),
        )
        self.init_model()

    def init_model(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname == 'Linear':
                nn.init.normal(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

        self.apply(weights_init)
        return self

    def forward(self, input):
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten to a 2d tensor
        x = self.classifier(features)
        if self.top_layer:
            x = self.top_layer(x)

        return x


import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = nn.functional.dropout2d(out, p=0.8, training=self.training)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def add_sobel(net, sobel=True):
    if sobel:
        grayscale = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0)
        grayscale.weight.data.fill_(1.0 / 3.0)
        grayscale.bias.data.zero_()
        sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
        sobel_filter.weight.data[0, 0].copy_(
            torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        )
        sobel_filter.weight.data[1, 0].copy_(
            torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        )
        sobel_filter.bias.data.zero_()
        net.sobel = nn.Sequential(grayscale, sobel_filter)
        for p in net.sobel.parameters():
            p.requires_grad = False
    else:
        net.sobel = None

import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        # if isinstance(kernel_size, numbers.Number):
        #     kernel_size = [kernel_size] * dim
        # if isinstance(sigma, numbers.Number):
        #     sigma = [sigma] * dim

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                        torch.exp(
                            -torch.sum((xy_grid - mean)**2., dim=-1) /\
                            (2*variance)
                        )

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False
        self.filter = gaussian_filter

        # return gaussian_filter
        

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return F.pad(self.filter(input), (2, 2, 2, 2), mode='reflect')


class ResNet(nn.Module):
    def __init__(self, block, layers, sobel=False, num_classes=1000, traj_enc='bow'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        n_inp_chan = 2 if sobel else 3
        print('ResNet has', sum(layers), 'layers')
        if sum(layers) > 8 or True:
            self.conv1 = nn.Conv2d(
                n_inp_chan, 64, kernel_size=5, stride=2, padding=2, bias=False)

            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        else:
            self.conv1 = nn.Conv2d(
                n_inp_chan, 64, kernel_size=5, stride=2, padding=2, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block,  64, layers[0], stride=2)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 128, layers[3], stride=2)

        self.features = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool,
            self.layer1, self.layer2, self.layer3, self.layer4
        )

        fcdim = 1024 if sum(layers) > 8 else 512
        self.avgpool = nn.AvgPool2d(2, 2)

        self.tc = TCBlock(fcdim, 8, 32)

        self.traj_enc = traj_enc
    
        if traj_enc == 'bow':
            self.classifier =  nn.Sequential(
                                # nn.Linear(256 * 4 * 4, 1024),
                                # nn.Linear(fcdim, fcdim),
                                # nn.Linear(1120, fcdim),
                                # nn.Linear(fcdim, fcdim),
                                nn.ReLU(inplace=True))
        else:
            self.classifier =  nn.Sequential(
                                # nn.Linear(256 * 4 * 4, 1024),
                                nn.Linear(self.tc.channel_count, fcdim),
                                # nn.Linear(1120, fcdim),
                                # nn.Linear(fcdim, fcdim),
                                nn.ReLU(inplace=True))

        self.top_layer = nn.Linear(fcdim, num_classes)
        self.printed = False
        self.blur = GaussianSmoothing(3, 7, 5)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        B, T, C, H, W = x.shape

        x = x.view(B*T, *x.shape[-3:])

        if self.blur is not None:
            x = self.blur(x)

        if self.sobel is not None:
            x = self.sobel(x)

        x = self.features(x)
        # # print(x.shape)
        # x = self.conv1(x)
        # # print(x.shape)
        # x = self.bn1(x)
        # # print(x.shape)
        # x = self.relu(x)
        # # print(x.shape)
        # x = self.maxpool(x)

        # # print(x.shape)
        # x = self.layer1(x)
        # # print(x.shape)
        # x = self.layer2(x)
        # # print(x.shape)
        # x = self.layer3(x)
        # # print(x.shape)
        # x = self.layer4(x)

        if not self.printed:
            print(x.shape)
            self.printed = True

        x = self.avgpool(x)
        # x = torch.nn.functional.avg_pool2d(x, kernel_size=x.shape[-1])
        x = x.view(x.size(0), -1)

        x = x.view(B, T, -1)

        # x = torch.tanh(x[:, 1:] - x[:, :-1])

        # import pdb; pdb.set_trace()
        if self.traj_enc == 'temp_conv':
            x = self.tc(x.transpose(1,2)).transpose(1,2)

        x = x.mean(1)
        # x = x[:, -1]

        # in_channels, seq_len, filters
        # import pdb; pdb.set_trace()
        if self.classifier is None:
            self.classifier = nn.Sequential(
                    nn.Linear(x.shape[-1], 1024),
                    # nn.Linear(256 * block.expansion, 1024),
                    nn.ReLU(inplace=True))
        x = self.classifier(x)

        if self.top_layer:
            x = self.top_layer(x)


        return x



def resnet18(sobel=False, bn=True, out=1000, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 1, 1], sobel=sobel, num_classes=out, **kwargs)
    add_sobel(model, sobel=sobel)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    return model

# def resnet18(pretrained=False, **kwargs):
#     """Constructs a ResNet-18 model.

#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
#     if pretrained:
#         model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
#     return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


import math
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt


def getCAM(feature_conv, weight_fc, class_idx):
    _, nc, h, w = feature_conv.shape
    cam = weight_fc[class_idx.cpu().numpy()].dot(
        feature_conv.reshape((nc, h * w)))
    # cam = weight_fc[class_idx.cpu().numpy()].dot(feature_conv.reshape(-1))

    # import pdb; pdb.set_trace()

    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return [cam_img]


class SaveFeatures():
    features = None

    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = ((output.cpu()).data).numpy()

    def remove(self):
        self.hook.remove()


def CAM(model, img):
    model.eval()
    final_layer = model.layer4
    activated_features = SaveFeatures(final_layer)
    x = Variable((img.unsqueeze(0)).cuda(), requires_grad=True)

    y = model(x)
    yp = F.softmax(y).data.squeeze()
    activated_features.remove()

    weight_softmax_params = list(model.fc.parameters())
    weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
    # weight_softmax_params
    class_idx = torch.topk(yp, 1)[1].int()
    overlay = getCAM(activated_features.features, weight_softmax, class_idx)

    from skimage import transform
    oo = transform.resize(overlay[0], img.shape[1:3])
    model.train()
    img = img.cpu().numpy()
    cmap = plt.cm.jet

    # import pdb; pdb.set_trace()
    return np.array(cmap(oo)).transpose([2, 0, 1
                                         ])[:3] * 0.5 + img / img.max() * 0.5
    # prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad=True)



"""A module containing the models described in the SNAIL paper
"""
import math
import torch.nn as nn

# class OmniglotEmbedding(nn.Module):
#     """A CNN which transforms a 1x28x28 image to a 64-dimensional vector
#     """

#     def __init__(self):
#         super(OmniglotEmbedding, self).__init__()
#         self.cnn1 = nn.Sequential(
#             nn.Conv2d(1, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, ceil_mode=True))
#         self.cnn2 = nn.Sequential(
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, ceil_mode=True))
#         self.cnn3 = nn.Sequential(
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, ceil_mode=True))
#         self.cnn4 = nn.Sequential(
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2, ceil_mode=True))
#         self.fc = nn.Linear(256, 64)

#     def forward(self, minibatch):
#         out = self.cnn1(minibatch)
#         out = self.cnn2(out)
#         out = self.cnn3(out)
#         out = self.cnn4(out)
#         return self.fc(out.view(minibatch.size(0), -1))


class SNAIL(nn.Module):
    """
    Arguments:
        N (int): number of classes
        K (int): k-shot. i.e. number of examples
    """
    def __init__(self, N, K, nin=65, nout=None):
        super(SNAIL, self).__init__()
        T = N * K + 1
        layer_count = math.ceil(math.log(T) / math.log(2))

        nout = nout if nout is not None else N/2
        nin = nin
        
        self.mod0 = AttentionBlock(nin, 64, 32)
        self.mod1 = TCBlock(nin+32, T, 128)
        self.mod2 = AttentionBlock(nin+32+128*layer_count, 256, 128)
        self.mod3 = TCBlock(nin+32+128*layer_count+128, T, 128)
        self.mod4 = AttentionBlock(nin+32+2*128*layer_count+128, 512, 256)
        self.out_layer = nn.Conv1d(nin+32+2*128*layer_count+128+256, nout, 1)

        # self.out_layer2 = nn.Linear(int(N/2), 1)

    def forward(self, mb):
        out = self.mod0(mb)
        out = self.mod1(out)
        out = self.mod2(out)
        out = self.mod3(out)
        out = self.mod4(out)

        out = self.out_layer(out) # HACK!!!!
        return out       
    # def forward(self, mb):
    #     out = self.mod0(mb)
    #     out = self.mod1(out)
    #     out = self.mod2(out)
    #     out = self.mod3(out)
    #     out = self.mod4(out)

    #     out = self.out_layer(out).squeeze() # HACK!!!!
    #     out = out.t()
    #     return F.leaky_relu(self.out_layer2(out).t())


"""A PyTorch implementation of the SNAIL building blocks.
This module implements the three blocks in the _A Simple Neural Attentive
Meta-Learner_ paper Mishra et al.
    URL: https://openreview.net/forum?id=B1DmUzWAW&noteId=B1DmUzWAW
The three building blocks are the following:
    - A dense block, built with causal convolutions.
    - A TC Block, built with a stack of dense blocks.
    - An attention block, similar to the attention mechanism described by
      Vaswani et al (2017).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CausalConv1d(nn.Module):
    """A 1D causal convolution layer.
    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions per step, and T is the number of steps.
    Output: (B, D_out, T), where B is the minibatch size, D_out is the number
        of dimensions in the output, and T is the number of steps.
    Arguments:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
    """

    def __init__(self, in_channels, out_channels, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = dilation
        self.causal_conv = nn.Conv1d(
            in_channels,
            out_channels,
            2,
            padding=self.padding,
            dilation=dilation)

    def forward(self, minibatch):
        return self.causal_conv(minibatch)[:, :, :-self.padding]


class DenseBlock(nn.Module):
    """Two parallel 1D causal convolution layers w/tanh and sigmoid activations
    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions of the input, and T is the number of steps.
    Output: (B, D_in+F, T), where where `B` is the minibatch size, `D_in` is the
        number of dimensions of the input, `F` is the number of filters, and `T`
        is the length of the input sequence.
    Arguments:
        in_channels (int): number of input channels
        filters (int): number of filters per channel
    """

    def __init__(self, in_channels, filters, dilation=1):
        super(DenseBlock, self).__init__()
        self.causal_conv1 = CausalConv1d(
            in_channels, filters, dilation=dilation)
        self.causal_conv2 = CausalConv1d(
            in_channels, filters, dilation=dilation)

    def forward(self, minibatch):
        tanh = F.tanh(self.causal_conv1(minibatch))
        sig = F.sigmoid(self.causal_conv2(minibatch))
        out = torch.cat([minibatch, tanh * sig], dim=1)
        return out


class TCBlock(nn.Module):
    """A stack of DenseBlocks which dilates to desired sequence length
    The TCBlock adds `ceil(log_2(seq_len))*filters` channels to the output.
    Input: (B, D_in, T), where B is the minibatch size, D_in is the number of
        dimensions of the input, and T is the number of steps.
    Output: (B, D_in+F, T), where where `B` is the minibatch size, `D_in` is the
        number of dimensions of the input, `F` is the number of filters, and `T`
        is the length of the input sequence.
    Arguments:
        in_channels (int): channels for the input
        seq_len (int): length of the sequence. The number of denseblock layers
            is log base 2 of `seq_len`.
        filters (int): number of filters per channel
    """

    def __init__(self, in_channels, seq_len, filters):
        super(TCBlock, self).__init__()
        layer_count = math.ceil(math.log(seq_len) / math.log(2))
        blocks = []
        channel_count = in_channels
        for layer in range(layer_count):
            block = DenseBlock(channel_count, filters, dilation=2**layer)
            blocks.append(block)
            channel_count += filters
        self.blocks = nn.Sequential(*blocks)
        self.channel_count = channel_count

    def forward(self, minibatch):
        return self.blocks(minibatch)


class AttentionBlock(nn.Module):
    """An attention mechanism similar to Vaswani et al (2017)
    The input of the AttentionBlock is `BxDxT` where `B` is the input
    minibatch size, `D` is the dimensions of each feature, `T` is the length of
    the sequence.
    The output of the AttentionBlock is `Bx(D+V)xT` where `V` is the size of the
    attention values.
    Arguments:
        input_dims (int): the number of dimensions (or channels) of each element
            in the input sequence
        k_size (int): the size of the attention keys
        v_size (int): the size of the attention values
    """

    def __init__(self, input_dims, k_size, v_size):
        super(AttentionBlock, self).__init__()
        self.key_layer = nn.Linear(input_dims, k_size)
        self.query_layer = nn.Linear(input_dims, k_size)
        self.value_layer = nn.Linear(input_dims, v_size)
        self.sqrt_k = math.sqrt(k_size)

    def forward(self, minibatch):
        minibatch = minibatch.permute(0, 2, 1)
        keys = self.key_layer(minibatch)
        queries = self.query_layer(minibatch)
        values = self.value_layer(minibatch)
        logits = torch.bmm(queries, keys.transpose(2, 1))
        mask = logits.data.new(logits.size(1), logits.size(2)).fill_(1).byte()
        mask = torch.triu(mask, 1)
        mask = mask.unsqueeze(0).expand_as(logits)
        logits.data.masked_fill_(mask, float('-inf'))
        probs = F.softmax(logits / self.sqrt_k, dim=2)
        read = torch.bmm(probs, values)
        return torch.cat([minibatch, read], dim=2).permute(0, 2, 1)
