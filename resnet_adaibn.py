import torch
import math
import warnings
import torch.nn as nn
from torch.autograd import Variable
from modules import IBN, AdaIBN
##################################################################################
# Resnet50_AdaIBN definition
##################################################################################


__all__ = ['ResNet_AdaIBN', 'resnet50_adaibn_a', 'pretrained_in_weight']

model_urls = {
    'resnet50_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
}


class BasicBlock_AdaIBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(BasicBlock_AdaIBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        if ibn == 'a':
            self.bn1 = AdaIBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.IN = nn.InstanceNorm2d(planes, affine=True) if ibn == 'b' else None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class Bottleneck_AdaIBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None, adain='a'):
        super(Bottleneck_AdaIBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        elif ibn == 'c':
            self.bn1 = AdaIBN(planes, cfg=adain)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.IN = nn.InstanceNorm2d(planes * 4, affine=True) if ibn == 'b' else None
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
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class ResNet_AdaIBN(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 ibn_cfg=('a', 'a', 'a', None),
                 adain='a',
                 num_classes=1000):
        self.inplanes = 64
        super(ResNet_AdaIBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0], adain=adain)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ibn=ibn_cfg[1], adain=adain)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, ibn=ibn_cfg[2], adain=adain)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, ibn=ibn_cfg[3], adain=adain)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None, adain='a'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        if ibn != None:
            ibn_ = ibn.split(',')
            if len(ibn_) == 1:
                ibn_1 = ibn_[0]
                ibn_2 = ibn_[0]
                ibn_3 = ibn_[0]
            elif len(ibn_) == 2:
                ibn_1 = ibn_[0]
                ibn_2 = ibn_[1]
                ibn_3 = ibn_1
            elif len(ibn_) == 3:
                ibn_1 = ibn_[0]
                ibn_2 = ibn_[1]
                ibn_3 = ibn_[2]
        else:
            ibn_1 = None
            ibn_2 = None
            ibn_3 = None
        layers = []
        layers.append(block(self.inplanes, planes,
                            None if ibn == 'b' else ibn_2,
                            stride, downsample, adain=adain))
        self.inplanes = planes * block.expansion
        # layers.append(block(self.inplanes, planes,
        #                     None if (ibn == 'b' and i < blocks - 1) else ibn_3, adain=adain))
        if blocks == 6:
            for i in range(1, 3):
                layers.append(block(self.inplanes, planes,
                                    None if (ibn == 'b' and i < blocks - 1) else ibn_1, adain=adain))
            layers.append(block(self.inplanes, planes,
                                None if (ibn == 'b' and i < blocks - 1) else ibn_1, adain=adain))
            layers.append(block(self.inplanes, planes,
                                None if (ibn == 'b' and i < blocks - 1) else ibn_1, adain=adain))
        else:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes,
                                    None if (ibn == 'b' and i < blocks-1) else ibn_1, adain=adain))
        layers.append(block(self.inplanes, planes,
                            None if (ibn == 'b' and i < blocks - 1) else ibn_3, adain=adain))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50_adaibn_a(pretrained=False, adain='a', **kwargs):
    """Constructs a ResNet-50-AdaIBN model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_AdaIBN(block=Bottleneck_AdaIBN,
                       layers=[3, 4, 6, 3],
                       ibn_cfg=('a', 'a', 'a', None),
                       adain=adain,
                       **kwargs)
    if pretrained:
        print('load ibn params:-----------------')
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet50_ibn_a']))
        # if torch.cuda.is_available():
        #     pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnet50_ibn_a'])
        # else:
        #     pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnet50_ibn_a'],
        #                                                          map_location=torch.device('cpu'))
        # model_dict = model.state_dict()
        # print('unload params:----------------', model_dict.keys() ^ pretrained_dict.keys())
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
    return model

def pretrained_in_weight(pretrained=True):
    if pretrained == True:
        if torch.cuda.is_available():
            pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnet50_ibn_a'])
        else:
            pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnet50_ibn_a'],
                                                                 map_location=torch.device('cpu'))
        w1_0 = pretrained_dict['layer1.0.bn1.IN.weight']
        b1_0 = pretrained_dict['layer1.0.bn1.IN.bias']
        w1_1 = pretrained_dict['layer1.1.bn1.IN.weight']
        b1_1 = pretrained_dict['layer1.1.bn1.IN.bias']
        w1_2 = pretrained_dict['layer1.2.bn1.IN.weight']
        b1_2 = pretrained_dict['layer1.2.bn1.IN.bias']

        w2_0 = pretrained_dict['layer2.0.bn1.IN.weight']
        b2_0 = pretrained_dict['layer2.0.bn1.IN.bias']
        # w2_1 = pretrained_dict['layer2.1.bn1.IN.weight']
        # b2_1 = pretrained_dict['layer2.1.bn1.IN.bias']
        # w2_2 = pretrained_dict['layer2.2.bn1.IN.weight']
        # b2_2 = pretrained_dict['layer2.2.bn1.IN.bias']
        w2_3 = pretrained_dict['layer2.3.bn1.IN.weight']
        b2_3 = pretrained_dict['layer2.3.bn1.IN.bias']

        w3_0 = pretrained_dict['layer3.0.bn1.IN.weight']
        b3_0 = pretrained_dict['layer3.0.bn1.IN.bias']
        # w3_1 = pretrained_dict['layer3.1.bn1.IN.weight']
        # b3_1 = pretrained_dict['layer3.1.bn1.IN.bias']
        # w3_2 = pretrained_dict['layer3.2.bn1.IN.weight']
        # b3_2 = pretrained_dict['layer3.2.bn1.IN.bias']
        w3_3 = pretrained_dict['layer3.3.bn1.IN.weight']
        b3_3 = pretrained_dict['layer3.3.bn1.IN.bias']
        # w3_4 = pretrained_dict['layer3.4.bn1.IN.weight']
        # b3_4 = pretrained_dict['layer3.4.bn1.IN.bias']
        w3_5 = pretrained_dict['layer3.5.bn1.IN.weight']
        b3_5 = pretrained_dict['layer3.5.bn1.IN.bias']

        # w1 = torch.cat([w1_0, w1_1, w1_2])
        # b1 = torch.cat([b1_0, b1_1, b1_2])
        # w2 = torch.cat([w2_0, w2_1, w2_2, w2_3])
        # b2 = torch.cat([b2_0, b2_1, b2_2, b2_3])
        # w3 = torch.cat([w3_0, w3_1, w3_2, w3_3, w3_4, w3_5])
        # b3 = torch.cat([b3_0, b3_1, b3_2, b3_3, b3_4, b3_5])
        w1 = [w1_0, w1_1, w1_2]
        b1 = [b1_0, b1_1, b1_2]
        w2 = [w2_0, w2_3]
        b2 = [b2_0, b2_3]
        w3 = [w3_0, w3_3, w3_5]
        b3 = [b3_0, b3_3, b3_5]
    else:
        w1 = [0, 0, 0]
        b1 = [0, 0, 0]
        w2 = [0, 0]
        b2 = [0, 0]
        w3 = [0, 0, 0]
        b3 = [0, 0, 0]
    return w1, b1, w2, b2, w3, b3
if __name__ == '__main__':

    def assign_adain_params(adain_params_w, adain_params_b, model, dim=32):
        # assign the adain_params to the AdaIN layers in model
        # dim = self.output_dim
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params_b[:, :dim].contiguous()
                std = adain_params_w[:, :dim].contiguous()
                m.bias = mean.view(-1)
                m.weight = std.view(-1)
                if adain_params_w.size(1) > dim:  # Pop the parameters
                    adain_params_b = adain_params_b[:, dim:]
                    adain_params_w = adain_params_w[:, dim:]
    # def encode_with_intermediate(input, layer):
    #     result = [input]
    #     m_result = []
    #     for l in layer.modules:
    #         func = getattr(layer, l)
    #         if l
    resnet50 = resnet50_adaibn_a(False, adain='a')
    # print(resnet50)
    x_test = Variable(torch.FloatTensor(1, 64, 5, 5))
    for layer in resnet50.layer1.children():
        for name, l in list(layer.named_children()):
            print(l)
            if hasattr(l, 'AdaIN'):
                print('yes')
            # if 'AdaIN' in l.named_modules().names():
            #     print('yes')
            # out = l(x_test)


    pretrained_dict = torch.hub.load_state_dict_from_url(model_urls['resnet50_ibn_a'], map_location=torch.device('cpu'))
    print(pretrained_dict.keys())
    # print(pretrained_dict['layer1.0.bn1.IN.weight'])
    w = pretrained_dict['layer1.0.bn1.IN.weight'].unsqueeze(0)
    b = pretrained_dict['layer1.0.bn1.IN.bias'].unsqueeze(0)
    model_dict = resnet50.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print(pretrained_dict.keys())
    # model_dict.update(pretrained_dict)
    # resnet50 = resnet50.load_state_dict(model_dict)
    input = Variable(torch.FloatTensor(1, 3, 256, 256))
    assign_adain_params(w, b, resnet50.layer1[0])
    out = resnet50(input)

    print(out.shape)








