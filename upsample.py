import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models


class Upsample(nn.Module):
    def __init__(self, inplanes, planes):
        super(Upsample, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x, size):
        x = F.upsample_bilinear(x, size=size)
        x = self.conv1(x)
        x = self.bn(x)
        return x


class Fusion(nn.Module):
    def __init__(self, inplanes):
        super(Fusion, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=1)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        out = self.bn(self.conv(x1)) + x2
        out = self.relu(out)

        return out


class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()

        self.num_classes = num_classes

        resnet = models.resnet101(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.upsample1 = Upsample(2048, 1024)
        self.upsample2 = Upsample(1024, 512)
        self.upsample3 = Upsample(512, 64)
        self.upsample4 = Upsample(64, 64)
        self.upsample5 = Upsample(64, 32)

        self.fs1 = Fusion(1024)
        self.fs2 = Fusion(512)
        self.fs3 = Fusion(256)
        self.fs4 = Fusion(64)
        self.fs5 = Fusion(64)

        self.out0 = self._classifier(2048)
        self.out1 = self._classifier(1024)
        self.out2 = self._classifier(512)
        self.out_e = self._classifier(256)
        self.out3 = self._classifier(64)
        self.out4 = self._classifier(64)
        self.out5 = self._classifier(32)

        self.transformer = nn.Conv2d(256, 64, kernel_size=1)

    def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes/2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes/2),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(inplanes/2, self.num_classes, 1),
        )

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        out32 = self.out0(fm4)

        fsfm1 = self.fs1(fm3, self.upsample1(fm4, fm3.size()[2:]))
        out16 = self.out1(fsfm1)

        fsfm2 = self.fs2(fm2, self.upsample2(fsfm1, fm2.size()[2:]))
        out8 = self.out2(fsfm2)

        fsfm3 = self.fs4(pool_x, self.upsample3(fsfm2, pool_x.size()[2:]))
        # print(fsfm3.size())
        out4 = self.out3(fsfm3)

        fsfm4 = self.fs5(conv_x, self.upsample4(fsfm3, conv_x.size()[2:]))
        out2 = self.out4(fsfm4)

        fsfm5 = self.upsample5(fsfm4, input.size()[2:])
        out = self.out5(fsfm5)

        return out, out2, out4, out8, out16, out32
