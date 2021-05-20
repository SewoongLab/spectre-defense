import torch.nn as nn
import torch.nn.functional as F
import resnet as resnet_mod
import pytorch_cifar.models.resnet as resnet
import collections

class BasicBlockNoReLU(nn.Module):
    expansion = 1

    def __init__(self, module):
        super().__init__()
        self.conv1 = module.conv1
        self.bn1 = module.bn1
        self.conv2 = module.conv2
        self.bn2 = module.bn2
        self.shortcut = module.shortcut

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out

class SequentialImageNetwork(nn.Sequential):
    def __init__(self, net=resnet.ResNet18()):
        if isinstance(net, collections.OrderedDict):
            super().__init__(net)
            return

        self.net_holder = (net,)
        i = 1
        layers = []
        name = f"layer{i}"
        while hasattr(net, name):
            layers.extend(list(getattr(net, name)))
            i += 1
            name = f"layer{i}"

        layers2 = []
        for layer in layers:
            if isinstance(layer, resnet.BasicBlock):
                layers2.append(BasicBlockNoReLU(layer))
                layers2.append(nn.ReLU())
            else:
                layers2.append(layer)

        super().__init__(
            net.conv1,
            net.bn1,
            nn.ReLU(),
            *layers2,
            nn.AvgPool2d(8 if net.in_planes == 64 else 4),
            nn.Flatten(),
            net.linear,
        )

    @property
    def net(self):
        return self.net_holder[0]


class BasicBlockSplitter(nn.Module):
    def __init__(self, block: resnet.BasicBlock, step="add"):
        super().__init__()
        self.block = block
        self.step = step

    def forward(self, x):
        if self.step == "identity":
            return x
        shortcut = self.block.shortcut(x)
        if self.step == "shortcut":
            return shortcut
        x = self.block.conv1(x)
        if self.step == "conv1":
            return x
        x = self.block.bn1(x)
        if self.step == "bn1":
            return x
        x = F.relu(x)
        if self.step == "relu1":
            return x
        x = self.block.conv2(x)
        if self.step == "conv2":
            return x
        x = self.block.bn2(x)
        if self.step == "bn2":
            return x
        x += shortcut
        if self.step == "add":
            return x
        x = F.relu(x)
        if self.step == "relu2":
            return x
        return x


class SequentialImageNetworkMod(nn.Sequential):
    def __init__(self, net=resnet_mod.resnet32()):
        if isinstance(net, collections.OrderedDict):
            super().__init__(net)
            return

        self.net_holder = (net,)
        i = 1
        layers = []
        name = f"layer{i}"
        while hasattr(net, name):
            layers.extend(list(getattr(net, name)))
            i += 1
            name = f"layer{i}"

        super().__init__(
            net.conv1,
            *layers,
            net.final_bn,
            nn.LeakyReLU(0.1),
            nn.AvgPool2d(8),
            nn.Flatten(),
            net.linear,
        )

    @property
    def net(self):
        return self.net_holder[0]
