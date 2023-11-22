import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision

class DenseNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super().__init__()
        preloaded = torchvision.models.densenet121(pretrained=False)
        self.features = preloaded.features
        self.features.conv0 = nn.Conv2d(in_channels, 64, 7, 2, 3)
        self.classifier = nn.Linear(1024, num_classes, bias=True)
        del preloaded
        
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out

class DenseNet_f(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super().__init__()
        preloaded = torchvision.models.densenet121(pretrained=False)
        self.features = preloaded.features
        self.features.conv0 = nn.Conv2d(in_channels, 64, 7, 2, 3)
        self.classifier = nn.Linear(1024, num_classes, bias=True)
        del preloaded
        
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        feat = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        # print(features.size(0))
        out = self.classifier(feat)
        return feat, out

def densenet121(in_channels = 3, num_classes = 1):
    return DenseNet(num_classes = num_classes, in_channels = in_channels)

def densenet121_f(in_channels = 3, num_classes = 1):
    return DenseNet_f(num_classes = num_classes, in_channels = in_channels)