import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu6', **kwargs):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        if activation == 'relu6':
            self.activation = nn.ReLU6(inplace=True)
        else:
            self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class CNNModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(CNNModel, self).__init__()
        self.features = torchvision.models.mobilenet_v2(pretrained=pretrained).features
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=1280, out_features=num_classes),
        )
        

    def forward(self, x):
        features = self.features(x)
        features = features.mean(dim=(2,3))
        preds = self.classifier(features)
        features = F.normalize(features, p=2, dim=-1)
        return preds, features
