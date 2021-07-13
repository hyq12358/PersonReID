import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)
        
        
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
    def __init__(self, num_classes, backbone='resnet50', pretrained=True):
        super(CNNModel, self).__init__()
        self.backbone_name = backbone
        if self.backbone_name == 'mobilenet_v2':
            self.backbone = torchvision.models.mobilenet_v2(pretrained=pretrained)
            in_features = 1280
        elif self.backbone_name=='resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=pretrained)
            in_features = 2048
        else:
            raise NameError("invalid backbone")
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=in_features, out_features=num_classes),
        )
        
        self.bn = nn.BatchNorm1d(in_features)
        
    def forward(self, x):
        if self.backbone_name == 'mobilenet_v2':
            features = self.backbone.features(x)
            features = features.mean(dim=(2,3))
        else:
            features = self.backbone.conv1(x)
            features = self.backbone.bn1(features)
            features = self.backbone.relu(features)
            features = self.backbone.maxpool(features)
            features = self.backbone.layer1(features)
            features = self.backbone.layer2(features)
            features = self.backbone.layer3(features)
            features = self.backbone.layer4(features)
            features = self.backbone.avgpool(features)
            features = torch.squeeze(features)

        features = self.bn(features)
        preds = self.classifier(features)
        features = F.normalize(features, p=2, dim=-1)
        return preds, features

