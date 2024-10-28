import torch
import torch.nn as nn
import timm
from torchvision import transforms
from src.config import cfg

class BirdModel(nn.Module):
    def __init__(self, model_name, pretrained, in_channels, num_classes, pool="default"):
        super().__init__()
        self.pool = pool
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        if pool == "default":
            self.backbone = timm.create_model(
                model_name=model_name, pretrained=pretrained,
                num_classes=0, in_chans=3)
        else:
            self.backbone = timm.create_model(
                model_name=model_name, pretrained=pretrained,
                num_classes=0, in_chans=3, global_pool="")

        in_features = self.backbone.num_features

        self.max_pooling = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(start_dim=1)
        )
        self.avg_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1)
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.expand(-1, 3, -1, -1)
        x = self.normalize(x)
        x = self.backbone(x)

        if self.pool == "max":
            x = self.max_pooling(x)
        elif self.pool == "avg":
            x = self.avg_pooling(x)
        elif self.pool == "both":
            x_max = self.max_pooling(x)
            x_avg = self.avg_pooling(x)
            x = x_max + x_avg

        x = self.head(x)
        return x
