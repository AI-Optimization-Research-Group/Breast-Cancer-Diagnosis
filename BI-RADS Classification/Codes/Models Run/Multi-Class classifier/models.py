from typing import Dict, Type

import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

from config import NUM_CLASSES


class DenseNet121Model(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model = models.densenet121(pretrained=False)  
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class EfficientNetB2Model(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model = EfficientNet.from_name("efficientnet-b2")
        in_features = self.model._fc.in_features
        self.model._fc = nn.Identity()

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


class ResNet34Model(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model = models.resnet34(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class VGG16Model(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model = models.vgg16(pretrained=False)
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

MODEL_FACTORY: Dict[str, Type[nn.Module]] = {
    "densenet121": DenseNet121Model,
    "efficientnet_b2": EfficientNetB2Model,
    "resnet34": ResNet34Model,
    "vgg16": VGG16Model,
}
