import torch.nn as nn
from torchvision import models
from efficientnet_pytorch import EfficientNet

from config import NUM_CLASSES


class DenseNet121Model(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.model = models.densenet121(pretrained=True)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class EfficientNetB2Model(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b2")
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
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.model = models.resnet34(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class VGG16Model(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.model = models.vgg16(pretrained=True)
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def build_model(arch_name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    if arch_name == "resnet34":
        return ResNet34Model(num_classes)
    if arch_name == "efficientnet_b2":
        return EfficientNetB2Model(num_classes)
    if arch_name == "vgg16":
        return VGG16Model(num_classes)
    if arch_name == "densenet121":
        return DenseNet121Model(num_classes)
    raise ValueError(f"Unknown architecture name: {arch_name}")
